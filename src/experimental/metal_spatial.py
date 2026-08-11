"""GPU (Metal/MPS) 批量 point-in-polygon 射线法判定。

城市设计中的大量判定 —— 地块归属、设施覆盖、生活圈 —— 本质都是
point-in-polygon。本模块用 PyTorch MPS 做奇偶规则射线法的批量判定：

  对每条边 (x0,y0)->(x1,y1) 与每个点 (px,py)：
    y_straddle = (y0 > py) != (y1 > py)          # 水平射线是否跨过该边
    left       = (x1-x0)*(py-y0) - (y1-y0)*(px-x0) > 0   # 点是否在边左侧
    crossing   = y_straddle & left

  奇偶规则：一个多边形全部 ring（外环 + 洞）的 crossing 总数为奇数 -> 在内部。
  叉积形式避免除法，水平边 y_straddle 恒 False，无除零问题。

精度策略与 08 模块一致：float64 里减去 bbox 中心（平移不变），float32 GPU
处理 ±10km 残差（ULP ~0.001m），叉积符号判定稳健。

数据布局：ring 扁平化为边数组 (E, 4) = [x0, y0, x1, y1]，N 点 (N, 2)。
GPU 工作量为 N x E 的逐元素广播 + 按行规约，按 chunk 分块把 (chunk x E)
中间量控制在 ~2^24 元素内。计数在 GPU 上 int32 累加，结束时一次性读回，
避免逐 chunk 回传。

注意：本路径只用逐元素运算 + 规约，不涉及 matmul，天然不受 08 模块发现的
MPS GEMM 转置 bug 影响（如未来引入基于 GEMM 的优化，需走 metal_projection
的 probe + 补偿机制）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

_DEVICE_OK: Optional[bool] = None


def mps_available() -> bool:
    """MPS 设备是否可用（结果缓存）。"""
    global _DEVICE_OK
    if _DEVICE_OK is None:
        _DEVICE_OK = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    return _DEVICE_OK


# ---------------------------------------------------------------------------
# 数据结构：ring -> 边数组
# ---------------------------------------------------------------------------


def ring_to_edges(ring: np.ndarray) -> np.ndarray:
    """单个 ring (V, 2) -> 边数组 (V, 4) [x0, y0, x1, y1]。

    环必须闭合（首尾重复）或按隐式闭合处理：相邻点相连，最后一条边
    由末点连回首点。开环的边数 = 顶点数（隐式闭合）。
    """
    ring = np.asarray(ring, dtype=np.float64)
    if ring.shape[1] != 2:
        raise ValueError(f"ring 应为 (V, 2)，got {ring.shape}")
    n = len(ring)
    if n < 3:
        raise ValueError("ring 至少需要 3 个顶点")
    closed = np.allclose(ring[0], ring[-1])
    starts = ring[:-1] if closed else ring
    ends = ring[1:] if closed else np.vstack([ring[1:], ring[:1]])
    return np.hstack([starts, ends]).astype(np.float64)


def polygon_to_edges(rings: Sequence[np.ndarray]) -> np.ndarray:
    """多边形（外环 + 洞）的全部 ring -> 一个边数组 (E, 4)。

    奇偶规则对全部 ring 统一计数，环方向无关，洞自然排除。
    """
    return np.vstack([ring_to_edges(r) for r in rings])


# ---------------------------------------------------------------------------
# 核心：射线法 crossing 计数
# ---------------------------------------------------------------------------


def raycast_counts_numpy(
    points: np.ndarray,
    edges: np.ndarray,
    chunk_elems: int = 2**24,
) -> np.ndarray:
    """CPU numpy 参考实现（与 GPU 同算法）：返回每点的 crossing 计数。

    points (N, 2), edges (E, 4)。分块控制 (chunk x E) 中间量。
    """
    points = np.asarray(points, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    n, e = len(points), len(edges)
    # 平移（float64 下完成），crossing 判定只需相对坐标
    shift = _pick_shift(points, edges)
    p = points - shift
    ed = edges - np.hstack([shift, shift])
    x0, y0, x1, y1 = ed.T
    dx, dy = x1 - x0, y1 - y0
    counts = np.zeros(n, dtype=np.int32)
    chunk = max(64, min(2**18, chunk_elems // max(e, 1)))
    for s in range(0, n, chunk):
        px = p[s : s + chunk, 0][:, None]
        py = p[s : s + chunk, 1][:, None]
        straddle = (y0[None, :] > py) != (y1[None, :] > py)
        # 叉积 cross>0 表示点在边左侧；射线法要求 px < x_inter，
        # 仅在边向上 (dy>0) 时等价；边向下时需 cross 与 dy 同号。
        cross = dx[None, :] * (py - y0[None, :]) - dy[None, :] * (px - x0[None, :])
        left = (cross > 0.0) == (dy[None, :] > 0.0)
        counts[s : s + chunk] = (straddle & left).sum(axis=1).astype(np.int32)
    return counts


def raycast_counts_mps(
    points: np.ndarray,
    edges: np.ndarray,
    chunk_elems: int = 2**24,
    device: str = "mps",
    shift: Optional[np.ndarray] = None,
) -> np.ndarray:
    """MPS 射线法：每点的 crossing 计数（int32）。

    平移 + float32 上传；计数在 GPU 累加，结束时一次性读回。
    shift 可传入以复用（归属场景多次调用避免重复 min/max）。
    """
    if not mps_available():
        raise RuntimeError("MPS 不可用")
    points = np.asarray(points, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    n, e = len(points), len(edges)
    shift = _pick_shift(points, edges) if shift is None else np.asarray(shift)
    p32 = (points - shift).astype(np.float32)
    ed32 = (edges - np.hstack([shift, shift])).astype(np.float32)

    x0 = torch.from_numpy(ed32[:, 0]).to(device)
    y0 = torch.from_numpy(ed32[:, 1]).to(device)
    y1 = torch.from_numpy(ed32[:, 3]).to(device)
    dx = torch.from_numpy(ed32[:, 2] - ed32[:, 0]).to(device)
    dy = torch.from_numpy(ed32[:, 3] - ed32[:, 1]).to(device)

    counts = torch.zeros(n, dtype=torch.int32, device=device)
    chunk = max(64, min(2**18, chunk_elems // max(e, 1)))
    pts_t = torch.from_numpy(p32).to(device)  # (N, 2) 一次上传
    for s in range(0, n, chunk):
        px = pts_t[s : s + chunk, 0][:, None]  # (C, 1)
        py = pts_t[s : s + chunk, 1][:, None]
        straddle = (y0[None, :] > py) != (y1[None, :] > py)
        cross = dx[None, :] * (py - y0[None, :]) - dy[None, :] * (px - x0[None, :])
        left = (cross > 0.0) == (dy[None, :] > 0.0)
        counts[s : s + chunk] = (straddle & left).sum(dim=1).to(torch.int32)
    return counts.cpu().numpy()


def points_in_polygon_numpy(points, rings, chunk_elems=2**24):
    """CPU numpy 参考：points 是否在 rings（外环+洞）内（奇偶规则）。"""
    edges = polygon_to_edges(rings)
    return (raycast_counts_numpy(points, edges, chunk_elems) % 2) == 1


def points_in_polygon_mps(points, rings, chunk_elems=2**24, device="mps", shift=None):
    """MPS：points 是否在 rings 定义的多边形内（奇偶规则，全 ring 统一计数）。"""
    edges = polygon_to_edges(rings)
    return (raycast_counts_mps(points, edges, chunk_elems, device, shift) % 2) == 1


def _pick_shift(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """平移中心：点与边包围盒共同中心（float64 计算）。"""
    lo = np.minimum(points.min(axis=0), edges[:, :2].min(axis=0))
    hi = np.maximum(points.max(axis=0), edges[:, 2:].max(axis=0))
    return (lo + hi) * 0.5


# ---------------------------------------------------------------------------
# 归属：N 点 vs P 个多边形
# ---------------------------------------------------------------------------


def points_in_polygons_mps(
    points: np.ndarray,
    polygons: Sequence[Sequence[np.ndarray]],
    chunk_elems: int = 2**24,
    device: str = "mps",
) -> np.ndarray:
    """归属判定：每个点属于哪个多边形（-1 = 不属于任何）。

    polygons: list，每个元素是多边形 = ring 列表（外环 + 洞）。
    策略：逐多边形整批 GPU 射线法（地块边数少时，全量 raycast 比
    bbox 预筛更快——预筛在 numpy 里做 P x N 次比较，反而成为瓶颈）。
    返回 (N,) int32 label；若多个多边形包含同一点（重叠），取第一个。
    """
    points = np.asarray(points, dtype=np.float64)
    labels = np.full(len(points), -1, dtype=np.int32)
    for pid, rings in enumerate(polygons):
        edges = polygon_to_edges(rings)
        # shift 只由边的 bbox 决定（点坐标与边同区域，残差量级一致）
        shift = (edges[:, :2].min(axis=0) + edges[:, 2:].max(axis=0)) * 0.5
        inside = points_in_polygon_mps(points, rings, chunk_elems, device, shift)
        hit = np.flatnonzero(inside)
        if len(hit):
            labels[hit] = np.where(labels[hit] < 0, pid, labels[hit])
    return labels


# ---------------------------------------------------------------------------
# Benchmark 工具（与 08 模块同一套方法论）
# ---------------------------------------------------------------------------


def timeit_median(fn, repeat: int = 7, min_total_s: float = 0.1) -> float:
    """自适应重复取中位数（秒）。小负载重复到总时长 >= min_total_s。"""
    times = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < min_total_s or len(times) < repeat:
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


@dataclass
class PipResult:
    """单次规模扫描的结果行。"""

    n_points: int
    n_edges: int
    inside_frac: float
    mps_ms: float = np.nan       # 端到端（含装配+传输+读回）
    mps_compute_ms: float = np.nan  # 仅 GPU 核函数 + 读回（tensor 已就位）
    cpu_shapely_ms: float = np.nan
    cpu_numpy_ms: float = np.nan  # 同算法 numpy 参考
    mismatch: int = -1            # 与 shapely 不一致的点数


def _synthetic_rings_polygon(edges_target: int, rng: np.random.Generator) -> List[np.ndarray]:
    """合成一个边数接近 edges_target 的复杂多边形（径向 jitter 的闭合环）。

    环严格闭合：theta 等距覆盖 [0, 2π)，r[-1] = r[0]。
    """
    v = max(8, int(np.ceil(edges_target / 4)) * 4)
    theta = np.linspace(0.0, 2.0 * np.pi, v + 1)[:-1]
    r = 5000.0 + 1500.0 * np.sin(3.0 * theta + 0.7) + 600.0 * rng.standard_normal(v)
    x = 116.30 + r * np.cos(theta) / (111000.0 * np.cos(np.deg2rad(39.95)))
    y = 39.94 + r * np.sin(theta) / 111000.0
    ring = np.column_stack([x, y])
    ring = np.vstack([ring, ring[0]])  # 显式闭合
    return [ring]


def benchmark_pip(
    sizes: Sequence[int],
    rings: Sequence[np.ndarray],
    rng: np.random.Generator,
    seed: int = 7,
    repeat: int = 5,
    min_total_s: float = 0.1,
    sample: str = "bbox",
    global_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[List[PipResult], np.ndarray, np.ndarray]:
    """规模扫描：MPS vs shapely(CPU) vs numpy 同算法(CPU)。

    采样方式（对结果影响很大——GEOS 有 bbox 快路径）：
    - 'bbox':   点密集分布在整个多边形 bbox 内（±0.001°），"边界密集"场景
    - 'global': 点在 global_bounds 指定的大范围均匀采样（多数点落在多边形外，
                "全域扫描"场景——GEOS bbox 快路径直接把外部点排除）

    返回 (results, points_cache, edges)。
    """
    import shapely

    edges = polygon_to_edges(rings)
    holes = [r for r in rings[1:]]
    poly = (
        shapely.polygons(list(rings[0]), holes=holes)
        if holes
        else shapely.polygons(list(rings[0]))
    )

    results: List[PipResult] = []
    cache: dict = {}
    for n in sizes:
        if sample == "bbox":
            lo = edges[:, :2].min(axis=0) - 0.001
            hi = edges[:, 2:].max(axis=0) + 0.001
        else:
            lo, hi = global_bounds
        pts = rng.uniform(lo, hi, size=(n, 2))
        cache[n] = pts
        inside = (raycast_counts_numpy(pts, edges) % 2) == 1
        inside_frac = float(inside.mean())

        res = PipResult(n_points=n, n_edges=len(edges), inside_frac=inside_frac)

        # CPU: shapely 向量化（生产 baseline）
        res.cpu_shapely_ms = (
            timeit_median(
                lambda: shapely.contains_xy(poly, pts[:, 0], pts[:, 1]),
                repeat,
                min_total_s,
            )
            * 1e3
        )

        # CPU: 同算法 numpy
        res.cpu_numpy_ms = (
            timeit_median(lambda: raycast_counts_numpy(pts, edges), repeat, min_total_s)
            * 1e3
        )

        # MPS：端到端
        res.mps_ms = (
            timeit_median(lambda: raycast_counts_mps(pts, edges), repeat, min_total_s)
            * 1e3
        )

        # MPS：仅核函数 + 读回（tensor 已就位）——单独测一次准备
        shift = _pick_shift(pts, edges)
        p32 = (pts - shift).astype(np.float32)
        ed32 = (edges - np.hstack([shift, shift])).astype(np.float32)
        x0 = torch.from_numpy(ed32[:, 0]).to("mps")
        y0 = torch.from_numpy(ed32[:, 1]).to("mps")
        y1 = torch.from_numpy(ed32[:, 3]).to("mps")
        dx = torch.from_numpy(ed32[:, 2] - ed32[:, 0]).to("mps")
        dy = torch.from_numpy(ed32[:, 3] - ed32[:, 1]).to("mps")
        pts_t = torch.from_numpy(p32).to("mps")
        counts = torch.zeros(len(pts), dtype=torch.int32, device="mps")
        chunk = max(64, min(2**18, 2**24 // max(len(edges), 1)))

        def _compute():
            for s in range(0, len(pts), chunk):
                px = pts_t[s : s + chunk, 0][:, None]
                py = pts_t[s : s + chunk, 1][:, None]
                straddle = (y0[None, :] > py) != (y1[None, :] > py)
                cross = dx[None, :] * (py - y0[None, :]) - dy[None, :] * (px - x0[None, :])
                left = (cross > 0.0) == (dy[None, :] > 0.0)
                counts[s : s + chunk] = (straddle & left).sum(dim=1).to(torch.int32)
            return counts.cpu().numpy()

        res.mps_compute_ms = timeit_median(_compute, repeat, min_total_s) * 1e3

        # 正确性门：与 shapely 对比 mismatch 数
        ref = shapely.contains_xy(poly, pts[:, 0], pts[:, 1])
        mine = (raycast_counts_mps(pts, edges) % 2) == 1
        res.mismatch = int(np.count_nonzero(mine != ref))

        results.append(res)
    return results, cache, edges


# 保持与 08 模块一致的导入风格
__all__ = [
    "mps_available",
    "ring_to_edges",
    "polygon_to_edges",
    "raycast_counts_numpy",
    "raycast_counts_mps",
    "points_in_polygon_numpy",
    "points_in_polygon_mps",
    "points_in_polygons_mps",
    "timeit_median",
    "PipResult",
    "benchmark_pip",
]
