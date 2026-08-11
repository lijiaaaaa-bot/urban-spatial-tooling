"""
MPS-accelerated batch coordinate projection (local affine approximation).

EPSG:4326 (WGS84 lon/lat) -> EPSG:4548 (CGCS2000 / 3-degree Gauss-Kruger CM 117E)
for the Beijing / Haidian region.

The Gauss-Kruger projection is locally near-affine: over a ~10 km urban region
the deviation from a linear map is on the order of a metre (dominated by the
meridian-convergence cross terms), and it shrinks to millimetres if the map is
promoted to a quadratic polynomial — still one batched matmul on the GPU, just
with a 6-column basis.  The pipeline is:

  1. fit_projection_model(): sample a regular grid over the data bounding box,
     transform it exactly with pyproj (float64), and least-squares fit either a
     linear (3x3 homogeneous) or quadratic (6x3) map;
  2. project_points_mps(): apply the map to every point in one batched
     ``torch.matmul`` on the MPS device (Metal Performance Shaders under the
     hood).  Coordinates are origin-shifted before the float32 matmul and the
     target origin offset (x ~ 4.4e5 m, y ~ 4.4e6 m) is added back in float64,
     so float32 rounding stays at millimetre level despite the large northing
     magnitude;
  3. project_points_pyproj(): vectorized exact transform (ground truth and the
     CPU baseline actually used in production via ``geopandas.to_crs``).

Workaround for an upstream PyTorch MPS bug
------------------------------------------
torch 2.12.0 and 2.13.0-dev on Apple silicon (observed on M5 Max) intermittently
mis-compute the batched matmul ``A @ B`` so that the right operand is read as if
it were stored transposed (column-major).  The failure is deterministic within
a process but flips between processes / over time (GPU-wide state, not
code-deterministic), and it hits every GEMM path (``@``, ``torch.mm``, ``bmm``,
``einsum``) at every size we tested.  When the bug is active, ``A @ B.t()`` is
correct, so this module probes the GPU state once per process with a tiny known
matmul, and transparently applies the transposed form when required.  The first
real batch is additionally verified against a CPU reference and the
compensation flipped if the probe was unlucky.  No compiled C++ extensions:
pure NumPy + PyTorch (MPS backend).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pyproj

# EPSG:4548 — CGCS2000 / 3-degree Gauss-Kruger CM 117E (metres)
# EPSG:4326 — WGS84 (degrees)
CRS_4326 = "EPSG:4326"
CRS_4548 = "EPSG:4548"

try:  # torch is required only for the MPS path
    import torch

    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ProjectionModel:
    """Best-fit map EPSG:4326 -> EPSG:4548 over a small region.

    ``M`` is a (basis, 2) float64 coefficient matrix, CPU:

    * linear (``degree=1``): ``[x, y] = [lon, lat, 1] @ M``
    * quadratic (``degree=2``): ``[x, y] = [lon^2, lon*lat, lat^2, lon, lat, 1] @ M``

    ``M_local`` is the same map in origin-shifted coordinates
    (``lon - lon0``, ``lat - lat0``), whose constant term is ~cm scale, so it
    can be applied in float32 without losing precision at the ~4.4e6 m northing
    magnitude.
    """

    M: np.ndarray          # (3 or 6, 2) float64 coefficients (full map)
    M_local: np.ndarray    # (3 or 6, 2) float64 coefficients (shifted coords)
    degree: int            # 1 (linear) or 2 (quadratic)
    lon0: float            # source origin (bbox centre, degrees)
    lat0: float
    x0: float              # target origin (metres, float64 exact from pyproj)
    y0: float
    fit_rmse_m: float      # fit error over the data points, metres
    fit_max_err_m: float
    n_fit_samples: int


def _get_transformer(src_crs: str = CRS_4326, dst_crs: str = CRS_4548) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _walk_coordinates(geom: Any) -> Iterable[tuple[float, float]]:
    """Yield every (lon, lat) coordinate pair in a GeoJSON geometry."""
    gtype = geom.get("type")
    if gtype == "Point":
        yield geom["coordinates"]
    elif gtype == "MultiPoint":
        yield from geom["coordinates"]
    elif gtype in ("LineString",):
        yield from geom["coordinates"]
    elif gtype == "MultiLineString":
        for part in geom["coordinates"]:
            yield from part
    elif gtype == "Polygon":
        for ring in geom["coordinates"]:
            yield from ring
    elif gtype == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                yield from ring
    elif gtype == "GeometryCollection":
        for g in geom.get("geometries", []):
            yield from _walk_coordinates(g)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported geometry type: {gtype!r}")


def extract_coordinates(source: str | dict | list) -> np.ndarray:
    """Extract every (lon, lat) point from a GeoJSON FeatureCollection.

    Parameters
    ----------
    source : str | dict | list
        Path to a .geojson file, a parsed GeoJSON object, or a list of features.

    Returns
    -------
    np.ndarray of shape (N, 2), dtype float64, in EPSG:4326 order (lon, lat).
    """
    if isinstance(source, (str, bytes)):
        with open(source) as fh:
            gj = json.load(fh)
    else:
        gj = source

    if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
        features = gj["features"]
    elif isinstance(gj, dict) and gj.get("type") == "Feature":
        features = [gj]
    else:
        features = gj

    pts = [xy for f in features for xy in _walk_coordinates(f["geometry"])]
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected 2D coordinates, got shape %r" % (arr.shape,))
    return arr


def sample_points_in_bbox(lons: np.ndarray, lats: np.ndarray, n: int, rng=None) -> np.ndarray:
    """Draw ``n`` uniform points from the bounding box of the given points.

    Used to synthesise urban-scale workloads (a city-wide GeoJSON easily holds
    millions of coordinates) from the small real boundary.
    """
    rng = np.random.default_rng() if rng is None else rng
    lo_lon, hi_lon = lons.min(), lons.max()
    lo_lat, hi_lat = lats.min(), lats.max()
    return np.column_stack(
        [rng.uniform(lo_lon, hi_lon, n), rng.uniform(lo_lat, hi_lat, n)]
    )


# ---------------------------------------------------------------------------
# Model fit (CPU, float64)
# ---------------------------------------------------------------------------


def _basis_columns(lon: np.ndarray, lat: np.ndarray, degree: int) -> np.ndarray:
    """Basis columns for the polynomial map, evaluated at (lon, lat)."""
    if degree == 1:
        return np.column_stack([lon, lat, np.ones_like(lon)])
    if degree == 2:
        return np.column_stack(
            [lon**2, lon * lat, lat**2, lon, lat, np.ones_like(lon)]
        )
    raise ValueError("degree must be 1 or 2")


def _shifted_basis_f32(coords: np.ndarray, model: ProjectionModel) -> np.ndarray:
    """Float32 basis columns in origin-shifted coordinates (GPU input buffer)."""
    lon = coords[:, 0] - model.lon0
    lat = coords[:, 1] - model.lat0
    n = coords.shape[0]
    d = model.degree
    out = np.empty((n, 3 if d == 1 else 6), dtype=np.float32)
    if d == 1:
        out[:, 0] = lon
        out[:, 1] = lat
        out[:, 2] = 1.0
    else:
        out[:, 0] = lon * lon
        out[:, 1] = lon * lat
        out[:, 2] = lat * lat
        out[:, 3] = lon
        out[:, 4] = lat
        out[:, 5] = 1.0
    return out


def fit_projection_model(
    coords: np.ndarray,
    transformer: Optional[pyproj.Transformer] = None,
    grid: int = 9,
    degree: int = 1,
    src_crs: str = CRS_4326,
    dst_crs: str = CRS_4548,
) -> ProjectionModel:
    """Fit the best polynomial map (linear or quadratic) from exact pyproj samples.

    A ``grid x grid`` regular lattice over the bounding box of *coords* is
    transformed exactly (float64) and the map is fitted by least squares.  The
    origin for the shifted form is the bbox centre, transformed exactly.

    Parameters
    ----------
    coords : np.ndarray (N, 2)
        Source coordinates in (lon, lat).
    transformer : pyproj.Transformer, optional
        Reusable transformer (created from *src_crs*/*dst_crs* if omitted).
    grid : int
        Lattice density for the fit samples (default 9 -> 81 samples).
    degree : 1 or 2
        Linear (affine) or quadratic map.
    """
    if transformer is None:
        transformer = _get_transformer(src_crs, dst_crs)
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")

    lon0 = float(coords[:, 0].mean())
    lat0 = float(coords[:, 1].mean())
    x0, y0 = transformer.transform(lon0, lat0)

    lon_s = np.linspace(coords[:, 0].min(), coords[:, 0].max(), grid)
    lat_s = np.linspace(coords[:, 1].min(), coords[:, 1].max(), grid)
    LON, LAT = np.meshgrid(lon_s, lat_s)
    X, Y = transformer.transform(LON.ravel(), LAT.ravel())

    # Full map on absolute coordinates
    B = _basis_columns(LON.ravel(), LAT.ravel(), degree)
    M, *_ = np.linalg.lstsq(B, np.column_stack([X, Y]), rcond=None)  # (basis, 2)

    # Shifted map: refit on origin-shifted samples (equivalent to the full map
    # for degree 1; exact expansion for degree 2).  The constant term of the
    # shifted form is the fit residual at the origin (~cm), so float32 can
    # represent it exactly.
    Bs = _basis_columns(LON.ravel() - lon0, LAT.ravel() - lat0, degree)
    M_local, *_ = np.linalg.lstsq(Bs, np.column_stack([X - x0, Y - y0]), rcond=None)

    approx = apply_projection_numpy(coords, M, degree)
    exact = project_points_pyproj(coords, transformer)
    err = np.linalg.norm(approx - exact, axis=1)
    return ProjectionModel(
        M=M,
        M_local=M_local,
        degree=degree,
        lon0=lon0,
        lat0=lat0,
        x0=float(x0),
        y0=float(y0),
        fit_rmse_m=float(np.sqrt(np.mean(err**2))),
        fit_max_err_m=float(err.max()),
        n_fit_samples=int(LON.size),
    )


def apply_projection_numpy(coords: np.ndarray, M: np.ndarray, degree: int) -> np.ndarray:
    """Apply a polynomial map (M, shape (3|6, 2)) to (N, 2) coords (CPU, float64)."""
    B = _basis_columns(coords[:, 0], coords[:, 1], degree)
    return B @ M


# ---------------------------------------------------------------------------
# Projection backends
# ---------------------------------------------------------------------------


def project_points_pyproj(
    coords: np.ndarray,
    transformer: Optional[pyproj.Transformer] = None,
    src_crs: str = CRS_4326,
    dst_crs: str = CRS_4548,
) -> np.ndarray:
    """Exact vectorized transform (ground truth / CPU baseline).

    This is the same engine ``geopandas.to_crs`` uses; single-threaded C code.
    """
    if transformer is None:
        transformer = _get_transformer(src_crs, dst_crs)
    x, y = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([x, y])


# --- MPS GEMM workaround ----------------------------------------------------

_MPS_GEMM_TRANSPOSED: Optional[bool] = None  # None=unknown, False=healthy, True=buggy
_MPS_VERIFIED: bool = False


def _mps_gemm(src: "torch.Tensor", mat: "torch.Tensor") -> "torch.Tensor":
    """Batched ``src @ mat`` on MPS with automatic workaround for the upstream
    transposed-RHS bug (see module docstring)."""
    global _MPS_GEMM_TRANSPOSED
    if _MPS_GEMM_TRANSPOSED is None:
        _MPS_GEMM_TRANSPOSED = _probe_mps_gemm_state()
    if _MPS_GEMM_TRANSPOSED:
        mat = mat.t()
    return src @ mat


def _probe_mps_gemm_state() -> bool:
    """Return True if the MPS backend reads the right matmul operand transposed.

    Runs one tiny (16, 3) @ (3, 3) matmul on MPS and compares against a CPU
    reference (a few microseconds, once per process).
    """
    rng = np.random.default_rng(0)
    a = torch.from_numpy(rng.standard_normal((16, 3)).astype(np.float32)).to("mps")
    m = torch.from_numpy(rng.standard_normal((3, 3)).astype(np.float32)).to("mps")
    expected = (a.cpu() @ m.cpu()).numpy()
    got = (a @ m).cpu().numpy()
    return float(np.abs(got - expected).max()) > 1e-3


def _verify_and_fix_first_call(coords, model, out_local) -> np.ndarray:
    """Cross-check the first real MPS projection against a CPU reference and
    flip the compensation flag if the probe was unlucky.

    ``out_local`` is the float32 local-coordinate output of the MPS matmul
    (before adding the target origin).  Reference uses the shifted basis, so
    both sides are in the same frame.  Runs once per process on the first
    batch (sample of <= 2000 points).
    """
    global _MPS_GEMM_TRANSPOSED, _MPS_VERIFIED
    if _MPS_VERIFIED:
        return out_local
    _MPS_VERIFIED = True

    k = min(coords.shape[0], 2000)
    ref = _shifted_basis_f32(coords[:k], model) @ np.asarray(model.M_local, dtype=np.float32)
    if np.abs(out_local[:k] - ref).max() > 1e-2:
        # probe state was wrong for the real workload: flip and recompute
        _MPS_GEMM_TRANSPOSED = not _MPS_GEMM_TRANSPOSED
        return _project_mps_core(coords, model, engine="matmul")
    return out_local


# --- MPS projection core -----------------------------------------------------


def _project_mps_core(coords: np.ndarray, model: ProjectionModel, engine: str = "matmul") -> np.ndarray:
    """Project *coords* with the shifted polynomial map on MPS; returns float32
    local coordinates (before adding the target origin)."""
    basis = _shifted_basis_f32(coords, model)
    M32 = np.asarray(model.M_local, dtype=np.float32)

    if engine == "elementwise":
        # deterministic fallback: broadcast multiply-add on MPS, no GEMM kernels
        b = torch.from_numpy(basis).to("mps")
        m = torch.from_numpy(M32).to("mps")
        out = b[:, :1] * m[0]
        for i in range(1, basis.shape[1]):
            out = out + b[:, i : i + 1] * m[i]
        return out.cpu().numpy()

    src = torch.from_numpy(basis).to("mps")
    mat = torch.from_numpy(M32).to("mps")
    return (_mps_gemm(src, mat)).cpu().numpy()


def project_points_mps(
    coords: np.ndarray,
    model: ProjectionModel,
    device: str = "mps",
    engine: str = "matmul",
    plain_float32: bool = False,
) -> np.ndarray:
    """Batched polynomial projection on the MPS (Metal) device.

    Parameters
    ----------
    coords : np.ndarray (N, 2)
        Source coordinates (lon, lat).
    model : ProjectionModel
        Fit from :func:`fit_projection_model`.
    device : str
        torch device, default ``"mps"``.  Any other value runs the exact
        float64 CPU path (``apply_projection_numpy`` on the shifted map) —
        useful as a non-GPU reference.
    engine : "matmul" | "elementwise"
        ``matmul`` uses the batched GEMM (MPSMatrixMultiplication-style) with
        the upstream-bug workaround; ``elementwise`` avoids GEMM entirely
        (deterministic) and exists as a comparison baseline.
    plain_float32 : bool
        If True, apply the full map in raw float32 without origin shifting —
        included to quantify the precision loss at ~4.4e6 m northing.

    Returns
    -------
    np.ndarray (N, 2) float64 metres in EPSG:4548.
    """
    if not _TORCH_OK:  # pragma: no cover
        raise RuntimeError("torch is required for the MPS projection path")

    if device != "mps":
        if plain_float32:
            raise ValueError("plain_float32 is only meaningful on the MPS device")
        # exact float64 CPU reference of the *shifted* map
        B = _basis_columns(coords[:, 0] - model.lon0, coords[:, 1] - model.lat0, model.degree)
        return B @ model.M_local + np.array([model.x0, model.y0])

    if plain_float32:
        src = torch.from_numpy(
            _basis_columns(coords[:, 0], coords[:, 1], model.degree).astype(np.float32)
        ).to("mps")
        mat = torch.from_numpy(np.asarray(model.M, dtype=np.float32)).to("mps")
        out = (_mps_gemm(src, mat)).cpu().numpy()[:, :2].astype(np.float64)
        return out

    out = _project_mps_core(coords, model, engine=engine)
    if engine == "matmul":
        out = _verify_and_fix_first_call(coords, model, out)
    return out.astype(np.float64) + np.array([model.x0, model.y0])


def _torch_device_ok(device: str = "mps") -> bool:
    return _TORCH_OK and device == "mps" and torch.backends.mps.is_available()


# ---------------------------------------------------------------------------
# Extension probe: shoelace area on MPS
# ---------------------------------------------------------------------------


def shoelace_area_mps(
    xy: np.ndarray,
    ring_starts: np.ndarray,
    shift: tuple[float, float] = (0.0, 0.0),
    device: str = "mps",
) -> np.ndarray:
    """Signed area per ring by the shoelace formula, computed on MPS.

    The shoelace cross-sum is invariant under translation, so coordinates are
    shifted by *shift* (e.g. the projected origin) in float64 before being
    cast to float32 — required because MPS supports no float64 and raw float32
    at the ~4.4e6 m northing magnitude would cost ~0.5% area precision.

    Parameters
    ----------
    xy : np.ndarray (N, 2) float64
        Ring vertices, concatenated, closed (first == last).
    ring_starts : np.ndarray (K,)
        Index into *xy* where each ring begins; the last ring ends at N.
    shift : (float, float)
        Origin subtracted before the float32 cast (default none).

    Returns
    -------
    np.ndarray (K,) absolute area in m^2 (rings are typically CCW/CW mixed).
    """
    if not _TORCH_OK:  # pragma: no cover
        raise RuntimeError("torch is required for the MPS area path")

    xy32 = np.asarray(xy, dtype=np.float64) - np.asarray(shift, dtype=np.float64)
    x = torch.from_numpy(xy32[:, 0].astype(np.float32)).to(device)
    y = torch.from_numpy(xy32[:, 1].astype(np.float32)).to(device)

    # shoelace: sum over ring i of (x_j * y_{j+1} - x_{j+1} * y_j) / 2
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    starts = torch.from_numpy(ring_starts.astype(np.int64)).to(device)
    ends = torch.zeros_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = xy.shape[0] - 1

    cs = torch.cat([torch.zeros(1, device=device), torch.cumsum(cross, 0)])
    area2 = cs[ends] - cs[starts]  # 2 x signed area per ring (ends inclusive)
    return (area2.abs() * 0.5).cpu().numpy()


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def timeit_median(fn, repeat: int = 7, min_total_s: float = 0.1) -> float:
    """Median wall time of *fn* in seconds, with adaptive repetitions.

    Small workloads are repeated enough times that the measured total spans at
    least ``min_total_s`` so the timer resolution does not dominate.
    """
    import time

    t0 = time.perf_counter()
    fn()
    per_call = time.perf_counter() - t0
    reps = max(repeat, int(min_total_s / max(per_call, 1e-9)) + 1)
    reps = min(reps, 2000)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return float(times[len(times) // 2])


def benchmark_projection(
    sizes: Iterable[int],
    coords_base: np.ndarray,
    model: ProjectionModel,
    transformer: pyproj.Transformer,
    repeat: int = 7,
    rng=None,
) -> "np.ndarray":
    """Benchmark CPU (pyproj vectorized) vs MPS (end-to-end) vs MPS (compute-only).

    Returns a structured array with one row per size: ``n, cpu_ms, mps_ms,
    mps_compute_ms, speedup, speedup_compute, mps_max_err_m`` where
    ``mps_max_err_m`` is the MPS result's max deviation from the exact pyproj
    transform for that batch (correctness gate).
    """
    rows = []
    for n in sizes:
        coords = sample_points_in_bbox(coords_base[:, 0], coords_base[:, 1], n, rng)
        exact = project_points_pyproj(coords, transformer)  # correctness reference

        cpu_ms = timeit_median(
            lambda c=coords: project_points_pyproj(c, transformer), repeat=repeat
        ) * 1e3

        # GPU-only: tensors pre-built once, timer covers kernel + readback
        basis = _shifted_basis_f32(coords, model)
        M32 = np.asarray(model.M_local, dtype=np.float32)
        src_t = torch.from_numpy(basis).to("mps")
        mat_t = torch.from_numpy(M32).to("mps")

        def mps_compute_only():
            return (_mps_gemm(src_t, mat_t)).cpu().numpy()

        def mps_run(c=coords, m=model):
            return project_points_mps(c, m, device="mps")

        mps_compute_ms = timeit_median(mps_compute_only, repeat=repeat) * 1e3
        mps_ms = timeit_median(mps_run, repeat=repeat) * 1e3

        mps_out = project_points_mps(coords, model, device="mps")
        mps_err = float(np.linalg.norm(mps_out - exact, axis=1).max())

        rows.append(
            (n, cpu_ms, mps_ms, mps_compute_ms, cpu_ms / mps_ms, cpu_ms / mps_compute_ms, mps_err)
        )

    dt = np.dtype(
        [
            ("n", "i8"),
            ("cpu_ms", "f8"),
            ("mps_ms", "f8"),
            ("mps_compute_ms", "f8"),
            ("speedup", "f8"),
            ("speedup_compute", "f8"),
            ("mps_max_err_m", "f8"),
        ]
    )
    return np.array(rows, dtype=dt)
