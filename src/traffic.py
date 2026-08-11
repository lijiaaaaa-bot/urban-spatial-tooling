"""Traffic capacity analysis (交通承载力).

Extracted from notebooks/19-traffic-capacity.ipynb.

Three complementary screening methods for road-network capacity work:

1. **V/C ratio** (Method 1, 路段饱和度) — per segment, the peak-hour
   volume-to-capacity ratio.  ``road_capacity`` gives the design capacity
   of a segment from its class and lane count; ``vc_ratio`` computes the
   ratio; ``saturation_level`` classifies it into 畅通 (good) /
   可接受 (acceptable) / 拥堵 (congested).
2. **Intersection level-of-service** (Method 2, 交叉口服务水平) —
   ``intersection_los`` computes per-approach control delay (uniform +
   overflow terms) and maps the flow-weighted average delay to LOS A-F.
3. **Network-wide saturation** (Method 3, 网络饱和度) — ``road_saturation``
   aggregates segment V/C into network statistics and flags bottleneck
   segments (V/C > 0.85), to be read against the road-density standard
   (GB/T 51328-2018: 5.4-7.1 km/km2 for 大城市建成区).

References
----------
- GB/T 51328-2018《城市综合交通体系规划标准》— road network density
  5.4-7.1 km/km2; road level-of-service V/C bands (0.6 / 0.85)
- CJJ 37-2012《城市道路工程设计规范》表 4.3.2 — design capacity per lane
  by road class.  Notebook 19 uses the conservative parameterization
  主干路 1500 / 次干路 800 / 支路 400 pcu/h/lane (lower edge of the
  standard's 1300-1700 / 900-1300 / 600-900 ranges).
- HCM 2010 Ch.18 — control-delay model (uniform + overflow terms) and
  LOS A-F delay thresholds (10 / 20 / 35 / 55 / 80 s), as adopted in
  Chinese signalized-intersection LOS practice.

Assumption sensitivity
----------------------
Trip-generation rates, through-traffic shares, signal green splits and
the 120 s cycle are reference parameters (C-type, parameter-sensitive).
Production use must calibrate them per site and record them in
``assumptions.json``.  All thresholds are module-level constants.

Pure numpy — no GPU, no geopandas dependency.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants — reference parameters (calibrate per site)
# ---------------------------------------------------------------------------

#: design capacity per lane by road class (pcu/h/lane) — conservative
#: parameterization, lower edge of CJJ 37-2012 表4.3.2
DESIGN_CAPACITY: dict[str, float] = {
    "arterial": 1500.0,
    "secondary": 800.0,
    "branch": 400.0,
}
#: default two-way lane counts by road class
DEFAULT_LANES: dict[str, int] = {"arterial": 6, "secondary": 4, "branch": 2}
#: V/C band thresholds: < GOOD 畅通, GOOD-ACCEPTABLE 可接受, > ACCEPTABLE 拥堵
VC_GOOD: float = 0.6
VC_ACCEPTABLE: float = 0.85
#: default saturation flow per lane (pcu/h) for intersection approaches
SAT_FLOW: float = 1800.0
#: default signal cycle length (s)
CYCLE: float = 120.0
#: effective green share per approach by road class (signal priority)
GREEN_RATIO: dict[str, float] = {
    "arterial": 0.45,
    "secondary": 0.30,
    "branch": 0.15,
}
#: LOS A-F upper delay bounds (s); ``delay <= bound`` -> that letter
LOS_BANDS: tuple[float, ...] = (10.0, 20.0, 35.0, 55.0, 80.0)
LOS_LETTERS: str = "ABCDEF"


def road_capacity(road_class: str, lanes: int | None = None) -> float:
    """Design capacity (pcu/h) of a road segment by class.

    ``capacity = lanes * per-lane design capacity``.  Per-lane values are
    ``DESIGN_CAPACITY`` (主干路 1500 / 次干路 800 / 支路 400 pcu/h/lane);
    ``lanes`` defaults to ``DEFAULT_LANES`` (6 / 4 / 2 two-way).
    """
    if road_class not in DESIGN_CAPACITY:
        raise ValueError(
            f"unknown road class {road_class!r}; choose from {sorted(DESIGN_CAPACITY)}")
    n = DEFAULT_LANES[road_class] if lanes is None else lanes
    return float(n * DESIGN_CAPACITY[road_class])


def vc_ratio(volume: float, capacity: float) -> float:
    """Volume-to-capacity ratio (饱和度) of a road segment."""
    return volume / capacity


def saturation_level(vc: float) -> str:
    """V/C band: ``good`` (< 0.6), ``acceptable`` (0.6-0.85), ``congested`` (> 0.85)."""
    if vc < VC_GOOD:
        return "good"
    if vc <= VC_ACCEPTABLE:
        return "acceptable"
    return "congested"


def approach_delay(volume: float, capacity: float, green_ratio: float,
                   cycle: float = CYCLE, period_h: float = 1.0) -> float:
    """Control delay of one signalized approach (HCM uniform + overflow).

    .. math:: d = \\frac{0.5 C (1-\\lambda)^2}{1 - \\min(1,x)\\lambda}
        + 900 T \\left[(x-1) + \\sqrt{(x-1)^2 + \\frac{4x}{c T}}\\right]

    with ``x = volume / capacity``, ``lambda = green_ratio``, pretimed
    calibration factor 0.5 and no upstream metering (I = 1).  The
    overflow term is 0 below saturation (x <= 1) and grows quickly
    above it — vehicles queue across multiple cycles.
    """
    x = volume / capacity
    lam = min(max(green_ratio, 1e-6), 0.99)
    xc = min(x, 0.999)
    d1 = 0.5 * cycle * (1.0 - lam) ** 2 / (1.0 - lam * xc)
    d2 = 900.0 * period_h * (
        (x - 1.0) + np.sqrt(max((x - 1.0) ** 2 + 4.0 * x / max(capacity * period_h, 1e-9), 0.0)))
    return float(d1 + d2)


def los_from_delay(delay: float) -> str:
    """HCM LOS letter from average control delay (s): A <=10, B <=20, C <=35,
    D <=55, E <=80, F > 80."""
    for letter, band in zip(LOS_LETTERS, LOS_BANDS):
        if delay <= band:
            return letter
    return "F"


def intersection_los(volumes: np.ndarray, lanes: np.ndarray,
                     green_ratios: np.ndarray | None = None,
                     cycle: float = CYCLE, lost_time: float = 12.0,
                     sat_flow: float = SAT_FLOW, period_h: float = 1.0) -> dict:
    """Level-of-service of a signalized intersection (Method 2).

    Parameters
    ----------
    volumes : array of per-approach peak-hour volumes (pcu/h)
    lanes : array of per-approach lane counts
    green_ratios : per-approach effective green share ``g/C``; defaults to
        an equal 4-phase split ``(cycle - lost_time) / (cycle * n)``
    cycle : signal cycle length (s)
    lost_time : total lost time per cycle (s), used for the default split
    sat_flow : saturation flow per lane (pcu/h)
    period_h : analysis period (h), 1.0 = peak hour

    Returns
    -------
    dict with ``x`` (approach v/c), ``capacities``, ``delays`` (s),
    ``per_approach_los`` (letters), and the flow-weighted intersection
    ``delay_s`` / ``los``.
    """
    volumes = np.asarray(volumes, dtype=float)
    lanes = np.asarray(lanes, dtype=float)
    if green_ratios is None:
        n = len(volumes)
        lam = np.full(n, max((cycle - lost_time) / (cycle * n), 1e-3))
    else:
        lam = np.asarray(green_ratios, dtype=float)
    capacities = lanes * sat_flow * lam
    delays = np.array([approach_delay(v, c, g, cycle, period_h)
                       for v, c, g in zip(volumes, capacities, lam)])
    x = volumes / np.maximum(capacities, 1e-9)
    weighted = float(np.sum(volumes * delays) / max(np.sum(volumes), 1e-9))
    return {
        "x": x,
        "capacities": capacities,
        "delays": delays,
        "per_approach_los": [los_from_delay(d) for d in delays],
        "delay_s": weighted,
        "los": los_from_delay(weighted),
    }


def road_saturation(volumes: np.ndarray, capacities: np.ndarray,
                    weights: np.ndarray | None = None) -> dict:
    """Network-wide saturation statistics (Method 3).

    Parameters
    ----------
    volumes : per-segment peak-hour volumes (pcu/h)
    capacities : per-segment design capacities (pcu/h)
    weights : per-segment weights for the weighted mean (default: the
        capacities themselves — exposure-weighted, heavier roads count
        more)

    Returns
    -------
    dict with ``vc`` (array), ``mean_vc``, ``weighted_mean_vc``,
    ``max_vc``, and the bottleneck set — segments with ``vc > 0.85``:
    ``bottleneck_indices``, ``n_bottleneck``, ``bottleneck_share``.
    """
    vc = np.asarray(volumes, dtype=float) / np.asarray(capacities, dtype=float)
    if weights is None:
        weights = np.asarray(capacities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean_vc = float(np.mean(vc))
    weighted_mean = float(np.sum(vc * weights) / np.sum(weights))
    bottleneck = vc > VC_ACCEPTABLE
    idx = np.flatnonzero(bottleneck)
    return {
        "vc": vc,
        "mean_vc": mean_vc,
        "weighted_mean_vc": weighted_mean,
        "max_vc": float(np.max(vc)),
        "bottleneck_indices": idx,
        "n_bottleneck": int(len(idx)),
        "bottleneck_share": float(len(idx) / max(len(vc), 1)),
    }
