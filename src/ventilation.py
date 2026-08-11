"""Ventilation / wind corridor analysis (通风分析).

Extracted from notebooks/17-ventilation-analysis.ipynb.

Three complementary screening methods for urban wind-environment work:

1. **Empirical wake model** (Method 1, 风压模型) — per-street-segment
   ventilation ratio ``V_reduced / V_ref = 1 - C_d * H / W`` from building
   height ``H`` and street width ``W``.  Instant, ties directly to street
   design (红线宽度); saturates once ``H/W > 1/C_d``.
2. **Frontal area density** (Method 2, 迎风面积密度) — per-cell
   ``lambda_f`` and connected through-corridor identification.  This is the
   metric behind 通风廊道管控 in Beijing 控规/城市设计 guidance:
   ``lambda_f < 0.3`` open (corridor candidate), ``0.3-0.5`` medium,
   ``> 0.5`` poor.
3. **Source-panel potential flow** (Method 3, 面元法, in the notebook) —
   2D inviscid CFD capturing Venturi speed-up in gaps and corner/wake
   effects; validated on a cylinder (max surface speed 1.944 vs analytic 2.0
   at 32 panels).  The panel machinery lives in the notebook; this module
   exposes the planning-standard screening functions.

References
----------
- 《北京城市总体规划(2016年—2035年)》: 5 条一级通风廊道 (>= 500 m),
  11 条二级通风廊道 (>= 200 m)
- GB 50009-2012《建筑结构荷载规范》: Beijing basic wind pressure
  ``w0 = 0.45 kN/m2`` (50-year return)
- Beijing wind climate: winter NW prevailing (heating season, pollution
  dispersion), summer SE prevailing; calm frequency ~30%
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wind_speed_reduction",
    "ventilation_ratio",
    "blowing_dir",
    "frontal_width",
    "frontal_area_density",
    "classify_ventilation",
    "ventilation_corridors",
    "windward_leeward",
]

#: drag coefficient of the empirical wake model (urban blocks, ~1.2)
C_D = 1.2
#: floor of the ventilation ratio — the wake never goes negative
RATIO_FLOOR = 0.05
#: frontal-area-density screening thresholds (planning standard)
GOOD_LF = 0.30
POOR_LF = 0.50
#: default grid cell size (m) for frontal area density
CELL = 75.0


def wind_speed_reduction(H: float, W: float, C_d: float = C_D,
                         V_ref: float = 1.0, floor: float = RATIO_FLOOR) -> float:
    """Reduced wind speed behind a building wake (empirical model).

    Parameters
    ----------
    H : float
        Building height (m).
    W : float
        Street width / right-of-way (m).
    C_d : float
        Drag coefficient of the empirical model.
    V_ref : float
        Reference (freestream) wind speed.
    floor : float
        Lower clamp — the ratio never goes below ``floor`` (wake fully
        separated once ``H/W > 1/C_d``).

    Returns
    -------
    float
        ``V_ref * max(floor, 1 - C_d * H / W)``.
    """
    return V_ref * max(floor, 1.0 - C_d * H / W)


def ventilation_ratio(H_mean: float, W_street: float, C_d: float = C_D,
                      floor: float = RATIO_FLOOR) -> float:
    """Ventilation ratio ``V_reduced / V_ref`` of a street segment.

    Same model as :func:`wind_speed_reduction` with ``V_ref = 1``; screening
    classes: >= 0.6 good, 0.35-0.6 medium, < 0.35 poor.
    """
    return max(floor, 1.0 - C_d * H_mean / W_street)


def blowing_dir(azimuth_from_north: float) -> np.ndarray:
    """Unit vector of the blowing direction for a wind **FROM** azimuth.

    Parameters
    ----------
    azimuth_from_north : float
        Wind azimuth in degrees from north, clockwise (meteorological
        convention: 315 = NW, 135 = SE, 0 = north wind).

    Returns
    -------
    np.ndarray
        Unit 2-vector the wind blows toward (``(-sin a, -cos a)``).
    """
    a = np.radians(azimuth_from_north)
    return np.array([-np.sin(a), -np.cos(a)])


def frontal_width(b, bhat: np.ndarray) -> float:
    """Silhouette width of a footprint perpendicular to the wind.

    ``w_x |bhat_y| + w_y |bhat_x|`` for an axis-aligned footprint of
    size ``(w_x, w_y)`` — the "wall" the wind hits.

    Parameters
    ----------
    b : shapely geometry with ``bounds`` (Polygon/box)
        Building footprint.
    bhat : np.ndarray
        Unit blowing-direction vector (see :func:`blowing_dir`).
    """
    minx, miny, maxx, maxy = b.bounds
    wx, wy = maxx - minx, maxy - miny
    return wx * abs(bhat[1]) + wy * abs(bhat[0])


def frontal_area_density(buildings, heights, wind_from_az: float,
                         cell: float = CELL,
                         domain=(0.0, 0.0, 900.0, 900.0)):
    """Building frontal area density ``lambda_f`` over a cell grid.

    Each building's frontal area ``(frontal_width * height)`` is attributed
    to the cell containing its centroid:

    .. math:: \\lambda_f = \\sum_i (\\text{frontal width}_i \\cdot h_i) /
        (\\text{cell area})

    Parameters
    ----------
    buildings : iterable of shapely geometries
        Building footprints (Polygon/box).
    heights : iterable of float
        Building heights (m), same order as ``buildings``.
    wind_from_az : float
        Wind azimuth the flow comes from (see :func:`blowing_dir`).
    cell : float
        Grid cell size (m).
    domain : tuple
        ``(minx, miny, maxx, maxy)`` of the analysis domain.

    Returns
    -------
    lf : np.ndarray (ny, nx)
        Frontal area density per cell.
    xs, ys : np.ndarray
        Cell-centre coordinates (m).
    """
    bhat = blowing_dir(wind_from_az)
    minx, miny, maxx, maxy = domain
    nx = int(round((maxx - minx) / cell))
    ny = int(round((maxy - miny) / cell))
    lf = np.zeros((ny, nx))
    for b, h in zip(buildings, heights):
        cx = (b.bounds[0] + b.bounds[2]) / 2.0
        cy = (b.bounds[1] + b.bounds[3]) / 2.0
        ci = int((cx - minx) // cell)
        cj = int((cy - miny) // cell)
        if 0 <= cj < ny and 0 <= ci < nx:
            lf[cj, ci] += frontal_width(b, bhat) * h / (cell * cell)
    xs = minx + cell * (np.arange(nx) + 0.5)
    ys = miny + cell * (np.arange(ny) + 0.5)
    return lf, xs, ys


def classify_ventilation(lf: np.ndarray, good: float = GOOD_LF,
                         poor: float = POOR_LF) -> np.ndarray:
    """Classify a lambda_f field: 0 = good, 1 = medium, 2 = poor.

    Thresholds (planning standard): ``lambda_f < good`` open, ``<= poor``
    medium, ``> poor`` blocked.
    """
    return np.where(lf < good, 0, np.where(lf <= poor, 1, 2))


def ventilation_corridors(lf: np.ndarray, threshold: float = GOOD_LF):
    """Connected components (8-connectivity) of cells with ``lambda_f <
    threshold`` — the open cells that can carry a ventilation corridor.

    Parameters
    ----------
    lf : np.ndarray (ny, nx)
        Frontal area density field.
    threshold : float
        Open-cell cut-off (default 0.30).

    Returns
    -------
    labels : np.ndarray (ny, nx), int
        Component id per cell; 0 = not an open cell.
    n_components : int
        Number of open components.
    """
    good = lf < threshold
    ny, nx = good.shape
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for j in range(ny):
        for i in range(nx):
            if good[j, i]:
                parent[(j, i)] = (j, i)
    for j in range(ny):
        for i in range(nx):
            if not good[j, i]:
                continue
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if dj == 0 and di == 0:
                        continue
                    j2, i2 = j + dj, i + di
                    if 0 <= j2 < ny and 0 <= i2 < nx and good[j2, i2]:
                        union((j, i), (j2, i2))
    labels = np.zeros((ny, nx), dtype=int)
    comp = {}
    n = 0
    for j in range(ny):
        for i in range(nx):
            if not good[j, i]:
                continue
            r = find((j, i))
            if r not in comp:
                n += 1
                comp[r] = n
            labels[j, i] = comp[r]
    return labels, n


def windward_leeward(labels: np.ndarray, wind_from_az: float):
    """Boundary-cell masks on the upwind (windward) and downwind (leeward)
    edges of the domain for the given wind.

    Parameters
    ----------
    labels : np.ndarray (ny, nx)
        Corridor-component labels (see :func:`ventilation_corridors`).
    wind_from_az : float
        Wind azimuth the flow comes from.

    Returns
    -------
    windward, leeward : np.ndarray (ny, nx), bool
        Masks of the domain-edge rows/columns facing the wind and away
        from it.
    """
    ny, nx = labels.shape
    bhat = blowing_dir(wind_from_az)
    ww = np.zeros((ny, nx), dtype=bool)
    ll = np.zeros((ny, nx), dtype=bool)
    # row 0 = y-min (south), row ny-1 = y-max (north); the wind FROM azimuth
    # enters the domain on the edges facing the source.  Epsilon guard:
    # cos(90 deg) is 6e-17 in floating point, not exactly 0.
    eps = 1e-12
    if bhat[1] < -eps:         # wind from the north, blowing south
        ww[-1, :] = True
        ll[0, :] = True
    elif bhat[1] > eps:        # wind from the south, blowing north
        ww[0, :] = True
        ll[-1, :] = True
    if bhat[0] < -eps:         # wind from the east, blowing west
        ww[:, -1] = True
        ll[:, 0] = True
    elif bhat[0] > eps:        # wind from the west, blowing east
        ww[:, 0] = True
        ll[:, -1] = True
    return ww, ll
