"""Vertical design / grading analysis (竖向设计).

Extracted from notebooks/21-vertical-design.ipynb.

Data-gated (needs a DEM); the notebook demonstrates the methods on
synthetic terrain (2D Gaussian hills), so every function here takes a
plain 2-D elevation array and is deterministic, numpy-only and CPU-only.

Grid convention (documented once here and used everywhere in this module):
the DEM is a 2-D array ``z[j, i]`` with **row 0 = north** (j increases
southward) and **col 0 = west** (i increases eastward).  Cell spacing is
``dx`` (east-west, m) and ``dy`` (north-south, m), uniform.

Three methods, each a standalone function:

1. :func:`slope_aspect` — slope (percent and degrees) from the numpy
   gradient, plus downhill aspect as azimuth and 8-way cardinal direction.
2. :func:`cut_fill_balance` — 挖方 (cut) / 填方 (fill) earthwork volumes
   for a proposed platform elevation; with no elevation given, returns
   the level that minimizes ``|cut - fill|`` (exactly the area-weighted
   mean elevation).
3. :func:`flow_direction_d8` — D8 steepest-descent flow direction
   (ESRI direction codes) and sink mask for drainage analysis.

References
----------
- CJJ 83-2016《城市用地竖向规划规范》: slope suitability classes used in
  grading design (flat < 5 %, gentle 5-15 %, moderate 15-25 %, steep > 25 %)
- GB 50007-2011《建筑地基基础设计规范》: 挖方/填方 balance target for
  场地平整 (minimize earthwork import/export)
- D8 algorithm: O'Callaghan & Mark (1984), "The extraction of drainage
  networks from digital elevation data", Computer Vision, Graphics and
  Image Processing 28(3).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "slope_aspect",
    "cut_fill_balance",
    "flow_direction_d8",
    "D8_DELTA",
    "D8_DIST",
]

#: ESRI D8 flow-direction codes: code -> (dj, di) grid step.
#: Iteration order fixes tie-breaking: the first (best) direction in this
#: order wins when two neighbours have an identical drop/distance.
D8_DELTA = {
    64: (-1, 0),   # N
    128: (-1, 1),  # NE
    1: (0, 1),     # E
    2: (1, 1),     # SE
    4: (1, 0),     # S
    8: (1, -1),    # SW
    16: (0, -1),   # W
    32: (-1, -1),  # NW
}

#: flow distance (m) for each D8 step — diagonal steps are sqrt(2) apart
D8_DIST = {code: np.hypot(dj, di) for code, (dj, di) in D8_DELTA.items()}

#: 8-way cardinal sector names, N = [337.5, 22.5), then 45 deg steps
#: (object dtype so "flat" can be assigned afterwards without truncation)
_SECTORS = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], dtype=object)
#: gradient magnitude below which a cell is treated as flat
_FLAT_EPS = 1e-9


def slope_aspect(dem, dx=1.0, dy=1.0):
    """Slope and aspect of a DEM (Method 1).

    Slope is the gradient magnitude of the elevation surface:
    ``sqrt(p^2 + q^2)`` with ``p = dz/dx`` (east) and ``q = dz/dy``
    (north, row 0 = north).  Aspect is the azimuth the surface slopes
    TOWARD (downhill), clockwise from north, binned into 8 cardinal
    directions.  Near-flat cells get ``aspect_deg = -1`` (undefined) and
    direction ``"flat"``.

    Parameters
    ----------
    dem : np.ndarray (ny, nx)
        Elevation array, row 0 = north, col 0 = west.
    dx, dy : float
        Cell size east-west / north-south (same units as ``dem``).

    Returns
    -------
    dict with keys
        slope_pct : slope in percent (``100 * grad``)
        slope_deg : slope in degrees (``arctan(grad)``)
        aspect_deg : downhill azimuth (0-360 clockwise from north; -1 flat)
        aspect_dir : cardinal direction per cell (N/NE/E/SE/S/SW/W/NW/flat)
    """
    z = np.asarray(dem, dtype=float)
    p = np.gradient(z, dx, axis=1)       # eastward slope
    q = -np.gradient(z, dy, axis=0)      # northward slope (row 0 = north)
    grad = np.hypot(p, q)
    flat = grad < _FLAT_EPS
    aspect_deg = np.degrees(np.arctan2(-p, -q)) % 360.0
    aspect_deg[flat] = -1.0
    sector = (((aspect_deg + 22.5) % 360.0) / 45.0).astype(int)
    aspect_dir = _SECTORS[sector].copy()
    aspect_dir[flat] = "flat"
    return {
        "slope_pct": 100.0 * grad,
        "slope_deg": np.degrees(np.arctan(grad)),
        "aspect_deg": aspect_deg,
        "aspect_dir": aspect_dir,
    }


def cut_fill_balance(dem, platform_elev=None, dx=1.0, dy=1.0, mask=None):
    """Cut / fill earthwork volumes for a proposed platform (Method 2).

    The platform is a horizontal surface at ``platform_elev`` covering the
    cells selected by ``mask`` (default: the whole grid).  Cells above the
    platform need 挖方 (cut — soil removed), cells below need 填方 (fill —
    soil imported); volumes are the area-weighted sum:

    ``cut = sum(max(z - platform_elev, 0)) * dx * dy``

    With ``platform_elev=None`` the level minimizing ``|cut - fill|`` is
    returned.  Because ``cut - fill = sum(z - platform_elev)``, that level
    is exactly the area-weighted mean elevation of the platform cells.

    Parameters
    ----------
    dem : np.ndarray (ny, nx)
        Elevation array (row 0 = north, col 0 = west).
    platform_elev : float or None
        Proposed platform elevation.  ``None`` -> balanced level returned.
    dx, dy : float
        Cell size east-west / north-south (m).
    mask : np.ndarray (ny, nx), bool, optional
        Platform footprint; only these cells take part in grading.

    Returns
    -------
    dict with keys platform_elev, cut, fill, net, balance (|cut - fill|),
    cell_area and cells (number of graded cells).
    """
    z = np.asarray(dem, dtype=float)
    if mask is None:
        zc = z.ravel()
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != z.shape:
            raise ValueError(f"mask shape {m.shape} != dem shape {z.shape}")
        zc = z[m]
    if zc.size == 0:
        raise ValueError("no cells selected by mask")
    cell = float(dx * dy)
    if platform_elev is None:
        platform_elev = float(zc.mean())
    cut = float(np.maximum(zc - platform_elev, 0.0).sum() * cell)
    fill = float(np.maximum(platform_elev - zc, 0.0).sum() * cell)
    return {
        "platform_elev": platform_elev,
        "cut": cut,
        "fill": fill,
        "net": cut - fill,
        "balance": abs(cut - fill),
        "cell_area": cell,
        "cells": int(zc.size),
    }


def flow_direction_d8(dem):
    """D8 steepest-descent flow direction and sink mask (Method 3).

    Every cell points to the in-grid neighbour with the maximum
    drop/distance ratio; cells with no strictly lower neighbour (pits and
    flat patches, where "flat" includes equal-elevation neighbours) are
    sinks (code 0).  Ties on identical drop/distance go to the first
    direction in ``D8_DELTA`` iteration order (N, NE, E, SE, S, SW, W, NW).

    Parameters
    ----------
    dem : np.ndarray (ny, nx)
        Elevation array (row 0 = north, col 0 = west).

    Returns
    -------
    codes : np.ndarray (ny, nx), int
        ESRI D8 direction code per cell (see ``D8_DELTA``; 0 = sink).
    sinks : np.ndarray (ny, nx), bool
        True where the cell has no downhill neighbour.
    """
    z = np.asarray(dem, dtype=float)
    ny, nx = z.shape
    codes = np.zeros((ny, nx), dtype=np.int16)
    for j in range(ny):
        for i in range(nx):
            best_code, best_slope = 0, 0.0
            for code, (dj, di) in D8_DELTA.items():
                j2, i2 = j + dj, i + di
                if not (0 <= j2 < ny and 0 <= i2 < nx):
                    continue
                drop = z[j, i] - z[j2, i2]
                if drop <= 0:
                    continue
                s = drop / D8_DIST[code]
                if s > best_slope:
                    best_slope, best_code = s, code
            codes[j, i] = best_code
    return codes, codes == 0
