"""
Solar geometry and daylight-spacing utilities for urban spatial analysis.

Extracted from notebooks/10-solar-analysis.ipynb.

Provides:
- SunPosition: solar altitude / azimuth at a given moment
- solar_declination, sun_position: NOAA-convention solar geometry
- compute_spacing_d: required north-south building spacing D from the
  spacing-coefficient method
- building_shadow_footprint, compute_insolation_grid: hourly shadow
  overlay producing cumulative insolation hours on a grid
- check_standard_spacing: compare an actual spacing against the Beijing
  D/H standard
- facade_insolation, check_per_window_compliance: per-window facade
  insolation (窗台面日照) — the review metric 规划局 actually checks
  (GB 50180-2018 表 4.0.9: 大寒日 8:00-16:00, 窗台面 0.9 m, 每窗 ≥ 2 h)

Standard references
-------------------
- GB 50180-2018《城市居住区规划设计标准》表 4.0.9 — residential daylight
  standard.  Beijing is in climate zone II, 大城市: **一般住宅 大寒日
  ≥ 2 h**; 冬至日 2 h applies only to 老年人/残疾人住宅.  The statutory
  analysis day is 大寒日 (approximately Jan 20), not 冬至日.
- DB11/T 1670-2019《北京市居住建筑日照标准》— Beijing local standard.
- The Beijing practice spacing coefficient D/H = 1.6-1.7 is calibrated on
  **大寒日** analysis; pure geometric 冬至日 coefficients (D/H = 2.2-2.8)
  are a conservative upper bound (lower sun, longer shadows) and are used
  for early screening only, never for a statutory report.

Azimuth convention: degrees from north, clockwise (0=N, 90=E, 180=S,
270=W).  The azimuth uses the ``arctan2`` formulation — the ``arccos``
formulation cannot resolve quadrants (morning/afternoon get swapped),
which flips shadow directions.

All computations are pure numpy + shapely — no GPU dependency.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon
from shapely.vectorized import contains as vec_contains

# Beijing coordinates
BEIJING_LAT = 39.9    # degrees north
BEIJING_LON = 116.3   # degrees east
BEIJING_TZ = 8        # UTC+8

# Key analysis days (day of year)
WINTER_SOLSTICE = 355  # Dec 21 — conservative upper-bound screening day
GREAT_COLD = 20        # 大寒日, approx Jan 20 (GB 50180-2018 statutory analysis day)
SUMMER_SOLSTICE = 172  # Jun 21

# Window sill height (m) used in the spacing coefficient method
H_SILL = 0.9

# Beijing practice spacing coefficient (大寒日-based, GB 50180-2018)
D_H_STANDARD_LOW = 1.6
D_H_STANDARD_HIGH = 1.7

# Effective winter insolation window (local hours) for shadow overlay
INSOLATION_WINDOW = (8.0, 16.0)


@dataclass
class SunPosition:
    """Solar position at a given moment.

    Attributes
    ----------
    altitude : float
        Degrees above horizon.
    azimuth : float
        Degrees from north, clockwise (0=N, 90=E, 180=S, 270=W).
    """

    altitude: float
    azimuth: float

    @property
    def altitude_rad(self) -> float:
        """Altitude in radians."""
        return np.radians(self.altitude)

    @property
    def is_above_horizon(self) -> bool:
        """True when the sun is above the horizon."""
        return self.altitude > 0

    @property
    def shadow_length_ratio(self) -> float:
        """Shadow length / building height ratio.

        Infinity when the sun is at or below the horizon.
        """
        if self.altitude <= 0:
            return float('inf')
        return 1.0 / np.tan(self.altitude_rad)


def solar_declination(day_of_year: int) -> float:
    """Solar declination for a given day of the year (1-365), in degrees.

    Uses the Spencer (1971) formula from the NOAA Solar Position
    Calculator.

    Parameters
    ----------
    day_of_year : int
        Day of year, 1-365.

    Returns
    -------
    float
        Declination in degrees.
    """
    B = 2 * np.pi * (day_of_year - 1) / 365.0
    return (0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
            - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
            - 0.002697 * np.cos(3 * B) + 0.001480 * np.sin(3 * B)) * (180 / np.pi)


def sun_position(lat: float, lon: float, tz: float,
                 day_of_year: int, hour: float) -> SunPosition:
    """Compute sun position at a given location and time.

    Uses the standard solar geometry formulas (NOAA/SPA convention).
    Azimuth is measured from north, clockwise (0=N, 90=E, 180=S, 270=W)
    via the ``arctan2`` formulation that resolves quadrants correctly.

    Note: this is a simplified model that treats solar noon as 12:00
    local time, so ``lon`` and ``tz`` are accepted for API compatibility
    but do not shift the hour angle.  For longitude-corrected apparent
    time, subtract the longitude time offset from ``hour`` before calling.

    Parameters
    ----------
    lat : float
        Latitude in degrees (positive = north).
    lon : float
        Longitude in degrees (positive = east).  Not used by the
        simplified hour-angle model.
    tz : float
        Timezone offset from UTC (hours).  Not used by the simplified
        hour-angle model.
    day_of_year : int
        Day of year, 1-365.
    hour : float
        Local time in hours (e.g. 12.5 = 12:30).

    Returns
    -------
    SunPosition
        Altitude and azimuth.
    """
    lat_rad = np.radians(lat)
    dec = np.radians(solar_declination(day_of_year))
    hour_angle = np.radians((hour - 12.0) * 15.0)

    # Solar altitude: sin(alpha) = sin(phi)sin(delta) + cos(phi)cos(delta)cos(omega)
    sin_alt = (np.sin(lat_rad) * np.sin(dec) +
               np.cos(lat_rad) * np.cos(dec) * np.cos(hour_angle))
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))

    # Solar azimuth: use arctan2 for proper quadrant resolution
    # tan(A_north) = sin(omega) / (sin(phi)cos(omega) - cos(phi)tan(delta))
    # where A_north is measured from north clockwise
    tan_dec = np.tan(dec)
    denom = (np.sin(lat_rad) * np.cos(hour_angle) -
             np.cos(lat_rad) * tan_dec)
    numer = np.sin(hour_angle)

    # atan2(y, x) with y=numer, x=denom gives azimuth from north
    # Convert: az_from_north = atan2(-numer, -denom) = atan2(numer, denom) + pi
    az_north_rad = np.arctan2(numer, denom) + np.pi

    return SunPosition(
        altitude=float(np.degrees(alt_rad)),
        azimuth=float(np.degrees(az_north_rad) % 360),
    )


def compute_spacing_d(building_height: float, day: int, hour: float,
                      building_azimuth: float = 180.0,
                      lat: float = BEIJING_LAT,
                      lon: float = BEIJING_LON,
                      tz: float = BEIJING_TZ) -> float:
    """Compute required north-south spacing D (metres, 间距系数法).

    The spacing-coefficient method is the planner's quick check: the
    southern building of height ``building_height`` casts a shadow of
    length (H - sill) / tan(altitude), projected onto the north-south
    axis by the cosine of the azimuth difference to the building
    orientation.

    Parameters
    ----------
    building_height : float
        Height of the southern building (m).
    day : int
        Day of year.  Use :data:`GREAT_COLD` for the statutory day,
        :data:`WINTER_SOLSTICE` for a conservative upper bound.
    hour : float
        Analysis time (local).
    building_azimuth : float, optional
        Building orientation (deg, 180 = south-facing).
    lat, lon, tz : float, optional
        Location parameters, defaulting to Beijing.

    Returns
    -------
    float
        Required north-south spacing (m); ``inf`` when the sun is below
        the horizon (analysis impossible).
    """
    pos = sun_position(lat, lon, tz, day, hour)
    if pos.altitude <= 0:
        return float('inf')
    beta = np.radians(abs(pos.azimuth - building_azimuth))
    D = (building_height - H_SILL) * (1.0 / np.tan(pos.altitude_rad)) * np.cos(beta)
    return max(0.0, float(D))


def building_shadow_footprint(building_xy: np.ndarray,
                              building_height: float,
                              sun_pos: SunPosition) -> np.ndarray:
    """Compute the ground shadow polygon of a building at a sun position.

    Projects each vertex along the sun's rays onto the ground plane.

    Parameters
    ----------
    building_xy : ndarray, shape (N, 2)
        Building outline vertices in meters.
    building_height : float
        Building height (m).
    sun_pos : SunPosition
        Sun position.

    Returns
    -------
    ndarray, shape (2N+1, 2)
        Closed shadow polygon vertices (building base + projected
        vertices + closing vertex).
    """
    if sun_pos.altitude <= 0:
        return building_xy.copy()

    az_rad = np.radians(sun_pos.azimuth)
    alt_rad = sun_pos.altitude_rad
    # Shadow displacement on the ground plane
    dx = -np.sin(az_rad) / np.tan(alt_rad)
    dy = -np.cos(az_rad) / np.tan(alt_rad)

    shadow_xy = building_xy + building_height * np.array([dx, dy])
    # Combine building base + shadow vertices into a closed polygon
    return np.vstack([building_xy, shadow_xy[::-1], building_xy[:1]])


def compute_insolation_grid(buildings: Sequence[Tuple[np.ndarray, float]],
                            site_bbox: Tuple[float, float, float, float],
                            grid_res: float = 5.0,
                            day: int = WINTER_SOLSTICE,
                            hours_range: Tuple[float, float] = INSOLATION_WINDOW,
                            time_step_min: int = 10,
                            lat: float = BEIJING_LAT,
                            lon: float = BEIJING_LON,
                            tz: float = BEIJING_TZ,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative insolation hours on a grid (hourly shadow overlay).

    For each time step in ``hours_range``, projects every building's
    shadow onto the ground plane and accumulates shadowed time per grid
    cell.  Cells with fewer than the standard 2 h (大寒日 statutory, or
    冬至日 screening) are deficient.

    Parameters
    ----------
    buildings : list of (outline_xy, height) tuples
        Building outlines (N, 2) in meters plus heights.
    site_bbox : (xmin, ymin, xmax, ymax)
        Grid extent in meters.
    grid_res : float, optional
        Grid resolution in meters.
    day : int, optional
        Day of year.  Default 冬至日 (conservative screening); use
        :data:`GREAT_COLD` for the statutory analysis day.
    hours_range : (start, end), optional
        Local-hour window to overlay.
    time_step_min : int, optional
        Time step in minutes.
    lat, lon, tz : float, optional
        Location parameters, defaulting to Beijing.

    Returns
    -------
    insolation : ndarray, shape (ny, nx)
        Cumulative insolation hours per cell.
    x_edges, y_edges : ndarray
        Grid edge arrays.
    """
    xmin, ymin, xmax, ymax = site_bbox
    x_edges = np.arange(xmin, xmax + grid_res, grid_res)
    y_edges = np.arange(ymin, ymax + grid_res, grid_res)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    insolation = np.zeros((ny, nx))

    t_start, t_end = hours_range
    n_steps = int((t_end - t_start) * 60 / time_step_min)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    for step in range(n_steps):
        hour = t_start + step * time_step_min / 60.0
        pos = sun_position(lat, lon, tz, day, hour)
        if pos.altitude <= 0.5:
            continue

        for outline, height in buildings:
            shadow_verts = building_shadow_footprint(outline, height, pos)
            inside = vec_contains(Polygon(shadow_verts), X, Y)
            insolation[inside] += time_step_min / 60.0

    return insolation, x_edges, y_edges


def check_standard_spacing(
    spacing_m: float,
    building_height: float,
    d_h_standard: Tuple[float, float] = (D_H_STANDARD_LOW, D_H_STANDARD_HIGH),
) -> dict:
    """Compare an actual spacing to the Beijing standard D/H coefficient.

    Parameters
    ----------
    spacing_m : float
        Actual north-south spacing between building rows (m).
    building_height : float
        Height of the southern building (m).
    d_h_standard : tuple of float, optional
        Standard D/H band (default 1.6-1.7, 大寒日-based Beijing
        practice, GB 50180-2018).

    Returns
    -------
    dict
        Keys ``d_h``, ``required_min_m``, ``required_max_m``, ``passes``.
    """
    lo, hi = d_h_standard
    d_h = spacing_m / building_height if building_height > 0 else float('inf')
    return {
        'd_h': round(d_h, 3),
        'required_min_m': round(building_height * lo, 1),
        'required_max_m': round(building_height * hi, 1),
        'passes': d_h >= lo,
    }


# Extraterrestrial solar constant in kW/m².  No atmospheric attenuation —
# an upper-bound beam-irradiation model (see facade_annual_irradiation).
SOLAR_CONSTANT_KW = 1.361


def facade_annual_irradiation(buildings: Sequence[Tuple[np.ndarray, float]],
                              lat: float = BEIJING_LAT,
                              day_step: int = 7,
                              lon: float = BEIJING_LON,
                              tz: float = BEIJING_TZ,
                              hour_step: float = 1.0,
                              ) -> dict:
    """Annual beam irradiation (kWh/m²) on each cardinal facade of each building.

    For every ``day_step``-th day of the year (day_step=7 → 53 sample days,
    ~weekly intervals) and every ``hour_step`` hours, evaluates the sun
    position and accumulates ``max(0, cos(incidence)) * SOLAR_CONSTANT_KW``.
    For a vertical facade whose outward normal points at azimuth ``az_f`` the
    incidence cosine is ``cos(altitude) * cos(azimuth - az_f)``; hours with
    the sun below the horizon or behind the facade contribute nothing.
    Daily totals are scaled by ``day_step`` so each sampled day stands for
    the days it represents — the result is the annual integral.  Neighbouring
    buildings and a building's own shadow are ignored: this is the
    unobstructed upper-bound irradiation.

    Facade edges are grouped by outward normal rounded to the nearest
    cardinal direction (0=N, 90=E, 180=S, 270=W).

    Parameters
    ----------
    buildings : list of (outline_xy, height)
        Same format as :func:`facade_insolation` (height is not used).
    lat, lon, tz : float, optional
        Location parameters, defaulting to Beijing.
    day_step : int, optional
        Sampling interval in days (default 7 → 53 samples/year).
    hour_step : float, optional
        Intra-day integration step in hours.

    Returns
    -------
    dict
        ``{bi: {orientation_deg: kWh_per_m2}}`` for each building index
        ``bi`` in the order of ``buildings``.
    """
    results = {}
    for bi, (outline, _height) in enumerate(buildings):
        per_orientation = {}
        for p0, p1 in _facade_edges(outline):
            az_f = int(90.0 * round(_outward_normal_azimuth(p0, p1) / 90.0)) % 360
            daily = 0.0
            for day in range(1, 366, day_step):
                for hour in np.arange(0.0, 24.0, hour_step):
                    pos = sun_position(lat, lon, tz, day, hour)
                    if pos.altitude <= 0:
                        continue
                    cos_i = (np.cos(pos.altitude_rad) *
                             np.cos(np.radians(pos.azimuth - az_f)))
                    daily += max(0.0, cos_i) * SOLAR_CONSTANT_KW * hour_step
            per_orientation[az_f] = max(per_orientation.get(az_f, 0.0),
                                        daily * day_step)
        results[bi] = per_orientation
    return results


# ---------------------------------------------------------------------------
# Per-window facade insolation (窗台面日照) — GB 50180-2018 表 4.0.9
# ---------------------------------------------------------------------------

# Window sampling grid on facades (m)
FACADE_H_SPACING = 1.5        # horizontal spacing between window positions
FACADE_V_SPACING = 1.2        # vertical spacing between window positions
FACADE_CORNER_INSET = 0.75    # first window inset from the facade corner
FACADE_TOP_MARGIN = 0.5       # no window positions above height - top_margin

# Steps below this solar altitude are skipped (deg), matching
# ``compute_insolation_grid``.
MIN_SUN_ALTITUDE = 0.5

# GB 50180-2018 表 4.0.9: 大寒日 >= 2 h per window (II 气候区, 大城市, 一般住宅)
MIN_WINDOW_HOURS = 2.0


def _polygon_is_ccw(xy: np.ndarray) -> bool:
    """True when the polygon vertices wind counter-clockwise (shoelace)."""
    x, y = xy[:, 0], xy[:, 1]
    area2 = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1]
    return area2 > 0


def _facade_edges(outline_xy: np.ndarray) -> list:
    """Facade segments ``(p0, p1)`` of a building outline, normalized to CCW.

    Each returned segment is a pair of (2,) float arrays; the outline is
    reversed first when needed so that outward normals point outwards.
    """
    xy = np.asarray(outline_xy, dtype=float)
    if not _polygon_is_ccw(xy):
        xy = xy[::-1]
    return [(xy[i], xy[(i + 1) % len(xy)]) for i in range(len(xy))]


def _outward_normal_azimuth(p0: np.ndarray, p1: np.ndarray) -> float:
    """Azimuth (deg from north, clockwise) of the outward normal of a CCW edge.

    For a counter-clockwise polygon the outward normal of edge (p0, p1) is
    (dy, -dx).  A south-facing facade therefore yields 180 deg, north 0 deg.
    """
    dx, dy = p1 - p0
    return float(np.degrees(np.arctan2(dy, -dx)) % 360)


def _sample_window_positions(p0: np.ndarray, p1: np.ndarray,
                             height: float, sill_height: float,
                             h_spacing: float, v_spacing: float) -> np.ndarray:
    """Window positions on one facade segment, shape (N, 3): (x, y, z).

    Horizontal positions run FACADE_CORNER_INSET + k * h_spacing along the
    wall; vertical positions run sill_height + k * v_spacing, stopping
    FACADE_TOP_MARGIN below the roof.
    """
    u = p1 - p0
    length = float(np.hypot(*u))
    u = u / length
    ts = np.arange(FACADE_CORNER_INSET, length - FACADE_CORNER_INSET + 1e-9, h_spacing)
    zs = np.arange(sill_height, height - FACADE_TOP_MARGIN + 1e-9, v_spacing)
    if len(ts) == 0 or len(zs) == 0:
        return np.empty((0, 3))
    xy = np.repeat(p0[None, :] + ts[:, None] * u[None, :], len(zs), axis=0)
    z = np.tile(zs, len(ts))
    return np.column_stack([xy, z])


def building_shadow_at_height(building_xy: np.ndarray,
                              building_height: float,
                              sun_pos: SunPosition,
                              z: float) -> Optional[np.ndarray]:
    """Shadow polygon of a building at height ``z`` above ground.

    A point at height ``z`` whose 2D position lies inside this polygon
    cannot see the sun: the ray to the sun is blocked by the building.
    The polygon is the footprint swept by ``(H - z)`` along the shadow
    direction — mathematically identical to tracing the ray from the point
    to the sun, but vectorizable (``shapely.vectorized.contains``).

    The shadow region at height ``z`` is the Minkowski sum of the footprint
    and the shadow-shift segment; for rectangular / convex footprints this
    equals the convex hull of (footprint ∪ shifted footprint), which is
    what is returned.  For concave footprints the hull is a conservative
    over-approximation.  (``building_shadow_footprint`` reuses the same
    shift math for the ground plane but assembles [base ∪ shifted] as a
    single vertex loop, which self-intersects for oblique shadow angles —
    fine for the grid overlay, not for point-in-shadow tests.)

    Parameters
    ----------
    building_xy : ndarray, shape (N, 2)
        Building outline in meters.
    building_height : float
        Building height (m).
    sun_pos : SunPosition
        Sun position.
    z : float
        Height above ground of the receiving points (window height, m).

    Returns
    -------
    ndarray or None
        Shadow polygon vertices (closed, convex hull), or None when ``z``
        is at/above the roof (nothing can be blocked) or the sun is below
        the horizon.
    """
    if z >= building_height or sun_pos.altitude <= 0:
        return None
    base = np.asarray(building_xy, dtype=float)
    # Same shadow-direction math as building_shadow_footprint: a point at
    # height z is blocked while the ray has not yet climbed above the roof.
    az_rad = np.radians(sun_pos.azimuth)
    dx = -np.sin(az_rad) / np.tan(sun_pos.altitude_rad)
    dy = -np.cos(az_rad) / np.tan(sun_pos.altitude_rad)
    shifted = base + (building_height - z) * np.array([dx, dy])
    hull = Polygon(np.vstack([base, shifted])).convex_hull
    return np.asarray(hull.exterior.coords)


def window_receives_sun(wx: float, wy: float, wz: float,
                        sun_pos: SunPosition,
                        facade_azimuth: float,
                        host_outline: np.ndarray,
                        buildings: Sequence[Tuple[np.ndarray, float]],
                        max_dist: float = 2500.0) -> bool:
    """Ray-cast reference: does the ray from a window to the sun stay clear?

    True when (a) the sun is in the facade's outward half-space and (b) the
    3D ray from the window position toward the sun does not intersect any
    building other than the host.  A building's own shadow on its own
    facade does not count — only neighbouring buildings block windows.

    This per-window brute-force formulation is used to cross-validate the
    vectorized shadow sweep in :func:`facade_insolation`.
    ``host_outline`` must be the exact array object used in ``buildings``
    (identity comparison excludes it from blocking).

    Parameters
    ----------
    wx, wy, wz : float
        Window position (m; ``wz`` above ground).
    sun_pos : SunPosition
        Sun position.
    facade_azimuth : float
        Outward normal azimuth of the facade (deg from north, CW).
    host_outline : ndarray
        Outline of the building the window belongs to.
    buildings : list of (outline_xy, height)
        All buildings, including the host.
    max_dist : float, optional
        Ray length for the 2D intersection test (m).

    Returns
    -------
    bool
        True when the window sees direct sun at this moment.
    """
    if sun_pos.altitude <= MIN_SUN_ALTITUDE:
        return False
    az_rad = np.radians(sun_pos.azimuth)
    # sun exactly on the facade plane (grazing) is not direct sun:
    # require a strictly positive cosine with a small float tolerance
    if np.cos(az_rad - np.radians(facade_azimuth)) <= 1e-6:
        return False
    dx, dy = np.sin(az_rad), np.cos(az_rad)
    ray = LineString([(wx, wy), (wx + dx * max_dist, wy + dy * max_dist)])
    for outline, height in buildings:
        if outline is host_outline:
            continue
        inter = ray.intersection(Polygon(np.asarray(outline, dtype=float)))
        if inter.is_empty:
            continue
        d = inter.distance(Point(wx, wy))
        if wz + d * np.tan(sun_pos.altitude_rad) <= height:
            return False  # blocked by this building
    return True


def facade_insolation(buildings: Sequence[Tuple[np.ndarray, float]],
                      day: int = GREAT_COLD,
                      hours_range: Tuple[float, float] = INSOLATION_WINDOW,
                      time_step_min: int = 10,
                      sill_height: float = H_SILL,
                      h_spacing: float = FACADE_H_SPACING,
                      v_spacing: float = FACADE_V_SPACING,
                      lat: float = BEIJING_LAT,
                      lon: float = BEIJING_LON,
                      tz: float = BEIJING_TZ,
                      ) -> dict:
    """Cumulative direct-sun hours for every window on every facade (窗台面日照).

    The GB 50180-2018 表 4.0.9 review method: on the statutory analysis day
    (:data:`GREAT_COLD`, 大寒日) within the effective insolation window
    (:data:`INSOLATION_WINDOW`, 8:00-16:00) each window must accumulate at
    least 2 h of direct sun (window sill 0.9 m, :data:`H_SILL`).

    Windows are sampled per facade on a grid: FACADE_CORNER_INSET +
    k*h_spacing horizontally along the wall, sill_height + k*v_spacing
    vertically.  At each time step a window receives sun iff the sun is in
    the facade's outward half-space AND the window's 2D position lies
    outside every other building's shadow polygon at the window's height
    (:func:`building_shadow_at_height`).  A building's own shadow on its
    own facade never counts — only neighbouring buildings block windows.
    Time steps are evaluated at ``t_start + k * dt`` for ``k`` in
    ``[0, n_steps)`` — the same half-open convention as
    :func:`compute_insolation_grid`.

    Parameters
    ----------
    buildings : list of (outline_xy, height)
        Same format as :func:`compute_insolation_grid`.
    day : int, optional
        Analysis day; default :data:`GREAT_COLD` (statutory).
    hours_range : (start, end), optional
        Effective insolation window in local hours.
    time_step_min : int, optional
        Time step in minutes.
    sill_height : float, optional
        Window sill height above ground (m).
    h_spacing, v_spacing : float, optional
        Window sampling spacing horizontally / vertically (m).
    lat, lon, tz : float, optional
        Location parameters, defaulting to Beijing.

    Returns
    -------
    dict
        Keyed by building index (0-based order of ``buildings``):
        ``{bid: {'min_hours': float, 'n_windows': int,
                 'facades': [{'orientation': float, 'length_m': float,
                              'n_windows': int, 'min_hours': float,
                              'min_window_index': int,
                              'windows': [float, ...]}, ...]}}``
        ``min_hours`` is the critical (worst) window on the facade;
        ``min_window_index`` locates it within ``windows``.
    """
    host_data = []
    for outline, height in buildings:
        facades = []
        for p0, p1 in _facade_edges(outline):
            windows = _sample_window_positions(p0, p1, height,
                                               sill_height, h_spacing, v_spacing)
            facades.append({
                'orientation': _outward_normal_azimuth(p0, p1),
                'length_m': float(np.hypot(*(p1 - p0))),
                'windows': windows,
                'hours': np.zeros(len(windows)),
            })
        host_data.append({'height': height, 'facades': facades})

    t_start, t_end = hours_range
    n_steps = int((t_end - t_start) * 60 / time_step_min)
    dt = time_step_min / 60.0

    for step in range(n_steps):
        hour = t_start + step * time_step_min / 60.0
        pos = sun_position(lat, lon, tz, day, hour)
        if pos.altitude <= MIN_SUN_ALTITUDE:
            continue

        for hi, host in enumerate(host_data):
            for f in host['facades']:
                windows = f['windows']
                if len(windows) == 0:
                    continue
                # (a) the sun must be in the facade's outward half-space;
                #     sun exactly on the facade plane (grazing) is not
                #     direct sun, hence the 1e-6 float tolerance
                if np.cos(np.radians(pos.azimuth - f['orientation'])) <= 1e-6:
                    continue
                zs = windows[:, 2]
                for z in np.unique(zs):
                    sel = zs == z
                    xy = windows[sel, :2]
                    blocked = np.zeros(len(xy), dtype=bool)
                    # (b) outside every OTHER building's shadow at height z
                    for bi, (outline, height) in enumerate(buildings):
                        if bi == hi or height <= z:
                            continue
                        shadow = building_shadow_at_height(outline, height, pos, z)
                        if shadow is None:
                            continue
                        blocked |= contains_xy(Polygon(shadow), xy[:, 0], xy[:, 1])
                        if blocked.all():
                            break
                    f['hours'][sel] += (1 - blocked) * dt

    results = {}
    for bi, host in enumerate(host_data):
        facades_out = []
        n_windows = 0
        b_min = float('inf')
        for f in host['facades']:
            hours = f['hours']
            n_windows += len(hours)
            if len(hours):
                fm = float(hours.min())
                wi = int(hours.argmin())
            else:
                fm, wi = float('nan'), -1
            b_min = min(b_min, fm)
            facades_out.append({
                'orientation': round(f['orientation'], 1),
                'length_m': round(f['length_m'], 1),
                'n_windows': int(len(hours)),
                'min_hours': round(fm, 3),
                'min_window_index': wi,
                'windows': [round(float(h), 3) for h in hours],
            })
        results[bi] = {
            'min_hours': round(b_min, 3),
            'n_windows': n_windows,
            'facades': facades_out,
        }
    return results


@dataclass
class CheckResult:
    """Per-window daylight compliance check (GB 50180-2018 表 4.0.9).

    Attributes
    ----------
    passes : bool
        True when no window falls below ``min_hours``.
    min_hours : float
        The threshold applied (default 2 h).
    total_windows : int
        Total number of sampled window positions.
    deficient_windows : int
        Windows below the threshold.
    deficient_facades : list of dict
        One dict per facade whose critical window is deficient, with keys
        ``building_id``, ``facade_index``, ``orientation``, ``min_hours``.
    summary : str
        One-line human-readable verdict.
    """

    passes: bool
    min_hours: float
    total_windows: int
    deficient_windows: int
    deficient_facades: list
    summary: str


def check_per_window_compliance(facade_results: dict,
                                min_hours: float = MIN_WINDOW_HOURS) -> CheckResult:
    """Check every window against the >= ``min_hours`` insolation standard.

    Parameters
    ----------
    facade_results : dict
        Output of :func:`facade_insolation`.
    min_hours : float, optional
        Minimum cumulative insolation per window (default 2.0 h —
        GB 50180-2018 表 4.0.9 for 大寒日, II 气候区, 大城市, 一般住宅).

    Returns
    -------
    CheckResult
    """
    total = 0
    deficient = 0
    deficient_facades = []
    for bid in sorted(facade_results, key=lambda k: str(k)):
        for fi, f in enumerate(facade_results[bid]['facades']):
            hours = f['windows']
            total += len(hours)
            if f['min_hours'] < min_hours:
                deficient += sum(1 for h in hours if h < min_hours)
                deficient_facades.append({
                    'building_id': bid,
                    'facade_index': fi,
                    'orientation': f['orientation'],
                    'min_hours': f['min_hours'],
                })
    passes = deficient == 0
    if passes:
        summary = f'PASS: all {total} windows >= {min_hours:g} h'
    else:
        summary = (f'FAIL: {deficient}/{total} windows < {min_hours:g} h '
                   f'across {len(deficient_facades)} deficient facade(s)')
    return CheckResult(passes=passes, min_hours=min_hours,
                       total_windows=total, deficient_windows=deficient,
                       deficient_facades=deficient_facades, summary=summary)


__all__ = [
    "BEIJING_LAT", "BEIJING_LON", "BEIJING_TZ",
    "WINTER_SOLSTICE", "GREAT_COLD", "SUMMER_SOLSTICE",
    "H_SILL", "D_H_STANDARD_LOW", "D_H_STANDARD_HIGH",
    "INSOLATION_WINDOW", "SunPosition", "solar_declination",
    "sun_position", "compute_spacing_d", "building_shadow_footprint",
    "compute_insolation_grid", "check_standard_spacing",
    "FACADE_H_SPACING", "FACADE_V_SPACING", "FACADE_CORNER_INSET",
    "FACADE_TOP_MARGIN", "MIN_SUN_ALTITUDE", "MIN_WINDOW_HOURS",
    "building_shadow_at_height", "window_receives_sun",
    "facade_insolation", "CheckResult", "check_per_window_compliance",
    "SOLAR_CONSTANT_KW", "facade_annual_irradiation",
]
