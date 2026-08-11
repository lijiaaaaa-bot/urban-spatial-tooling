"""
Spatial-compliance checks: view corridors and three-lines (三区三线).

Extracted from notebooks/11-view-corridor-analysis.ipynb and
notebooks/15-three-lines-compliance.ipynb.

View-corridor machinery (三山五园-style sightlines):
- Viewpoint / ViewCorridor — analytic height envelope: the max building
  height at any point is a linear interpolation between the viewpoint
  and target elevations (O(1) per point, no ray tracing).
- generate_view_fan — fan of sightlines producing a 2D height surface.
- combine_corridors — the binding height at any point is the minimum
  across all corridors (multi-corridor composition).
- terrain_corrected_height — sightline height above terrain elevation,
  so the envelope adapts to DEM ground heights instead of a flat plane.
- height_envelope — combined constraint surface over a 2D grid.
- check_building_compliance — building-level pass/fail against a set of
  corridors.

Three-lines machinery:
- ThreeLinesChecker — hard binary containment (legally safest, final
  approval check).
- ProportionalChecker — configurable overlap tolerances (de minimis
  thresholds used in planning review).
- generate_with_constraints — rejection-sampling generator that prevents
  violations at generation time.

Standard references
-------------------
- 北京城市总体规划 (2016-2035) 第52条 — 三山五园视廊保护;
  DB11/T 1945-2021《城市设计导则补充图则》.
- 中发〔2019〕18号 / 自然资发〔2022〕142号 — 三区三线划定规则;
  《土地管理法》(2019) — 永久基本农田特殊保护.

Data note: official corridor / three-lines GIS is unavailable for the
exploration fixture; the algorithms are production-ready and the
coordinates are swappable.

All geometry checks are shapely-only — no GPU dependency.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon


# ---------------------------------------------------------------------------
# View corridor: height envelope from sightline constraints
# ---------------------------------------------------------------------------


@dataclass
class Viewpoint:
    """A viewpoint or landmark with 3D coordinates."""

    name: str
    x: float    # Easting (m, projected CRS)
    y: float    # Northing (m)
    z: float    # Elevation + observer height (m)


@dataclass
class ViewCorridor:
    """A single protected sightline (e.g. Fragrant Hills -> city center)."""

    name: str
    viewpoint: Viewpoint
    target: Viewpoint
    corridor_width: float = 50.0  # horizontal buffer width (m)

    @property
    def sightline(self) -> LineString:
        """Horizontal projection of the sightline."""
        return LineString([(self.viewpoint.x, self.viewpoint.y),
                           (self.target.x, self.target.y)])

    @property
    def distance(self) -> float:
        """3D distance between viewpoint and target (m)."""
        dx = self.target.x - self.viewpoint.x
        dy = self.target.y - self.viewpoint.y
        return np.sqrt(dx * dx + dy * dy)

    def max_height_at(self, px: float, py: float,
                      ground_elevation: float = 0.0) -> float:
        """Maximum allowed building height at point (px, py).

        Returns the height above ground elevation that preserves the
        sightline: the sightline elevation at the projection of the
        point onto the segment, minus the ground elevation.  Points
        behind the viewpoint or beyond the target are clipped to the
        segment ends (the constraint only applies along the corridor).

        Parameters
        ----------
        px, py : float
            Point easting / northing (m).
        ground_elevation : float, optional
            Ground elevation at the point (m).

        Returns
        -------
        float
            Max allowed building height (m), >= 0.
        """
        vx, vy = self.viewpoint.x, self.viewpoint.y
        tx, ty = self.target.x, self.target.y
        dx, dy = tx - vx, ty - vy
        d2 = dx * dx + dy * dy
        if d2 < 1e-10:
            return float('inf')
        t = ((px - vx) * dx + (py - vy) * dy) / d2
        t = np.clip(t, 0.0, 1.0)

        # Height of the sightline at the projected point
        z_sightline = self.viewpoint.z + t * (self.target.z - self.viewpoint.z)
        h_max = z_sightline - ground_elevation
        return max(0.0, float(h_max))

    def distance_along_sightline(self, px: float, py: float) -> float:
        """Distance from the viewpoint along the sightline to (px, py)."""
        vx, vy = self.viewpoint.x, self.viewpoint.y
        tx, ty = self.target.x, self.target.y
        dx, dy = tx - vx, ty - vy
        d2 = dx * dx + dy * dy
        t = ((px - vx) * dx + (py - vy) * dy) / max(d2, 1e-10)
        return float(np.clip(t, 0.0, 1.0) * np.sqrt(d2))


def generate_view_fan(viewpoint: Viewpoint,
                      target_center: Viewpoint,
                      fan_angle_deg: float = 10.0,
                      n_rays: int = 9,
                      corridor_width: float = 100.0) -> List[ViewCorridor]:
    """Generate a fan of sightlines between a viewpoint and a target area.

    Real corridors are wedge-shaped sectors, not single lines; the
    combined height constraint at any point is the minimum across all
    rays.

    Parameters
    ----------
    viewpoint : Viewpoint
        The observation point.
    target_center : Viewpoint
        Center of the target area (its ``z`` is used for all rays).
    fan_angle_deg : float, optional
        Total angular spread (degrees).
    n_rays : int, optional
        Number of sightlines.
    corridor_width : float, optional
        Width of each corridor (m).

    Returns
    -------
    list of ViewCorridor
        The fan rays.
    """
    dx = target_center.x - viewpoint.x
    dy = target_center.y - viewpoint.y
    base_angle = np.arctan2(dy, dx)
    half_fan = np.radians(fan_angle_deg / 2)

    corridors = []
    for i in range(n_rays):
        frac = (i - (n_rays - 1) / 2) / ((n_rays - 1) / 2) if n_rays > 1 else 0
        angle = base_angle + frac * half_fan
        dist = np.sqrt(dx * dx + dy * dy)
        tx = viewpoint.x + dist * np.cos(angle)
        ty = viewpoint.y + dist * np.sin(angle)
        target = Viewpoint(f'T_{i}', float(tx), float(ty), target_center.z)
        corridors.append(ViewCorridor(
            f'Ray {i}', viewpoint, target, corridor_width=corridor_width))
    return corridors


def combine_corridors(corridors: Sequence[ViewCorridor],
                      px: float, py: float,
                      ground_elevation: float = 0.0) -> float:
    """Binding max building height at (px, py): min across all corridors.

    When several view corridors overlap the same point, every corridor
    must remain clear, so the allowed height is the most restrictive
    one — the element-wise minimum of the per-corridor envelopes.

    Parameters
    ----------
    corridors : list of ViewCorridor
        Active corridors.
    px, py : float
        Point easting / northing (m).
    ground_elevation : float, optional
        Ground elevation at the point (m).

    Returns
    -------
    float
        Max allowed building height above ground (m), >= 0.
    """
    return float(min(c.max_height_at(px, py, ground_elevation)
                     for c in corridors))


def terrain_corrected_height(corridors: Sequence[ViewCorridor],
                             px: float, py: float,
                             ground_elevation) -> float:
    """Combined max height above terrain at (px, py).

    Terrain correction: the sightline is fixed in space, so on rising
    ground the allowed building height shrinks by the ground elevation
    -- ``h_max = z_sightline - z_ground`` (the same relation used by
    ``ViewCorridor.max_height_at``).  The result is clamped to >= 0.

    Parameters
    ----------
    corridors : list of ViewCorridor
        Active corridors.
    px, py : float
        Point easting / northing (m).
    ground_elevation : float or callable(x, y) -> float
        Ground elevation (m) at the point, or a terrain model
        (e.g. a DEM interpolation) evaluated at (px, py).

    Returns
    -------
    float
        Max allowed building height above terrain (m), >= 0.
    """
    z_ground = (ground_elevation(px, py) if callable(ground_elevation)
                else float(ground_elevation))
    return combine_corridors(corridors, px, py, ground_elevation=z_ground)


def height_envelope(corridors: Sequence[ViewCorridor],
                    gx, gy,
                    ground_elevation=0.0):
    """Combined height constraint surface over a 2D grid.

    ``H[i, j]`` is the binding max building height at ``(gx[j], gy[i])``:
    the element-wise minimum across all corridors.  Cells no corridor
    constrains (infinite height) become NaN.

    Parameters
    ----------
    corridors : list of ViewCorridor
        Active corridors.
    gx, gy : array-like
        Grid coordinates (m) along easting / northing.
    ground_elevation : float, array-like, or callable, optional
        Ground elevation (m).  A 2D array must have shape
        ``(len(gy), len(gx))``; a callable is evaluated as
        ``ground_elevation(GX, GY)``.

    Returns
    -------
    GX, GY, H : ndarray
        Meshgrids and the combined height surface (m); H has shape
        ``(len(gy), len(gx))`` and NaN outside the constraint region.
    """
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    GX, GY = np.meshgrid(gx, gy)
    if callable(ground_elevation):
        Z = ground_elevation(GX, GY)
    elif np.ndim(ground_elevation) == 0:
        Z = np.full(GX.shape, float(ground_elevation))
    else:
        Z = np.asarray(ground_elevation, dtype=float)

    H = np.full(GX.shape, float('inf'))
    for cor in corridors:
        h_cor = np.array([
            [cor.max_height_at(GX[i, j], GY[i, j], Z[i, j])
             for j in range(len(gx))]
            for i in range(len(gy))
        ])
        H = np.minimum(H, h_cor)
    H[H > 1e18] = np.nan
    return GX, GY, H


def check_building_compliance(building_footprint: np.ndarray,
                              building_height: float,
                              corridors: Sequence[ViewCorridor],
                              tolerance_m: float = 0.5) -> dict:
    """Check if a building complies with all view corridor constraints.

    Parameters
    ----------
    building_footprint : ndarray, shape (N, 2)
        Building outline vertices (easting, northing) in m.
    building_height : float
        Building height above ground (m).
    corridors : list of ViewCorridor
        Active view corridors.
    tolerance_m : float, optional
        Height tolerance (m) before a violation is recorded.

    Returns
    -------
    dict
        Keys ``compliant`` (bool), ``violations`` (list of dict),
        ``margin`` (float, min h_max - height over all checks).
    """
    violations = []
    min_margin = float('inf')

    # Check each vertex and the centroid
    check_points = np.vstack(
        [building_footprint, building_footprint.mean(axis=0, keepdims=True)])

    for cor in corridors:
        for pt in check_points:
            h_max = cor.max_height_at(pt[0], pt[1])
            margin = h_max - building_height
            if margin < min_margin:
                min_margin = margin
            if margin < -tolerance_m:
                violations.append({
                    'corridor': cor.name,
                    'point': tuple(pt),
                    'h_max': h_max,
                    'building_h': building_height,
                    'excess': -margin,
                })

    return {
        'compliant': len(violations) == 0,
        'violations': violations,
        'margin': min_margin,
    }


# ---------------------------------------------------------------------------
# Three-lines (三区三线) compliance
# ---------------------------------------------------------------------------


@dataclass
class ThreeLinesChecker:
    """Check parcels against three-lines constraints (binary containment).

    Attributes
    ----------
    urban_boundary : shapely Polygon
        城镇开发边界 — parcels must be inside.
    basic_farmland : shapely Polygon
        永久基本农田 — parcels must not intersect.
    ecological_redline : shapely Polygon
        生态保护红线 — parcels must not intersect.
    """

    urban_boundary: Polygon
    basic_farmland: Polygon
    ecological_redline: Polygon

    def check_parcel(self, parcel: Polygon, parcel_id: str) -> Optional[dict]:
        """Check one parcel.

        Parameters
        ----------
        parcel : shapely Polygon
            Proposed land-use parcel.
        parcel_id : str
            Identifier used in the result.

        Returns
        -------
        dict or None
            Violation dict, or None when compliant.  All three
            violation types carry ``severity: 'critical'``.
        """
        # Must be within the urban development boundary
        if not self.urban_boundary.contains(parcel):
            outside = parcel.difference(self.urban_boundary)
            return {
                'parcel_id': parcel_id,
                'violation': 'outside_urban_boundary',
                'severity': 'critical',
                'area_outside_m2': outside.area,
                'pct_outside': outside.area / parcel.area * 100,
            }
        # Must not intersect permanent basic farmland
        if parcel.intersects(self.basic_farmland):
            overlap = parcel.intersection(self.basic_farmland)
            return {
                'parcel_id': parcel_id,
                'violation': 'on_basic_farmland',
                'severity': 'critical',
                'overlap_area_m2': overlap.area,
                'pct_overlap': overlap.area / parcel.area * 100,
            }
        # Must not intersect the ecological conservation redline
        if parcel.intersects(self.ecological_redline):
            overlap = parcel.intersection(self.ecological_redline)
            return {
                'parcel_id': parcel_id,
                'violation': 'in_ecological_redline',
                'severity': 'critical',
                'overlap_area_m2': overlap.area,
                'pct_overlap': overlap.area / parcel.area * 100,
            }
        return None

    def check_all(self, parcels: Sequence[Tuple[str, Polygon]]) -> dict:
        """Check all parcels.

        Parameters
        ----------
        parcels : list of (parcel_id, Polygon)

        Returns
        -------
        dict
            Keys ``total_parcels``, ``violations``, ``compliant``,
            ``details`` (list of violation dicts).
        """
        violations = []
        for pid, geom in parcels:
            v = self.check_parcel(geom, pid)
            if v:
                violations.append(v)
        return {
            'total_parcels': len(parcels),
            'violations': len(violations),
            'compliant': len(parcels) - len(violations),
            'details': violations,
        }


class ProportionalChecker:
    """Check parcels with configurable overlap tolerances.

    Real planning bureaus apply 'de minimis' thresholds: a parcel that
    is 99% inside the urban boundary and 1% outside (survey / boundary
    digitization error) should not fail.  Tolerance values are policy
    choices, not mathematics — document them with every run.
    """

    def __init__(self, urban_boundary: Polygon,
                 basic_farmland: Polygon,
                 ecological_redline: Polygon,
                 max_outside_pct: float = 2.0,
                 max_farmland_pct: float = 0.5,
                 max_redline_pct: float = 0.5):
        self.urban_boundary = urban_boundary
        self.basic_farmland = basic_farmland
        self.ecological_redline = ecological_redline
        self.max_outside_pct = max_outside_pct
        self.max_farmland_pct = max_farmland_pct
        self.max_redline_pct = max_redline_pct

    def check_parcel(self, parcel: Polygon, parcel_id: str):
        """Check with tolerance.

        Parameters
        ----------
        parcel : shapely Polygon
        parcel_id : str

        Returns
        -------
        (passed, violations_list)
            ``violations_list`` holds (code, pct) tuples for each
            tolerance exceeded.
        """
        violations = []
        # Urban boundary check with tolerance
        if not self.urban_boundary.contains(parcel):
            outside = parcel.difference(self.urban_boundary)
            pct = outside.area / parcel.area * 100
            if pct > self.max_outside_pct:
                violations.append(('outside_urban', pct))
        # Farmland check with tolerance
        if parcel.intersects(self.basic_farmland):
            overlap = parcel.intersection(self.basic_farmland)
            pct = overlap.area / parcel.area * 100
            if pct > self.max_farmland_pct:
                violations.append(('on_farmland', pct))
        # Redline check with tolerance
        if parcel.intersects(self.ecological_redline):
            overlap = parcel.intersection(self.ecological_redline)
            pct = overlap.area / parcel.area * 100
            if pct > self.max_redline_pct:
                violations.append(('in_redline', pct))
        return len(violations) == 0, violations


def generate_with_constraints(urban_boundary: Polygon,
                              basic_farmland: Polygon,
                              ecological_redline: Polygon,
                              n_parcels: int = 50,
                              min_size: float = 10000,
                              max_size: float = 50000,
                              max_attempts_per_parcel: int = 100,
                              seed: int = 123) -> Tuple[List[Tuple[str, Polygon]], int]:
    """Generate land-use parcels respecting three-lines constraints.

    Uses rejection sampling: propose a random parcel, accept only if it
    lies entirely within the buildable area (urban boundary minus
    farmland minus redline).  This prevents violations at generation
    time instead of detecting them post-hoc.

    Parameters
    ----------
    urban_boundary, basic_farmland, ecological_redline : shapely Polygon
        Three-lines layers.
    n_parcels : int, optional
        Target number of parcels.
    min_size, max_size : float, optional
        Parcel area bounds (m²).
    max_attempts_per_parcel : int, optional
        Acceptance-attempt budget.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    parcels : list of (parcel_id, Polygon)
        Constraint-compliant parcels.
    attempts : int
        Total proposal attempts (acceptance rate = len(parcels)/attempts).

    Raises
    ------
    ValueError
        When the constraints leave no buildable area.
    """
    rng = np.random.default_rng(seed)
    buildable = urban_boundary.difference(basic_farmland).difference(ecological_redline)
    if buildable.is_empty:
        raise ValueError('No buildable area — constraints cover entire site')

    parcels = []
    attempts = 0

    while len(parcels) < n_parcels and attempts < n_parcels * max_attempts_per_parcel:
        attempts += 1
        # Random center within the buildable area bbox
        minx, miny, maxx, maxy = buildable.bounds
        cx = rng.uniform(minx + 100, maxx - 100)
        cy = rng.uniform(miny + 100, maxy - 100)
        area = rng.uniform(min_size, max_size)
        aspect = rng.uniform(0.5, 2.0)
        w = np.sqrt(area * aspect)
        h = area / w
        candidate = Polygon([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                             (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)])
        # Must be entirely within the buildable area
        if buildable.contains(candidate):
            parcels.append((f'GEN-{len(parcels):03d}', candidate))

    return parcels, attempts


__all__ = [
    "Viewpoint", "ViewCorridor", "generate_view_fan",
    "combine_corridors", "terrain_corrected_height", "height_envelope",
    "check_building_compliance", "ThreeLinesChecker",
    "ProportionalChecker", "generate_with_constraints",
]
