"""
Building setback (建筑退线) and buildable-envelope tools.

Extracted from notebooks/14-building-setback.ipynb.

Provides:
- setback_envelope_by_edges: per-edge inward offsets (mitered corners)
- edge_setbacks_by_side: assign setbacks to ring edges by compass side,
  robust to ring winding and starting vertex
- road_constraint, buildable_from_roads: road-network-aware envelope —
  the CAD/GIS-standard construction (buffer centerline by redline
  half-width + class setback, union zones, subtract from the
  property-setback-shrunk parcel)
- generate_footprints, validate_footprints: footprint generation inside
  an envelope with south-facing preference and fire-separation spacing
- check_fire_separation, fire_separation_buffer, required_fire_separation,
  fire_grade_of, is_high_rise: inter-building fire separation (建筑防火
  间距), GB 50016-2014 — pairwise distance check (exact), half-distance
  buffer zones, and grade classification

Standard references
-------------------
- DB11/T 996-2013 (Beijing construction project planning design rules)
- GB 50180-2018《城市居住区规划设计标准》
- GB 50016-2014《建筑设计防火规范》(fire separation)

All numbers are typical reference values — the exact values always come
from the site's regulatory plan (控规).  Nothing here is hard-coded as a
regulation: every parameter is a table entry.

Method note: the redline, not the parcel edge, is the constraint.  The
per-edge method (Method 1) is only valid when the redline coincides with
the parcel edge; the road-aware method (Method 2) measures from the
redline and is always correct for angled/curved roads.

All geometry is shapely — no GPU dependency.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import LineString, Polygon, box

# Road class -> (Chinese name, redline half-width, redline setback)
ROAD_CLASS_PARAMS: Dict[str, dict] = {
    'arterial':  {'name_zh': '主干路', 'half_width': 30.0, 'setback': 15.0},
    'secondary': {'name_zh': '次干路', 'half_width': 20.0, 'setback': 10.0},
    'local':     {'name_zh': '支路',   'half_width': 10.0, 'setback': 5.0},
}

# Land use -> property-line setback (typical)
PROPERTY_SETBACK_BY_USE: Dict[str, float] = {
    'R': 5.0,   # residential
    'A': 5.0,   # public service
    'B': 3.0,   # commercial
    'M': 3.0,   # industrial
}

# Fire separation (GB 50016-2014) — minimum clear distance between buildings
FIRE_SEPARATION: Dict[str, float] = {
    'low_low':   6.0,   # 多层-多层
    'high_low':  9.0,   # 高层-多层
    'high_high': 13.0,  # 高层-高层
}

# Building palette: (building_type, width, depth, floors)
# Aspect ratios within [0.35, 1.0] (typical 1:2 to 1:2.5 slabs)
PALETTE: List[tuple] = [
    ('residential_slab',      30.0, 12.0, 6),   # d/w = 0.40 (1:2.5 slab)
    ('residential_slab_slim', 24.0, 11.0, 6),   # d/w = 0.46 (1:2.2 slab)
    ('residential_tower',     22.0, 22.0, 12),  # d/w = 1.00 (tower)
]

# Min clear distance between footprints (fire separation target, m)
DEFAULT_SPACING = (6.0, 6.0)


def setback_envelope_by_edges(parcel: Polygon,
                              edge_setbacks: Sequence[float],
                              return_lines: bool = False):
    """Buildable envelope via per-edge inward offsets (mitered corners).

    For each edge of the parcel exterior ring, offset the edge inward by
    its setback; the buildable envelope is the intersection of all
    interior half-planes, clipped to the parcel.

    Parameters
    ----------
    parcel : shapely Polygon
        Parcel boundary.
    edge_setbacks : list of float
        Setback per exterior-ring edge, aligned with the ring order.
    return_lines : bool, optional
        Also return the offset LineStrings (for drawing).

    Returns
    -------
    shapely Polygon
        Buildable envelope, contained in ``parcel``.
    (Polygon, list of (LineString, float)) if ``return_lines`` is True.
    """
    coords = list(parcel.exterior.coords)[:-1]
    n = len(coords)
    assert len(edge_setbacks) == n, 'one setback per edge'
    interior = parcel.representative_point()
    ip = np.array([interior.x, interior.y])

    minx, miny, maxx, maxy = parcel.bounds
    diag = float(np.hypot(maxx - minx, maxy - miny))
    big_box = Polygon([(minx - diag, miny - diag), (maxx + diag, miny - diag),
                       (maxx + diag, maxy + diag), (minx - diag, maxy + diag)])

    envelope = big_box
    lines = []
    for i in range(n):
        a = np.array(coords[i])
        b = np.array(coords[(i + 1) % n])
        d = edge_setbacks[i]
        if d <= 1e-9:
            continue
        e = b - a
        norm = np.linalg.norm(e)
        t = e / norm                      # along-edge direction
        L = np.array([-e[1], e[0]]) / norm  # left normal
        s = np.sign(np.dot(ip - a, L))    # sign so L points to the interior
        L = s * L
        o = a + d * L                     # point on the inward offset line
        ext = diag
        # strip beyond the offset line (the 'outside' half of the line)
        strip = Polygon([o - t * ext, o + t * ext,
                         o + t * ext - L * ext, o - t * ext - L * ext])
        envelope = envelope.difference(strip)
        lines.append((LineString([o - t * ext, o + t * ext]), d))

    envelope = envelope.intersection(parcel)
    if return_lines:
        return envelope, lines
    return envelope


def edge_setbacks_by_side(parcel: Polygon,
                          rules: Dict[str, float]) -> Tuple[List[float], List[str]]:
    """Assign a setback to each ring edge by its compass side.

    Parameters
    ----------
    parcel : shapely Polygon
        Parcel boundary.
    rules : dict
        Maps 'E'/'N'/'W'/'S' to setbacks.  Robust to ring winding and to
        which vertex the ring starts at (shapely's box() starts at the
        east edge, so positional lists are easy to get wrong).

    Returns
    -------
    setbacks : list of float
        Aligned with the ring order.
    side_names : list of str
        Per-edge compass side.
    """
    ring = list(parcel.exterior.coords)[:-1]
    n = len(ring)
    minx, miny, maxx, maxy = parcel.bounds
    setbacks = []
    sides = []
    for i in range(n):
        a = np.array(ring[i])
        b = np.array(ring[(i + 1) % n])
        mx, my = (a + b) / 2
        if abs(mx - minx) < 1e-6:
            side = 'W'
        elif abs(mx - maxx) < 1e-6:
            side = 'E'
        elif abs(my - miny) < 1e-6:
            side = 'S'
        elif abs(my - maxy) < 1e-6:
            side = 'N'
        else:
            side = '?'
        setbacks.append(rules.get(side, 0.0))
        sides.append(side)
    return setbacks, sides


def road_constraint(centerline: LineString, road_class: str) -> Polygon:
    """Forbidden zone around a road: redline width + class setback.

    Parameters
    ----------
    centerline : shapely LineString
        Road centerline.
    road_class : str
        One of :data:`ROAD_CLASS_PARAMS` keys.

    Returns
    -------
    shapely Polygon
        Buffer of the centerline by redline half-width + setback.
    """
    p = ROAD_CLASS_PARAMS[road_class]
    return centerline.buffer(p['half_width'] + p['setback'])


def buildable_from_roads(parcel: Polygon,
                         road_constraints,
                         property_setback: float) -> Optional[Polygon]:
    """Buildable area = parcel shrunk by property setback, minus road zones.

    Parameters
    ----------
    parcel : shapely Polygon
        Parcel boundary.
    road_constraints : shapely geometry
        Union of road constraint zones (see :func:`road_constraint`);
        may be an empty polygon.
    property_setback : float
        Property-line setback (m) for non-road edges.

    Returns
    -------
    shapely Polygon or None
        Buildable envelope; None if the constraints leave no area.
    """
    if property_setback > 0:
        base = parcel.buffer(-property_setback)
    else:
        base = parcel
    if road_constraints.is_empty:
        return base
    out = base.difference(road_constraints)
    return out if not out.is_empty else None


def generate_footprints(envelope: Polygon,
                        palette: Sequence[tuple] = PALETTE,
                        spacing: Tuple[float, float] = DEFAULT_SPACING,
                        target_coverage: float = 1.0,
                        parcel_area: Optional[float] = None,
                        rng: Optional[np.random.Generator] = None,
                        p_south: float = 0.85,
                        jitter: float = 0.3,
                        max_tries: int = 8000):
    """Place rectangular footprints inside ``envelope`` on a jittered grid.

    Generation rules:
    - South-facing preference (住宅朝南): angle 0 means the long axis runs
      E-W (85% of placements), occasional N-S slabs for variety.
    - Aspect ratios are enforced on the palette (depth/width within
      [0.35, 1.0]).
    - Minimum clear spacing between footprints (fire separation); the
      jitter reduces it to spacing - jitter (e.g. 6 m -> >= 5.4 m).
    - Stop when the covered footprint area reaches the target fraction
      of the parcel.

    Parameters
    ----------
    envelope : shapely Polygon
        Buildable envelope (e.g. from :func:`buildable_from_roads`).
    palette : list of (building_type, width, depth, floors), optional
        Building palette, defaulting to :data:`PALETTE`.
    spacing : (sx, sy), optional
        Grid spacing / min clear distance between footprints (m).
    target_coverage : float, optional
        Stop when covered footprint area reaches this fraction of the
        parcel area.
    parcel_area : float, optional
        Parcel area used for the coverage target (defaults to the
        envelope area).
    rng : numpy Generator, optional
        Random number generator (deterministic seeds reproduce layouts).
    p_south : float, optional
        Probability of a south-facing (E-W long axis) placement.
    jitter : float, optional
        Placement jitter (m).  Must not exceed the spacing guarantee.
    max_tries : int, optional
        Candidate attempts before giving up.

    Returns
    -------
    placed : list of (rect, angle, btype, floors)
        Feasible footprints.
    covered : float
        Total covered footprint area (m²).
    """
    if rng is None:
        rng = np.random.default_rng()
    sx, sy = spacing
    minx, miny, maxx, maxy = envelope.bounds
    wmax = max(p[1] for p in palette)
    dmax = max(p[2] for p in palette)
    if parcel_area is None:
        parcel_area = envelope.area
    target_area = target_coverage * parcel_area

    placed = []
    placed_buffers = []
    covered = 0.0
    tries = 0
    # Note: jitter must be applied to copies — mutating the loop variable
    # would let it drift across columns (a random walk) and break the
    # spacing guarantee.
    for cy0 in np.arange(miny + dmax / 2 + jitter, maxy - dmax / 2, dmax + sy):
        for cx0 in np.arange(minx + wmax / 2 + jitter, maxx - wmax / 2, wmax + sx):
            if covered >= target_area or tries >= max_tries:
                break
            tries += 1
            cx = cx0 + rng.uniform(-jitter, jitter)
            cy = cy0 + rng.uniform(-jitter, jitter)
            btype, w, d, floors = palette[rng.integers(0, len(palette))]
            angs = [0.0] if rng.random() < p_south else [90.0]
            for ang in angs:
                if ang == 0.0:
                    rect = box(cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2)
                else:
                    rect = box(cx - d / 2, cy - w / 2, cx + d / 2, cy + w / 2)
                if not envelope.buffer(-0.05).covers(rect):
                    continue
                if any(rect.intersects(buff) for buff in placed_buffers):
                    continue
                placed.append((rect, ang, btype, floors))
                placed_buffers.append(rect.buffer(min(sx, sy) / 2))
                covered += rect.area
                break
        if covered >= target_area:
            break
    return placed, covered


def validate_footprints(placed: Sequence[tuple],
                        envelope: Polygon,
                        spacing: Tuple[float, float] = DEFAULT_SPACING,
                        jitter: float = 0.3) -> Tuple[int, float]:
    """Containment + pairwise spacing check for placed footprints.

    Parameters
    ----------
    placed : list of (rect, angle, btype, floors)
        Footprints from :func:`generate_footprints`.
    envelope : shapely Polygon
        Envelope each footprint must sit inside.
    spacing : (sx, sy), optional
        Spacing used at generation.
    jitter : float, optional
        Jitter used at generation (min gap = min(spacing) - jitter).

    Returns
    -------
    issues : int
        Number of containment failures.
    min_gap : float
        Minimum pairwise distance between footprints (m).
    """
    issues = 0
    min_gap = float('inf')
    for i in range(len(placed)):
        rect = placed[i][0]
        if not envelope.buffer(-0.05).covers(rect):
            issues += 1
        for j in range(i + 1, len(placed)):
            min_gap = min(min_gap, rect.distance(placed[j][0]))
    return issues, min_gap


def summarize(placed: Sequence[tuple], parcel_area: float,
              label: str) -> Tuple[int, float, float]:
    """Summarize a footprint layout: count, coverage, FAR.

    Parameters
    ----------
    placed : list of (rect, angle, btype, floors)
    parcel_area : float
        Parcel area (m²) used as the FAR denominator.
    label : str
        Label printed with the summary.

    Returns
    -------
    (n, area, far) : count, footprint area (m²), floors-weighted FAR.
    """
    n = len(placed)
    area = sum(p[0].area for p in placed)
    far = sum(p[0].area * p[3] for p in placed) / parcel_area
    print(f'{label}: {n} footprints, footprint area {area:7.0f} m², '
          f'coverage {area / parcel_area * 100:5.1f}%, FAR {far:5.2f} (floors-weighted)')
    return n, area, far


# ---------------------------------------------------------------------------
# Inter-building fire separation (建筑防火间距) — GB 50016-2014
# ---------------------------------------------------------------------------
# Minimum clear distance between two buildings by fire resistance grade
# (耐火等级).  The lower the grade number the better the fire resistance,
# hence the smaller the required separation.
FIRE_GRADE_SEPARATION: Dict[int, float] = {
    1: 6.0,   # 一、二级耐火等级
    2: 6.0,
    3: 9.0,   # 三级
    4: 13.0,  # 四级
}
DEFAULT_FIRE_GRADE: int = 2
# High-rise residential (高层住宅): height > 27 m or more than 8 storeys.
FIRE_HIGH_RISE: str = 'high_rise'
FIRE_HIGH_RISE_SEPARATION: float = 13.0


def is_high_rise(height_m: Optional[float] = None,
                 floors: Optional[int] = None) -> bool:
    """High-rise residential per GB 50016-2014: height > 27 m or > 8 storeys.

    Parameters
    ----------
    height_m : float, optional
        Building height (m).
    floors : int, optional
        Number of storeys.

    Returns
    -------
    bool
        True if either criterion marks the building high-rise.
    """
    if height_m is not None and height_m > 27.0:
        return True
    if floors is not None and floors > 8:
        return True
    return False


def fire_grade_of(height_m: Optional[float] = None,
                  floors: Optional[int] = None,
                  default: int = DEFAULT_FIRE_GRADE) -> Union[int, str]:
    """Fire grade (int 1-4) or ``FIRE_HIGH_RISE`` for a building.

    High-rise buildings (see :func:`is_high_rise`) report ``FIRE_HIGH_RISE``;
    everything else takes ``default`` (Grade 2 unless overridden).

    Parameters
    ----------
    height_m : float, optional
        Building height (m).
    floors : int, optional
        Number of storeys.
    default : int, optional
        Grade used when the building is not high-rise (default 2).

    Returns
    -------
    int or str
        Fire grade of the building.
    """
    if is_high_rise(height_m, floors):
        return FIRE_HIGH_RISE
    return default


def required_fire_separation(grade_a: Union[int, str],
                             grade_b: Union[int, str]) -> float:
    """Minimum clear distance (m) between two buildings, GB 50016-2014.

    A grade is an int 1-4 (fire resistance grade) or ``FIRE_HIGH_RISE``.
    Rules:
    - high-rise vs anything: 13 m (:data:`FIRE_HIGH_RISE_SEPARATION`);
    - otherwise the pair takes the requirement of the less fire-resistant
      (larger grade number) building: ``max(grade_a, grade_b)`` looked up
      in :data:`FIRE_GRADE_SEPARATION`.

    Parameters
    ----------
    grade_a, grade_b : int (1-4) or 'high_rise'
        Fire grades of the two buildings.

    Returns
    -------
    float
        Required minimum distance (m).

    Raises
    ------
    ValueError
        If a grade is not 1-4 or ``FIRE_HIGH_RISE``.
    """
    if FIRE_HIGH_RISE in (grade_a, grade_b):
        return FIRE_HIGH_RISE_SEPARATION
    ga, gb = int(grade_a), int(grade_b)
    for g in (ga, gb):
        if g not in FIRE_GRADE_SEPARATION:
            raise ValueError(f'invalid fire grade {g!r}; expected 1-4 or '
                             f'{FIRE_HIGH_RISE!r}')
    return FIRE_GRADE_SEPARATION[max(ga, gb)]


def fire_separation_buffer(building: Polygon,
                           grade: Union[int, str] = DEFAULT_FIRE_GRADE) -> Polygon:
    """Half the required separation as a buffer zone around a footprint.

    The radius is half the separation this building must keep from a default
    (Grade 2) neighbour: ``required_fire_separation(grade, DEFAULT_FIRE_GRADE)
    / 2`` (Grade 2 -> 3 m, Grade 3 -> 4.5 m, Grade 4 -> 6.5 m, high-rise ->
    6.5 m).

    Two buildings are too close iff their zones intersect.  The test is
    *exact* when both buildings share one grade (sum of half-zones equals
    the pair requirement); with mixed grades the sum of half-zones is
    <= the true requirement, so the zone test can only *miss* violations,
    never invent them.  Use :func:`check_fire_separation` for the exact
    pair check.

    Parameters
    ----------
    building : shapely Polygon
        Building footprint.
    grade : int (1-4) or 'high_rise', optional
        Fire grade of the building (default: Grade 2).

    Returns
    -------
    shapely Polygon
        ``building`` buffered by half its required separation.
    """
    radius = required_fire_separation(grade, DEFAULT_FIRE_GRADE) / 2.0
    return building.buffer(radius)


def check_fire_separation(buildings: Sequence[Polygon],
                          fire_grade_map: Optional[Dict[int, Union[int, str]]] = None) -> Dict:
    """Pairwise fire-separation check (exact, O(N²)).

    For every pair of footprints the minimum clear distance is measured
    with shapely and compared against the GB 50016-2014 requirement of the
    pair's grades (see :func:`required_fire_separation`).

    Parameters
    ----------
    buildings : sequence of shapely Polygon
        Building footprints.
    fire_grade_map : dict, optional
        Maps building index -> fire grade (int 1-4) or ``FIRE_HIGH_RISE``.
        Missing indices default to Grade 2 (:data:`DEFAULT_FIRE_GRADE`).

    Returns
    -------
    dict
        - n_buildings, n_pairs: layout size
        - violations: list of dicts with keys i, j (indices), distance (m),
          required (m), grade_i, grade_j
        - n_violations: len(violations)
        - min_distance: minimum pairwise distance (m); inf with < 2 buildings
    """
    if fire_grade_map is None:
        fire_grade_map = {}
    n = len(buildings)
    violations = []
    min_distance = float('inf')
    for i in range(n):
        gi = fire_grade_map.get(i, DEFAULT_FIRE_GRADE)
        for j in range(i + 1, n):
            gj = fire_grade_map.get(j, DEFAULT_FIRE_GRADE)
            d = buildings[i].distance(buildings[j])
            min_distance = min(min_distance, d)
            req = required_fire_separation(gi, gj)
            if d < req:
                violations.append({
                    'i': i, 'j': j, 'distance': d, 'required': req,
                    'grade_i': gi, 'grade_j': gj,
                })
    return {
        'n_buildings': n,
        'n_pairs': n * (n - 1) // 2,
        'violations': violations,
        'n_violations': len(violations),
        'min_distance': min_distance,
    }


__all__ = [
    "ROAD_CLASS_PARAMS", "PROPERTY_SETBACK_BY_USE", "FIRE_SEPARATION",
    "PALETTE", "DEFAULT_SPACING", "setback_envelope_by_edges",
    "edge_setbacks_by_side", "road_constraint", "buildable_from_roads",
    "generate_footprints", "validate_footprints", "summarize",
    "FIRE_GRADE_SEPARATION", "DEFAULT_FIRE_GRADE", "FIRE_HIGH_RISE",
    "FIRE_HIGH_RISE_SEPARATION", "is_high_rise", "fire_grade_of",
    "required_fire_separation", "fire_separation_buffer",
    "check_fire_separation",
]
