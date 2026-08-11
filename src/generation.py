"""
Reusable generation helpers for urban spatial data.

Provides:
- Grid-based subdivision of a boundary polygon
- Building footprint generation within parcels
- Three-level grid road network generation
"""

import random
from typing import List, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union

# Area tolerance (m²) used to classify polygonize faces as inside the
# boundary.  Faces are kept when the area poking outside the boundary is
# below this threshold (i.e. pure floating-point noise).  See also
# ``src.topology.AREA_TOLERANCE_SQM``.
_FACE_INSIDE_TOL_SQM = 1e-6


# Mapping from land-use code to typical building types
#
# Codes follow GB 50137-2011 (城市用地分类与规划建设用地标准):
#   E 非建设用地 (non-construction land) intentionally has no buildings.
#   Education (中小学/教育科研) is A3, part of A 公共管理与公共服务用地.
_LAND_USE_TO_BUILDING = {
    "R": ["residential_tower", "residential_slab"],
    "A": ["office_tower", "office_podium", "school_building", "sports_hall"],
    "B": ["commercial_block", "mixed_use_tower"],
    "M": ["factory_shed", "workshop"],
    "W": ["warehouse", "logistics_center"],
    "S": ["bus_station", "metro_station"],
    "U": ["utility_station", "substation"],
    "G": ["park_pavilion", "sports_venue"],
}


# Standard land-use colour palette for Chinese planning
LAND_USE_COLORS = {
    "R": "#FFF5E1",  # Residential — warm cream
    "A": "#E8F0FE",  # Administration / public service — light blue
    "B": "#FFE4E1",  # Commercial — misty rose
    "M": "#E8E8E8",  # Industrial — light grey
    "W": "#D3D3D3",  # Logistics — medium grey
    "S": "#F0E68C",  # Transportation — khaki
    "U": "#C8E6C9",  # Utilities — pale green
    "G": "#A5D6A7",  # Green space — green
    "E": "#F5F5F5",  # Non-construction land (GB 50137-2011) — neutral
}

# Longer descriptive labels (GB 50137-2011)
LAND_USE_LABELS = {
    "R": "R 居住用地",
    "A": "A 公共管理与公共服务用地",
    "B": "B 商业服务业设施用地",
    "M": "M 工业用地",
    "W": "W 物流仓储用地",
    "S": "S 道路与交通设施用地",
    "U": "U 公用设施用地",
    "G": "G 绿地与广场用地",
    "E": "E 非建设用地",
}


def subdivide_boundary(
    boundary: Polygon, n_rows: int, n_cols: int
) -> List[Polygon]:
    """
    Subdivide a boundary polygon into a grid of cells clipped to the boundary.

    The boundary ring and all grid lines are merged in a **single planar
    noding pass** (`unary_union` + `polygonize`).  Every edge of the result
    comes from one noded graph, so adjacent cells share *exactly* the same
    coordinates and the cells form a true partition of the boundary:

    - no overlaps — neighbouring cells touch only along shared edges
      (intersections are 0-area LineStrings, never sliver polygons);
    - no gaps — the union of the cells covers the boundary exactly;
    - containment — every cell lies inside the boundary.

    This avoids the floating-point slivers, tiny gaps and edge poking that
    arise when each cell is independently intersected with the boundary.

    Parameters
    ----------
    boundary : Polygon
        The boundary to subdivide.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.

    Returns
    -------
    list of Polygon
        Sub-polygons that lie within the boundary.  Some edge cells may be
        irregular (clipped) polygons.
    """
    minx, miny, maxx, maxy = boundary.bounds
    cell_width = (maxx - minx) / n_cols
    cell_height = (maxy - miny) / n_rows

    # Build the planar line network: the boundary ring plus all grid lines
    # spanning the boundary's bounding box.
    lines = [boundary.boundary]
    for r in range(1, n_rows):
        y = miny + r * cell_height
        lines.append(LineString([(minx, y), (maxx, y)]))
    for c in range(1, n_cols):
        x = minx + c * cell_width
        lines.append(LineString([(x, miny), (x, maxy)]))

    # unary_union nodes the network (single noding pass, snap-rounding);
    # polygonize then extracts the faces of the arrangement.  All faces of
    # the same arrangement share exactly the same edge coordinates.
    noded = unary_union(lines)
    faces = [g for g in polygonize(noded) if g.geom_type == "Polygon"]

    # Keep only faces inside the boundary.  polygonize also returns bounded
    # faces that lie between the ring and the grid's bounding box; classify
    # by measured area outside the boundary so sub-microscopic floating-point
    # poking at the ring does not drop boundary-touching cells (unlike the
    # coordinate-exact `.within()` predicate).
    return [f for f in faces if f.difference(boundary).area <= _FACE_INSIDE_TOL_SQM]


def generate_buildings_in_parcel(
    parcel: Polygon,
    coverage_ratio: float = 0.35,
    bldg_types: Optional[List[str]] = None,
    bldg_width: float = 15.0,
    bldg_depth: float = 12.0,
    random_seed: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Generate simple building footprints within a parcel.

    Buildings are placed on a grid within the parcel, with random offsets,
    and are clipped to ensure they stay within the parcel boundary.

    Parameters
    ----------
    parcel : Polygon
        The parent parcel.
    coverage_ratio : float
        Target building coverage ratio (0 to 1).  Building count is derived
        from this.
    bldg_types : list of str, optional
        Candidate building types.  One is chosen randomly per building.
    bldg_width : float
        Building footprint width in metres.
    bldg_depth : float
        Building footprint depth in metres.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    GeoDataFrame
        With columns ``building_type``, ``parcel_id``, and geometry (Polygon).
    """
    rng = random.Random(random_seed)

    if bldg_types is None:
        bldg_types = ["generic"]

    minx, miny, maxx, maxy = parcel.bounds
    parcel_area = parcel.area

    # Calculate number of buildings to roughly hit the coverage ratio
    bldg_area = bldg_width * bldg_depth
    target_count = max(1, int((parcel_area * coverage_ratio) / bldg_area))

    # Grid spacing
    cols = max(1, int(np.sqrt(target_count * (bldg_width / bldg_depth))))
    rows = max(1, int(np.ceil(target_count / cols)))

    x_spacing = (maxx - minx) / (cols + 1)
    y_spacing = (maxy - miny) / (rows + 1)

    buildings = []
    for r_idx in range(rows):
        for c_idx in range(cols):
            if len(buildings) >= target_count:
                break

            cx = minx + (c_idx + 1) * x_spacing + rng.uniform(
                -x_spacing * 0.2, x_spacing * 0.2
            )
            cy = miny + (r_idx + 1) * y_spacing + rng.uniform(
                -y_spacing * 0.2, y_spacing * 0.2
            )

            half_w = bldg_width / 2
            half_d = bldg_depth / 2
            bldg = box(cx - half_w, cy - half_d, cx + half_w, cy + half_d)

            if parcel.contains(bldg):
                b_type = rng.choice(bldg_types)
                buildings.append(
                    {"building_type": b_type, "geometry": bldg}
                )
            elif parcel.intersects(bldg):
                # Clip to parcel
                clipped = bldg.intersection(parcel)
                if not clipped.is_empty and isinstance(clipped, Polygon):
                    b_type = rng.choice(bldg_types)
                    buildings.append(
                        {"building_type": b_type, "geometry": clipped}
                    )

        if len(buildings) >= target_count:
            break

    if not buildings:
        return gpd.GeoDataFrame(
            columns=["building_type", "geometry"],
            crs=parcel._crs if hasattr(parcel, "_crs") else None,
        )

    gdf = gpd.GeoDataFrame(buildings, crs="EPSG:4548")
    return gdf


def assign_land_use(
    cells: List[Polygon],
    land_use_mix: Optional[dict] = None,
    random_seed: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Assign land-use codes to a list of cells based on a target mix.

    Parameters
    ----------
    cells : list of Polygon
        Sub-parcels to assign.
    land_use_mix : dict, optional
        Mapping of land_use_code → target fraction (0..1).  Defaults to a
        typical mixed urban mix.  Codes follow GB 50137-2011: education
        (A3) belongs to A 公共管理与公共服务用地, and E is 非建设用地
        (non-construction land), so E is not used for construction parcels.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    GeoDataFrame
        With columns ``land_use_code`` and geometry (Polygon).
    """
    rng = random.Random(random_seed)

    if land_use_mix is None:
        land_use_mix = {
            "R": 0.35,  # Residential — 居住用地
            "A": 0.18,  # Administration + education (A3) — 公共管理与公共服务用地
            "B": 0.12,  # Commercial — 商业服务业设施用地
            "G": 0.20,  # Green space — 绿地与广场用地
            "S": 0.15,  # Transportation — 道路与交通设施用地
        }

    codes = list(land_use_mix.keys())
    weights = list(land_use_mix.values())

    total_cells = len(cells)
    # Weighted random assignment
    assignments = rng.choices(codes, weights=weights, k=total_cells)

    records = []
    for cell, code in zip(cells, assignments):
        records.append({"land_use_code": code, "geometry": cell})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4548")
    return gdf


def _generate_grid_roads(
    boundary: Polygon, spacing_m: float, offset_m: float = 0
) -> List[LineString]:
    """
    Generate grid lines at a given spacing, clipped to the boundary.

    This is the core of the grid-road algorithm used in notebook 04
    (``generate_grid_roads``) and notebook 07 (``generate_grid_roads_gdf``):
    vertical (南北向) and horizontal (东西向) lines are laid over the
    boundary's bounding box and each is clipped to the boundary.  A line
    clipped into multiple pieces (e.g. at a re-entrant boundary) is
    flattened into individual LineString segments.

    Parameters
    ----------
    boundary : Polygon
        The boundary polygon in a projected, metre-based CRS.
    spacing_m : float
        Spacing between parallel grid lines, in metres.
    offset_m : float
        Offset of the first grid line from the boundary's minimum extent.

    Returns
    -------
    list of LineString
        Clipped road segments within the boundary.
    """
    minx, miny, maxx, maxy = boundary.bounds
    lines: List[LineString] = []

    # Vertical lines (南北向)
    x = minx + offset_m
    while x <= maxx:
        clipped = LineString([(x, miny), (x, maxy)]).intersection(boundary)
        if not clipped.is_empty:
            if clipped.geom_type == "LineString":
                lines.append(clipped)
            elif clipped.geom_type == "MultiLineString":
                lines.extend(list(clipped.geoms))
        x += spacing_m

    # Horizontal lines (东西向)
    y = miny + offset_m
    while y <= maxy:
        clipped = LineString([(minx, y), (maxx, y)]).intersection(boundary)
        if not clipped.is_empty:
            if clipped.geom_type == "LineString":
                lines.append(clipped)
            elif clipped.geom_type == "MultiLineString":
                lines.extend(list(clipped.geoms))
        y += spacing_m

    return lines


def generate_road_network(
    boundary: Polygon,
    secondary_spacing: float = 800,
    branch_spacing: float = 400,
) -> dict:
    """
    Generate a three-level grid road network within a boundary.

    Roads are generated as grid lines over the boundary's bounding box and
    clipped to the boundary (extracted from notebook 04 cell 4 and notebook
    07 cell 7).  Three hierarchical levels are produced:

    - ``'secondary'`` — 次干道, spaced ``secondary_spacing`` m apart
    - ``'branch'``    — 支路,   spaced ``branch_spacing`` m apart
    - ``'slow'``      — 慢行道, spaced ``branch_spacing / 2`` m apart

    The 4:2:1 spacing hierarchy mirrors the exploratory values used in
    notebook 04 (1200/600/250 m); the defaults here follow the usual
    次干路/支路 convention of 800/400 m.  The input boundary is expected to
    be in a projected, metre-based CRS such as EPSG:4548.

    Parameters
    ----------
    boundary : Polygon
        The boundary (in a projected, metre-based CRS, e.g. EPSG:4548)
        within which the road network is clipped.
    secondary_spacing : float
        Spacing in metres between 次干道 (secondary) roads.
    branch_spacing : float
        Spacing in metres between 支路 (branch) roads; the 慢行道 (slow)
        level uses half of this spacing.

    Returns
    -------
    dict
        Mapping ``'secondary'``, ``'branch'``, ``'slow'`` to lists of
        clipped LineString segments.
    """
    slow_spacing = branch_spacing / 2
    return {
        "secondary": _generate_grid_roads(boundary, secondary_spacing),
        "branch": _generate_grid_roads(boundary, branch_spacing),
        "slow": _generate_grid_roads(boundary, slow_spacing),
    }
