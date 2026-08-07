"""
Reusable generation helpers for urban spatial data.

Provides:
- Grid-based subdivision of a boundary polygon
- Building footprint generation within parcels
"""

import random
from typing import List, Optional

import geopandas as gpd
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box


# Mapping from land-use code to typical building types
_LAND_USE_TO_BUILDING = {
    "R": ["residential_tower", "residential_slab"],
    "A": ["office_tower", "office_podium"],
    "B": ["commercial_block", "mixed_use_tower"],
    "M": ["factory_shed", "workshop"],
    "W": ["warehouse", "logistics_center"],
    "S": ["bus_station", "metro_station"],
    "U": ["utility_station", "substation"],
    "G": ["park_pavilion", "sports_venue"],
    "E": ["school_building", "sports_hall"],
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
    "E": "#81C784",  # Education / public facilities — medium green
}

# Longer descriptive labels
LAND_USE_LABELS = {
    "R": "R 居住用地",
    "A": "A 公共管理与公共服务用地",
    "B": "B 商业服务业设施用地",
    "M": "M 工业用地",
    "W": "W 物流仓储用地",
    "S": "S 道路与交通设施用地",
    "U": "U 公用设施用地",
    "G": "G 绿地与广场用地",
    "E": "E 其他建设用地",
}


def subdivide_boundary(
    boundary: Polygon, n_rows: int, n_cols: int
) -> List[Polygon]:
    """
    Subdivide a boundary polygon into a grid of cells clipped to the boundary.

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

    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = minx + c * cell_width
            y0 = miny + r * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height
            cell = box(x0, y0, x1, y1)
            inter = cell.intersection(boundary)
            if not inter.is_empty:
                # Normalise to Polygons
                if inter.geom_type == "Polygon":
                    cells.append(inter)
                elif inter.geom_type == "MultiPolygon":
                    cells.extend(list(inter.geoms))
                elif inter.geom_type == "GeometryCollection":
                    cells.extend(
                        [g for g in inter.geoms if g.geom_type == "Polygon"]
                    )
    return cells


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
        typical mixed urban mix.
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
            "R": 0.35,  # Residential
            "A": 0.10,  # Administration
            "B": 0.12,  # Commercial
            "G": 0.20,  # Green space
            "S": 0.15,  # Transportation
            "E": 0.08,  # Education / other
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
