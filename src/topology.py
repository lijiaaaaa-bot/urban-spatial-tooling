"""
Reusable topology-check functions for urban spatial data.

Checks implemented:
- Coverage: do parcels fully cover a boundary?
- Overlap: do any parcels intersect each other?
- Containment: are all features inside the boundary?

All checks use shapely spatial predicates and return structured results
suitable for both programmatic use and human-readable reporting.
"""

from typing import List, Optional, Tuple

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union


def check_coverage(
    parcels: gpd.GeoDataFrame, boundary: Polygon
) -> Tuple[bool, List[Polygon]]:
    """
    Check whether *parcels* fully cover *boundary*.

    Parameters
    ----------
    parcels : GeoDataFrame
        Polygons representing land parcels / zones.
    boundary : Polygon
        The reference boundary polygon.

    Returns
    -------
    (covered, gaps)
        covered : bool
            True if parcels fully cover the boundary.
        gaps : list of Polygon
            Polygons representing uncovered areas within the boundary.
    """
    union = unary_union(parcels.geometry.values)
    if union.covers(boundary):
        return True, []

    gaps_polygon = boundary.difference(union)
    if gaps_polygon.is_empty:
        return True, []

    # Normalise to list of Polygons
    if gaps_polygon.geom_type == "Polygon":
        gaps = [gaps_polygon]
    elif gaps_polygon.geom_type == "MultiPolygon":
        gaps = list(gaps_polygon.geoms)
    elif gaps_polygon.geom_type == "GeometryCollection":
        gaps = [g for g in gaps_polygon.geoms if g.geom_type == "Polygon"]
    else:
        gaps = []

    return len(gaps) == 0, gaps


def check_overlaps(
    parcels: gpd.GeoDataFrame,
) -> Tuple[bool, List[Polygon]]:
    """
    Check whether any parcels overlap each other.

    Uses pairwise intersection detection.  Adjacent parcels that share an edge
    (1-D overlap) are NOT considered overlaps.

    Parameters
    ----------
    parcels : GeoDataFrame
        Polygons to check.  Must have at least 2 rows.

    Returns
    -------
    (clean, overlaps)
        clean : bool
            True if no 2-D overlaps were found.
        overlaps : list of Polygon
            Overlapping regions (intersections with area > 0).
    """
    n = len(parcels)
    overlap_regions: List[Polygon] = []

    for i in range(n):
        for j in range(i + 1, n):
            inter = parcels.geometry.iloc[i].intersection(
                parcels.geometry.iloc[j]
            )
            if not inter.is_empty and inter.area > 0:
                overlap_regions.append(inter)

    return len(overlap_regions) == 0, overlap_regions


def check_containment(
    features: gpd.GeoDataFrame, boundary: Polygon
) -> Tuple[bool, gpd.GeoDataFrame]:
    """
    Check whether all features are fully contained within *boundary*.

    Parameters
    ----------
    features : GeoDataFrame
        Features to check.
    boundary : Polygon
        The containing boundary.

    Returns
    -------
    (all_inside, violators)
        all_inside : bool
            True if every feature is within the boundary.
        violators : GeoDataFrame
            Subset of *features* that fall (even partially) outside the boundary.
    """
    mask = features.geometry.within(boundary)
    violators = features[~mask].copy()
    return len(violators) == 0, violators
