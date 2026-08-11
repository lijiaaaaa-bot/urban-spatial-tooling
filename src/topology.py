"""
Reusable topology-check functions for urban spatial data.

Checks implemented:
- Coverage: do parcels fully cover a boundary?
- Overlap: do any parcels intersect each other?
- Containment: are all features inside the boundary?

All checks use shapely spatial predicates and return structured results
suitable for both programmatic use and human-readable reporting.
"""

from typing import List, Tuple

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Area tolerance (m²) below which gaps / outside-poking are treated as
# floating-point noise rather than as real topological violations.
#
# GEOS predicates such as `.covers()` and `.within()` compare coordinates
# exactly, so two geometries that agree to the last bit still fail the
# predicate even when the true (area-based) difference is zero.  Planners
# care about *material* gaps/overlaps/poking, so coverage and containment
# are judged on measured area difference with this tolerance.  A genuine
# violation (a missing parcel, a displaced polygon) is orders of magnitude
# larger than 1e-4 m² (0.01 cm²) on a km²-scale site.
AREA_TOLERANCE_SQM = 1e-4


def check_coverage(
    parcels: gpd.GeoDataFrame,
    boundary: Polygon,
    tol: float = AREA_TOLERANCE_SQM,
) -> Tuple[bool, List[Polygon]]:
    """
    Check whether *parcels* fully cover *boundary*.

    Coverage is judged by the measured area of the uncovered region
    ``boundary - union(parcels)``: a gap smaller than *tol* m² (i.e.
    floating-point noise) is not considered a violation.  This is
    deliberately more robust than the coordinate-exact ``union.covers()``
    predicate, which fails on last-bit rounding even for a perfect
    partition.

    Parameters
    ----------
    parcels : GeoDataFrame
        Polygons representing land parcels / zones.
    boundary : Polygon
        The reference boundary polygon.
    tol : float, optional
        Maximum acceptable uncovered area in m² (default 1e-4).

    Returns
    -------
    (covered, gaps)
        covered : bool
            True if parcels fully cover the boundary (within *tol*).
        gaps : list of Polygon
            Polygons representing uncovered areas within the boundary.
    """
    union = unary_union(parcels.geometry.values)
    gaps_polygon = boundary.difference(union)
    if gaps_polygon.is_empty:
        return True, []
    if gaps_polygon.area <= tol:
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
    features: gpd.GeoDataFrame,
    boundary: Polygon,
    tol: float = AREA_TOLERANCE_SQM,
) -> Tuple[bool, gpd.GeoDataFrame]:
    """
    Check whether all features are fully contained within *boundary*.

    Containment is judged by the measured area ``feature - boundary``
    (i.e. how much of the feature pokes outside the boundary).  A feature
    whose outside area is below *tol* m² (floating-point noise) is not a
    violator.  This is deliberately more robust than the coordinate-exact
    ``.within()`` predicate, which fails on last-bit rounding for
    boundary-sharing features of an otherwise perfect partition.

    Parameters
    ----------
    features : GeoDataFrame
        Features to check.
    boundary : Polygon
        The containing boundary.
    tol : float, optional
        Maximum acceptable outside area per feature in m² (default 1e-4).

    Returns
    -------
    (all_inside, violators)
        all_inside : bool
            True if every feature is inside the boundary (within *tol*).
        violators : GeoDataFrame
            Subset of *features* that fall materially (>= *tol* m²) outside
            the boundary.
    """
    mask = features.geometry.apply(
        lambda g: g.difference(boundary).area <= tol
    )
    violators = features[~mask].copy()
    return len(violators) == 0, violators
