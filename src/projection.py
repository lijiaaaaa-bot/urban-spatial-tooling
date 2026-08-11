"""
Reusable projection utilities for urban spatial analysis.

Provides:
- CRS transformation of shapely geometries via pyproj
- Area calculation in EPSG:4548 (CGCS2000 / 3-degree Gauss-Kruger CM 117E)
- Area breakdown by group
"""

import geopandas as gpd
import numpy as np
import pyproj
from shapely.geometry import base
from shapely.ops import transform


# EPSG:4548 — CGCS2000 / 3-degree Gauss-Kruger CM 117E (metres)
# EPSG:4326 — WGS84 (degrees)
CRS_4548 = "EPSG:4548"
CRS_4326 = "EPSG:4326"


def _get_transformer(src_crs: str, dst_crs: str) -> pyproj.Transformer:
    """Build a pyproj Transformer, always including time dimension for robustness."""
    return pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    )


def transform_geometry(
    geom: base.BaseGeometry, src_crs: str, dst_crs: str
) -> base.BaseGeometry:
    """
    Transform a shapely geometry from src_crs to dst_crs.

    Parameters
    ----------
    geom : shapely geometry
        Input geometry in src_crs.
    src_crs : str
        Source CRS (e.g. "EPSG:4326").
    dst_crs : str
        Target CRS (e.g. "EPSG:4548").

    Returns
    -------
    shapely geometry
        Transformed geometry in dst_crs.
    """
    transformer = _get_transformer(src_crs, dst_crs)
    return transform(transformer.transform, geom)


def compute_area_4548(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute area in square metres for each geometry in EPSG:4548.

    If the GeoDataFrame is not already in EPSG:4548, it will be reprojected first.
    A new column ``area_sqm`` is added.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must have a geometry column.  Any CRS is accepted; if not EPSG:4548 the
        frame is reprojected (the original is not modified).

    Returns
    -------
    GeoDataFrame
        Copy of *gdf* with an additional ``area_sqm`` column.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set")

    working = gdf.copy()
    if working.crs.to_epsg() != 4548:
        working = working.to_crs(CRS_4548)

    working["area_sqm"] = working.geometry.area
    return working


def compute_area_by_group(
    gdf: gpd.GeoDataFrame, group_col: str
) -> gpd.GeoDataFrame:
    """
    Compute area breakdown by a grouping column.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must contain an ``area_sqm`` column (see :func:`compute_area_4548`).
    group_col : str
        Column name to group by.

    Returns
    -------
    GeoDataFrame
        Indexed by *group_col* with columns ``area_sqm``, ``area_ha``, ``area_km2``,
        and ``pct`` (percentage of total).
    """
    if "area_sqm" not in gdf.columns:
        gdf = compute_area_4548(gdf)

    total = gdf["area_sqm"].sum()
    grouped = gdf.groupby(group_col)["area_sqm"].sum().reset_index()
    grouped["area_ha"] = grouped["area_sqm"] / 10_000.0
    grouped["area_km2"] = grouped["area_sqm"] / 1_000_000.0
    grouped["pct"] = (grouped["area_sqm"] / total * 100).round(2)
    return grouped


def format_area(area_sqm: float) -> dict:
    """
    Format a square-metre area into human-readable units.

    Returns
    -------
    dict
        Keys: ``sqm``, ``ha``, ``km2``.
    """
    return {
        "sqm": round(area_sqm, 2),
        "ha": round(area_sqm / 10_000.0, 4),
        "km2": round(area_sqm / 1_000_000.0, 6),
    }
