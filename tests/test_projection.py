"""Tests for src/projection.py — CRS transform and area computation."""

import math

import pytest
from shapely.geometry import Point, Polygon, box

import geopandas as gpd

from src.projection import (
    CRS_4326,
    CRS_4548,
    compute_area_4548,
    compute_area_by_group,
    format_area,
    transform_geometry,
)

# Beijing / Haidian reference point (degrees)
BEIJING_LON, BEIJING_LAT = 116.3, 39.9


@pytest.fixture
def lonlat_polygon() -> Polygon:
    """~1 km square in WGS84 near Beijing (0.01 deg x 0.01 deg)."""
    return box(BEIJING_LON, BEIJING_LAT, BEIJING_LON + 0.01, BEIJING_LAT + 0.01)


@pytest.fixture
def metre_square() -> Polygon:
    """Exactly 1000 m x 1000 m square in EPSG:4548 coordinates."""
    return box(444000.0, 4415000.0, 445000.0, 4416000.0)


def test_transform_lands_in_beijing_utm_zone(lonlat_polygon):
    """EPSG:4548 is CGCS2000 Gauss-Kruger CM 117E: Beijing easting ~4.4e5,
    northing ~4.4e6 metres. Geometry type is preserved."""
    out = transform_geometry(lonlat_polygon, CRS_4326, CRS_4548)
    assert out.geom_type == "Polygon"
    assert out.is_valid
    minx, miny, maxx, maxy = out.bounds
    assert 4.4e5 < minx < maxx < 4.6e5
    assert 4.41e6 < miny < maxy < 4.43e6


def test_transform_roundtrip_4326(lonlat_polygon):
    """Transform to 4548 and back to 4326 should recover the original."""
    there = transform_geometry(lonlat_polygon, CRS_4326, CRS_4548)
    back = transform_geometry(there, CRS_4548, CRS_4326)
    assert back.equals_exact(lonlat_polygon, tolerance=1e-9)


def test_transform_point_types(lonlat_polygon):
    pt = Point(BEIJING_LON, BEIJING_LAT)
    out = transform_geometry(pt, CRS_4326, CRS_4548)
    assert out.geom_type == "Point"
    assert abs(out.x - 440137.0) < 100.0  # Beijing (116.3E, 39.9N) easting ~ 440.1 km
    assert abs(out.y - 4418660.0) < 100.0


def test_compute_area_4548_exact_square(metre_square):
    gdf = gpd.GeoDataFrame({"geometry": [metre_square]}, crs=CRS_4548)
    out = compute_area_4548(gdf)
    assert out["area_sqm"].iloc[0] == pytest.approx(1_000_000.0, abs=1e-6)
    assert out.crs.to_epsg() == 4548


def test_compute_area_4548_reprojects_from_4326(lonlat_polygon):
    """Area in EPSG:4548 of a 0.01 x 0.01 deg square near lat 39.9:
    width ~ 0.01 * 111320 * cos(39.9 deg), height ~ 0.01 * 111320 m."""
    gdf = gpd.GeoDataFrame({"geometry": [lonlat_polygon]}, crs=CRS_4326)
    out = compute_area_4548(gdf)
    width_m = 0.01 * 111_320.0 * math.cos(math.radians(BEIJING_LAT))  # ~854 m
    height_m = 0.01 * 111_320.0  # ~1113 m
    area_expected = width_m * height_m  # ~ 0.95 km2
    assert out["area_sqm"].iloc[0] == pytest.approx(area_expected, rel=0.02)


def test_compute_area_4548_requires_crs(metre_square):
    gdf = gpd.GeoDataFrame({"geometry": [metre_square]})  # no crs
    with pytest.raises(ValueError):
        compute_area_4548(gdf)


def test_compute_area_by_group(metre_square):
    gdf = gpd.GeoDataFrame(
        {
            "land_use": ["R", "R", "G"],
            "geometry": [metre_square, metre_square, metre_square],
        },
        crs=CRS_4548,
    )
    grouped = compute_area_by_group(gdf, "land_use")
    assert set(grouped["land_use"]) == {"R", "G"}
    assert grouped.loc[grouped["land_use"] == "R", "area_sqm"].iloc[0] == pytest.approx(
        2_000_000.0
    )
    assert grouped.loc[grouped["land_use"] == "G", "area_sqm"].iloc[0] == pytest.approx(
        1_000_000.0
    )
    assert grouped["pct"].sum() == pytest.approx(100.0)
    assert grouped.loc[grouped["land_use"] == "R", "area_ha"].iloc[0] == pytest.approx(200.0)


def test_format_area_units(metre_square):
    gdf = gpd.GeoDataFrame({"geometry": [metre_square]}, crs=CRS_4548)
    sqm = compute_area_4548(gdf)["area_sqm"].iloc[0]
    fmt = format_area(sqm)
    assert fmt["sqm"] == pytest.approx(1_000_000.0)
    assert fmt["ha"] == pytest.approx(100.0)
    assert fmt["km2"] == pytest.approx(1.0)
