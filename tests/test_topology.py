"""Tests for src/topology.py — coverage / overlap / containment checks."""

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from src.topology import (
    AREA_TOLERANCE_SQM,
    check_containment,
    check_coverage,
    check_overlaps,
)

CRS = "EPSG:4548"


def make_gdf(polygons):
    return gpd.GeoDataFrame({"geometry": list(polygons)}, crs=CRS)


@pytest.fixture
def boundary() -> Polygon:
    """100 m x 100 m square boundary, bottom-left at origin."""
    return box(0.0, 0.0, 100.0, 100.0)


@pytest.fixture
def perfect_2x2_grid() -> gpd.GeoDataFrame:
    """Four 50 x 50 squares that exactly partition the 100 x 100 boundary."""
    return make_gdf([
        box(0.0, 0.0, 50.0, 50.0),    # SW
        box(50.0, 0.0, 100.0, 50.0),  # SE
        box(0.0, 50.0, 50.0, 100.0),  # NW
        box(50.0, 50.0, 100.0, 100.0),  # NE
    ])


def test_perfect_grid_covers_boundary(boundary, perfect_2x2_grid):
    covered, gaps = check_coverage(perfect_2x2_grid, boundary)
    assert covered is True
    assert gaps == []


def test_perfect_grid_has_no_overlaps(perfect_2x2_grid):
    clean, overlaps = check_overlaps(perfect_2x2_grid)
    assert clean is True
    assert overlaps == []


def test_perfect_grid_is_contained(boundary, perfect_2x2_grid):
    all_inside, violators = check_containment(perfect_2x2_grid, boundary)
    assert all_inside is True
    assert len(violators) == 0


def test_removed_square_detects_gap(boundary, perfect_2x2_grid):
    """Remove the NE square: a 50 x 50 = 2500 m2 gap must be reported."""
    gdf = perfect_2x2_grid.iloc[:-1].copy()
    covered, gaps = check_coverage(gdf, boundary)
    assert covered is False
    assert len(gaps) == 1
    assert gaps[0].area == pytest.approx(2500.0)


def test_overlapping_squares_detected(boundary, perfect_2x2_grid):
    """Shift the NE square (+10, -10): it now overlaps only the SE square
    in a single 40 x 10 region = 400 m2."""
    gdf = perfect_2x2_grid.copy()
    shifted = box(60.0, 40.0, 110.0, 90.0)
    gdf.loc[gdf.index[-1], "geometry"] = shifted
    clean, overlaps = check_overlaps(gdf)
    assert clean is False
    assert len(overlaps) == 1
    assert overlaps[0].area == pytest.approx(40.0 * 10.0)


def test_outside_feature_detected(boundary, perfect_2x2_grid):
    """A square completely outside the boundary must be a violator."""
    gdf = make_gdf(list(perfect_2x2_grid.geometry) + [box(200.0, 200.0, 210.0, 210.0)])
    all_inside, violators = check_containment(gdf, boundary)
    assert all_inside is False
    assert len(violators) == 1
    assert violators.geometry.iloc[0].area == pytest.approx(100.0)


def test_partial_poke_outside_detected(boundary, perfect_2x2_grid):
    """A square straddling the east boundary by 10 m pokes out 10 x 40 m."""
    gdf = make_gdf(list(perfect_2x2_grid.geometry) + [box(90.0, 30.0, 110.0, 70.0)])
    all_inside, violators = check_containment(gdf, boundary)
    assert all_inside is False
    outside_area = violators.geometry.iloc[0].difference(boundary).area
    assert outside_area == pytest.approx(10.0 * 40.0)


def test_sub_tolerance_gap_is_noise(boundary, perfect_2x2_grid):
    """A sub-tolerance gap (1e-5 m2) is floating-point noise: still covered.
    The same gap with tol=0 is a violation — proves the tolerance drives it.
    Also pins the exported default tolerance constant."""
    assert AREA_TOLERANCE_SQM == 1e-4
    gdf = perfect_2x2_grid.copy()
    gdf.loc[gdf.index[-1], "geometry"] = box(50.0 + 1e-7, 50.0 + 1e-7, 100.0, 100.0)
    covered_default, _ = check_coverage(gdf, boundary)
    assert covered_default is True
    covered_zero, _ = check_coverage(gdf, boundary, tol=0.0)
    assert covered_zero is False
