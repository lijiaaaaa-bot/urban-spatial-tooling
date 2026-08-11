"""Tests for src/generation.py — boundary subdivision and land-use assignment."""

import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from src.generation import (
    assign_land_use,
    generate_buildings_in_parcel,
    subdivide_boundary,
)

BOUNDARY = box(0, 0, 100, 100)


def test_subdivide_boundary_creates_expected_cell_count():
    """A 100x100 boundary on a 20 m grid -> 5 rows x 5 cols = 25 cells."""
    cells = subdivide_boundary(BOUNDARY, n_rows=5, n_cols=5)
    assert len(cells) == 25


def test_cells_do_not_overlap_and_cover_boundary():
    """Adjacent cells share only edges: union area == sum of cell areas,
    and the union covers the boundary exactly (no gaps, no overlap)."""
    cells = subdivide_boundary(BOUNDARY, n_rows=5, n_cols=5)
    union = unary_union(cells)
    assert union.area == pytest.approx(sum(c.area for c in cells))
    assert union.area == pytest.approx(BOUNDARY.area)
    assert all(cell.within(BOUNDARY) for cell in cells)


def test_assign_land_use_produces_valid_codes():
    """Every assigned code must come from the requested mix (GB 50137-2011)."""
    cells = subdivide_boundary(BOUNDARY, n_rows=5, n_cols=5)
    gdf = assign_land_use(cells, random_seed=42)
    assert len(gdf) == len(cells)
    assert set(gdf["land_use_code"]).issubset({"R", "A", "B", "G", "S"})


def test_generate_buildings_in_parcel_stays_inside():
    """Generated footprints are contained in the parcel and typed."""
    parcel = box(0, 0, 200, 200)
    gdf = generate_buildings_in_parcel(parcel, coverage_ratio=0.3, random_seed=1)
    assert len(gdf) > 0
    assert all(parcel.covers(geom) for geom in gdf.geometry)
    assert all(t == "generic" for t in gdf["building_type"])
