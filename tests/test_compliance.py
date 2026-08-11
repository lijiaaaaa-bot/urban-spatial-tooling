"""Tests for src/compliance.py — three-lines (三区三线) and multi-corridor
height-envelope checks."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from src.compliance import (
    ThreeLinesChecker,
    ViewCorridor,
    Viewpoint,
    combine_corridors,
    height_envelope,
    terrain_corrected_height,
)

# Constraint layers on a 100 x 100 m synthetic site (EPSG:4548-like metres)
URBAN_BOUNDARY = box(0.0, 0.0, 100.0, 100.0)
FARMLAND = box(10.0, 10.0, 20.0, 20.0)          # 10 x 10 = 100 m2
REDLINE = box(80.0, 80.0, 90.0, 90.0)           # 10 x 10 = 100 m2


@pytest.fixture
def checker() -> ThreeLinesChecker:
    return ThreeLinesChecker(
        urban_boundary=URBAN_BOUNDARY,
        basic_farmland=FARMLAND,
        ecological_redline=REDLINE,
    )


@pytest.fixture
def inside_parcel() -> Polygon:
    """40 x 40 m parcel strictly inside the boundary, clear of both layers."""
    return box(30.0, 30.0, 70.0, 70.0)


def test_parcel_inside_returns_none(checker, inside_parcel):
    assert checker.check_parcel(inside_parcel, "P1") is None


def test_parcel_on_farmland_is_violation(checker):
    parcel = box(15.0, 15.0, 40.0, 40.0)  # overlaps farmland corner
    v = checker.check_parcel(parcel, "P2")
    assert v is not None
    assert v["parcel_id"] == "P2"
    assert v["violation"] == "on_basic_farmland"
    assert v["severity"] == "critical"
    assert v["overlap_area_m2"] == pytest.approx(25.0)  # 5 x 5 corner
    assert v["pct_overlap"] == pytest.approx(25.0 / 625.0 * 100)


def test_parcel_outside_boundary_is_violation(checker):
    parcel = box(150.0, 150.0, 160.0, 160.0)  # fully outside
    v = checker.check_parcel(parcel, "P3")
    assert v is not None
    assert v["violation"] == "outside_urban_boundary"
    assert v["severity"] == "critical"
    assert v["area_outside_m2"] == pytest.approx(100.0)
    assert v["pct_outside"] == pytest.approx(100.0)


def test_parcel_on_redline_is_violation(checker):
    parcel = box(85.0, 85.0, 95.0, 95.0)  # overlaps redline corner
    v = checker.check_parcel(parcel, "P4")
    assert v is not None
    assert v["violation"] == "in_ecological_redline"
    assert v["overlap_area_m2"] == pytest.approx(25.0)


def test_parcel_straddling_boundary_is_violation(checker):
    """Parcels that cross the urban boundary are violations even when they
    also overlap farmland — the boundary check has precedence."""
    parcel = box(90.0, 90.0, 110.0, 110.0)  # crosses NE boundary corner
    v = checker.check_parcel(parcel, "P5")
    assert v is not None
    assert v["violation"] == "outside_urban_boundary"
    # outside strips: x>100 (10x20) plus y>100, x<=100 (10x10)
    assert v["area_outside_m2"] == pytest.approx(300.0)


def test_check_all_summary(checker, inside_parcel):
    parcels = [
        ("P1", inside_parcel),
        ("P2", box(15.0, 15.0, 40.0, 40.0)),   # farmland
        ("P3", box(150.0, 150.0, 160.0, 160.0)),  # outside
        ("P4", box(85.0, 85.0, 95.0, 95.0)),   # redline
    ]
    result = checker.check_all(parcels)
    assert result["total_parcels"] == 4
    assert result["violations"] == 3
    assert result["compliant"] == 1
    assert len(result["details"]) == 3
    assert {v["violation"] for v in result["details"]} == {
        "on_basic_farmland", "outside_urban_boundary", "in_ecological_redline",
    }


def test_check_all_empty(checker):
    result = checker.check_all([])
    assert result["total_parcels"] == 0
    assert result["violations"] == 0
    assert result["compliant"] == 0
    assert result["details"] == []


def test_checker_keeps_constraint_layers(checker):
    assert checker.urban_boundary.equals(URBAN_BOUNDARY)
    assert checker.basic_farmland.equals(FARMLAND)
    assert checker.ecological_redline.equals(REDLINE)


# ---------------------------------------------------------------------------
# Multi-corridor combination (D-03)
# ---------------------------------------------------------------------------

# Two crossing corridors with known analytic envelopes:
#   A: (0,0,100) -> (100,0,50)   h at (50, 0) = 75.0; at (25, 0) = 87.5
#   B: (0,0,100) -> (0,100,50)   h at (0, 50) = 75.0; at (0, 25) = 87.5
# On flat ground at (50, 50): both project to t=0.5 -> z = 75.0.
# At (100, 50) corridor A clips to t=1 -> z = 50.0.
COR_A = ViewCorridor(
    "A", Viewpoint("V", 0.0, 0.0, 100.0), Viewpoint("T", 100.0, 0.0, 50.0))
COR_B = ViewCorridor(
    "B", Viewpoint("V", 0.0, 0.0, 100.0), Viewpoint("T", 0.0, 100.0, 50.0))


def test_combine_corridors_is_elementwise_min():
    """The binding height at any point is min(h_i) across corridors."""
    # (50, 0): only corridor A constrains strongly (75.0); B clips to t=1 (50.0)
    px, py = 50.0, 0.0
    h_a = COR_A.max_height_at(px, py)
    h_b = COR_B.max_height_at(px, py)
    assert combine_corridors([COR_A, COR_B], px, py) == pytest.approx(
        min(h_a, h_b))
    # The most restrictive corridor binds even when another allows more
    assert combine_corridors([COR_A, COR_B], px, py) <= h_a + 1e-12
    assert combine_corridors([COR_A, COR_B], px, py) <= h_b + 1e-12
    # A single corridor is its own envelope
    assert combine_corridors([COR_A], px, py) == pytest.approx(h_a)
    # Adding a corridor never raises the allowed height
    assert combine_corridors([COR_A, COR_B], 25.0, 25.0) <= combine_corridors(
        [COR_A], 25.0, 25.0) + 1e-12


def test_terrain_corrected_height_uses_ground_elevation():
    """h_max = z_sightline - z_ground: rising ground shrinks the envelope."""
    px, py = 50.0, 0.0
    flat = combine_corridors([COR_A], px, py)  # 75.0
    assert flat == pytest.approx(75.0)
    # 25 m of ground eats 25 m off the allowed height
    assert terrain_corrected_height([COR_A], px, py, 25.0) == pytest.approx(50.0)
    # callable terrain model behaves identically to a scalar
    assert terrain_corrected_height([COR_A], px, py, lambda x, y: 25.0) == \
        pytest.approx(50.0)
    # ground above the sightline clamps to zero (no negative heights)
    assert terrain_corrected_height([COR_A], px, py, 200.0) == pytest.approx(0.0)
    # multi-corridor: terrain correction applies per point, then min
    assert terrain_corrected_height([COR_A, COR_B], 50.0, 50.0, 25.0) == \
        pytest.approx(50.0)


def test_height_envelope_grid_combination_and_terrain():
    """Grid surface shape, min composition, NaN outside, terrain variant."""
    gx = np.array([0.0, 25.0, 50.0, 100.0])
    gy = np.array([0.0, 50.0, 100.0])
    GX, GY, H = height_envelope([COR_A, COR_B], gx, gy)
    assert H.shape == (len(gy), len(gx))
    assert np.allclose(GX[0], gx) and np.allclose(GY[:, 0], gy)

    # H == element-wise min of the individual corridor surfaces
    _, _, H_a = height_envelope([COR_A], gx, gy)
    _, _, H_b = height_envelope([COR_B], gx, gy)
    assert np.allclose(H, np.minimum(H_a, H_b), equal_nan=True)

    # Known analytic value: at (50, 50), both corridors sit at t=0.5,
    # where the sightline is at z = 100 + 0.5 * (50 - 100) = 75.0
    i, j = 1, 2  # gy=50, gx=50
    assert H[i, j] == pytest.approx(75.0)

    # Terrain variant: a 10 m ground grid lowers every cell by 10 m
    Z_ground = np.full_like(GX, 10.0)
    _, _, H_terr = height_envelope([COR_A, COR_B], gx, gy, Z_ground)
    assert np.allclose(H_terr, H - 10.0, equal_nan=True)

    # NaN when a corridor yields no constraint (degenerate zero-length
    # corridor: max_height_at returns inf everywhere)
    degenerate = ViewCorridor(
        "zero", Viewpoint("V", 0.0, 0.0, 100.0), Viewpoint("T", 0.0, 0.0, 100.0))
    _, _, H3 = height_envelope([degenerate], gx, gy)
    assert np.isnan(H3).all()
