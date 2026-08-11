"""Tests for src/setback.py — 建筑退线 buildable envelopes + 防火间距."""

import pytest
from shapely.geometry import LineString, box

from src.setback import (
    buildable_from_roads,
    check_fire_separation,
    edge_setbacks_by_side,
    fire_grade_of,
    fire_separation_buffer,
    is_high_rise,
    required_fire_separation,
    road_constraint,
    setback_envelope_by_edges,
)

PARCEL = box(0, 0, 100, 100)


def test_setback_envelope_5m_all_sides_is_90x90():
    """100x100 parcel with 5 m setbacks on every edge -> 90x90 envelope."""
    envelope = setback_envelope_by_edges(PARCEL, [5.0, 5.0, 5.0, 5.0])
    assert envelope.bounds == pytest.approx((5.0, 5.0, 95.0, 95.0))
    assert envelope.area == pytest.approx(90.0 * 90.0)


def test_envelope_strictly_inside_original_parcel():
    envelope = setback_envelope_by_edges(PARCEL, [5.0, 5.0, 5.0, 5.0])
    assert envelope.within(PARCEL)
    assert PARCEL.covers(envelope)
    assert envelope.area < PARCEL.area


def test_edge_setbacks_by_side_maps_compass_sides():
    """Side detection is robust to ring order: each edge gets its side's rule."""
    rules = {"E": 5.0, "N": 10.0, "W": 5.0, "S": 10.0}
    setbacks, sides = edge_setbacks_by_side(PARCEL, rules)
    assert dict(zip(sides, setbacks)) == rules
    # shapely's box() ring starts at the east edge
    assert sides == ["E", "N", "W", "S"]


def test_buildable_from_roads_subtracts_road_zone():
    """Parcel shrunk by 5 m property setback, minus the 30 m local-road zone."""
    road_zone = road_constraint(LineString([(50, -50), (50, 150)]), "local")
    envelope = buildable_from_roads(PARCEL, road_zone, property_setback=5.0)
    assert envelope is not None
    # 90x90 base minus a 30 m wide x 90 m tall road strip
    assert envelope.area == pytest.approx(90.0 * 90.0 - 30.0 * 90.0)


# ---- Inter-building fire separation (GB 50016-2014) ----


def test_required_fire_separation_grades():
    """6 / 9 / 13 m by fire grade; high-rise is 13 m from anything."""
    assert required_fire_separation(2, 2) == 6.0
    assert required_fire_separation(1, 2) == 6.0
    assert required_fire_separation(2, 3) == 9.0
    assert required_fire_separation(3, 4) == 13.0
    assert required_fire_separation(4, 4) == 13.0
    assert required_fire_separation("high_rise", 2) == 13.0
    assert required_fire_separation("high_rise", "high_rise") == 13.0


def test_check_fire_separation_flags_close_pairs():
    """Two Grade-2 boxes 4 m apart violate the 6 m rule; 7 m apart is fine."""
    a = box(0, 0, 10, 10)
    close = box(14, 0, 24, 10)   # 4 m clear gap
    far = box(17, 0, 27, 10)     # 7 m clear gap
    res = check_fire_separation([a, close])
    assert res["n_buildings"] == 2
    assert res["n_pairs"] == 1
    assert res["n_violations"] == 1
    v = res["violations"][0]
    assert (v["i"], v["j"]) == (0, 1)
    assert v["distance"] == pytest.approx(4.0)
    assert v["required"] == 6.0
    assert v["grade_i"] == 2 and v["grade_j"] == 2
    assert check_fire_separation([a, far])["n_violations"] == 0


def test_check_fire_separation_high_rise_requires_13m():
    """High-rise tower 10 m from a Grade-2 slab violates the 13 m rule."""
    slab = box(0, 0, 20, 20)
    tower = box(30, 0, 40, 10)   # 10 m clear gap
    res = check_fire_separation([slab, tower], {1: "high_rise"})
    assert res["n_violations"] == 1
    v = res["violations"][0]
    assert v["required"] == 13.0
    assert v["grade_i"] == 2 and v["grade_j"] == "high_rise"
    assert v["distance"] == pytest.approx(10.0)


def test_fire_separation_buffer_half_distance():
    """Zone radius = half the required separation (Grade 2 -> 3 m, high-rise -> 6.5 m)."""
    a = box(0, 0, 10, 10)
    z_a = fire_separation_buffer(a, 2)
    # gap 5 m < 6 m required -> 3 m zones overlap
    b_close = box(15, 0, 25, 10)
    assert z_a.intersects(fire_separation_buffer(b_close, 2))
    # gap 8 m > 6 m required -> zones stay apart
    b_far = box(18, 0, 28, 10)
    assert not z_a.intersects(fire_separation_buffer(b_far, 2))
    # zone extends exactly the radius beyond the footprint (box at origin)
    assert fire_separation_buffer(a, 2).bounds == pytest.approx((-3.0, -3.0, 13.0, 13.0))
    zh = fire_separation_buffer(a, "high_rise")   # half of the 13 m rule = 6.5 m
    assert zh.bounds == pytest.approx((-6.5, -6.5, 16.5, 16.5))


def test_is_high_rise_and_fire_grade_of():
    """High-rise: height > 27 m or > 8 storeys; everything else defaults to Grade 2."""
    assert is_high_rise(height_m=36.0)
    assert is_high_rise(floors=9)
    assert not is_high_rise(height_m=18.0, floors=6)
    assert not is_high_rise()
    assert fire_grade_of(floors=12) == "high_rise"
    assert fire_grade_of(floors=6) == 2
    assert fire_grade_of(default=3) == 3
