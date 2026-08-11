"""Tests for src/traffic.py — traffic capacity analysis (交通承载力)."""

import numpy as np
import pytest

from src.traffic import (
    road_capacity,
    vc_ratio,
    saturation_level,
    los_from_delay,
    intersection_los,
    road_saturation,
)


def test_road_capacity_values_and_lanes():
    # design capacity = lanes x per-lane capacity; default lanes 6/4/2
    assert road_capacity("arterial") == pytest.approx(6 * 1500.0)      # 9000
    assert road_capacity("secondary") == pytest.approx(4 * 800.0)      # 3200
    assert road_capacity("branch") == pytest.approx(2 * 400.0)         # 800
    # lane override scales linearly
    assert road_capacity("arterial", lanes=4) == pytest.approx(6000.0)
    assert road_capacity("branch", lanes=1) == pytest.approx(400.0)
    # unknown classes are rejected loudly
    with pytest.raises(ValueError):
        road_capacity("highway")


def test_vc_ratio_saturation_levels_and_network():
    # V/C is plain volume / capacity
    assert vc_ratio(900, 1500) == pytest.approx(0.6)
    assert vc_ratio(0, 1000) == 0.0
    # band edges: <0.6 good, 0.6-0.85 acceptable, >0.85 congested
    assert saturation_level(0.0) == "good"
    assert saturation_level(0.59) == "good"
    assert saturation_level(0.6) == "acceptable"
    assert saturation_level(0.85) == "acceptable"
    assert saturation_level(0.851) == "congested"
    assert saturation_level(3.0) == "congested"

    # network aggregation: mean, bottlenecks (>0.85), exposure weighting
    vols = np.array([300.0, 500.0, 900.0, 1000.0])
    caps = np.full(4, 1000.0)
    net = road_saturation(vols, caps)
    assert net["mean_vc"] == pytest.approx(0.675)
    assert net["max_vc"] == pytest.approx(1.0)
    assert net["n_bottleneck"] == 2
    assert net["bottleneck_indices"].tolist() == [2, 3]
    assert net["bottleneck_share"] == pytest.approx(0.5)
    # equal capacities -> weighted mean equals the plain mean
    assert net["weighted_mean_vc"] == pytest.approx(0.675)
    # explicit weights change the mean: heavy weight on a 1.0-V/C segment
    w = np.array([1.0, 1.0, 1.0, 10.0])
    assert road_saturation(vols, caps, weights=w)["weighted_mean_vc"] == pytest.approx(0.9)


def test_intersection_los_delay_and_bands():
    # unsaturated approaches: uniform delay only, all four approaches LOS B
    res = intersection_los([300, 300, 300, 300], [3, 3, 3, 3],
                           green_ratios=[0.45] * 4)
    assert res["x"][0] == pytest.approx(300 / (3 * 1800 * 0.45))       # ~0.123
    assert res["per_approach_los"] == ["B", "B", "B", "B"]
    assert res["los"] == "B" and res["delay_s"] == pytest.approx(res["delays"][0])

    # one saturated approach -> that approach LOS F and pulls the
    # flow-weighted intersection delay into F
    res2 = intersection_los([3000, 500, 500, 500], [3, 3, 3, 3],
                            green_ratios=[0.45] * 4)
    assert res2["x"][0] > 1.0
    assert res2["per_approach_los"][0] == "F"
    assert res2["per_approach_los"][1:] == ["C", "C", "C"]
    assert res2["los"] == "F"
    assert res2["delay_s"] > 80.0
    # delay grows with volume (monotone)
    assert res2["delays"][0] > res["delays"][0]

    # default green split: equal 4-phase share (cycle - lost) / (cycle * n)
    res3 = intersection_los([200, 200, 200, 200], [1, 1, 1, 1])
    assert res3["capacities"][0] == pytest.approx(1800 * (120 - 12) / (120 * 4))
    assert res3["los"] == "D"

    # HCM delay bands: A <=10, B <=20, C <=35, D <=55, E <=80, F > 80
    assert [los_from_delay(d) for d in (10, 20, 35, 55, 80, 80.1)] == \
        ["A", "B", "C", "D", "E", "F"]
