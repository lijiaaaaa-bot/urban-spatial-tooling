"""Tests for src/renewal.py — 城市更新 留-改-拆 classification."""

import numpy as np
from shapely.geometry import box

from src.renewal import (
    capacity_assessment,
    classify_building,
    spatial_cluster,
)


def _line(count, pitch=30.0, size=20.0, origin_x=0.0):
    """A row of `count` 20x20 m buildings with 10 m gaps (adjacent at 12 m)."""
    return [box(origin_x + i * pitch, 0, origin_x + i * pitch + size, size)
            for i in range(count)]


def test_classify_building_rule_scores():
    """Old masonry with mismatched use -> 拆 (6 pts); new rc matching -> 留 (2)."""
    old = classify_building(40, "masonry", "residential", "commercial")
    assert old["score"] == 6 and old["treatment"] == "demolish"

    new = classify_building(10, "rc", "residential", "residential")
    assert new["score"] == 2 and new["treatment"] == "retain"

    mid = classify_building(20, "rc", "residential", "residential")
    assert mid["treatment"] == "renovate"          # 15-30 years -> 改

    # 危旧房 override: young + strong structure still demolished
    unsafe = classify_building(5, "rc", "residential", "residential",
                               unsafe=True)
    assert unsafe["treatment"] == "demolish"


def test_capacity_assessment_bands():
    """FAR-ratio bands: <0.5 拆, [0.5,0.8) 改, >=0.8 留; no-target guard."""
    assert capacity_assessment(1.0, 2.5)["treatment"] == "demolish"    # 0.40
    assert capacity_assessment(1.5, 2.5)["treatment"] == "renovate"    # 0.60
    assert capacity_assessment(2.2, 2.5)["treatment"] == "retain"      # 0.88
    no_target = capacity_assessment(2.0, 0.0)
    assert no_target["treatment"] == "retain"
    assert no_target["ratio"] is None


def test_spatial_cluster_prevents_tooth_extraction():
    """Isolated 拆-scored buildings inside a 留-majority row are absorbed."""
    buildings = _line(6)
    scores = [2, 2, 2, 6, 2, 6]           # Method 1: two isolated 拆
    treatments, cluster_ids = spatial_cluster(buildings, scores)
    assert np.all(treatments == "retain")     # one cluster, majority 留
    assert len(np.unique(cluster_ids)) == 1


def test_spatial_cluster_keeps_separate_blocks_and_merges_minor_clusters():
    """Separate groups keep separate fates; under-sized clusters merge away."""
    group_a = _line(4)                                 # 4 x 拆 buildings
    group_b = _line(4, origin_x=200.0)                 # 4 x 留 buildings
    isolated = box(160, 0, 172, 12)                    # 28 m from group B
    buildings = group_a + group_b + [isolated]
    scores = [6] * 4 + [2] * 4 + [2]

    treatments, cluster_ids = spatial_cluster(buildings, scores,
                                              adjacency_dist=12.0,
                                              min_cluster_size=3)
    # group A keeps demolition; group B + isolated merge into one 留 cluster
    assert np.all(treatments[:4] == "demolish")
    assert np.all(treatments[4:] == "retain")
    assert len(np.unique(cluster_ids)) == 2
