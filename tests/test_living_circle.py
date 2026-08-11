"""Tests for src/living_circle.py — walking-network distances and
isochrones, plus the multi-modal (cycling / bus transit) extension."""

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point

from src.living_circle import (
    BIKE_SPEED_KMH,
    BUS_SPEED_KMH,
    WALK_SPEED_KMH,
    WalkingNetwork,
    cycling_isochrone,
    decay_linear,
    decay_step,
    multi_modal_score,
    transit_accessibility,
)

# 3 vertical roads (x) x 3 horizontal roads (y) on a 1000 m grid,
# routed every 100 m -> a connected 11 x 11 node lattice.
V_ROADS = [0.0, 500.0, 1000.0]
H_ROADS = [0.0, 500.0, 1000.0]


def make_grid_network() -> WalkingNetwork:
    return WalkingNetwork.from_road_grid(V_ROADS, H_ROADS, node_step=100.0)


def test_from_road_grid_creates_connected_graph():
    net = make_grid_network()
    assert nx.is_connected(net.graph)


def test_from_road_grid_with_river_stays_connected_via_bridge():
    """A river barrier at x=500 cuts the grid except at the bridge row."""
    net = WalkingNetwork.from_road_grid(
        V_ROADS, H_ROADS, node_step=100.0,
        river_x=500.0, bridges=[(500.0, 500.0)],
    )
    assert nx.is_connected(net.graph)


def test_network_dist_returns_finite_manhattan_distance():
    """(0,0) -> (1000,1000) walks 1000 m east + 1000 m north = 2000 m."""
    net = make_grid_network()
    d = net.network_dist(Point(0, 0), Point(1000, 1000))
    assert d == pytest.approx(2000.0)
    assert d != float("inf")


def test_isochrone_nodes_returns_nodes_within_radius():
    """Nodes only exist along roads (x in {0,500,1000}, y in {0,500,1000})."""
    net = make_grid_network()
    nodes = net.isochrone_nodes(Point(0, 0), radius=250.0)
    assert (0.0, 0.0) in nodes
    assert (1000.0, 1000.0) not in nodes
    assert all(net.network_dist(Point(0, 0), Point(*n)) <= 250.0 for n in nodes)
    # Road nodes within 250 m walking distance of (0,0)
    assert len(nodes) == 5


def test_decay_functions_behave():
    d = np.array([100.0, 300.0])
    assert decay_step(d, 300.0).tolist() == [1.0, 1.0]
    assert decay_step(np.array([400.0]), 300.0).tolist() == [0.0]
    assert decay_linear(np.array([100.0]), 300.0)[0] == pytest.approx(1.0 - 100.0 / 300.0)


# ---------------------------------------------------------------------------
# Multi-modal: cycling isochrones, bus transit, combined scores
# ---------------------------------------------------------------------------


def test_cycling_isochrone_covers_4x_walking_radius_in_same_time():
    """15 min on foot reaches 1250 m; 15 min by bike (15 km/h) reaches
    3750 m, so (1000,1000) at walking distance 2000 m is bike-reachable
    but NOT walk-reachable within 15 minutes."""
    net = make_grid_network()
    cyc = cycling_isochrone(net, Point(0, 0), max_minutes=15.0)
    assert (1000.0, 1000.0) in cyc
    walk = net.isochrone_nodes(Point(0, 0), radius=15.0 * WALK_SPEED_KMH * 1000.0 / 60.0)
    assert (1000.0, 1000.0) not in walk
    # time budget scales with speed: 8 min at 15 km/h = 2000 m exactly
    assert (1000.0, 1000.0) in cycling_isochrone(net, Point(0, 0), max_minutes=8.0)
    # distance-based cutoff is identical to the walking isochrone of the
    # same radius (same graph, same radii — only speed differs)
    assert cycling_isochrone(net, Point(0, 0), radius_m=2000.0) == (
        net.isochrone_nodes(Point(0, 0), radius=2000.0))


def test_transit_accessibility_faster_than_walking_to_remote_node():
    """Bus edge (0,0)->(1000,1000) at 30 km/h + headway/2 wait beats the
    24-min walk; travel time == crow-flies spacing / speed + wait."""
    net = make_grid_network()
    stops = [("A", 0.0, 0.0), ("B", 1000.0, 1000.0)]
    routes = [("L1", ["A", "B"])]
    field = transit_accessibility(net, Point(0, 0), stops, routes)
    t_transit = field[(1000.0, 1000.0)]
    spacing = Point(0, 0).distance(Point(1000, 1000))
    expected = spacing / (BUS_SPEED_KMH * 1000.0 / 60.0) + 10.0 / 2.0
    assert t_transit == pytest.approx(expected, rel=1e-6)
    assert t_transit < net.network_dist(Point(0, 0), Point(1000, 1000)) / (
        WALK_SPEED_KMH * 1000.0 / 60.0)


def test_multi_modal_score_bike_rescues_walk_desert():
    """Facility 1000 m away with a 500 m walking radius: walk score 0
    (desert), but the bike score (radius x3 at 15 km/h) lifts the
    combined score to 0.8 * bike — exactly the METHODS formula
    max(walk, 0.8 * bike, 0.7 * transit)."""
    net = make_grid_network()
    facs = [("公园绿地", 1000.0, 0.0, "park", 2, 500.0, 1.0)]
    res = {"R": (0.0, 0.0)}
    mm = multi_modal_score(net, facs, res)
    r = mm["R"]
    assert r["walk"] == pytest.approx(0.0)
    bike_expected = 100.0 * (1.0 - 1000.0 / (500.0 * BIKE_SPEED_KMH / WALK_SPEED_KMH))
    assert r["bike"] == pytest.approx(bike_expected, rel=1e-6)
    assert r["combined"] == pytest.approx(0.8 * r["bike"], rel=1e-6)
    assert r["combined"] > 0.0
