"""
15-minute living circle (15分钟生活圈) evaluation.

Extracted from notebooks/13-fifteen-minute-living-circle.ipynb.

Provides:
- WalkingNetwork: street-network graph with barriers (river / rail)
  crossed only at bridges, Dijkstra walking distances and isochrones
- decay_step / decay_linear / decay_gaussian: distance-decay functions
- euclidean_coverage: crow-flies buffer screening
- evaluate_living_circles: score a facility layout against living-circle
  coverage and return a PASS/REVISE verdict — the evaluation loop for
  src/generation.py layout proposals
- Multi-modal extension (walking + cycling + bus):
  cycling_isochrone (same graph, 15 km/h, wider radius per minute),
  build_transit_graph / transit_accessibility (bus routes at 30 km/h
  between transit stops, headway/2 expected wait),
  multi_modal_score (combined score = max(walk, 0.8*bike, 0.7*transit))

Standard references
-------------------
- TD/T 1062-2021《社区生活圈规划技术指南》(Ministry of Natural
  Resources, effective 2022-01-01) — 15-min (800-1000 m), 10-min
  (500 m) and 5-min (300 m) urban community life circles.
- GB 50180-2018《城市居住区规划设计标准》— the tiered living-circle
  system: 居住街坊 → 五分钟生活圈居住区 (300 m) → 十分钟生活圈居住区
  (500 m) → 十五分钟生活圈居住区 (800-1000 m).
- Walking speed assumption: 60 m/min planning standard.

Method note: Euclidean buffers over-claim coverage near obstacles (river,
rail, walls).  Network walking distance is the truth for formal review;
the gap between the two is the detour penalty.

All computation is networkx + numpy + shapely — no GPU dependency.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import Point


@dataclass
class WalkingNetwork:
    """Street-network walking graph with barrier crossings.

    Built from a road grid; barrier lines (e.g. a river) are crossed
    only at explicit bridge nodes.  Edge weight = segment length (m).

    Attributes
    ----------
    graph : networkx.Graph
        Nodes are (x, y) tuples, edges carry ``weight`` in metres.
    """

    graph: nx.Graph = field(default_factory=nx.Graph)

    @classmethod
    def from_road_grid(cls,
                       v_roads: Sequence[float],
                       h_roads: Sequence[float],
                       node_step: float = 100.0,
                       river_x: Optional[float] = None,
                       bridges: Optional[Sequence[Tuple[float, float]]] = None,
                       ) -> 'WalkingNetwork':
        """Build a walking network from a road grid and river barriers.

        Parameters
        ----------
        v_roads : sequence of float
            x coordinates of vertical roads.
        h_roads : sequence of float
            y coordinates of horizontal roads.
        node_step : float, optional
            Routing node spacing along each road (m).
        river_x : float, optional
            x of a N-S barrier (e.g. river); vertical road segments
            exactly on it are removed except at bridge rows.
        bridges : sequence of (x, y), optional
            Crossing points where the barrier can be crossed.

        Returns
        -------
        WalkingNetwork
            Connected graph (bridges keep it connected).
        """
        graph = nx.Graph()
        bridge_ys = {b[1] for b in (bridges or [])}

        # Vertical roads at fixed x
        for x in v_roads:
            ys = np.arange(0.0, max(h_roads) + 1e-9, node_step)
            for i in range(len(ys) - 1):
                y1, y2 = ys[i], ys[i + 1]
                # segment crosses the barrier? blocked unless it is a bridge
                if river_x is not None and abs(x - river_x) < 1e-9 and not any(
                        abs(y - by) < 1e-9 for y in (y1, y2) for by in bridge_ys):
                    continue
                graph.add_edge((x, float(y1)), (x, float(y2)), weight=abs(y2 - y1))
        # Horizontal roads at fixed y: the node exactly on the barrier
        # is removed unless y is a bridge row, so crossings exist only
        # at bridges.
        for y in h_roads:
            xs = np.arange(0.0, max(v_roads) + 1e-9, node_step)
            for i in range(len(xs) - 1):
                x1, x2 = xs[i], xs[i + 1]
                if river_x is not None and y not in bridge_ys and (
                        abs(x1 - river_x) < 1e-9 or abs(x2 - river_x) < 1e-9):
                    continue  # barrier crossing at non-bridge y -> no edge
                graph.add_edge((float(x1), y), (float(x2), y), weight=abs(x2 - x1))
        return cls(graph=graph)

    @property
    def nodes_array(self) -> np.ndarray:
        """All graph nodes as an (N, 2) float array."""
        return np.array(list(self.graph.nodes))

    def snap(self, point: Point) -> Tuple[float, float]:
        """Nearest graph node to a point.

        Parameters
        ----------
        point : shapely Point

        Returns
        -------
        tuple of float
            Nearest node coordinates.
        """
        nodes = self.nodes_array
        d = np.hypot(nodes[:, 0] - point.x, nodes[:, 1] - point.y)
        return tuple(nodes[int(np.argmin(d))])

    def network_dist(self, point_a: Point, point_b: Point) -> float:
        """Walking distance between two points (inf if unreachable).

        Parameters
        ----------
        point_a, point_b : shapely Point

        Returns
        -------
        float
            Shortest walking distance (m).
        """
        try:
            return nx.shortest_path_length(
                self.graph, self.snap(point_a), self.snap(point_b), weight='weight')
        except nx.NetworkXNoPath:
            return float('inf')

    def isochrone_nodes(self, source_pt: Point, radius: float) -> set:
        """All graph nodes within walking ``radius`` of ``source_pt``.

        Parameters
        ----------
        source_pt : shapely Point
        radius : float
            Walking radius (m).

        Returns
        -------
        set
            Node coordinates reachable within the radius.
        """
        src = self.snap(source_pt)
        lengths = nx.single_source_dijkstra_path_length(
            self.graph, src, cutoff=radius, weight='weight')
        return set(lengths.keys())

    def distance_field(self, source_pt: Point) -> Dict[Tuple[float, float], float]:
        """Dijkstra distance from ``source_pt`` to every graph node.

        Parameters
        ----------
        source_pt : shapely Point

        Returns
        -------
        dict
            {node: walking distance (m)}.
        """
        src = self.snap(source_pt)
        return nx.single_source_dijkstra_path_length(
            self.graph, src, weight='weight')


def decay_step(d, r):
    """Step decay: 1 inside the radius, 0 outside (soft buffer)."""
    return np.where(d <= r, 1.0, 0.0)


def decay_linear(d, r):
    """Linear decay: max(0, 1 - d / r)."""
    return np.clip(1.0 - d / r, 0.0, 1.0)


def decay_gaussian(d, r, sigma_ratio: float = 0.5):
    """Gaussian decay: exp(-d² / 2 sigma²) with sigma = sigma_ratio * r."""
    sigma = sigma_ratio * r
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def euclidean_coverage(res_point: Point, facs: Sequence[tuple],
                       required_radius: float,
                       category: str) -> Tuple[bool, float]:
    """Is ``category`` covered within ``required_radius`` of ``res_point``?

    Parameters
    ----------
    res_point : shapely Point
        Residential point.
    facs : list of (name, Point, category, level, radius, weight)
        Facilities.
    required_radius : float
        Required service radius (m) for the category.
    category : str
        Category to check.

    Returns
    -------
    (covered, distance_to_nearest_in_category)
    """
    d_min = float('inf')
    for _name, pt, cat, _lvl, _rad, _w in facs:
        if cat == category:
            d_min = min(d_min, pt.distance(res_point))
    return (d_min <= required_radius, d_min)


def _nearest_category_distance(dist_fn, res_point: Point, facs: Sequence[tuple],
                               category: str) -> float:
    """Distance from ``res_point`` to the nearest facility of ``category``.

    Parameters
    ----------
    dist_fn : callable (Point, Point) -> float
        Distance function (crow-flies or network).
    res_point : shapely Point
    facs : list of (name, Point, category, level, radius, weight)
    category : str

    Returns
    -------
    float
        Minimum distance (m); inf if no facility of the category.
    """
    d_min = float('inf')
    for _name, pt, cat, _lvl, _rad, _w in facs:
        if cat == category:
            d_min = min(d_min, dist_fn(res_point, pt))
    return d_min


def evaluate_living_circles(residential: Dict[str, Tuple[float, float]],
                            facility_list: Sequence[tuple],
                            mode: str = 'network',
                            desert_fraction: float = 0.5,
                            network: Optional[WalkingNetwork] = None) -> dict:
    """Score a facility layout against TD/T 1062-2021 living circles.

    Parameters
    ----------
    residential : dict
        {name: (x, y)} neighborhoods in projected metres.
    facility_list : list of tuples
        ``(name_zh, x, y, category, level, required_radius_m, weight)``.
    mode : str, optional
        ``'network'`` (full walking graph, formal check) or
        ``'euclidean'`` (fast screening).
    desert_fraction : float, optional
        A neighborhood whose covered category share is below this value
        is a service desert.
    network : WalkingNetwork, optional
        Required when ``mode == 'network'``; build one with
        :meth:`WalkingNetwork.from_road_grid`.

    Returns
    -------
    dict
        ``mode``, ``neighborhoods``, ``mean_coverage_pct``, ``deserts``,
        ``verdict``.  Verdict is ``PASS`` when mean coverage >= 90% and
        no deserts, else ``REVISE``.
    """
    facs = [(n, Point(x, y), c, l, r, w) for n, x, y, c, l, r, w in facility_list]
    rpts = {k: Point(xy) for k, xy in residential.items()}
    cats = sorted({f[2] for f in facs})
    rad_of = {f[2]: f[4] for f in facs}

    if mode == 'euclidean':
        def dist(a, b):
            return a.distance(b)
    elif mode == 'network':
        if network is None:
            raise ValueError(
                'mode="network" requires a WalkingNetwork (build one with '
                'WalkingNetwork.from_road_grid)')
        def dist(a, b):
            return network.network_dist(a, b)
    else:
        raise ValueError(f'unknown mode: {mode!r}')

    per_hood = {}
    for rname, rpt in rpts.items():
        n_cov = sum(_nearest_category_distance(dist, rpt, facs, c) <= rad_of[c]
                    for c in cats)
        per_hood[rname] = n_cov / len(cats)

    desert_names = [r for r, f in per_hood.items() if f < desert_fraction]
    mean_cov = float(np.mean(list(per_hood.values())))
    verdict = 'PASS' if mean_cov >= 0.9 and not desert_names else 'REVISE'
    return {'mode': mode, 'neighborhoods': len(rpts),
            'mean_coverage_pct': 100.0 * mean_cov, 'deserts': desert_names,
            'verdict': verdict}


def accessibility_scores(network: WalkingNetwork,
                         facility_list: Sequence[tuple],
                         grid: int = 50,
                         decay: str = 'linear') -> Tuple[dict, np.ndarray, np.ndarray]:
    """Distance-decay weighted accessibility score over a grid.

    Per category, precompute the network distance from every graph node
    to the nearest facility (one Dijkstra per facility), snap grid cells
    to nodes, then score = sum_cat weight_cat * decay(d_min,c),
    normalized to 0-100.

    Parameters
    ----------
    network : WalkingNetwork
        Walking graph.
    facility_list : list of tuples
        ``(name_zh, x, y, category, level, required_radius_m, weight)``.
    grid : int, optional
        Grid resolution (cells per side of the study area).
    decay : str, optional
        One of ``'step'``, ``'linear'``, ``'gaussian'``.

    Returns
    -------
    scores : dict of {decay_name: ndarray (grid, grid)}
    xs, ys : ndarray
        Grid coordinate arrays (length ``grid``).
    """
    facs = [(n, Point(x, y), c, l, r, w) for n, x, y, c, l, r, w in facility_list]
    cats = sorted({f[2] for f in facs})
    rad_of = {f[2]: f[4] for f in facs}
    weights = {f[2]: f[5] for f in facs}

    node_index = {tuple(n): i for i, n in enumerate(network.nodes_array)}
    n_nodes = len(node_index)
    cat_fields = {}
    for cat in cats:
        cat_fields[cat] = np.full(n_nodes, np.inf)
        for _name, pt, c2, _lvl, _rad, _w in facs:
            if c2 == cat:
                field = network.distance_field(pt)
                idx = np.array([node_index[k] for k in field])
                vals = np.array(list(field.values()))
                cat_fields[cat][idx] = np.minimum(cat_fields[cat][idx], vals)

    xs = np.linspace(0, max(x for x, _ in network.graph.nodes) or 1, grid)
    ys = np.linspace(0, max(y for _, y in network.graph.nodes) or 1, grid)
    GX, GY = np.meshgrid(xs, ys)
    cell_pts = [Point(x, y) for y in ys for x in xs]
    cell_idx = np.array([node_index[network.snap(p)] for p in cell_pts]).reshape(grid, grid)

    decays = {'step': decay_step, 'linear': decay_linear, 'gaussian': decay_gaussian}
    scores = {}
    for dname, dfun in decays.items():
        S = np.zeros((grid, grid))
        for cat in cats:
            S += weights[cat] * dfun(cat_fields[cat][cell_idx], rad_of[cat])
        scores[dname] = 100.0 * S / sum(weights[c] for c in cats)
    return scores, xs, ys


# ---------------------------------------------------------------------------
# Multi-modal living circle: walking + cycling + bus transit
# ---------------------------------------------------------------------------

WALK_SPEED_KMH = 5.0
BIKE_SPEED_KMH = 15.0
BUS_SPEED_KMH = 30.0
DEFAULT_HEADWAY_MIN = 10.0


def _time_weighted_graph(network: WalkingNetwork, speed_kmh: float) -> nx.Graph:
    """Copy the street graph with edge weights in travel *minutes*.

    Parameters
    ----------
    network : WalkingNetwork
    speed_kmh : float
        Mode speed used to convert metre weights to minutes.

    Returns
    -------
    networkx.Graph
        Same topology, ``weight`` attribute in minutes.
    """
    m_per_min = speed_kmh * 1000.0 / 60.0
    G = nx.Graph()
    for u, v, d in network.graph.edges(data=True):
        G.add_edge(u, v, weight=d['weight'] / m_per_min)
    return G


def cycling_isochrone(network: WalkingNetwork,
                      source_pt: Point,
                      max_minutes: float = 15.0,
                      speed_kmh: float = BIKE_SPEED_KMH,
                      radius_m: Optional[float] = None) -> set:
    """Graph nodes reachable by bicycle within a time budget.

    Uses the *same* street graph as walking; only the edge weights change
    (metres converted to minutes at ``speed_kmh``).  At 15 km/h a cyclist
    covers 4x the walking radius in the same 15 minutes, so 1 / 2 / 3 km
    cycling isochrones are 4 / 8 / 12 minute trips.

    Parameters
    ----------
    network : WalkingNetwork
    source_pt : shapely Point
    max_minutes : float, optional
        Time budget (min).  Default 15 min.
    speed_kmh : float, optional
        Cycling speed.  Default 15 km/h.
    radius_m : float, optional
        If given, overrides the time budget with a distance cutoff
        (identical result to :meth:`WalkingNetwork.isochrone_nodes`).

    Returns
    -------
    set
        Node coordinates reachable by bike within the budget.
    """
    if radius_m is not None:
        # distance cutoff: identical result to the walking isochrone
        # (same graph, same radii) — only the time to cover it differs
        return network.isochrone_nodes(source_pt, radius_m)
    src = network.snap(source_pt)
    G = _time_weighted_graph(network, speed_kmh)
    # +1e-6 min absorbs float rounding in minute edge weights so nodes
    # exactly on the time budget are not spuriously excluded
    lengths = nx.single_source_dijkstra_path_length(
        G, src, cutoff=max_minutes + 1e-6, weight='weight')
    return set(lengths.keys())


def build_transit_graph(network: WalkingNetwork,
                        stops: Sequence[Tuple[str, float, float]],
                        routes: Sequence[Tuple[str, Sequence[str]]],
                        headway_min: float = DEFAULT_HEADWAY_MIN,
                        speed_kmh: float = BUS_SPEED_KMH,
                        access_kmh: float = WALK_SPEED_KMH
                        ) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
    """Two-layer walk + bus graph (directed) with minute edge weights.

    Nodes are ``(coord, layer)`` tuples: layer 0 is the street/access
    layer, layer 1 the bus layer.  This layering charges the expected
    wait exactly once per boarding:

    - street edges (layer 0-0) timed at ``access_kmh``
    - boarding edge (n, 0) -> (n, 1) at each stop: ``headway_min / 2``
    - alighting edge (n, 1) -> (n, 0): free
    - bus edges (n, 1) -> (n, 1) between consecutive stops of a route at
      ``speed_kmh`` (default 30 km/h), spacing = stop-to-stop distance
      (transit alignment abstracted as a direct connector)

    Stops are snapped to the nearest street node, so bus edges always
    connect into the access network.

    Parameters
    ----------
    network : WalkingNetwork
    stops : sequence of (name, x, y)
    routes : sequence of (route_name, [stop names in travel order])
    headway_min : float, optional
        Bus headway; half of it is charged as the expected wait.
    speed_kmh : float, optional
        Bus cruising speed.  Default 30 km/h.
    access_kmh : float, optional
        Walking (or cycling) access speed.  Default 5 km/h.

    Returns
    -------
    (graph, stop_nodes)
        ``graph`` (a directed graph) has minute weights and
        ``(coord, layer)`` nodes; ``stop_nodes`` maps stop name to its
        snapped node coordinates.
    """
    access_m_per_min = access_kmh * 1000.0 / 60.0
    bus_m_per_min = speed_kmh * 1000.0 / 60.0
    G = nx.DiGraph()
    for u, v, d in network.graph.edges(data=True):
        w = d['weight'] / access_m_per_min
        G.add_edge((u, 0), (v, 0), weight=w)
        G.add_edge((v, 0), (u, 0), weight=w)
    stop_nodes: Dict[str, Tuple[float, float]] = {
        name: network.snap(Point(x, y)) for name, x, y in stops}
    wait = headway_min / 2.0
    for name, n in stop_nodes.items():
        # directed boarding/alighting edges keep the wait charged exactly
        # once per boarding (an undirected graph would merge them)
        G.add_edge((n, 0), (n, 1), weight=wait)   # boarding (wait once)
        G.add_edge((n, 1), (n, 0), weight=0.0)    # alighting (free)
    for route_name, stop_names in routes:
        for a, b in zip(stop_names, stop_names[1:]):
            na, nb = stop_nodes[a], stop_nodes[b]
            spacing = Point(na).distance(Point(nb))
            w = spacing / bus_m_per_min
            G.add_edge((na, 1), (nb, 1), weight=w)
            G.add_edge((nb, 1), (na, 1), weight=w)
    return G, stop_nodes


def transit_accessibility(network: WalkingNetwork,
                          source_pt: Point,
                          stops: Sequence[Tuple[str, float, float]],
                          routes: Sequence[Tuple[str, Sequence[str]]],
                          headway_min: float = DEFAULT_HEADWAY_MIN,
                          speed_kmh: float = BUS_SPEED_KMH,
                          access_kmh: float = WALK_SPEED_KMH
                          ) -> Dict[Tuple[float, float], float]:
    """Door-to-door travel minutes from ``source_pt`` by walk + bus.

    Parameters
    ----------
    network : WalkingNetwork
    source_pt : shapely Point
    stops, routes, headway_min, speed_kmh, access_kmh
        See :func:`build_transit_graph`.

    Returns
    -------
    dict
        {street node: total travel minutes} — access walk, one expected
        wait, in-vehicle bus time, egress walk.  Only street-layer nodes
        are returned (bus-layer times fold in through the free
        alighting edge).
    """
    G, _ = build_transit_graph(
        network, stops, routes, headway_min, speed_kmh, access_kmh)
    src = network.snap(source_pt)
    field = nx.single_source_dijkstra_path_length(G, (src, 0), weight='weight')
    return {n: t for (n, layer), t in field.items() if layer == 0}


def multi_modal_score(network: WalkingNetwork,
                      facility_list: Sequence[tuple],
                      residential: Dict[str, Tuple[float, float]],
                      stops: Optional[Sequence[Tuple[str, float, float]]] = None,
                      routes: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
                      headway_min: float = DEFAULT_HEADWAY_MIN,
                      bike_kmh: float = BIKE_SPEED_KMH,
                      transit_kmh: float = BUS_SPEED_KMH,
                      bike_factor: float = 0.8,
                      transit_factor: float = 0.7,
                      decay: str = 'linear') -> Dict[str, Dict[str, float]]:
    """Combined multi-modal accessibility score per residential point.

    Per mode, per point: ``score = 100 * sum_cat w_cat * decay(t_cat) /
    sum_cat w_cat`` where ``t_cat`` is the travel time to the nearest
    facility of the category measured in that mode, and the decay radius
    is the category's required *time* budget.  All modes therefore share
    the same time budget per category — distance scales with speed:

    - walk:    distance (m) vs the required radius (m)
    - bike:    same distance vs ``radius * bike_kmh / WALK_SPEED_KMH``
    - transit: door-to-door minutes (walk + wait + ride + walk) vs
      ``radius / walk m-per-min``

    Combined score per METHODS spec:
    ``score = max(walk_score, bike_factor * bike_score,
                  transit_factor * transit_score)``
    with default factors 0.8 / 0.7.

    Parameters
    ----------
    network : WalkingNetwork
    facility_list : list of tuples
        ``(name_zh, x, y, category, level, required_radius_m, weight)``.
    residential : dict
        {name: (x, y)} residential points.
    stops, routes : optional
        Transit network; when omitted the transit score is 0.
    headway_min, bike_kmh, transit_kmh : float
        Transit headway and mode speeds.
    bike_factor, transit_factor : float
        Combined-score weights for bike and transit.
    decay : str
        One of ``'step'``, ``'linear'``, ``'gaussian'``.

    Returns
    -------
    dict
        {residential name: {'walk': .., 'bike': .., 'transit': ..,
                            'combined': ..}} with all scores in 0-100.
    """
    facs = [(n, Point(x, y), c, l, r, w) for n, x, y, c, l, r, w in facility_list]
    rpts = {k: Point(xy) for k, xy in residential.items()}
    cats = sorted({f[2] for f in facs})
    rad_of = {f[2]: f[4] for f in facs}
    weights = {f[2]: f[5] for f in facs}
    total_w = sum(weights[c] for c in cats)
    dfun = {'step': decay_step, 'linear': decay_linear,
            'gaussian': decay_gaussian}[decay]

    bike_ratio = bike_kmh / WALK_SPEED_KMH
    walk_m_per_min = WALK_SPEED_KMH * 1000.0 / 60.0
    fac_nodes = {i: network.snap(Point(f[1])) for i, f in enumerate(facs)}
    fac_cat = {i: f[2] for i, f in enumerate(facs)}

    out: Dict[str, Dict[str, float]] = {}
    for rname, rpt in rpts.items():
        # walk distance to nearest facility per category
        d_walk = {c: _nearest_category_distance(
            lambda a, b: network.network_dist(a, b), rpt, facs, c)
            for c in cats}
        t_walk = {c: d_walk[c] / walk_m_per_min for c in cats}

        # transit field: one dijkstra per source point on the layered graph
        t_transit = {}
        if stops and routes:
            field = transit_accessibility(
                network, rpt, stops, routes, headway_min, transit_kmh)
            for c in cats:
                t_transit[c] = min(
                    (field.get(fac_nodes[i], float('inf'))
                     for i, f in enumerate(facs) if fac_cat[i] == c),
                    default=float('inf'))

        walk_s = sum(weights[c] * dfun(np.array([t_walk[c]]), rad_of[c] / walk_m_per_min)[0]
                     for c in cats) / total_w
        bike_s = sum(weights[c] * dfun(np.array([d_walk[c]]), rad_of[c] * bike_ratio)[0]
                     for c in cats) / total_w
        transit_s = (sum(weights[c] * dfun(np.array([t_transit[c]]), rad_of[c] / walk_m_per_min)[0]
                         for c in cats) / total_w if t_transit else 0.0)
        combined = max(walk_s, bike_factor * bike_s, transit_factor * transit_s)
        out[rname] = {'walk': 100.0 * walk_s, 'bike': 100.0 * bike_s,
                      'transit': 100.0 * transit_s, 'combined': 100.0 * combined}
    return out


__all__ = [
    "WalkingNetwork", "decay_step", "decay_linear", "decay_gaussian",
    "euclidean_coverage", "evaluate_living_circles", "accessibility_scores",
    "WALK_SPEED_KMH", "BIKE_SPEED_KMH", "BUS_SPEED_KMH", "DEFAULT_HEADWAY_MIN",
    "cycling_isochrone", "build_transit_graph", "transit_accessibility",
    "multi_modal_score",
]
