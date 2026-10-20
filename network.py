"""Graph-based digital railway-network state model.

The network is a synthetic but operationally realistic layout containing:
  * a 20 km double-track main line (Up line eastbound, Down line westbound,
    5 m track-centre spacing) with two stations, each with an Up passing loop;
  * a flat junction (J on the Up line, DJ on the Down line) where Up trains
    diverging to the branch cross the Down line, and branch trains converge
    onto the Down line (crossing and converging fouling-point conflicts);
  * a curved single-line branch (R = 1500 m) with a passing loop, used for
    head-on and loop-overrun conflicts.

Routes are defined as interlocking-style node sequences (a "route table"), and
all safety distances are measured *along the route* (track distance), not as
straight-line (Euclidean) distance. NetworkX stores the topology and is used to
validate routes and to compute track-referenced chainage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

TRACK_SPACING = 5.0  # m, centre-to-centre distance of parallel tracks
ARC_RADIUS = 1500.0  # m, branch curve radius


def _arc_points(cx, cy, r, a0, a1, n=24):
    ts = np.linspace(a0, a1, n)
    return [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in ts]


@dataclass
class Route:
    name: str
    nodes: list
    line_type: str          # main | loop | branch | crossing
    speed_limit_kmh: float


def build_network() -> nx.Graph:
    """Return the undirected track graph with geometry on every edge."""
    G = nx.Graph()
    xs = {"0": 0, "1": 4600, "2": 5400, "3": 9800, "4": 14600, "5": 15400, "6": 20000}
    # Up main (y = 0) and Down main (y = TRACK_SPACING)
    for k, x in xs.items():
        G.add_node(f"U{k}", pos=(x, 0.0))
        G.add_node(f"D{k}", pos=(x, TRACK_SPACING))
    G.add_node("J", pos=(10000.0, 0.0))     # flat junction on Up line
    G.add_node("DJ", pos=(10000.0, TRACK_SPACING))
    # station loops on the Up side (y = -5)
    G.add_node("L1w", pos=(4700.0, -TRACK_SPACING)); G.add_node("L1e", pos=(5300.0, -TRACK_SPACING))
    G.add_node("L2w", pos=(14700.0, -TRACK_SPACING)); G.add_node("L2e", pos=(15400.0 - 100, -TRACK_SPACING))
    # branch: quarter-circle from DJ curving north-east, then straight north
    cx, cy = 10000.0, TRACK_SPACING + ARC_RADIUS
    G.add_node("BA", pos=(cx + ARC_RADIUS, cy))           # end of curve
    G.add_node("Pw", pos=(cx + ARC_RADIUS, cy + 2500.0))  # passing-loop south end
    G.add_node("Pe", pos=(cx + ARC_RADIUS, cy + 3300.0))  # passing-loop north end
    G.add_node("PL1", pos=(cx + ARC_RADIUS + TRACK_SPACING, cy + 2600.0))
    G.add_node("PL2", pos=(cx + ARC_RADIUS + TRACK_SPACING, cy + 3200.0))
    G.add_node("T", pos=(cx + ARC_RADIUS, cy + 8000.0))

    def straight(u, v, grad=0.0, line="main"):
        (x0, y0), (x1, y1) = G.nodes[u]["pos"], G.nodes[v]["pos"]
        G.add_edge(u, v, length=math.hypot(x1 - x0, y1 - y0), geom=[(x0, y0), (x1, y1)],
                   gradient=grad, curvature=0.0, line=line)

    order = ["0", "1", "2", "3"]
    grads = {("0", "1"): 0.002, ("1", "2"): 0.0, ("2", "3"): -0.004}
    for a, b in zip(order[:-1], order[1:]):
        straight(f"U{a}", f"U{b}", grads[(a, b)]); straight(f"D{a}", f"D{b}", grads[(a, b)])
    straight("U3", "J", -0.004); straight("D3", "DJ", -0.004)
    straight("J", "U4", 0.010); straight("DJ", "D4", 0.010)      # 1 % gradient section
    straight("U4", "U5", 0.0); straight("D4", "D5", 0.0)
    straight("U5", "U6", 0.006); straight("D5", "D6", 0.006)
    for (w, e, a, b) in [("L1w", "L1e", "U1", "U2"), ("L2w", "L2e", "U4", "U5")]:
        straight(a, w, 0.0, "loop"); straight(w, e, 0.0, "loop"); straight(e, b, 0.0, "loop")
    # flat-junction diamond: branch crosses Up line at J and joins Down line at DJ
    straight("J", "DJ", 0.0, "crossing")
    # curved branch from DJ (heading east) turning north
    pts = _arc_points(cx, cy, ARC_RADIUS, -math.pi / 2, 0.0)
    pts[0] = G.nodes["DJ"]["pos"]
    G.add_edge("DJ", "BA", length=ARC_RADIUS * math.pi / 2, geom=pts,
               gradient=0.012, curvature=1.0 / ARC_RADIUS, line="branch")
    straight("BA", "Pw", 0.015, "branch")
    straight("Pw", "Pe", 0.0, "branch")
    straight("Pw", "PL1", 0.0, "loop"); straight("PL1", "PL2", 0.0, "loop"); straight("PL2", "Pe", 0.0, "loop")
    straight("Pe", "T", -0.008, "branch")
    return G


def route_table() -> dict:
    """Interlocking-style route table. Direction is implied by node order."""
    up = ["U0", "U1", "U2", "U3", "J", "U4", "U5", "U6"]
    return {
        "UP_MAIN": Route("UP_MAIN", up, "main", 130),
        "UP_LOOP1": Route("UP_LOOP1", ["U0", "U1", "L1w", "L1e", "U2", "U3", "J", "U4", "U5", "U6"], "loop", 130),
        "UP_LOOP2": Route("UP_LOOP2", ["U0", "U1", "U2", "U3", "J", "U4", "L2w", "L2e", "U5", "U6"], "loop", 130),
        "DN_MAIN": Route("DN_MAIN", ["D6", "D5", "D4", "DJ", "D3", "D2", "D1", "D0"], "main", 130),
        "UP_TO_BRANCH": Route("UP_TO_BRANCH", ["U0", "U1", "U2", "U3", "J", "DJ", "BA", "Pw", "Pe", "T"], "branch", 90),
        "BRANCH_DN": Route("BRANCH_DN", ["T", "Pe", "Pw", "BA", "DJ", "D3", "D2", "D1", "D0"], "branch", 90),
        "BRANCH_DN_LOOP": Route("BRANCH_DN_LOOP", ["T", "Pe", "PL2", "PL1", "Pw", "BA", "DJ", "D3", "D2", "D1", "D0"], "branch", 90),
        "BRANCH_UP_LOOP": Route("BRANCH_UP_LOOP", ["U0", "U1", "U2", "U3", "J", "DJ", "BA", "Pw", "PL1", "PL2", "Pe", "T"], "branch", 90),
    }


class RouteGeometry:
    """Chainage <-> coordinate and attribute lookup along a route."""

    def __init__(self, G: nx.Graph, route: Route):
        self.route = route
        self.G = G
        for u, v in zip(route.nodes[:-1], route.nodes[1:]):
            if not G.has_edge(u, v):
                raise ValueError(f"route {route.name}: missing edge {u}-{v}")
        self.edges = list(zip(route.nodes[:-1], route.nodes[1:]))
        self.cum = np.concatenate([[0.0], np.cumsum([G.edges[e]["length"] for e in self.edges])])
        self.node_chainage = {n: c for n, c in zip(route.nodes, self.cum)}
        self.length = float(self.cum[-1])

    def _edge_at(self, s):
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.edges) - 1))
        return i, s - self.cum[i]

    def xy(self, s):
        i, off = self._edge_at(s)
        u, v = self.edges[i]
        geom = self.G.edges[u, v]["geom"]
        if tuple(geom[0]) != tuple(self.G.nodes[u]["pos"]):
            geom = geom[::-1]
        seg = np.array(geom)
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
        frac = off / self.G.edges[u, v]["length"] * d[-1]
        j = int(np.clip(np.searchsorted(d, frac, side="right") - 1, 0, len(seg) - 2))
        t = (frac - d[j]) / max(d[j + 1] - d[j], 1e-9)
        return seg[j] + t * (seg[j + 1] - seg[j])

    def attrs(self, s):
        i, _ = self._edge_at(s)
        u, v = self.edges[i]
        e = self.G.edges[u, v]
        # gradient sign convention: positive = downhill in direction of travel
        sign = 1.0 if self.G.nodes[u]["pos"][0] <= self.G.nodes[v]["pos"][0] else -1.0
        if e["line"] in ("branch",) and self.G.nodes[u]["pos"][1] > self.G.nodes[v]["pos"][1]:
            sign = -1.0
        return {"gradient": -sign * e["gradient"], "curvature": e["curvature"], "line": e["line"]}


def euclidean(p, q) -> float:
    return float(math.hypot(p[0] - q[0], p[1] - q[1]))
