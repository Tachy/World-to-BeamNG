"""
Weiche Breitenübergänge zwischen aneinanderstoßenden DecalRoads.

Ändert sich an einem Stoßpunkt zweier Straßen, die geradeaus ineinander übergehen, die Breite (z.B. lanes=2 ->
lanes=3), springt sie nicht mehr hart um: über ROAD_WIDTH_TRANSITION_LENGTH (je die Hälfte vor und nach dem
Stoßpunkt) wird sie mit einem kubischen Hermite-Spline (smoothstep, Steigung 0 an beiden Zonenenden) übergeblendet.
Die Breite steckt im 4. Eintrag jedes DecalRoad-Knotens [x, y, z, width]; BeamNG interpoliert zwischen den Knoten,
deshalb bekommt die Übergangszone zusätzliche Knoten im Abstand `step`.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

Endpoint = Tuple[int, str]  # (Index der Straße, "start" | "end")


def smoothstep(t: float) -> float:
    """Kubischer Hermite-Spline 3t^2 - 2t^3 auf [0, 1] (außerhalb geklemmt)."""
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _arc_lengths(nodes: Sequence[Sequence[float]]) -> np.ndarray:
    xy = np.asarray(nodes, dtype=float)[:, :2]
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def outward_direction(nodes: Sequence[Sequence[float]], end: str) -> Optional[np.ndarray]:
    """Einheitsvektor vom Stoßpunkt in die Straße hinein (None bei einer Straße ohne Ausdehnung)."""
    xy = np.asarray(nodes, dtype=float)[:, :2]
    if end == "end":
        xy = xy[::-1]
    for other in xy[1:]:
        vector = other - xy[0]
        length = float(np.linalg.norm(vector))
        if length > 1e-9:
            return vector / length
    return None


def find_continuations(
    roads: Sequence[Sequence[Sequence[float]]], endpoint_tol: float, max_angle_deg: float
) -> List[Tuple[Endpoint, Endpoint]]:
    """
    Paare von Straßenenden, die am selben Punkt liegen (Abstand <= endpoint_tol) und geradeaus ineinander übergehen
    (Knick <= max_angle_deg). An einer Einmündung gewinnt das gestreckteste Paar; eine schräg abzweigende Rampe
    bleibt ungepaart. Eine Straße wird nie mit sich selbst gepaart (Ring).
    """
    endpoints = []  # ((road_idx, end), (x, y), outward_direction)
    for road_idx, nodes in enumerate(roads):
        if len(nodes) < 2:
            continue
        for end in ("start", "end"):
            direction = outward_direction(nodes, end)
            if direction is None:
                continue
            point = nodes[0] if end == "start" else nodes[-1]
            endpoints.append(((road_idx, end), (float(point[0]), float(point[1])), direction))
    if len(endpoints) < 2:
        return []

    parent = list(range(len(endpoints)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(np.array([e[1] for e in endpoints])).query_pairs(endpoint_tol):
        parent[root(a)] = root(b)
    clusters: Dict[int, List[int]] = {}
    for i in range(len(endpoints)):
        clusters.setdefault(root(i), []).append(i)

    min_opposition = float(np.cos(np.radians(max_angle_deg)))
    pairs = []
    for members in clusters.values():
        candidates = []
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = endpoints[members[x]], endpoints[members[y]]
                if a[0][0] == b[0][0]:
                    continue
                opposition = -float(np.dot(a[2], b[2]))  # 1 = exakt geradeaus
                if opposition >= min_opposition:
                    candidates.append((opposition, members[x], members[y]))
        used = set()
        for _, a, b in sorted(candidates, key=lambda c: -c[0]):
            if a in used or b in used:
                continue
            used.update((a, b))
            pairs.append((endpoints[a][0], endpoints[b][0]))
    return pairs


def continuation_partners(pairs: Sequence[Tuple[Endpoint, Endpoint]]) -> Dict[int, Set[int]]:
    """Straßen-Index -> Indizes der Straßen, in die sie geradeaus übergeht."""
    partners: Dict[int, Set[int]] = {}
    for (a, _), (b, _) in pairs:
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def _insert_nodes(nodes: List[List[float]], distances: Sequence[float], min_spacing: float) -> List[List[float]]:
    """Zusätzliche Knoten bei den Bogenlängen `distances` (ab nodes[0]), linear interpoliert (x, y, z, Breite).
    Wo schon ein Knoten näher als min_spacing liegt, wird nichts eingefügt (BeamNG verwirft zu kurze Segmente)."""
    arr = np.asarray(nodes, dtype=float)
    cum = _arc_lengths(nodes)
    entries = [(float(cum[i]), [float(v) for v in arr[i]]) for i in range(len(arr))]
    taken = list(cum)
    for d in distances:
        if d <= 0.0 or d >= cum[-1] or min(abs(t - d) for t in taken) < min_spacing:
            continue
        seg = int(np.searchsorted(cum, d)) - 1
        f = (d - cum[seg]) / (cum[seg + 1] - cum[seg])
        entries.append((float(d), [float(v) for v in arr[seg] + f * (arr[seg + 1] - arr[seg])]))
        taken.append(d)
    entries.sort(key=lambda e: e[0])
    return [node for _, node in entries]


def _blend_end(nodes, end, own_width, other_width, half, step, min_spacing):
    """Breiten in der Übergangszone am Ende `end`: Spline von der mittleren Breite (Stoßpunkt) zur eigenen (half)."""
    work = [list(n) for n in (nodes if end == "start" else nodes[::-1])]
    distances = [float(d) for d in np.arange(step, half, step)] + [half]
    work = _insert_nodes(work, distances, min_spacing)
    for node, s in zip(work, _arc_lengths(work)):
        if s <= half + 1e-9:
            node[3] = own_width + (other_width - own_width) * smoothstep((half - s) / (2.0 * half))
    return work if end == "start" else work[::-1]


def apply_width_transitions(
    roads: Sequence[Sequence[Sequence[float]]],
    transition_length: float,
    step: float,
    endpoint_tol: float,
    max_angle_deg: float,
    min_delta: float,
    min_spacing: float,
) -> List[List[List[float]]]:
    """
    Neue Knotenlisten ([x, y, z, width] je Knoten) mit weichen Breitenübergängen an allen Geradeaus-Stößen, deren
    Breiten sich um mindestens min_delta unterscheiden. Am Stoßpunkt liegt die mittlere Breite, transition_length / 2
    davor und dahinter wieder die eigene. Ist eine der beiden Straßen kürzer als transition_length, schrumpft die Zone
    auf beiden Seiten symmetrisch auf die halbe Länge der kürzeren Straße.
    """
    result = [[[float(v) for v in n] for n in nodes] for nodes in roads]
    for (ia, ea), (ib, eb) in find_continuations(roads, endpoint_tol, max_angle_deg):
        wa = float(roads[ia][0 if ea == "start" else -1][3])
        wb = float(roads[ib][0 if eb == "start" else -1][3])
        if abs(wa - wb) < min_delta:
            continue
        half = min(transition_length / 2.0, _arc_lengths(roads[ia])[-1] / 2.0, _arc_lengths(roads[ib])[-1] / 2.0)
        if half <= 0.0:
            continue
        result[ia] = _blend_end(result[ia], ea, wa, wb, half, step, min_spacing)
        result[ib] = _blend_end(result[ib], eb, wb, wa, half, step, min_spacing)
    return result


def close_continuation_gaps(
    roads: Sequence[Sequence[Sequence[float]]], endpoint_tol: float, max_angle_deg: float, min_angle_deg: float = 1.0
) -> List[List[List[float]]]:
    """
    Neue Knotenlisten, in denen an jedem geknickten Geradeaus-Stoß beide Enden über den Stoßpunkt hinaus verlängert
    sind. Ein DecalRoad endet mit einer flachen Kante senkrecht zu seinem eigenen letzten Segment; bei einem Knick um t
    bleibt deshalb außen ein Keil frei, der bis zur Fahrbahnkante reicht (im Spiel sichtbar). Der Endknoten wird dafür
    um (Breite / 2) * tan(t) + 5 cm in seiner eigenen Richtung verschoben - verschoben statt angehängt, damit kein
    Segment unter DECAL_ROAD_MIN_NODE_SPACING entsteht (BeamNG verwirft sonst das ganze Decal). Innen überlappen die
    Decals; Stöße mit weniger als min_angle_deg Knick bleiben unverändert.
    """
    result = [[[float(v) for v in n] for n in nodes] for nodes in roads]
    for (ia, ea), (ib, eb) in find_continuations(roads, endpoint_tol, max_angle_deg):
        da, db = outward_direction(roads[ia], ea), outward_direction(roads[ib], eb)
        kink = float(np.degrees(np.arccos(np.clip(-np.dot(da, db), -1.0, 1.0))))
        if kink < min_angle_deg:
            continue
        for road_idx, end, outward in ((ia, ea, da), (ib, eb, db)):
            node = result[road_idx][0 if end == "start" else -1]
            extension = node[3] / 2.0 * float(np.tan(np.radians(kink))) + 0.05
            node[0] -= float(outward[0]) * extension
            node[1] -= float(outward[1]) * extension
    return result
