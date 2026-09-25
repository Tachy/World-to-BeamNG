"""
Smooth width transitions between abutting DecalRoads.

If the width changes (e.g. lanes=2 -> lanes=3) at the joint of two roads that continue straight into each other, it
no longer jumps abruptly: over ROAD_WIDTH_TRANSITION_LENGTH (half before and half after the joint) it is blended
with a cubic Hermite spline (smoothstep, slope 0 at both ends of the zone).
The width is stored in the 4th entry of each DecalRoad node [x, y, z, width]; BeamNG interpolates between the nodes,
so the transition zone gets additional nodes at spacing `step`.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

Endpoint = Tuple[int, str]  # (road index, "start" | "end")


def smoothstep(t: float) -> float:
    """Cubic Hermite spline 3t^2 - 2t^3 on [0, 1] (clamped outside)."""
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _arc_lengths(nodes: Sequence[Sequence[float]]) -> np.ndarray:
    xy = np.asarray(nodes, dtype=float)[:, :2]
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def outward_direction(nodes: Sequence[Sequence[float]], end: str) -> Optional[np.ndarray]:
    """Unit vector from the joint into the road (None for a road with zero extent)."""
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
    Pairs of road ends that lie at the same point (distance <= endpoint_tol) and continue straight into each other
    (kink <= max_angle_deg). At a T-junction the straightest pair wins; a ramp branching off at an angle
    stays unpaired. A road is never paired with itself (ring).
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
                opposition = -float(np.dot(a[2], b[2]))  # 1 = exactly straight
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
    """Road index -> indices of the roads it continues straight into."""
    partners: Dict[int, Set[int]] = {}
    for (a, _), (b, _) in pairs:
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def _insert_nodes(nodes: List[List[float]], distances: Sequence[float], min_spacing: float) -> List[List[float]]:
    """Additional nodes at the arc lengths `distances` (from nodes[0]), linearly interpolated (x, y, z, width).
    Where a node is already closer than min_spacing, nothing is inserted (BeamNG discards too-short segments)."""
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
    """Widths in the transition zone at end `end`: spline from the mean width (joint) to its own width (half)."""
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
    New node lists ([x, y, z, width] per node) with smooth width transitions at all straight joints whose widths
    differ by at least min_delta. The mean width applies at the joint, and transition_length / 2 before and after
    it the road's own width applies again. If one of the two roads is shorter than transition_length, the zone
    shrinks symmetrically on both sides to half the length of the shorter road.
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
    New node lists in which both ends are extended past the joint at every kinked straight joint. A DecalRoad ends
    with a flat edge perpendicular to its own last segment; with a kink of t, a wedge therefore stays uncovered on the
    outside, reaching up to the road edge (visible in game). To fix this, the end node is shifted by
    (width / 2) * tan(t) + 5 cm in its own direction - shifted instead of appended, so that no segment shorter than
    DECAL_ROAD_MIN_NODE_SPACING arises (BeamNG otherwise discards the whole decal). On the inside the decals
    overlap; joints with a kink of less than min_angle_deg stay unchanged.
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
