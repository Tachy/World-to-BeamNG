"""
Lane splits: nodes where the lanes of one road (the trunk) continue exactly onto several branches - a motorway exit
and entrance where the outer lanes of a four-lane road become the ramps, a road that splits into straight-on and turn
lanes. OSM draws all these ways into ONE node, so their carriageways would overlap there; with the split known, every
branch can start in its own lanes of the trunk's cross-section and only then turn into its own course and width.

Lanes per direction come from the OSM tags (`lanes`, `lanes:forward`, `lanes:backward`, `oneway`). A node is a split
when the lanes flowing from the trunk into the node equal the sum of the lanes flowing out into the branches and vice
versa, and when the branches, ordered from left to right, keep all lanes towards the node on the left and all lanes
away from it on the right (right-hand traffic) - otherwise the lanes would cross and the node is left alone.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .road_markings import lane_count, parse_lanes

ONEWAY_FORWARD = ("yes", "true", "1")
ONEWAY_BACKWARD = ("-1", "reverse")
DEFAULT_WIDTH = 6.0  # lane_count() fallback when the width is unknown: wide enough for two lanes
TRUNK_PROBE = 5.0  # the trunk axis is taken over its first meters from the node
BRANCH_PROBE = 15.0  # the branch order is taken from where the branch has turned into its own course


class Lanes(NamedTuple):
    forward: int  # in digitization direction
    backward: int

    @property
    def total(self) -> int:
        return self.forward + self.backward


def directional_lanes(tags: dict, width: float = DEFAULT_WIDTH, min_two_lane_width: float = 5.5) -> Lanes:
    """Lanes per direction; without lanes:forward/backward a two-way road gives the larger half to the forward
    direction (like marking_layout())."""
    tags = tags or {}
    total = lane_count(tags, width, min_two_lane_width)
    oneway = str(tags.get("oneway", "")).strip().lower()
    if oneway in ONEWAY_FORWARD:
        return Lanes(total, 0)
    if oneway in ONEWAY_BACKWARD:
        return Lanes(0, total)
    forward, backward = parse_lanes(tags.get("lanes:forward")), parse_lanes(tags.get("lanes:backward"))
    if forward is not None and backward is not None and forward + backward == total:
        return Lanes(forward, backward)
    if forward is not None and forward < total:
        return Lanes(forward, total - forward)
    if backward is not None and backward < total:
        return Lanes(total - backward, backward)
    return Lanes((total + 1) // 2, total // 2)


@dataclass
class Branch:
    road: Dict
    at_start: bool  # the branch starts at the node (otherwise it ends there)
    slot_offset: np.ndarray  # (x, y): from the trunk centre to the centre of the branch's lanes at the node
    slot_width: float  # width of the branch's lanes in the trunk's cross-section


@dataclass
class LaneSplit:
    node: Tuple[float, float]
    trunk: Dict
    trunk_at_start: bool
    trunk_width: float
    left_normal: Tuple[float, float]  # left of the direction from the node into the branches
    branches: List[Branch]


@dataclass
class _End:
    road: Dict
    at_start: bool
    lanes_in: int  # lanes flowing into the node
    lanes_out: int
    near_dir: np.ndarray  # away from the node, over TRUNK_PROBE
    far_dir: np.ndarray  # away from the node, over BRANCH_PROBE


def _direction_from_node(coords: np.ndarray, at_start: bool, probe: float) -> Optional[np.ndarray]:
    xy = coords[:, :2] if at_start else coords[::-1, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] < 1e-6:
        return None
    s = min(probe, arc[-1])
    point = np.array([np.interp(s, arc, xy[:, 0]), np.interp(s, arc, xy[:, 1])])
    vector = point - xy[0]
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else None


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def find_lane_splits(
    roads: List[Dict], width_of: Callable[[Dict], float], endpoint_tol: float = 0.5,
    lanes_of: Optional[Callable[[Dict], Lanes]] = None,
) -> List[LaneSplit]:
    """
    All lane splits among `roads` ({"coords": [(x, y, z), ...], "osm_tags"}); `width_of(road)` is the carriageway
    width, `lanes_of(road)` the lanes per direction (default: directional_lanes() of the tags and that width).
    """
    if lanes_of is None:
        lanes_of = lambda road: directional_lanes(road.get("osm_tags", {}), width_of(road))

    ends, points = [], []
    for road in roads:
        coords = np.asarray(road.get("coords", []), dtype=float)
        if len(coords) < 2:
            continue
        lanes = lanes_of(road)
        for at_start in (True, False):
            near = _direction_from_node(coords, at_start, TRUNK_PROBE)
            far = _direction_from_node(coords, at_start, BRANCH_PROBE)
            if near is None or far is None:
                continue
            # digitization direction points away from the node at the start of the way
            lanes_in, lanes_out = (lanes.backward, lanes.forward) if at_start else (lanes.forward, lanes.backward)
            ends.append(_End(road, at_start, lanes_in, lanes_out, near, far))
            points.append(coords[0, :2] if at_start else coords[-1, :2])
    if len(ends) < 3:
        return []

    parent = list(range(len(ends)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(np.asarray(points)).query_pairs(endpoint_tol):
        parent[find(i)] = find(j)
    clusters: Dict[int, List[int]] = {}
    for i in range(len(ends)):
        clusters.setdefault(find(i), []).append(i)

    splits = []
    for members in clusters.values():
        if len(members) < 3 or len({id(ends[i].road) for i in members}) != len(members):
            continue
        node = tuple(float(v) for v in np.mean([points[i] for i in members], axis=0))
        candidates = [s for s in (_split_with_trunk(ends, members, t, node, width_of) for t in members) if s]
        if len(candidates) == 1:
            splits.append(candidates[0])
        elif candidates:
            candidates.sort(key=lambda s: -lanes_of(s.trunk).total)
            if lanes_of(candidates[0].trunk).total > lanes_of(candidates[1].trunk).total:
                splits.append(candidates[0])
    return splits


def _split_with_trunk(ends, members, trunk_index, node, width_of) -> Optional[LaneSplit]:
    trunk = ends[trunk_index]
    branches = [ends[i] for i in members if i != trunk_index]
    axis = -trunk.near_dir  # from the node into the branches
    if any(float(np.dot(axis, b.far_dir)) <= 0.0 for b in branches):
        return None  # a branch leaves backwards past the trunk: no split of the trunk's lanes
    if trunk.lanes_in != sum(b.lanes_out for b in branches) or trunk.lanes_out != sum(b.lanes_in for b in branches):
        return None
    if any(b.lanes_in + b.lanes_out == 0 for b in branches):
        return None

    # left to right as seen along the axis; right-hand traffic: lanes towards the node lie left, lanes away right
    ordered = sorted(branches, key=lambda b: -_cross(axis, b.far_dir))
    sequence = [flow for b in ordered for flow in ["in"] * b.lanes_in + ["out"] * b.lanes_out]
    if sequence != sorted(sequence):  # "in" < "out": all lanes towards the node first
        return None

    total = len(sequence)
    width = float(width_of(trunk.road))
    lane_width = width / total
    left_normal = np.array([-axis[1], axis[0]])
    result, start = [], 0
    for b in ordered:
        count = b.lanes_in + b.lanes_out
        centre_from_left = (start + count / 2.0) * lane_width
        result.append(Branch(b.road, b.at_start, left_normal * (width / 2.0 - centre_from_left), count * lane_width))
        start += count
    return LaneSplit(node, trunk.road, trunk.at_start, width, (float(left_normal[0]), float(left_normal[1])), result)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _branch_pieces(roads, first, at_start, length, endpoint_tol, max_angle_deg):
    """[(road, enters_at_start, arc offset from the node)] along the branch and its straight continuations until
    `length` is covered or the branch ends; plus the covered length."""
    min_opposition = float(np.cos(np.radians(max_angle_deg)))
    pieces, offset, road, entry, visited = [], 0.0, first, at_start, set()
    while True:
        visited.add(id(road))
        coords = np.asarray(road["coords"], dtype=float)
        oriented = coords if entry else coords[::-1]
        piece_length = float(np.sum(np.linalg.norm(np.diff(oriented[:, :2], axis=0), axis=1)))
        pieces.append((road, entry, offset))
        offset += piece_length
        if offset >= length:
            return pieces, offset
        far = oriented[-1, :2]
        direction = _direction_from_node(oriented, False, TRUNK_PROBE)  # from the far end back into the piece
        best = None
        for other in roads:
            if id(other) in visited or len(other.get("coords", [])) < 2:
                continue
            other_coords = np.asarray(other["coords"], dtype=float)
            for other_start in (True, False):
                point = other_coords[0 if other_start else -1, :2]
                if np.linalg.norm(point - far) > endpoint_tol:
                    continue
                other_dir = _direction_from_node(other_coords, other_start, TRUNK_PROBE)
                if direction is None or other_dir is None:
                    continue
                opposition = -float(np.dot(direction, other_dir))
                if opposition >= min_opposition and (best is None or opposition > best[0]):
                    best = (opposition, other, other_start)
        if best is None:
            return pieces, offset
        road, entry = best[1], best[2]


def _densify(oriented: np.ndarray, offset: float, until: float, step: float) -> np.ndarray:
    """Adds points every `step` meters (x, y, z linear) wherever the arc length from the node (`offset` at the first
    point) is below `until` - the held stretch runs straight, not along the few OSM points of a diverging ramp."""
    result = [oriented[0]]
    s = offset
    for a, b in zip(oriented[:-1], oriented[1:]):
        seg = float(np.linalg.norm(b[:2] - a[:2]))
        if seg > step and s < until:
            count = int(np.ceil(seg / step))
            result.extend(a + (b - a) * (k / count) for k in range(1, count))
        result.append(b)
        s += seg
    return np.asarray(result)


def _connector_length(arc: np.ndarray, lateral: np.ndarray, slot: float, max_connector: float,
                      tol: float = 0.25, plateau: float = 0.1, window: float = 2.0) -> float:
    """Arc length at which OSM's branch has reached its lane centre: its lateral offset from the trunk axis comes within
    `tol` of the slot offset, or stops growing (less than `plateau` over `window` meters) after covering half of it -
    OSM's lane centre is not exactly our slot. At most max_connector (and the branch's length)."""
    limit = min(max_connector, float(arc[-1]))
    target = abs(slot)
    if target < tol:
        return 0.0
    sign = 1.0 if slot > 0.0 else -1.0
    for s in np.arange(0.0, limit, 0.5):
        here = sign * float(np.interp(s, arc, lateral))
        if here >= target - tol:
            return float(s)
        ahead = sign * float(np.interp(s + window, arc, lateral))
        if s >= window and here >= 0.5 * target and ahead - here < plateau:
            return float(s)
    return limit


def shift_branches_into_slots(
    splits: List[LaneSplit], roads: List[Dict], max_connector: float, length: float, endpoint_tol: float = 0.5,
    max_angle_deg: float = 60.0, step: float = 2.0,
) -> None:
    """
    OSM draws every branch from the node to the centre of its lane first - a symbolic connector - and only from there
    along the lane. That connector is replaced by a straight run in the branch's slot (its lanes of the trunk's
    cross-section, starting at the node) up to where OSM's branch reaches its lane centre (at most max_connector
    meters, see _connector_length()); from there the branch follows its OSM course. Only the small difference between
    OSM's lane centre and the slot fades out with a smoothstep over `length` meters, so the carriageways lie side by side
    at first. The branch is followed along straight continuations where one piece is shorter; bridges and ground roads
    are treated alike; the kink where OSM's connector turns into the lane may be up to max_angle_deg. Marks the first piece of every branch with "lane_split_branch" (the node end, its slot width,
    the connector length as "hold", the fade length and the node) and the trunk with "lane_split_trunk" (its ends at
    split nodes) and "lane_split_trunk_nodes".
    """
    for split in splits:
        split.trunk.setdefault("lane_split_trunk", set()).add("start" if split.trunk_at_start else "end")
        split.trunk.setdefault("lane_split_trunk_nodes", []).append(split.node)
        node = np.asarray(split.node, dtype=float)
        normal = np.asarray(split.left_normal, dtype=float)
        for branch in split.branches:
            pieces, reach = _branch_pieces(roads, branch.road, branch.at_start, max_connector + length, endpoint_tol, max_angle_deg)
            oriented_pieces = []
            for road, entry, offset in pieces:
                coords = np.asarray(road["coords"], dtype=float)
                oriented = coords if entry else coords[::-1]
                arc = offset + np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(oriented[:, :2], axis=0), axis=1))])
                oriented_pieces.append((road, entry, offset, oriented, arc))
            chain_arc = np.concatenate([p[4] for p in oriented_pieces])
            chain_xy = np.vstack([p[3][:, :2] for p in oriented_pieces])
            lateral = (chain_xy - node) @ normal
            slot = float(branch.slot_offset @ normal)
            connector = _connector_length(chain_arc, lateral, slot, max_connector)
            blend = max(0.0, min(length, reach - connector))
            branch.road["lane_split_branch"] = {
                "end": "start" if branch.at_start else "end", "slot_width": branch.slot_width, "hold": connector,
                "length": blend, "node": split.node,
            }
            at_lane = np.array([np.interp(connector, chain_arc, chain_xy[:, 0]), np.interp(connector, chain_arc, chain_xy[:, 1])])
            correction = (slot - float((at_lane - node) @ normal)) * normal
            start, lane_point = node + branch.slot_offset, at_lane + correction
            if connector + blend <= 0.0 and float(np.linalg.norm(correction)) < 1e-9:
                continue
            for road, entry, offset, oriented, _ in oriented_pieces:
                oriented = _densify(oriented, offset, connector + blend, step)
                arc = offset + np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(oriented[:, :2], axis=0), axis=1))])
                fade = 1.0 - (_smoothstep((arc - connector) / blend) if blend > 0.0 else (arc > connector).astype(float))
                shifted = oriented[:, :2] + fade[:, None] * correction[None, :]
                on_connector = arc < connector
                if connector > 0.0:
                    t = (arc[on_connector] / connector)[:, None]
                    shifted[on_connector] = start[None, :] + t * (lane_point - start)[None, :]
                oriented = oriented.copy()
                oriented[:, :2] = shifted
                result = oriented if entry else oriented[::-1]
                road["coords"] = [tuple(float(v) for v in point) for point in result]
