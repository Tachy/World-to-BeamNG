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


def _branch_pieces(roads, first, at_start, length, endpoint_tol, max_angle_deg, keep=None):
    """[(road, enters_at_start, arc offset from the node)] along the branch and its straight continuations until
    `length` is covered or the branch ends (or the next piece fails `keep`); plus the covered length."""
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
        if best is None or (keep is not None and not keep(best[1])):
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


class _Reference:
    """The main axis as a function of the arc length from the node: position and left normal (clamped at its ends)."""

    def __init__(self, arc: np.ndarray, xy: np.ndarray):
        keep = np.concatenate([[True], np.diff(arc) > 1e-9])
        self.arc, self.xy = arc[keep], xy[keep]

    def position(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), self.arc[0], self.arc[-1])
        return np.column_stack([np.interp(s, self.arc, self.xy[:, 0]), np.interp(s, self.arc, self.xy[:, 1])])

    def normal(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        tangent = self.position(s + 0.5) - self.position(s - 0.5)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)[:, None]
        return np.column_stack([-tangent[:, 1], tangent[:, 0]])


class _SideProfile:
    """
    Lateral offset of a side branch from the main axis along its arc length: its slot while it waits beside the main
    axis, then its OSM offset - without a kink. The rest between the slot and OSM's lane centre fades out over `blend`
    meters, the change of direction where the branch leaves its slot over `turn` meters (C1: no corner in the outline);
    OSM's own corners in that zone are rounded by a Gaussian of `smoothing` meters, which hands back to the exact OSM
    offset over `smoothing_end` meters behind the zone.
    """

    def __init__(self, chain: Dict, main_xy: np.ndarray, slot: float, connector: float, blend: float, smoothing: float,
                 turn: float = 10.0, smoothing_end: float = 10.0):
        from scipy.ndimage import gaussian_filter1d

        _, raw = path_frame(main_xy, chain["xy"])
        self.grid = np.arange(0.0, float(chain["arc"][-1]) + 1.0, 1.0)
        profile = np.interp(self.grid, chain["arc"], raw)
        # OSM's symbolic connector is ignored: it must not bend the smoothed course behind it
        profile[self.grid < connector] = float(np.interp(connector, chain["arc"], raw))
        self.smooth = gaussian_filter1d(profile, sigma=smoothing, mode="nearest")
        self.slot, self.connector, self.blend, self.turn, self.smoothing_end = slot, connector, blend, turn, smoothing_end
        at_lane = float(np.interp(connector, self.grid, self.smooth))
        self.offset = slot - at_lane
        self.slope = (float(np.interp(connector + 2.0, self.grid, self.smooth)) - at_lane) / 2.0

    def lateral(self, arc: np.ndarray, raw: np.ndarray) -> np.ndarray:
        smooth = np.interp(arc, self.grid, self.smooth)
        osm = smooth + (raw - smooth) * _smoothstep((arc - self.connector - self.blend) / self.smoothing_end)
        since = arc - self.connector
        fade = 1.0 - (_smoothstep(since / self.blend) if self.blend > 0.0 else (since > 0.0).astype(float))
        turning = 1.0 - _smoothstep(since / self.turn)
        target = osm + self.offset * fade - self.slope * since * turning
        return np.where(since < 0.0, self.slot, target)


def _is_bridge(road: Dict) -> bool:
    from .road_structures import classify_structure

    return classify_structure(road.get("osm_tags", {})) == "bridge"


def _level_side_bridge(roads, chain, main, main_z, endpoint_tol, max_angle_deg) -> None:
    """
    Heights of a side branch on a bridge: while it is part of the main deck (up to where it leaves it, see
    chain["leaves_at"]) it lies at the main deck's height - one flat cross-section; from there to the end of its bridge
    (its abutment) it runs linear, like every bridge between its supports.
    """
    branch = chain["branch"]
    pieces, _ = _branch_pieces(roads, branch.road, branch.at_start, float("inf"), endpoint_tol, max_angle_deg, keep=_is_bridge)
    oriented = []
    for road, entry, _ in pieces:
        coords = np.asarray(road["coords"], dtype=float)
        oriented.append((road, entry, coords if entry else coords[::-1]))
    lengths = [float(np.sum(np.linalg.norm(np.diff(o[2][:, :2], axis=0), axis=1))) for o in oriented]
    total = float(sum(lengths))
    abutment_z = float(oriented[-1][2][-1, 2])
    leaves_at = min(chain["leaves_at"], total)
    offset = 0.0
    for (road, entry, piece), piece_length in zip(oriented, lengths):
        arc = offset + np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(piece[:, :2], axis=0), axis=1))])
        along, _ = path_frame(main["xy"], piece[:, :2])
        on_deck = main_z(along)
        if leaves_at >= total:
            z = on_deck  # never leaves the deck on the bridge: one piece up to the abutment
        else:
            leave_z = float(main_z(path_frame(main["xy"], _point_at(oriented, lengths, leaves_at)[None, :])[0])[0])
            z = np.where(arc <= leaves_at, on_deck, leave_z + (abutment_z - leave_z) * (arc - leaves_at) / (total - leaves_at))
        piece = piece.copy()
        piece[:, 2] = z
        result = piece if entry else piece[::-1]
        road["coords"] = [tuple(float(v) for v in point) for point in result]
        offset += piece_length


def _point_at(oriented, lengths, s: float) -> np.ndarray:
    """(x, y) at arc length `s` along the oriented pieces."""
    offset = 0.0
    for (_, _, piece), piece_length in zip(oriented, lengths):
        if s <= offset + piece_length or piece is oriented[-1][2]:
            arc = offset + np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(piece[:, :2], axis=0), axis=1))])
            return np.array([np.interp(s, arc, piece[:, 0]), np.interp(s, arc, piece[:, 1])])
        offset += piece_length
    return oriented[-1][2][-1, :2]


def shift_branches_into_slots(
    splits: List[LaneSplit], roads: List[Dict], max_connector: float, length: float, endpoint_tol: float = 0.5,
    max_angle_deg: float = 60.0, step: float = 2.0, smoothing: float = 5.0, smoothing_end: float = 10.0,
    leave_gap: float = 0.8,
) -> None:
    """
    The main axis of a split - the branch that continues the trunk most straight - keeps its OSM course everywhere; its
    OSM line goes on from the trunk's centre line. Only where its lanes do not lie in the middle of the trunk, that
    offset fades out with a smoothstep over `length` meters.

    OSM draws every other branch from the node to the centre of its lane first - a symbolic connector. It is replaced
    by a run beside the main axis in the branch's slot (its lanes of the trunk's cross-section) up to where OSM's branch
    reaches its lane centre (at most max_connector meters, see _connector_length()); from there the branch follows its
    OSM course again - smoothly, see _SideProfile: the small difference between OSM's lane centre and the slot fades
    out over `length` meters, the turn out of the slot has no corner, and OSM's corners in that zone are rounded.

    The stem - where the trunk goes on as one piece - follows the main axis as far as the shortest connector of the
    side branches. Heights: the main axis keeps its own (a bridge is linear between its abutments). A side branch on a
    bridge lies at the main deck's height until it has moved `leave_gap` out of its slot (where it leaves the deck - the
    cut) and then runs linear to its abutment (_level_side_bridge()); on the ground it takes the main road's height on
    the stem and returns to its own over `length` meters.

    The branches are followed along straight continuations where one piece is shorter (the kink where OSM's connector
    turns into the lane may be up to max_angle_deg); bridges and ground roads are treated alike. Marks the first piece
    of every branch with "lane_split_branch" (node end, slot width and offset, connector length as "hold", fade length,
    stem length and path [(x, y, z)], node, left normal and trunk width) and the trunk with "lane_split_trunk" (its
    ends at split nodes) and "lane_split_trunk_nodes".
    """
    for split in splits:
        split.trunk.setdefault("lane_split_trunk", set()).add("start" if split.trunk_at_start else "end")
        split.trunk.setdefault("lane_split_trunk_nodes", []).append(split.node)
        node = np.asarray(split.node, dtype=float)
        normal = np.asarray(split.left_normal, dtype=float)
        axis = np.array([normal[1], -normal[0]])  # from the node into the branches
        chains = []
        for branch in split.branches:
            pieces, reach = _branch_pieces(roads, branch.road, branch.at_start, max_connector + length, endpoint_tol, max_angle_deg)
            oriented = []
            for road, entry, offset in pieces:
                coords = np.asarray(road["coords"], dtype=float)
                piece = coords if entry else coords[::-1]
                arc = offset + np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(piece[:, :2], axis=0), axis=1))])
                oriented.append((road, entry, offset, piece, arc))
            chain_arc = np.concatenate([o[4] for o in oriented])
            chain_xy = np.vstack([o[3][:, :2] for o in oriented])
            chain_z = np.concatenate([o[3][:, 2] for o in oriented])
            probe = np.array([np.interp(BRANCH_PROBE, chain_arc, chain_xy[:, 0]), np.interp(BRANCH_PROBE, chain_arc, chain_xy[:, 1])]) - node
            straightness = float(probe @ axis) / max(float(np.linalg.norm(probe)), 1e-9)
            chains.append({"branch": branch, "reach": reach, "pieces": oriented, "arc": chain_arc, "xy": chain_xy,
                           "z": chain_z, "slot": float(branch.slot_offset @ normal), "straightness": straightness})

        main = max(chains, key=lambda c: (round(c["straightness"], 3), c["branch"].slot_width))
        reference = _Reference(main["arc"], main["xy"])
        for chain in chains:
            if chain is main:
                chain["connector"] = 0.0
                continue
            lateral = np.einsum("ij,ij->i", chain["xy"] - reference.position(chain["arc"]), reference.normal(chain["arc"]))
            chain["connector"] = _connector_length(chain["arc"], lateral, chain["slot"], max_connector)
        side = [c["connector"] for c in chains if c is not main]
        stem_length = min(side) if side else 0.0

        def stem_z(arc: np.ndarray) -> np.ndarray:
            """Height of the main deck along the main axis - the cross-section of the stem is one surface."""
            return np.interp(arc, main["arc"], main["z"])

        stations = np.linspace(0.0, stem_length, max(2, int(np.ceil(stem_length)) + 1)) if stem_length > 0.0 else np.zeros(0)
        stem_path = [(float(x), float(y), float(z)) for (x, y), z in zip(reference.position(stations), stem_z(stations))]

        for chain in chains:
            branch, connector = chain["branch"], chain["connector"]
            blend = max(0.0, min(length, chain["reach"] - connector))
            branch.road["lane_split_branch"] = {
                "end": "start" if branch.at_start else "end", "slot_width": branch.slot_width, "hold": connector,
                "length": blend, "stem_length": stem_length, "stem_path": stem_path, "node": split.node,
                "left_normal": split.left_normal, "slot_offset": (float(branch.slot_offset[0]), float(branch.slot_offset[1])),
                "trunk_width": split.trunk_width,
            }
            slot = chain["slot"]
            profile = None if chain is main else _SideProfile(chain, main["xy"], slot, connector, blend, smoothing)
            if profile is not None:
                # where the branch has moved `leave_gap` out of its slot, it leaves the main deck (the cut on a bridge)
                probe = np.arange(connector, float(chain["arc"][-1]), 0.5)
                _, raw = path_frame(main["xy"], np.column_stack([np.interp(probe, chain["arc"], chain["xy"][:, k]) for k in (0, 1)]))
                away = np.abs(profile.lateral(probe, raw) - slot) >= leave_gap
                chain["leaves_at"] = float(probe[np.argmax(away)]) if away.any() else float(chain["arc"][-1])
                branch.road["lane_split_branch"]["leaves_at"] = chain["leaves_at"]
            zone_end = connector + blend + smoothing_end
            for road, entry, offset, piece, _ in chain["pieces"]:
                piece = _densify(piece, offset, max(connector, stem_length) + length + smoothing_end, step)
                arc = offset + np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(piece[:, :2], axis=0), axis=1))])
                if chain is main:
                    # the slot offset of the main axis at the node fades out along its own OSM course
                    fade = 1.0 - (_smoothstep(arc / blend) if blend > 0.0 else np.ones(len(arc)))
                    shifted = piece[:, :2] + (fade * slot)[:, None] * reference.normal(arc)
                else:
                    shifted = piece[:, :2].copy()
                    zone = arc < zone_end
                    along, lateral = path_frame(main["xy"], piece[zone, :2])
                    target = profile.lateral(arc[zone], lateral)
                    shifted[zone] = reference.position(along) + target[:, None] * reference.normal(along)
                piece = piece.copy()
                piece[:, :2] = shifted
                if chain is not main and not _is_bridge(road):
                    # on the ground the branch takes the main road's height on the stem and returns to its own
                    own_height = _smoothstep((arc - stem_length) / length) if length > 0.0 else (arc > stem_length).astype(float)
                    along, _ = path_frame(main["xy"], piece[:, :2])
                    piece[:, 2] = stem_z(along) * (1.0 - own_height) + piece[:, 2] * own_height
                result = piece if entry else piece[::-1]
                road["coords"] = [tuple(float(v) for v in point) for point in result]
            if chain is not main and _is_bridge(branch.road):
                _level_side_bridge(roads, chain, main, stem_z, endpoint_tol, max_angle_deg)


def path_frame(path_xy, points) -> Tuple[np.ndarray, np.ndarray]:
    """(along, lateral) of `points` relative to the polyline `path_xy`: arc length of the nearest point and signed
    distance (positive = left of the path's direction); before its start and beyond its end the path goes on straight,
    so `along` is negative there or larger than its length."""
    path = np.asarray(path_xy, dtype=float)[:, :2]
    points = np.asarray(points, dtype=float)[:, :2]
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-9
    starts, seg, seg_len = path[:-1][keep], seg[keep], seg_len[keep]
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
    unit = seg / seg_len[:, None]
    rel = points[:, None, :] - starts[None, :, :]  # (points, segments, 2)
    t = np.einsum("psk,sk->ps", rel, unit)
    first, last = np.zeros(len(seg), dtype=bool), np.zeros(len(seg), dtype=bool)
    first[0], last[-1] = True, True
    low = np.where(first[None, :], -np.inf, 0.0)
    high = np.where(last[None, :], np.inf, seg_len[None, :])
    clamped = np.clip(t, low, high)
    nearest = starts[None, :, :] + clamped[..., None] * unit[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    best = np.argmin(distance, axis=1)
    rows = np.arange(len(points))
    along = cum[best] + clamped[rows, best]
    d = unit[best]
    offset = points - nearest[rows, best]
    lateral = d[:, 0] * offset[:, 1] - d[:, 1] * offset[:, 0]
    return along, lateral


def stem_marking_masks(node_lists, stems, tol: float = 0.05) -> Dict[int, Dict]:
    """
    Markings on the stem of a lane split (the trunk goes on as one piece for the hold length, see
    shift_branches_into_slots()): where two branches lie side by side on it, the boundary between them is the trunk's
    lane divider - drawn as ONE block marking (by the outer branch, the one further from the stem's middle; on a tie the
    left one) instead of two edge lines. From the end of the stem on, each branch has its edge lines again, and they
    move apart with the carriageways.

    `node_lists`: DecalRoad nodes [x, y, z, width] per road, `stems`: [{"stem_path": [(x, y, z)] along the main axis,
    "stem_length", "trunk_width"}]. Returns {road index: {"edge_keep": {+1/-1: mask per node}, "blocks": [(mask, sign, 0.0)],
    "siblings": indices of the other roads on the same stem}} for every road that reaches onto a stem.
    """
    result: Dict[int, Dict] = {}
    for stem in stems:
        path = np.asarray(stem["stem_path"], dtype=float)
        hold, half_width = float(stem["stem_length"]), float(stem["trunk_width"]) / 2.0
        if hold <= 0.0 or len(path) < 2:
            continue
        members, spans, frames = [], {}, {}
        for index, nodes in enumerate(node_lists):
            arr = np.asarray(nodes, dtype=float)
            if len(arr) < 2:
                continue
            along, lateral = path_frame(path, arr[:, :2])
            on_stem = (along >= -tol) & (along < hold) & (np.abs(lateral) <= half_width + tol)
            if not on_stem.any():
                continue
            members.append(index)
            frames[index] = (arr, along, lateral, on_stem)
            if along[on_stem].max() > tol:  # a branch running on the stem (not the trunk that only touches the node)
                centre, half = float(np.median(lateral[on_stem])), float(np.median(arr[on_stem, 3])) / 2.0
                spans[index] = (centre - half, centre + half)
        masks = {}
        for index, (arr, along, lateral, on_stem) in frames.items():
            if index not in spans:
                continue
            direction = np.gradient(arr[:, :2], axis=0)
            direction /= np.maximum(np.linalg.norm(direction, axis=1), 1e-9)[:, None]
            own_left = np.column_stack([-direction[:, 1], direction[:, 0]])
            centre = abs(0.5 * (spans[index][0] + spans[index][1]))
            owner = centre > half_width / 2.0 + tol or (abs(centre - half_width / 2.0) <= tol and sum(spans[index]) > 0.0)
            entry = {"edge_keep": {}, "blocks": []}
            for sign in (1.0, -1.0):
                edge = path_frame(path, arr[:, :2] + own_left * (sign * arr[:, 3] / 2.0)[:, None])[1]
                beside = np.zeros(len(arr), dtype=bool)
                for other, (low, high) in spans.items():
                    if other != index:
                        beside |= (edge >= low - 0.1) & (edge <= high + 0.1)
                interior = on_stem & beside
                if not interior.any():
                    continue
                entry["edge_keep"][sign] = ~interior
                if owner:
                    entry["blocks"].append((interior, sign, 0.0))
            masks[index] = entry
        for index, entry in masks.items():
            existing = result.setdefault(index, {"edge_keep": {}, "blocks": [], "siblings": set()})
            for sign, keep in entry["edge_keep"].items():
                existing["edge_keep"][sign] = existing["edge_keep"].get(sign, np.ones(len(keep), dtype=bool)) & keep
            existing["blocks"] += entry["blocks"]
            existing["siblings"] |= set(members) - {index}
    return result
