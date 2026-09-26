"""
Road markings as separate, narrow DecalRoads above the carriageway - the way BeamNG's own levels do it
(west_coast_usa: ~3100 `line_white` and ~200 `line_dashed_short` DecalRoads with 0.15-0.2 m width). White
edge lines on the left and right, dashed lane dividers at the lane boundaries; on two-way roads with three or more
lanes a solid double line separates the directions. The lines follow the node width
of the carriageway (including the smooth width transitions from road_width_transitions.py). For background and rules
see docs/OSM_ROAD_ANALYSIS.md.
"""

from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import numpy as np

EDGE = "edge"
DIVIDER = "divider"
CENTER = "center"  # one of the two solid lines between the directions
MAX_MITRE_FACTOR = 2.0  # sharp kinks: offset at most twice as far as requested
BOUNDARY_EPS = 0.01  # shrink obstacle areas by 1 cm, see junction_obstacles()


@dataclass(frozen=True)
class MarkingLayout:
    lanes: int
    forward: Optional[int] = None  # lanes in digitization direction if a double centre line separates the directions


def parse_lanes(value) -> Optional[int]:
    """OSM `lanes` as a positive integer, otherwise None (missing, "2;3", "", "0", ...)."""
    try:
        lanes = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return lanes if lanes >= 1 else None


def marking_layout(
    tags: dict,
    width: float,
    internal_name: str,
    marked_highways: Collection[str],
    marked_surface: str,
    min_two_lane_width: float,
    double_center_min_lanes: Optional[int] = None,
    force_double_center: bool = False,
) -> Optional[MarkingLayout]:
    """
    Marking layout of a road, or None (no marking): only road types from `marked_highways` with the surface
    `marked_surface` (asphalt) and without `lane_markings=no`. Lanes come from `lanes`; if the tag is missing, a
    ramp (*_link) or a road narrower than min_two_lane_width is single-lane, everything else is two-lane.

    double_center_min_lanes: two-way roads (not oneway) with at least this many lanes get a solid double line between
    the directions; `forward` (lanes in digitization direction) from lanes:forward / lanes:backward, otherwise the
    larger half. force_double_center: two-way roads with two lanes get it as well (tunnels and galleries).
    """
    tags = tags or {}
    highway = str(tags.get("highway", ""))
    if tags.get("lane_markings") == "no" or highway not in marked_highways or internal_name != marked_surface:
        return None
    lanes = lane_count(tags, width, min_two_lane_width)
    oneway = str(tags.get("oneway", "")).lower() in ("yes", "true", "1", "-1")
    wanted = force_double_center and lanes >= 2
    if oneway or not (wanted or (double_center_min_lanes is not None and lanes >= double_center_min_lanes)):
        return MarkingLayout(lanes=lanes)
    forward, backward = parse_lanes(tags.get("lanes:forward")), parse_lanes(tags.get("lanes:backward"))
    if forward is None or forward >= lanes:
        forward = lanes - backward if backward is not None and backward < lanes else (lanes + 1) // 2
    return MarkingLayout(lanes=lanes, forward=forward)


def lane_count(tags: dict, width: float, min_two_lane_width: float) -> int:
    """Lanes from `lanes`; without the tag a ramp (*_link) or a road narrower than min_two_lane_width has one lane,
    everything else two."""
    tags = tags or {}
    lanes = parse_lanes(tags.get("lanes"))
    if lanes is None:
        lanes = 1 if str(tags.get("highway", "")).endswith("_link") or width < min_two_lane_width else 2
    return lanes


def line_offsets(
    widths: np.ndarray,
    lanes: int,
    edge_inset: float,
    forward: Optional[int] = None,
    center_gap: float = 0.0,
    line_width: float = 0.0,
    boundary_shift: Optional[np.ndarray] = None,
) -> List[Tuple[str, np.ndarray]]:
    """(kind, lateral offset per node), positive = left of the travel direction. Edge lines at +-(width/2 -
    edge_inset), dividers at the lanes-1 lane boundaries. With `forward` (right-hand traffic: the forward lanes lie on
    the right) the boundary after `forward` lanes from the right edge becomes two CENTER lines, `center_gap` apart.
    `boundary_shift` (per node, see boundary_shifts()) moves the boundary between the directions - the CENTER lines, or
    without `forward` the middle divider - so that it runs onto the double line of a wider neighbour."""
    widths = np.asarray(widths, dtype=float)
    half = widths / 2.0
    shift = np.zeros(len(widths)) if boundary_shift is None else np.asarray(boundary_shift, dtype=float)
    boundary_lane = forward if forward is not None else lanes // 2
    lines = [(EDGE, half - edge_inset), (EDGE, -(half - edge_inset))]
    for k in range(1, lanes):
        boundary = -half + k * widths / lanes
        if k == boundary_lane:
            boundary = boundary + shift
        if k == forward:
            gap = (center_gap + line_width) / 2.0
            lines += [(CENTER, boundary - gap), (CENTER, boundary + gap)]
        else:
            lines.append((DIVIDER, boundary))
    return lines


def direction_boundary_offset(width: float, lanes: int, forward: int) -> float:
    """Lateral offset (positive = left) of the boundary between the directions: after `forward` lanes from the right edge."""
    return -width / 2.0 + forward * width / lanes


def boundary_shifts(roads, layouts, own_widths, pairs, fixed=None) -> Dict[int, np.ndarray]:
    """
    Lateral shift of the direction boundary per node for roads that continue straight into a road with MORE lanes
    and a double centre line (`pairs` from find_continuations(), `layouts` per road, `own_widths` = the roads' own,
    unblended widths). Without it the single centre line of a 2-lane road ends in the middle of the carriageway while
    the double line of the 3-lane road starts a third of the width to the side. At the joint the boundary lies exactly
    where the wider road's double line begins; it follows the width blend (0 where the road has its own width again)
    over the first half of the road, mirrored if the two roads are digitized in opposite directions. Joints with a
    structure (`fixed` True: bridge, tunnel, gallery) are left to structure_boundary_shifts().
    """
    result: Dict[int, np.ndarray] = {}
    for (ia, ea), (ib, eb) in pairs:
        if fixed is not None and (fixed[ia] or fixed[ib]):
            continue
        la, lb = layouts[ia], layouts[ib]
        if la is None or lb is None or la.lanes == lb.lanes:
            continue
        (big, big_end), (small, small_end) = ((ia, ea), (ib, eb)) if la.lanes > lb.lanes else ((ib, eb), (ia, ea))
        big_layout, small_layout = layouts[big], layouts[small]
        if big_layout.forward is None or small_layout.lanes < 2:
            continue
        joint_width = float(roads[big][0 if big_end == "start" else -1][3])
        target = direction_boundary_offset(joint_width, big_layout.lanes, big_layout.forward)
        if big_end == small_end:  # opposite digitization: left and right swap
            target = -target
        own = float(own_widths[small])
        if abs(joint_width - own) < 1e-9:
            continue
        nodes = np.asarray(roads[small], dtype=float)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(nodes[:, :2], axis=0), axis=1))])
        from_joint = arc if small_end == "start" else arc[-1] - arc
        factor = np.clip((nodes[:, 3] - own) / (joint_width - own), 0.0, 1.0)
        shift = target * factor * (from_joint <= arc[-1] / 2.0 + 1e-9)
        result[small] = result.get(small, 0.0) + shift
    return result


def _boundary_line_offset(width: float, layout: "MarkingLayout") -> float:
    """Offset of the line between the directions of a layout: the double line after `forward` lanes, otherwise the
    middle divider."""
    return direction_boundary_offset(width, layout.lanes, layout.forward if layout.forward is not None else layout.lanes // 2)


def _approach_pieces(roads, partner, fixed, road: int, road_end: str, span: float):
    """Road pieces up to `span` meters before a structure, walking back along straight continuations from `road` (which
    touches the structure with end `road_end`): (piece index, end facing the structure, offset = distance of that end
    from the structure, arc length per node from that end + offset)."""
    offset, entry, current, seen = 0.0, road_end, road, set()
    while current is not None and current not in seen and offset < span:
        seen.add(current)
        nodes = np.asarray(roads[current], dtype=float)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(nodes[:, :2], axis=0), axis=1))])
        yield current, entry, offset, offset + (arc if entry == "start" else arc[-1] - arc)
        offset += float(arc[-1])
        nxt = partner.get((current, "end" if entry == "start" else "start"))
        if nxt is None or fixed[nxt[0]]:
            return
        current, entry = nxt


def _structure_joints(pairs, fixed):
    """(structure, structure end, road, road end) for every joint of a structure with a normal road."""
    for (ia, ea), (ib, eb) in pairs:
        if fixed[ia] != fixed[ib]:
            yield (ia, ea, ib, eb) if fixed[ia] else (ib, eb, ia, ea)


def structure_boundary_shifts(roads, layouts, fixed, pairs, span: float, done_at: float) -> Dict[int, np.ndarray]:
    """
    Lateral shift of the direction-boundary line per node for roads that end at a structure (`fixed`: bridge, tunnel,
    gallery; `pairs` from find_continuations()). The structure keeps its lines; the road's line moves onto the
    structure's boundary and has done so `done_at` meters before the structure (spline from `span` meters before it),
    while the carriageway width itself still changes up to the structure. Reaches across the road pieces the road is
    split into at junctions; mirrored if the two are digitized in opposite directions. Roads without a line between
    directions (fewer than two lanes) are left alone.
    """
    partner = {}
    for a, b in pairs:
        partner[a], partner[b] = b, a
    result: Dict[int, np.ndarray] = {}
    for struct, struct_end, road, road_end in _structure_joints(pairs, fixed):
        struct_layout = layouts[struct]
        if struct_layout is None or struct_layout.lanes < 2 or layouts[road] is None or layouts[road].lanes < 2:
            continue
        struct_width = float(roads[struct][0 if struct_end == "start" else -1][3])
        target = _boundary_line_offset(struct_width, struct_layout)
        if struct_end == road_end:  # opposite digitization: left and right swap
            target = -target
        for current, _, _, distance in _approach_pieces(roads, partner, fixed, road, road_end, span):
            layout = layouts[current]
            if layout is None:
                continue
            f = np.clip((span - distance) / (span - done_at), 0.0, 1.0)
            f = f * f * (3.0 - 2.0 * f)  # cubic Hermite spline, 1 within done_at of the structure
            own = np.array([_boundary_line_offset(w, layout) for w in np.asarray(roads[current], dtype=float)[:, 3]])
            result[current] = result.get(current, 0.0) + f * (target - own)
    return result


def _divider_count(layout) -> int:
    """Dashed dividers of a layout (all lane boundaries except the one between the directions)."""
    return max(layout.lanes - 2, 0) if layout.lanes >= 2 and (layout.forward is not None or layout.lanes == 2) else max(layout.lanes - 1, 0)


def divider_masks(roads, layouts, fixed, pairs, done_at: float) -> Dict[int, np.ndarray]:
    """
    Nodes (per road piece) that may carry the dashed lane dividers: the dividers of a road end `done_at` meters before
    a structure that has none of its own (a lane is dropped there and the lines are aligned by then, see
    structure_boundary_shifts()). Only roads with dividers next to a structure without any get a mask.
    """
    partner = {}
    for a, b in pairs:
        partner[a], partner[b] = b, a
    result: Dict[int, np.ndarray] = {}
    for struct, _, road, road_end in _structure_joints(pairs, fixed):
        struct_layout, road_layout = layouts[struct], layouts[road]
        if struct_layout is None or road_layout is None or _divider_count(struct_layout) > 0 or _divider_count(road_layout) == 0:
            continue
        for current, _, _, distance in _approach_pieces(roads, partner, fixed, road, road_end, done_at):
            keep = distance >= done_at - 1e-9
            result[current] = result[current] & keep if current in result else keep
    return result


def offset_polyline(
    xy: np.ndarray, offsets: np.ndarray, start_normal: Optional[np.ndarray] = None, end_normal: Optional[np.ndarray] = None
) -> np.ndarray:
    """Polyline with an offset per node (positive = left), mitered at kinks. Zero-length segments (duplicate nodes)
    take the direction of the neighboring segment. `start_normal`/`end_normal` (unit vectors, left) replace the
    miter direction at the first/last node - at the joint of two roads (see joint_normals())."""
    xy = np.asarray(xy, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    segments = np.diff(xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-9
    if not valid.any():
        return xy.copy()
    normals = np.zeros_like(segments)
    normals[valid] = np.column_stack([-segments[valid, 1], segments[valid, 0]]) / lengths[valid, None]
    last = normals[int(np.argmax(valid))]
    for i in range(len(normals)):
        if valid[i]:
            last = normals[i]
        else:
            normals[i] = last

    result = np.empty_like(xy)
    count = len(xy)
    for i in range(count):
        before, after = normals[max(i - 1, 0)], normals[min(i, count - 2)]
        if i == 0 and start_normal is not None:
            miter = np.asarray(start_normal, dtype=float)
        elif i == count - 1 and end_normal is not None:
            miter = np.asarray(end_normal, dtype=float)
        else:
            miter = before + after
            norm = float(np.linalg.norm(miter))
            miter = after if norm < 1e-9 else miter / norm
        scale = 1.0 / max(float(np.dot(miter, after)), 1.0 / MAX_MITRE_FACTOR)
        result[i] = xy[i] + miter * offsets[i] * scale
    return result


def forward_indices(offset_xy: np.ndarray, center_xy: np.ndarray) -> np.ndarray:
    """Indices of the line nodes that advance in the travel direction. In tight hairpins the inner line would
    otherwise run backwards (offset larger than the curve radius) - those nodes are dropped."""
    center_xy = np.asarray(center_xy, dtype=float)
    count = len(center_xy)
    kept = [0]
    for j in range(1, count):
        tangent = center_xy[min(j + 1, count - 1)] - center_xy[max(j - 1, 0)]
        if float(np.dot(offset_xy[j] - offset_xy[kept[-1]], tangent)) > 1e-9:
            kept.append(j)
    return np.array(kept, dtype=int)


def build_marking_lines(
    nodes: Sequence[Sequence[float]],
    layout: MarkingLayout,
    edge_inset: float,
    start_normal: Optional[np.ndarray] = None,
    end_normal: Optional[np.ndarray] = None,
    center_gap: float = 0.0,
    line_width: float = 0.0,
    boundary_shift: Optional[np.ndarray] = None,
    divider_keep: Optional[np.ndarray] = None,
) -> List[Tuple[str, np.ndarray]]:
    """(kind, (N, 3) line) for all marking lines of a road from its DecalRoad nodes [x, y, z, width];
    z per line node from the corresponding carriageway node (BeamNG projects the line onto the terrain anyway).
    `start_normal`/`end_normal`: shared joint normal with the straight continuation (joint_normals()), so that the
    lines of both roads connect exactly at a kinked joint."""
    arr = np.asarray(nodes, dtype=float)
    center_xy = arr[:, :2]
    lines = []
    for kind, offsets in line_offsets(
        arr[:, 3], layout.lanes, edge_inset, layout.forward, center_gap, line_width, boundary_shift
    ):
        offset_xy = offset_polyline(center_xy, offsets, start_normal, end_normal)
        kept = forward_indices(offset_xy, center_xy)
        if kind == DIVIDER and divider_keep is not None:
            kept = kept[np.asarray(divider_keep, dtype=bool)[kept]]  # dashed dividers end where the mask does
        if len(kept) >= 2:
            lines.append((kind, np.column_stack([offset_xy[kept], arr[kept, 2]])))
    return lines


def joint_normals(roads: Sequence[Sequence[Sequence[float]]], pairs) -> Dict[Tuple[int, str], np.ndarray]:
    """
    Shared left normal per road end at a straight joint (`pairs` from find_continuations()): the
    bisector of both travel directions, each in the travel direction of the road's own side. Without it, each
    road is offset perpendicular to its own last segment, and at a kink of angle t the line ends
    gape apart by about 2 * offset * sin(t/2) (gap on the outside, overlap on the inside).
    """
    from .road_width_transitions import outward_direction

    result: Dict[Tuple[int, str], np.ndarray] = {}
    for (ia, ea), (ib, eb) in pairs:
        da, db = outward_direction(roads[ia], ea), outward_direction(roads[ib], eb)
        if da is None or db is None:
            continue
        travel_a = da if ea == "start" else -da  # travel direction of the road at the joint
        travel_b = db if eb == "start" else -db
        sign = 1.0 if ea != eb else -1.0  # same travel direction (end -> start) or opposing
        joint = travel_a + sign * travel_b
        length = float(np.linalg.norm(joint))
        if length < 1e-9:
            continue
        joint = joint / length
        result[(ia, ea)] = np.array([-joint[1], joint[0]])
        result[(ib, eb)] = sign * np.array([-joint[1], joint[0]])
    return result


def clip_line(line: np.ndarray, obstacles, min_length: float) -> List[np.ndarray]:
    """Parts of an (N, 3) line outside `obstacles` (shapely area, e.g. the carriageways of joining roads)
    with a length of at least min_length; z linear along the original line."""
    from shapely.geometry import LineString, Point

    shape = LineString(line[:, :2])
    if obstacles is None or obstacles.is_empty:
        pieces = [shape]
    else:
        rest = shape.difference(obstacles)
        pieces = list(getattr(rest, "geoms", [rest]))
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line[:, :2], axis=0), axis=1))])
    result = []
    for piece in pieces:
        if piece.is_empty or piece.geom_type != "LineString" or piece.length < min_length:
            continue
        xy = np.asarray(piece.coords, dtype=float)
        s = np.array([shape.project(Point(p)) for p in xy])
        result.append(np.column_stack([xy, np.interp(s, cum, line[:, 2])]))
    return result


def road_surface_polygon(nodes: Sequence[Sequence[float]], clearance: float):
    """Carriageway area of a DecalRoad (buffer around the centerline with the largest node width, flat ends),
    widened by `clearance`."""
    from shapely.geometry import LineString

    arr = np.asarray(nodes, dtype=float)
    return LineString(arr[:, :2]).buffer(float(arr[:, 3].max()) / 2.0 + clearance, cap_style="flat")




def _side_junction(polygon, main_line, side_xy: np.ndarray, endpoint_tol: float):
    """
    (junction point, side, half-plane on its own side) of a road that joins `main_line` (centerline of the marked
    road) with exactly one end - side +1 left, -1 right. None for roads that cross the centerline or merely
    touch it.
    """
    from shapely.geometry import Point

    start, end = Point(side_xy[0]), Point(side_xy[-1])
    at_start = main_line.distance(start) <= endpoint_tol
    at_end = main_line.distance(end) <= endpoint_tol
    if at_start == at_end:
        return None
    reach = polygon.length
    left = main_line.buffer(reach, single_sided=True)
    right = main_line.buffer(-reach, single_sided=True)
    if polygon.intersection(left).area >= polygon.intersection(right).area:
        return (start if at_start else end), 1, left
    return (start if at_start else end), -1, right


def junction_obstacles(
    index: int,
    polygons: Sequence,
    tree,
    excluded: Collection[int],
    centerlines: Optional[Sequence[np.ndarray]] = None,
    endpoint_tol: float = 0.5,
):
    """
    Union of the carriageway areas that touch area `index` - excluding itself and `excluded`
    (straight-continuation partners, paths without a marking gap). None if none remain. `tree`: shapely.STRtree over
    `polygons`.

    With `centerlines` ((N, 2) per road), the area of a T-junction is restricted to its own side of the marked road's
    centerline: its flat end is perpendicular to itself, not to the main road, and for an oblique
    junction it would otherwise reach across the centerline - the divider and the opposite edge line would get a
    gap. If a road also joins from the opposite side at the same point (crossing, split into two ways in OSM at the
    main road), both areas are kept whole, so that the divider is interrupted in the crossing.

    Shrunk by BOUNDARY_EPS: a side road starts at the shared node on the main road's centerline,
    so its flat edge lies exactly on the main road's divider. Without the shrinking, the divider would get a gap at
    every T-junction.
    """
    from shapely import unary_union
    from shapely.geometry import LineString

    candidates = [j for j in tree.query(polygons[index]) if j != index and j not in excluded]
    if not candidates:
        return None
    if centerlines is None:
        others = [polygons[j] for j in candidates]
    else:
        main_line = LineString(centerlines[index])
        junctions = {
            j: _side_junction(polygons[j], main_line, np.asarray(centerlines[j], dtype=float), endpoint_tol)
            for j in candidates
        }
        others = []
        for j in candidates:
            junction = junctions[j]
            if junction is None:
                others.append(polygons[j])
                continue
            point, side, own_half = junction
            crossing = any(
                other is not None and other[1] != side and other[0].distance(point) <= 2.0 * endpoint_tol
                for k, other in junctions.items()
                if k != j
            )
            others.append(polygons[j] if crossing else polygons[j].intersection(own_half))
    others = [polygon for polygon in others if not polygon.is_empty]
    if not others:
        return None
    return unary_union(others).buffer(-BOUNDARY_EPS)
