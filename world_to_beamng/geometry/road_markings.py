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
) -> Optional[MarkingLayout]:
    """
    Marking layout of a road, or None (no marking): only road types from `marked_highways` with the surface
    `marked_surface` (asphalt) and without `lane_markings=no`. Lanes come from `lanes`; if the tag is missing, a
    ramp (*_link) or a road narrower than min_two_lane_width is single-lane, everything else is two-lane.

    double_center_min_lanes: two-way roads (not oneway) with at least this many lanes get a solid double line between
    the directions; `forward` (lanes in digitization direction) from lanes:forward / lanes:backward, otherwise the
    larger half.
    """
    tags = tags or {}
    highway = str(tags.get("highway", ""))
    if tags.get("lane_markings") == "no" or highway not in marked_highways or internal_name != marked_surface:
        return None
    lanes = lane_count(tags, width, min_two_lane_width)
    oneway = str(tags.get("oneway", "")).lower() in ("yes", "true", "1", "-1")
    if double_center_min_lanes is None or oneway or lanes < double_center_min_lanes:
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
) -> List[Tuple[str, np.ndarray]]:
    """(kind, lateral offset per node), positive = left of the travel direction. Edge lines at +-(width/2 -
    edge_inset), dividers at the lanes-1 lane boundaries. With `forward` (right-hand traffic: the forward lanes lie on
    the right) the boundary after `forward` lanes from the right edge becomes two CENTER lines, `center_gap` apart."""
    widths = np.asarray(widths, dtype=float)
    half = widths / 2.0
    lines = [(EDGE, half - edge_inset), (EDGE, -(half - edge_inset))]
    for k in range(1, lanes):
        boundary = -half + k * widths / lanes
        if k == forward:
            shift = (center_gap + line_width) / 2.0
            lines += [(CENTER, boundary - shift), (CENTER, boundary + shift)]
        else:
            lines.append((DIVIDER, boundary))
    return lines


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
) -> List[Tuple[str, np.ndarray]]:
    """(kind, (N, 3) line) for all marking lines of a road from its DecalRoad nodes [x, y, z, width];
    z per line node from the corresponding carriageway node (BeamNG projects the line onto the terrain anyway).
    `start_normal`/`end_normal`: shared joint normal with the straight continuation (joint_normals()), so that the
    lines of both roads connect exactly at a kinked joint."""
    arr = np.asarray(nodes, dtype=float)
    center_xy = arr[:, :2]
    lines = []
    for kind, offsets in line_offsets(arr[:, 3], layout.lanes, edge_inset, layout.forward, center_gap, line_width):
        offset_xy = offset_polyline(center_xy, offsets, start_normal, end_normal)
        kept = forward_indices(offset_xy, center_xy)
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
