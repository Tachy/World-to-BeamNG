"""
Kerb lines of the sidewalks: the carriageway edge of a road on each sidewalk side, cut where another road's carriageway
(widened by `clearance`) crosses it.

With `clearance` = kerb + sidewalk width, a joining street leaves a gap of its own width plus both of its sidewalks, and
its own sidewalk ends at the outer edge of the through road's sidewalk - the two bands never overlap. Only `blocking`
roads cut: footways, paths and tracks run through sidewalks without interrupting them. A straight-through continuation
(the same street split at a node, e.g. where a footway joins) never cuts: its widened carriageway would otherwise trim
the inner side of every kinked joint.

At rounded junction corners (junctions/corners.py) the kerb line of a corner arm ends at the arc's tangent point instead
of being cut by the other arm; where both arms of a corner have a sidewalk on the corner side, their runs are joined
through the arc into one run.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..geometry.guardrails import _left_normals
from ..geometry.polyline import arc_lengths


def plan_sidewalk_runs(
    roads: Sequence[Sequence[Sequence[float]]],
    sides: Sequence[Dict[str, str]],
    blocking: Sequence[bool],
    clearance: float,
    min_length: float,
    endpoint_tol: float,
    max_angle_deg: float,
    corners: Sequence[Dict] = (),
    road_ids: Optional[Sequence] = None,
) -> List[Dict]:
    """
    Args:
        roads: DecalRoad nodes [[x, y, z, width], ...] of all surface roads
        sides: per road {side: surface type} ("left"/"right" relative to the node order), empty without sidewalk
        blocking: per road, whether its carriageway interrupts the sidewalks of other roads
        clearance: added to the half width of a blocking road for the cut, in meters
        min_length: shorter pieces are dropped, in meters
        endpoint_tol, max_angle_deg: which road ends continue straight into each other (see find_continuations())
        corners: junction corner dicts of find_junction_corners() - the kerb lines of their arms end at the tangent points
        road_ids: road id per entry of `roads`, to match the corner arms

    Returns:
        [{"road_index", "side", "surface", "points"}] - points (M, 3) along the carriageway edge at road height, ordered
        so that the sidewalk lies on the LEFT of the run direction
    """
    from shapely import STRtree, union_all
    from shapely.geometry import LineString, Point
    from shapely.ops import substring

    from ..geometry.road_markings import road_surface_polygon
    from ..geometry.road_width_transitions import find_continuations

    arrays = [np.asarray(r, dtype=float) for r in roads]
    partners = {i: {i} for i in range(len(arrays))}
    for (a, _), (b, _) in find_continuations([r.tolist() for r in arrays], endpoint_tol, max_angle_deg):
        partners[a].add(b)
        partners[b].add(a)
    blockers = [i for i, a in enumerate(arrays) if blocking[i] and len(a) >= 2]
    surfaces = [road_surface_polygon(arrays[i], clearance) for i in blockers]
    tree = STRtree(surfaces) if surfaces else None

    index_of = {rid: i for i, rid in enumerate(road_ids or [])}
    trims, exempt = {}, {}  # (road index, side) -> {end: distance}, (road index, side) -> {indices that do not cut}
    usable = []
    for corner in corners:
        ia, ib = (index_of.get(arm["road_id"]) for arm in corner["arms"])
        if ia is None or ib is None:
            continue
        usable.append((corner, ia, ib))
        for arm, own, other in ((corner["arms"][0], ia, ib), (corner["arms"][1], ib, ia)):
            trims.setdefault((own, arm["side"]), {})[arm["end"]] = arm["trim"]
            exempt.setdefault((own, arm["side"]), set()).add(other)

    runs = []
    for index, road_sides in enumerate(sides):
        nodes = arrays[index]
        if not road_sides or len(nodes) < 2:
            continue
        xy, z, half = nodes[:, :2], nodes[:, 2], nodes[:, 3] / 2.0
        normals = _left_normals(xy)
        for side, surface in road_sides.items():
            sign = 1.0 if side == "left" else -1.0
            edge = xy + sign * normals * half[:, None]
            line = LineString(edge)
            if line.length < min_length:
                continue
            trim = trims.get((index, side), {})
            lo, hi = trim.get("start", 0.0), line.length - trim.get("end", 0.0)
            if hi - lo < min_length:
                continue
            kept = substring(line, lo, hi) if trim else line
            skip = partners[index] | exempt.get((index, side), set())
            cutters = [] if tree is None else [surfaces[k] for k in tree.query(kept) if blockers[k] not in skip]
            pieces = kept.difference(union_all(cutters)) if cutters else kept
            cum = arc_lengths(edge)
            for piece in getattr(pieces, "geoms", [pieces]):
                if piece.is_empty or piece.length < min_length:
                    continue
                points = np.asarray(piece.coords, dtype=float)[:, :2]
                s = np.array([line.project(Point(p)) for p in points])
                points = np.column_stack([points, np.interp(s, cum, z)])
                runs.append({"road_index": index, "side": side, "surface": surface, "points": points if sign > 0 else points[::-1]})
    return _join_through_corners(runs, usable)


def _corner_link(runs: List[Dict], corner: Dict, ia: int, ib: int, tol: float):
    """(index of the run arriving at the corner, arc path, index of the run leaving it) or None - only runs of the two
    corner arms on their corner-facing sides, with an end at the respective tangent point."""
    arm_a, arm_b = corner["arms"]
    arc = np.asarray(corner["arc"], dtype=float)

    def find(index, side, point, at_end):
        for k, run in enumerate(runs):
            if run["road_index"] == index and run["side"] == side:
                p = run["points"][-1] if at_end else run["points"][0]
                if np.linalg.norm(p[:2] - point[:2]) <= tol:
                    return k
        return None

    for (ia_, sa, ta), (ib_, sb, tb), path in (
        ((ia, arm_a["side"], arc[0]), (ib, arm_b["side"], arc[-1]), arc),
        ((ib, arm_b["side"], arc[-1]), (ia, arm_a["side"], arc[0]), arc[::-1]),
    ):
        incoming, outgoing = find(ia_, sa, ta, True), find(ib_, sb, tb, False)
        if incoming is not None and outgoing is not None:
            return incoming, path, outgoing
    return None


def _join_through_corners(runs: List[Dict], usable, tol: float = 0.5) -> List[Dict]:
    """
    Joins runs through the arcs of the corners where both arms have a sidewalk on the corner side. All links are found
    on the unjoined runs first and the chains are built afterwards, so the result does not depend on the corner order
    (a block with sidewalks all around becomes one closed run).
    """
    links = {}  # incoming run index -> (arc path, outgoing run index)
    for corner, ia, ib in usable:
        link = _corner_link(runs, corner, ia, ib, tol)
        if link is not None and link[0] not in links and link[2] not in {out for _, out in links.values()}:
            links[link[0]] = (link[1], link[2])
    incoming_of = {out: inc for inc, (_, out) in links.items()}

    def chain_from(start: int, seen: set) -> Dict:
        run = dict(runs[start])
        points, k = [run["points"]], start
        seen.add(start)
        while k in links:
            path, nxt = links[k]
            if nxt in seen:  # closed ring: end on the arc back at the start
                points[-1] = points[-1][:-1]
                points.append(path)
                break
            points[-1] = points[-1][:-1]
            points.append(path)
            points.append(runs[nxt]["points"][1:])
            seen.add(nxt)
            k = nxt
        run["points"] = np.vstack(points)
        return run

    result, seen = [], set()
    for k in range(len(runs)):  # chains start at runs nobody leads into
        if k not in seen and k not in incoming_of:
            result.append(chain_from(k, seen))
    for k in range(len(runs)):  # what is left are rings
        if k not in seen:
            result.append(chain_from(k, seen))
    return result
