"""
Kerb lines of the sidewalks: the carriageway edge of a road on each sidewalk side, cut where another road's carriageway
(widened by `clearance`) crosses it.

With `clearance` = kerb + sidewalk width, a joining street leaves a gap of its own width plus both of its sidewalks, and
its own sidewalk ends at the outer edge of the through road's sidewalk - the two bands never overlap. Only `blocking`
roads cut: footways, paths and tracks run through sidewalks without interrupting them.
"""

from typing import Dict, List, Sequence

import numpy as np

from ..geometry.guardrails import _left_normals
from ..geometry.polyline import arc_lengths


def plan_sidewalk_runs(
    roads: Sequence[Sequence[Sequence[float]]],
    sides: Sequence[Dict[str, str]],
    blocking: Sequence[bool],
    clearance: float,
    min_length: float,
) -> List[Dict]:
    """
    Args:
        roads: DecalRoad nodes [[x, y, z, width], ...] of all surface roads
        sides: per road {side: surface type} ("left"/"right" relative to the node order), empty without sidewalk
        blocking: per road, whether its carriageway interrupts the sidewalks of other roads
        clearance: added to the half width of a blocking road for the cut, in meters
        min_length: shorter pieces are dropped, in meters

    Returns:
        [{"road_index", "side", "surface", "points"}] - points (M, 3) along the carriageway edge at road height, ordered
        so that the sidewalk lies on the LEFT of the run direction
    """
    from shapely import STRtree, union_all
    from shapely.geometry import LineString, Point

    from ..geometry.road_markings import road_surface_polygon

    arrays = [np.asarray(r, dtype=float) for r in roads]
    blockers = [i for i, a in enumerate(arrays) if blocking[i] and len(a) >= 2]
    surfaces = [road_surface_polygon(arrays[i], clearance) for i in blockers]
    tree = STRtree(surfaces) if surfaces else None

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
            cutters = [] if tree is None else [surfaces[k] for k in tree.query(line) if blockers[k] != index]
            pieces = line.difference(union_all(cutters)) if cutters else line
            cum = arc_lengths(edge)
            for piece in getattr(pieces, "geoms", [pieces]):
                if piece.is_empty or piece.length < min_length:
                    continue
                points = np.asarray(piece.coords, dtype=float)[:, :2]
                s = np.array([line.project(Point(p)) for p in points])
                points = np.column_stack([points, np.interp(s, cum, z)])
                runs.append({"road_index": index, "side": side, "surface": surface, "points": points if sign > 0 else points[::-1]})
    return runs
