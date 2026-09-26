"""
Guard rails along roads with a drop beside them.

Rule: where the finished terrain `probe_offset` (4 m) beside the carriageway edge lies more than `min_drop` (2 m) below
the road, that side gets a guard rail whose traffic face stands `edge_gap` (20 cm) beside the carriageway edge. Each
guarded stretch is extended by `extension` (20 m) in both directions - along straight continuations into the next road
piece, but never past the end of the road chain (a structure or a junction without continuation starts there). Where
another road joins, the rail is interrupted.

The rail itself consists of BeamNG's stock guard rail segments (forest items, 3 m each) plus end caps
(place_guardrail_items()).
"""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _chains(roads: Sequence[np.ndarray], eligible: Sequence[bool], endpoint_tol: float, max_angle_deg: float) -> List[List[Tuple[int, bool]]]:
    """Eligible roads joined along straight continuations: [[(road index, reversed), ...], ...] in driving order."""
    from .road_width_transitions import find_continuations

    partner = {}
    for a, b in find_continuations([r.tolist() for r in roads], endpoint_tol, max_angle_deg):
        if eligible[a[0]] and eligible[b[0]]:
            partner[a], partner[b] = b, a

    chains, seen = [], set()

    def walk(index: int, entry: str) -> List[Tuple[int, bool]]:
        chain = []
        while index not in seen:
            seen.add(index)
            reversed_ = entry == "end"
            chain.append((index, reversed_))
            exit_end = "start" if reversed_ else "end"
            nxt = partner.get((index, exit_end))
            if nxt is None:
                break
            index, entry = nxt
        return chain

    # Start at free ends first, rings (every end paired) afterwards
    for index in range(len(roads)):
        if not eligible[index] or index in seen:
            continue
        if (index, "start") not in partner:
            chains.append(walk(index, "start"))
        elif (index, "end") not in partner:
            chains.append(walk(index, "end"))
    for index in range(len(roads)):
        if eligible[index] and index not in seen:
            chains.append(walk(index, "start"))
    return chains


def _left_normals(xy: np.ndarray) -> np.ndarray:
    """Unit normals left of the driving direction per point (averaged at interior points)."""
    d = np.diff(xy, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    t = np.vstack([d[:1], d[:-1] + d[1:], d[-1:]])
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return np.column_stack([-t[:, 1], t[:, 0]])


def _merged_intervals(centers: np.ndarray, reach: float, length: float) -> List[Tuple[float, float]]:
    intervals = []
    for s in np.sort(centers):
        lo, hi = max(0.0, s - reach), min(length, s + reach)
        if intervals and lo <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], hi))
        else:
            intervals.append((lo, hi))
    return intervals


def _cut(points: np.ndarray, cum: np.ndarray, lo: float, hi: float) -> np.ndarray:
    inner = points[(cum > lo) & (cum < hi)]
    at = lambda s: np.array([np.interp(s, cum, points[:, k]) for k in range(points.shape[1])])
    return np.vstack([at(lo), inner, at(hi)])


def plan_guardrail_runs(
    roads: Sequence[Sequence[Sequence[float]]],
    eligible: Sequence[bool],
    height_at: HeightAt,
    probe_offset: float,
    min_drop: float,
    edge_gap: float,
    extension: float,
    junction_clearance: float,
    endpoint_tol: float,
    max_angle_deg: float,
    min_length: float,
) -> List[np.ndarray]:
    """
    Guard rail runs of all eligible roads.

    Args:
        roads: DecalRoad nodes [[x, y, z, width], ...] of all surface roads (ineligible ones only interrupt rails)
        eligible: per road, whether it may get guard rails
        height_at: terrain height of the finished heightmap
        junction_clearance: gap between a joining road's carriageway and the rail end, in meters
        min_length: shorter pieces (after the junction cut) are dropped, in meters

    Returns:
        Runs as (M, 3) arrays of the rail's traffic face line at road level, ordered so that the road lies on the
        RIGHT of the run direction (the rail's back is on the left).
    """
    from shapely import STRtree
    from shapely.geometry import LineString, Point

    from .road_markings import road_surface_polygon

    arrays = [np.asarray(r, dtype=float) for r in roads]
    surfaces = [road_surface_polygon(a, junction_clearance) if len(a) >= 2 else None for a in arrays]
    valid = [i for i, s in enumerate(surfaces) if s is not None]
    tree = STRtree([surfaces[i] for i in valid])

    runs = []
    for chain in _chains(arrays, eligible, endpoint_tol, max_angle_deg):
        parts = [arrays[i][::-1] if rev else arrays[i] for i, rev in chain]
        nodes = np.vstack([parts[0]] + [p[1:] for p in parts[1:]])
        if len(nodes) < 2:
            continue
        xy, z, half = nodes[:, :2], nodes[:, 2], nodes[:, 3] / 2.0
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        normals = _left_normals(xy)
        members = {i for i, _ in chain}

        for side in (1.0, -1.0):  # left, right of the driving direction
            probe = xy + side * normals * (half + probe_offset)[:, None]
            ground = np.asarray(height_at(probe[:, 0], probe[:, 1]), dtype=float)
            guarded = z - ground > min_drop
            if not guarded.any():
                continue
            face = np.column_stack([xy + side * normals * (half + edge_gap)[:, None], z])
            for lo, hi in _merged_intervals(cum[guarded], extension, float(cum[-1])):
                if hi - lo < min_length:
                    continue
                line = LineString(_cut(face, cum, lo, hi))
                blockers = [surfaces[valid[k]] for k in tree.query(line) if valid[k] not in members]
                pieces = line.difference(_union(blockers)) if blockers else line
                for piece in getattr(pieces, "geoms", [pieces]):
                    if piece.is_empty or piece.length < min_length:
                        continue
                    run = np.asarray(piece.coords, dtype=float)[:, :2]
                    # heights from the arc length along the uncut line (independent of how GEOS carries Z)
                    line_xyz = np.asarray(line.coords, dtype=float)
                    line_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line_xyz[:, :2], axis=0), axis=1))])
                    s = np.array([line.project(Point(p)) for p in run])
                    run = np.column_stack([run, np.interp(s, line_cum, line_xyz[:, 2])])
                    # the road must lie on the right: a left rail runs with the road, a right rail against it
                    runs.append(run if side > 0 else run[::-1])
    return runs


def _union(geometries):
    from shapely import union_all

    return union_all(geometries)


def place_guardrail_items(
    runs: Sequence[np.ndarray],
    segment_length: float,
    beam_offset: float,
    segment_item: str,
    start_item: str,
    end_item: str,
) -> List[Dict]:
    """
    Forest items along the runs: floor(length / segment_length) segments, centered on the run (the ends stay at most half
    a segment short), plus an end cap at the start and at the end.

    Model convention of the stock segment (art/shapes/objects/italy_guardrails_basic.dae): x along the rail,
    -segment_length/2..+segment_length/2; y toward the back (the end caps flare to +y), the beam's traffic face at
    y = -beam_offset; z up, origin at ground level. rotationMatrix rows = model axes (see
    forest/vineyard_generator._rotation_matrices()).

    Returns:
        [{"type", "pos", "rotationMatrix", "scale"}, ...]
    """
    items = []
    for run in runs:
        run = np.asarray(run, dtype=float)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(run[:, :2], axis=0), axis=1))])
        count = int(cum[-1] // segment_length)
        if count < 1:
            continue
        start = (cum[-1] - count * segment_length) / 2.0
        at = lambda s: np.array([np.interp(s, cum, run[:, k]) for k in range(3)])

        def item(kind: str, a: np.ndarray, b: np.ndarray, origin: np.ndarray) -> Dict:
            forward = b - a
            forward /= np.linalg.norm(forward)
            up = np.array([0.0, 0.0, 1.0]) - forward[2] * forward
            up /= np.linalg.norm(up)
            back = np.cross(up, forward)  # left of the run = away from the road
            pos = origin + back * beam_offset
            matrix = np.vstack([forward, back, up]).reshape(-1)
            return {"type": kind, "pos": [float(v) for v in pos], "rotationMatrix": [float(v) for v in matrix], "scale": 1.0}

        segments = []
        for k in range(count):
            a, b = at(start + k * segment_length), at(start + (k + 1) * segment_length)
            segments.append(item(segment_item, a, b, (a + b) / 2.0))
        first_a, first_b = at(start), at(start + segment_length)
        last_a, last_b = at(start + (count - 1) * segment_length), at(start + count * segment_length)
        items.append(item(start_item, first_a, first_b, first_a))
        items.extend(segments)
        items.append(item(end_item, last_a, last_b, last_b))
    return items
