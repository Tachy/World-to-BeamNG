"""
Cap slabs on top of the walls: individual stone slabs that project a few centimeters beyond the wall body.

Each slab is its own box strip (top, both long sides, bottom, end faces). The slabs are staggered randomly
in the longitudinal direction (length ±25 %, joint in between) and follow the terrain profile of the wall
crown. At corners they end exactly on the miter and do not bend around the corner; at open ends they project beyond
the end face.
"""

import math
from typing import List, Sequence, Tuple

import numpy as np

from .mesh_parts import MeshBuilder, offset_points, unit_vector

CORNER_DEG = 15.0  # from this change of direction on a bend counts as a corner (slab joint on the miter)
LENGTH_VARIATION = 0.5  # slab length = target value * (0.75 ... 1.25)
MIN_REST_FRACTION = 0.3  # a remainder shorter than this fraction of the target length is added to the last slab
_EPS = 1e-9


def corner_arcs(points: np.ndarray, closed: bool, arc: np.ndarray) -> List[float]:
    """
    Arc lengths of the corners (changes of direction above CORNER_DEG).

    Args:
        points: (N, 2) centerline (without closing point if `closed`)
        arc: cumulative arc length per point (if `closed`, with an entry for the closing point)
    """
    count = len(points)
    ring = np.vstack([points, points[:1]]) if closed else points
    directions = np.diff(ring, axis=0)
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    indices = range(count) if closed else range(1, count - 1)
    limit = math.cos(math.radians(CORNER_DEG))
    return [float(arc[i]) for i in indices if float(np.dot(directions[i - 1], directions[i % len(directions)])) < limit]


def plate_spans(start: float, end: float, plate_length: float, joint: float, rng: np.random.Generator) -> List[Tuple[float, float]]:
    """
    Arc-length intervals of the slabs between `start` and `end`; a joint `joint` lies between two slabs.
    Shorter than one slab: a single small slab; a remainder that is too short is added to the last slab.
    """
    spans: List[Tuple[float, float]] = []
    position = start
    while position < end - _EPS:
        plate_end = min(position + plate_length * (1.0 - LENGTH_VARIATION / 2 + LENGTH_VARIATION * rng.random()), end)
        if end - plate_end < MIN_REST_FRACTION * plate_length:
            plate_end = end
        spans.append((position, plate_end))
        position = plate_end + joint
    return spans


def _run_intervals(corners: Sequence[float], total: float, closed: bool, joint: float) -> List[Tuple[float, float, float, float]]:
    """(from, to, joint at start, joint at end) per straight run between two corners or the wall ends."""
    if not closed:
        bounds = [0.0, *corners, total]
        return [(a, b, 0.0, 0.0) for a, b in zip(bounds[:-1], bounds[1:])]  # slab joint on the miter, no gap
    if not corners:  # ring without a corner: one run all the way around, a joint at the ring closure
        return [(0.0, total, 0.0, joint)]
    wrapped = [*corners, corners[0] + total]
    return [(a, b, 0.0, 0.0) for a, b in zip(wrapped[:-1], wrapped[1:])]


def add_cap(
    builder: MeshBuilder,
    points: np.ndarray,
    top: np.ndarray,
    closed: bool,
    wall_thickness: float,
    cap_thickness: float,
    overhang: float,
    plate_length: float,
    joint: float,
    tile_m: float,
    rng: np.random.Generator,
) -> None:
    """
    Adds the cap slabs to the mesh.

    Args:
        points: (N, 2) densified centerline of the wall (without closing point if `closed`)
        top: (N,) height of the slab top per point
        wall_thickness: thickness of the wall body; the slabs are wider by `overhang` on each side
        tile_m: texture tile size for the UVs (in texture tiles)
    """
    half = wall_thickness / 2.0 + overhang
    if closed:
        ring, ring_top = np.vstack([points, points[:1]]), np.append(top, top[0])
        left, right = offset_points(points, half, closed=True)
        left, right = np.vstack([left, left[:1]]), np.vstack([right, right[:1]])
    else:  # open ends: slabs project beyond the end face
        first = unit_vector(np.append(points[1] - points[0], 0.0))[:2]
        last = unit_vector(np.append(points[-1] - points[-2], 0.0))[:2]
        ring = np.vstack([points[0] - np.array(first) * overhang, points, points[-1] + np.array(last) * overhang])
        ring_top = np.concatenate([[top[0]], top, [top[-1]]])
        left, right = offset_points(ring, half, closed=False)

    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))])
    total = float(arc[-1])
    corner_points = ring[:-1] if closed else ring
    corners = corner_arcs(corner_points, closed, arc)
    runs = _run_intervals(corners, total, closed, joint)

    if closed:  # two laps so that runs across the ring start (corner -> corner) are continuous
        arc = np.concatenate([arc, arc[1:] + total])
        left, right, ring_top = (np.concatenate([a, a[1:]]) for a in (left, right, ring_top))

    for run_start, run_end, gap_start, gap_end in runs:
        for start, end in plate_spans(run_start + gap_start, run_end - gap_end, plate_length, joint, rng):
            _add_plate(builder, arc, left, right, ring_top, start, end, cap_thickness, tile_m)


def _add_plate(
    builder: MeshBuilder,
    arc: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    top: np.ndarray,
    start: float,
    end: float,
    thickness: float,
    tile_m: float,
) -> None:
    """One slab between the arc lengths `start` and `end`; bend points of the wall crown in between are kept."""
    inner = arc[(arc > start + _EPS) & (arc < end - _EPS)]
    stations = np.concatenate([[start], inner, [end]])
    lefts = np.column_stack([np.interp(stations, arc, left[:, 0]), np.interp(stations, arc, left[:, 1])])
    rights = np.column_stack([np.interp(stations, arc, right[:, 0]), np.interp(stations, arc, right[:, 1])])
    tops = np.interp(stations, arc, top)
    bottoms = tops - thickness
    across = float(np.linalg.norm(lefts[0] - rights[0])) / tile_m

    def p3(xy: np.ndarray, z: float) -> List[float]:
        return [float(xy[0]), float(xy[1]), float(z)]

    centers = (lefts + rights) / 2.0
    for k in range(len(stations) - 1):
        j = k + 1
        u0, u1 = stations[k] / tile_m, stations[j] / tile_m
        direction = centers[j] - centers[k]
        direction = direction / np.linalg.norm(direction)
        left_normal = [float(-direction[1]), float(direction[0]), 0.0]

        corners = [p3(lefts[k], tops[k]), p3(lefts[j], tops[j]), p3(rights[j], tops[j]), p3(rights[k], tops[k])]
        top_normal = unit_vector(np.cross(np.array(corners[1]) - np.array(corners[0]), np.array(corners[3]) - np.array(corners[0])))
        if top_normal[2] < 0:
            top_normal = [-c for c in top_normal]
        builder.quad(corners, [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]], top_normal)
        builder.quad(
            [p3(lefts[k], bottoms[k]), p3(lefts[j], bottoms[j]), p3(rights[j], bottoms[j]), p3(rights[k], bottoms[k])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, -1.0],
        )
        for edge, normal in ((lefts, left_normal), (rights, [-left_normal[0], -left_normal[1], 0.0])):
            builder.quad(
                [p3(edge[k], bottoms[k]), p3(edge[j], bottoms[j]), p3(edge[j], tops[j]), p3(edge[k], tops[k])],
                [[u0, bottoms[k] / tile_m], [u1, bottoms[j] / tile_m], [u1, tops[j] / tile_m], [u0, tops[k] / tile_m]],
                normal,
            )

    for index in (0, len(stations) - 1):  # end faces (joints): normal points out of the slab
        neighbour = 1 if index == 0 else index - 1
        direction = centers[index] - centers[neighbour]
        direction = direction / np.linalg.norm(direction)
        builder.quad(
            [p3(lefts[index], bottoms[index]), p3(rights[index], bottoms[index]), p3(rights[index], tops[index]), p3(lefts[index], tops[index])],
            [[0.0, bottoms[index] / tile_m], [across, bottoms[index] / tile_m], [across, tops[index] / tile_m], [0.0, tops[index] / tile_m]],
            [float(direction[0]), float(direction[1]), 0.0],
        )
