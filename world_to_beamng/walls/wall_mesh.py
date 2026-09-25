"""
Rubble stone walls from OSM lines (`barrier=wall`, `barrier=retaining_wall`).

Only walls with a `height` tag are built (without a height value the height would be a guess). The wall is a band of
~50 cm thickness along the line that follows the terrain: top edge = base at the centerline + height, bottom edge
on both sides below the respective ground, so that no gap remains at the foot. Corners are mitered.

Cap slabs (wall_cap.py) sit on top: stone slabs of `cap_thickness` thickness that project a few centimeters.
The total height (slab top edge) is the OSM height, the wall body ends lower by the slab thickness.

Base = ground at the centerline. If a point of the centerline is at most N meters from a road centerline
(`road_base_at`, see road_base.py), the base is instead the height of that centerline; the bottom edge there
reaches down below the terrain or the road, whichever is lower.

Every face has its own vertices (hard edges) and explicit normals. UVs are given in texture tiles
(tile size `tile_m` meters), so the material repeats with tiling_scale 1.
"""

import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mesh_parts import MeshBuilder, offset_points, unit_vector
from .wall_cap import add_cap

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]
ToLocal = Callable[[Sequence[Dict]], List[Tuple[float, float]]]

WALL_BARRIERS = ("wall", "retaining_wall")
STONE_MATERIALS = (None, "stone")  # without a material tag stone is assumed; brick, concrete, wood ... are not
MAX_PLAUSIBLE_HEIGHT = 10.0  # above this the value is a typo


def parse_wall_height(value: Optional[str]) -> Optional[float]:
    """OSM `height` such as "1.50", "3", "2 m" or "2,5" -> meters; None if missing or implausible."""
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*m?\s*", str(value))
    if not match:
        return None
    height = float(match.group(1).replace(",", "."))
    return height if 0.0 < height <= MAX_PLAUSIBLE_HEIGHT else None


def _is_stone_wall(element: Dict) -> bool:
    tags = element.get("tags") or {}
    return element.get("type") == "way" and tags.get("barrier") in WALL_BARRIERS and tags.get("material") in STONE_MATERIALS


def select_walls(osm_data: Sequence[Dict], to_local: ToLocal) -> List[Dict]:
    """
    Walls with a height value from OSM ways in local coordinates.

    Returns:
        [{"osm_id", "height", "coords": [(x, y), ...]}] - a wall without a `height` tag (or with a material other than stone) is dropped
    """
    walls = []
    for element in osm_data:
        if not _is_stone_wall(element):
            continue
        height = parse_wall_height(element["tags"].get("height"))
        coords = to_local(element.get("geometry") or [])
        if height is None or len(coords) < 2:
            continue
        walls.append({"osm_id": element.get("id"), "height": height, "coords": coords})
    return walls


def _densify(points: np.ndarray, max_step: float, closed: bool) -> np.ndarray:
    """Additional points so that no segment is longer than `max_step` (the wall is meant to follow the terrain)."""
    ring = np.vstack([points, points[:1]]) if closed else points
    result = [ring[0]]
    for start, end in zip(ring[:-1], ring[1:]):
        steps = max(1, int(math.ceil(np.linalg.norm(end - start) / max_step - 1e-9)))
        for i in range(1, steps + 1):
            result.append(start + (end - start) * i / steps)
    dense = np.array(result)
    return dense[:-1] if closed else dense


def build_wall_mesh(
    coords: Sequence[Tuple[float, float]],
    height: float,
    ground_at: HeightAt,
    thickness: float = 0.5,
    sink: float = 0.3,
    max_step: float = 1.0,
    tile_m: float = 1.2,
    road_base_at: Optional[HeightAt] = None,
    cap_thickness: float = 0.05,
    cap_overhang: float = 0.04,
    cap_plate_length: float = 0.8,
    cap_joint: float = 0.01,
    seed: int = 0,
) -> Dict:
    """
    Mesh of a wall along `coords` (local x, y; closed if first = last point).

    Args:
        road_base_at: (x, y) -> height of the road centerline within snap distance, NaN otherwise (RoadBaseHeight)
        cap_thickness, cap_overhang, cap_plate_length, cap_joint: cap slabs (thickness 0 = no slabs)
        seed: start value of the slab lengths (different per wall, but identical on every run)

    Returns:
        {"vertices": (N, 3), "uvs": (N, 2), "normals": (N, 3), "faces": [[a, b, c], ...]}; Z absolute like the terrain
    """
    points = np.array(coords, dtype=float)
    closed = len(points) > 3 and np.allclose(points[0], points[-1])
    if closed:
        points = points[:-1]
    points = _densify(points, max_step, closed)
    left, right = offset_points(points, thickness / 2.0, closed)

    ground_center = np.asarray(ground_at(points[:, 0], points[:, 1]), dtype=float)
    ground_left = np.asarray(ground_at(left[:, 0], left[:, 1]), dtype=float)
    ground_right = np.asarray(ground_at(right[:, 0], right[:, 1]), dtype=float)
    road = np.asarray(road_base_at(points[:, 0], points[:, 1]), dtype=float) if road_base_at else np.full(len(points), np.nan)
    on_road = ~np.isnan(road)
    base = np.where(on_road, road, ground_center)
    crown = base + height  # top edge of the cap slabs = OSM height
    top = crown - cap_thickness  # the wall body ends below the slabs
    bottom_left = np.where(on_road, np.minimum(base, ground_left), ground_left) - sink
    bottom_right = np.where(on_road, np.minimum(base, ground_right), ground_right) - sink

    steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)]) / tile_m  # u per point; for a ring the closing point holds the total value
    across = thickness / tile_m

    def p3(xy: np.ndarray, z: float) -> List[float]:
        return [float(xy[0]), float(xy[1]), float(z)]

    builder = MeshBuilder()
    segments = len(points) if closed else len(points) - 1
    for i in range(segments):
        j = (i + 1) % len(points)
        u0, u1 = along[i], along[i + 1]
        direction = points[j] - points[i]
        direction = direction / np.linalg.norm(direction)
        left_normal = [float(-direction[1]), float(direction[0]), 0.0]

        # Long sides: from the bottom edge (below the ground) to the top edge
        builder.quad(
            [p3(left[i], bottom_left[i]), p3(left[j], bottom_left[j]), p3(left[j], top[j]), p3(left[i], top[i])],
            [[u0, bottom_left[i] / tile_m], [u1, bottom_left[j] / tile_m], [u1, top[j] / tile_m], [u0, top[i] / tile_m]],
            left_normal,
        )
        builder.quad(
            [p3(right[i], bottom_right[i]), p3(right[j], bottom_right[j]), p3(right[j], top[j]), p3(right[i], top[i])],
            [[u0, bottom_right[i] / tile_m], [u1, bottom_right[j] / tile_m], [u1, top[j] / tile_m], [u0, top[i] / tile_m]],
            [-left_normal[0], -left_normal[1], 0.0],
        )
        # Top face
        corners = [p3(left[i], top[i]), p3(left[j], top[j]), p3(right[j], top[j]), p3(right[i], top[i])]
        top_normal = unit_vector(np.cross(np.array(corners[1]) - np.array(corners[0]), np.array(corners[3]) - np.array(corners[0])))
        if top_normal[2] < 0:
            top_normal = [-c for c in top_normal]
        builder.quad(corners, [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]], top_normal)

    if not closed:  # end faces at both ends
        for index, sign, neighbour in ((0, -1.0, 1), (len(points) - 1, 1.0, len(points) - 2)):
            direction = points[1] - points[0] if index == 0 else points[-1] - points[neighbour]
            direction = direction / np.linalg.norm(direction)
            builder.quad(
                [
                    p3(left[index], bottom_left[index]),
                    p3(right[index], bottom_right[index]),
                    p3(right[index], top[index]),
                    p3(left[index], top[index]),
                ],
                [
                    [0.0, bottom_left[index] / tile_m],
                    [across, bottom_right[index] / tile_m],
                    [across, top[index] / tile_m],
                    [0.0, top[index] / tile_m],
                ],
                [float(sign * direction[0]), float(sign * direction[1]), 0.0],
            )

    if cap_thickness > 0:
        add_cap(builder, points, crown, closed, thickness, cap_thickness, cap_overhang, cap_plate_length, cap_joint, tile_m, np.random.default_rng(seed))

    return {
        "vertices": np.array(builder.vertices, dtype=float),
        "uvs": np.array(builder.uvs, dtype=float),
        "normals": np.array(builder.normals, dtype=float),
        "faces": builder.faces,
    }


def build_walls(
    osm_data: Sequence[Dict],
    to_local: ToLocal,
    ground_at: HeightAt,
    material: str,
    thickness: float = 0.5,
    sink: float = 0.3,
    max_step: float = 1.0,
    tile_m: float = 1.2,
    road_base_at: Optional[HeightAt] = None,
    cap_thickness: float = 0.05,
    cap_overhang: float = 0.04,
    cap_plate_length: float = 0.8,
    cap_joint: float = 0.01,
) -> Tuple[List[Dict], Dict]:
    """
    All walls with a height value as mesh dicts for the DAE export (`{"id", "vertices", "uvs", "normals", "faces": {material: [...]}}`).

    Returns:
        (meshes, statistics {"built", "length", "without_height"}); "without_height" counts stone walls that were
        skipped for lack of a height value
    """
    walls = select_walls(osm_data, to_local)
    meshes, length = [], 0.0
    for wall in walls:
        mesh = build_wall_mesh(
            wall["coords"],
            wall["height"],
            ground_at,
            thickness,
            sink,
            max_step,
            tile_m,
            road_base_at,
            cap_thickness,
            cap_overhang,
            cap_plate_length,
            cap_joint,
            seed=int(wall["osm_id"] or 0),
        )
        meshes.append(
            {
                "id": f"wall_{wall['osm_id']}",
                "vertices": mesh["vertices"],
                "uvs": mesh["uvs"],
                "normals": mesh["normals"],
                "faces": {material: mesh["faces"]},
            }
        )
        length += float(np.linalg.norm(np.diff(np.array(wall["coords"]), axis=0), axis=1).sum())
    total = sum(1 for element in osm_data if _is_stone_wall(element))
    return meshes, {"built": len(meshes), "length": length, "without_height": total - len(walls)}
