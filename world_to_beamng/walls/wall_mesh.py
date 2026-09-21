"""
Bruchsteinmauern aus OSM-Linien (`barrier=wall`, `barrier=retaining_wall`).

Nur Mauern mit `height`-Tag werden gebaut (ohne Höhenangabe wäre die Höhe geraten). Die Mauer ist ein Band von
~50 cm Dicke entlang der Linie, das dem Gelände folgt: Oberkante = Basis an der Mittellinie + Höhe, Unterkante
beidseitig unter dem jeweiligen Boden, damit am Fuß kein Spalt bleibt. Ecken sind auf Gehrung geschnitten.

Oben liegen Abdeckplatten (wall_cap.py): Steinplatten von `cap_thickness` Dicke, die ein paar Zentimeter überstehen.
Die Gesamthöhe (Plattenoberkante) ist die OSM-Höhe, der Mauerkörper endet um die Plattendicke tiefer.

Basis = Boden an der Mittellinie. Liegt ein Punkt der Mittellinie höchstens N Meter neben einer Straßen-Centerline
(`road_base_at`, siehe road_base.py), ist die Basis stattdessen die Höhe dieser Centerline; die Unterkante reicht
dort bis unter das Gelände oder die Straße, je nachdem, was tiefer liegt.

Jede Fläche hat eigene Eckpunkte (harte Kanten) und explizite Normalen. UVs sind in Textur-Kacheln angegeben
(Kachelgröße `tile_m` Meter), das Material wiederholt sich also mit tiling_scale 1.
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
STONE_MATERIALS = (None, "stone")  # ohne material-Tag wird Stein angenommen; Ziegel, Beton, Holz ... nicht
MAX_PLAUSIBLE_HEIGHT = 10.0  # darüber ist die Angabe ein Tippfehler


def parse_wall_height(value: Optional[str]) -> Optional[float]:
    """OSM-`height` wie "1.50", "3", "2 m" oder "2,5" -> Meter; None bei fehlender oder unplausibler Angabe."""
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
    Mauern mit Höhenangabe aus OSM-Ways in lokalen Koordinaten.

    Returns:
        [{"osm_id", "height", "coords": [(x, y), ...]}] - ohne `height`-Tag (oder mit anderem Material als Stein) entfällt die Mauer
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
    """Zusätzliche Punkte, damit kein Segment länger als `max_step` ist (die Mauer soll dem Gelände folgen)."""
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
    Mesh einer Mauer entlang `coords` (lokale x, y; geschlossen, wenn erster = letzter Punkt).

    Args:
        road_base_at: (x, y) -> Höhe der Straßen-Centerline in Snap-Distanz, NaN sonst (RoadBaseHeight)
        cap_thickness, cap_overhang, cap_plate_length, cap_joint: Abdeckplatten (Dicke 0 = keine Platten)
        seed: Startwert der Plattenlängen (je Mauer verschieden, aber bei jedem Lauf gleich)

    Returns:
        {"vertices": (N, 3), "uvs": (N, 2), "normals": (N, 3), "faces": [[a, b, c], ...]}; Z absolut wie das Gelände
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
    crown = base + height  # Oberkante der Abdeckplatten = OSM-Höhe
    top = crown - cap_thickness  # der Mauerkörper endet unter den Platten
    bottom_left = np.where(on_road, np.minimum(base, ground_left), ground_left) - sink
    bottom_right = np.where(on_road, np.minimum(base, ground_right), ground_right) - sink

    steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)]) / tile_m  # u je Punkt; beim Ring hat der Schlusspunkt den Gesamtwert
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

        # Längsseiten: von der Unterkante (unter dem Boden) bis zur Oberkante
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
        # Oberseite
        corners = [p3(left[i], top[i]), p3(left[j], top[j]), p3(right[j], top[j]), p3(right[i], top[i])]
        top_normal = unit_vector(np.cross(np.array(corners[1]) - np.array(corners[0]), np.array(corners[3]) - np.array(corners[0])))
        if top_normal[2] < 0:
            top_normal = [-c for c in top_normal]
        builder.quad(corners, [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]], top_normal)

    if not closed:  # Stirnflächen an den beiden Enden
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
    Alle Mauern mit Höhenangabe als Mesh-Dicts für den DAE-Export (`{"id", "vertices", "uvs", "normals", "faces": {material: [...]}}`).

    Returns:
        (Meshes, Statistik {"built", "length", "without_height"}); "without_height" zählt Steinmauern, die mangels
        Höhenangabe übersprungen wurden
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
