"""
Bruchsteinmauern aus OSM-Linien (`barrier=wall`, `barrier=retaining_wall`).

Nur Mauern mit `height`-Tag werden gebaut (ohne Höhenangabe wäre die Höhe geraten). Die Mauer ist ein Band von
~50 cm Dicke entlang der Linie, das dem Gelände folgt: Oberkante = Boden an der Mittellinie + Höhe, Unterkante
beidseitig unter dem jeweiligen Boden, damit am Fuß kein Spalt bleibt. Ecken sind auf Gehrung geschnitten.

Jede Fläche hat eigene Eckpunkte (harte Kanten) und explizite Normalen. UVs sind in Textur-Kacheln angegeben
(Kachelgröße `tile_m` Meter), das Material wiederholt sich also mit tiling_scale 1.
"""

import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]
ToLocal = Callable[[Sequence[Dict]], List[Tuple[float, float]]]

WALL_BARRIERS = ("wall", "retaining_wall")
STONE_MATERIALS = (None, "stone")  # ohne material-Tag wird Stein angenommen; Ziegel, Beton, Holz ... nicht
MAX_PLAUSIBLE_HEIGHT = 10.0  # darüber ist die Angabe ein Tippfehler
MAX_MITRE_FACTOR = 2.0  # spitze Ecken: Gehrungslänge höchstens doppelte halbe Dicke


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


def _offset_points(points: np.ndarray, half: float, closed: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Linke und rechte Kante im Abstand `half`, an Knicken auf Gehrung geschnitten."""
    count = len(points)
    directions = np.roll(points, -1, axis=0) - points
    if not closed:
        directions = directions[:-1]
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    normals = np.column_stack([-directions[:, 1], directions[:, 0]])  # links der Laufrichtung

    left, right = np.zeros((count, 2)), np.zeros((count, 2))
    for i in range(count):
        if closed:
            before, after = normals[(i - 1) % count], normals[i % count]
        else:
            before, after = normals[max(i - 1, 0)], normals[min(i, count - 2)]
        miter = before + after
        norm = np.linalg.norm(miter)
        miter = after if norm < 1e-9 else miter / norm
        scale = min(half / max(float(np.dot(miter, after)), 1.0 / MAX_MITRE_FACTOR), half * MAX_MITRE_FACTOR)
        left[i] = points[i] + miter * scale
        right[i] = points[i] - miter * scale
    return left, right


class _Builder:
    def __init__(self):
        self.vertices: List[List[float]] = []
        self.uvs: List[List[float]] = []
        self.normals: List[List[float]] = []
        self.faces: List[List[int]] = []

    def quad(self, corners: Sequence[Sequence[float]], uvs: Sequence[Sequence[float]], normal: Sequence[float]) -> None:
        """Viereck mit eigenen Eckpunkten; der Umlaufsinn wird so gewählt, dass die Fläche zur Normalen zeigt."""
        base = len(self.vertices)
        self.vertices.extend([list(c) for c in corners])
        self.uvs.extend([list(u) for u in uvs])
        self.normals.extend([list(normal)] * 4)
        for tri in ([0, 1, 2], [0, 2, 3]):
            a, b, c = (np.array(corners[i], dtype=float) for i in tri)
            if np.dot(np.cross(b - a, c - a), normal) < 0:
                tri = [tri[0], tri[2], tri[1]]
            self.faces.append([base + tri[0], base + tri[1], base + tri[2]])


def _unit(vector: np.ndarray) -> List[float]:
    length = np.linalg.norm(vector)
    return [0.0, 0.0, 1.0] if length < 1e-12 else [float(c) for c in vector / length]


def build_wall_mesh(
    coords: Sequence[Tuple[float, float]],
    height: float,
    ground_at: HeightAt,
    thickness: float = 0.5,
    sink: float = 0.3,
    max_step: float = 1.0,
    tile_m: float = 1.2,
) -> Dict:
    """
    Mesh einer Mauer entlang `coords` (lokale x, y; geschlossen, wenn erster = letzter Punkt).

    Returns:
        {"vertices": (N, 3), "uvs": (N, 2), "normals": (N, 3), "faces": [[a, b, c], ...]}; Z absolut wie das Gelände
    """
    points = np.array(coords, dtype=float)
    closed = len(points) > 3 and np.allclose(points[0], points[-1])
    if closed:
        points = points[:-1]
    points = _densify(points, max_step, closed)
    left, right = _offset_points(points, thickness / 2.0, closed)

    ground_center = np.asarray(ground_at(points[:, 0], points[:, 1]), dtype=float)
    ground_left = np.asarray(ground_at(left[:, 0], left[:, 1]), dtype=float)
    ground_right = np.asarray(ground_at(right[:, 0], right[:, 1]), dtype=float)
    top = ground_center + height
    bottom_left, bottom_right = ground_left - sink, ground_right - sink

    steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)]) / tile_m  # u je Punkt; beim Ring hat der Schlusspunkt den Gesamtwert
    across = thickness / tile_m

    def p3(xy: np.ndarray, z: float) -> List[float]:
        return [float(xy[0]), float(xy[1]), float(z)]

    builder = _Builder()
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
        top_normal = _unit(np.cross(np.array(corners[1]) - np.array(corners[0]), np.array(corners[3]) - np.array(corners[0])))
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
        mesh = build_wall_mesh(wall["coords"], wall["height"], ground_at, thickness, sink, max_step, tile_m)
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
