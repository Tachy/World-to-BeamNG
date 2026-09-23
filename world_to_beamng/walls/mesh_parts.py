"""
Gemeinsame Bausteine der Mauer-Meshes (Mauerkörper und Abdeckplatten): Mesh-Sammler, Einheitsvektor, Randlinien.
"""

from typing import List, Sequence, Tuple

import numpy as np

MAX_MITRE_FACTOR = 2.0  # spitze Ecken: Gehrungslänge höchstens doppelte halbe Dicke


def offset_points(points: np.ndarray, half: float, closed: bool) -> Tuple[np.ndarray, np.ndarray]:
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


class MeshBuilder:
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


def unit_vector(vector: np.ndarray) -> List[float]:
    length = np.linalg.norm(vector)
    return [0.0, 0.0, 1.0] if length < 1e-12 else [float(c) for c in vector / length]


def add_box_column(
    builder: "MeshBuilder",
    cx: float,
    cy: float,
    bottom_z: float,
    top_z: float,
    size: float,
    tile_m: float,
    direction: Tuple[float, float] = (1.0, 0.0),
) -> None:
    """Rechteckige Stütze (4 Seitenflächen) von `bottom_z` bis `top_z`, quadratischer Querschnitt `size` - für
    Brücken-Pfeiler (bridges/bridge_mesh.py) und Galerie-Stützen (tunnels/gallery_mesh.py).

    Args:
        direction: (dx, dy) Fahrtrichtung an der Stützen-Position (muss nicht normiert sein) - das Profil
            ist relativ dazu ausgerichtet (Kanten parallel/quer zur Straße/Galerie), nicht an den Welt-
            Achsen. Default (1, 0) = achsenparallel, für Aufrufer ohne Richtungsinformation.
    """
    half = size / 2.0
    dx, dy = float(direction[0]), float(direction[1])
    norm = (dx * dx + dy * dy) ** 0.5
    dx, dy = (dx / norm, dy / norm) if norm > 1e-9 else (1.0, 0.0)
    fwd = (dx * half, dy * half)
    left = (-dy * half, dx * half)
    corners = [
        (cx - fwd[0] - left[0], cy - fwd[1] - left[1]),
        (cx + fwd[0] - left[0], cy + fwd[1] - left[1]),
        (cx + fwd[0] + left[0], cy + fwd[1] + left[1]),
        (cx - fwd[0] + left[0], cy - fwd[1] + left[1]),
    ]
    height_tiles = (top_z - bottom_z) / tile_m
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        direction = np.array([b[0] - a[0], b[1] - a[1]])
        direction = direction / np.linalg.norm(direction)
        normal = [float(direction[1]), float(-direction[0]), 0.0]
        builder.quad(
            [[a[0], a[1], bottom_z], [b[0], b[1], bottom_z], [b[0], b[1], top_z], [a[0], a[1], top_z]],
            [[0.0, 0.0], [size / tile_m, 0.0], [size / tile_m, height_tiles], [0.0, height_tiles]],
            normal,
        )
