"""
Shared building blocks of the wall meshes (wall body and cap slabs): mesh collector, unit vector, border lines.
"""

from typing import List, Sequence, Tuple

import numpy as np

MAX_MITRE_FACTOR = 2.0  # sharp corners: miter length at most twice the half thickness


def offset_points(points: np.ndarray, half: float, closed: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Left and right edge at distance `half`, mitered at bends."""
    count = len(points)
    directions = np.roll(points, -1, axis=0) - points
    if not closed:
        directions = directions[:-1]
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    normals = np.column_stack([-directions[:, 1], directions[:, 0]])  # left of the direction of travel

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
        """Quad with its own corner vertices; the winding order is chosen so that the face points along the normal.

        Cross/dot product deliberately in pure Python instead of numpy: quad() runs in tight
        loops (bridges/tunnels/walls) with tens of thousands of calls on 3-vectors - there
        numpy's dispatch overhead (among others moveaxis() in np.cross()) clearly dominates the
        actual computation (see the pyinstrument profile of the 4x4 km export: MeshBuilder.quad
        alone in _build_bridges() ~5 s of 67 s total time).
        """
        base = len(self.vertices)
        self.vertices.extend([list(c) for c in corners])
        self.uvs.extend([list(u) for u in uvs])
        self.normals.extend([list(normal)] * 4)
        nx, ny, nz = normal
        for tri in ([0, 1, 2], [0, 2, 3]):
            a, b, c = corners[tri[0]], corners[tri[1]], corners[tri[2]]
            bax, bay, baz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            cax, cay, caz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            cross_x = bay * caz - baz * cay
            cross_y = baz * cax - bax * caz
            cross_z = bax * cay - bay * cax
            if cross_x * nx + cross_y * ny + cross_z * nz < 0:
                tri = [tri[0], tri[2], tri[1]]
            self.faces.append([base + tri[0], base + tri[1], base + tri[2]])

    def triangle(self, corners: Sequence[Sequence[float]], uvs: Sequence[Sequence[float]], normal: Sequence[float]) -> None:
        """Triangle with its own corner vertices, winding order matching the normal as in quad()."""
        base = len(self.vertices)
        self.vertices.extend([list(c) for c in corners])
        self.uvs.extend([list(u) for u in uvs])
        self.normals.extend([list(normal)] * 3)
        a, b, c = corners
        cross = np.cross(np.subtract(b, a), np.subtract(c, a))
        order = [0, 1, 2] if float(np.dot(cross, normal)) >= 0 else [0, 2, 1]
        self.faces.append([base + order[0], base + order[1], base + order[2]])


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
    """Rectangular column (4 side faces) from `bottom_z` to `top_z`, square cross-section `size` - for
    bridge piers (bridges/bridge_mesh.py) and gallery columns (tunnels/gallery_mesh.py).

    Args:
        direction: (dx, dy) direction of travel at the column position (need not be normalized) - the profile
            is aligned relative to it (edges parallel/across the road/gallery), not to the world
            axes. Default (1, 0) = axis-aligned, for callers without direction information.
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
