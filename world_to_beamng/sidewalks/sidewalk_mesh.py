"""
Mesh of one sidewalk run: cross-section from the carriageway edge outward (the sidewalk lies LEFT of the run direction):

    kerb face (base - skirt_depth .. base + kerb_height) - kerb top (kerb_width) - sidewalk top (sidewalk_width, flat at
    base + kerb_height) - outer skirt (down to base - skirt_depth)

plus an end cap at both ends. Base = road height at the carriageway edge (the terrain under the band is embedded at road
height, see sidewalks/terrain.py). The kerb face and skirt reach below the road so that no gap shows where the terrain
between two raster cells lies a little lower. Every face has its own vertices and normals (hard edges); UVs are metric
in texture tiles of `tile_m`.
"""

from typing import Dict

import numpy as np

from ..geometry.polyline import arc_lengths
from ..walls.mesh_parts import MeshBuilder, offset_points, point3, unit_vector


def _densify(points: np.ndarray, max_step: float) -> np.ndarray:
    """Inserts points so that no segment is longer than `max_step` (x, y and z interpolated linearly)."""
    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / max_step - 1e-9)))
        result.extend(a + (b - a) * (k / count) for k in range(1, count + 1))
    return np.array(result)


def _upward(corners) -> list:
    normal = unit_vector(np.cross(np.subtract(corners[1], corners[0]), np.subtract(corners[3], corners[0])))
    return normal if normal[2] >= 0 else [-c for c in normal]


def build_sidewalk_mesh(
    points: np.ndarray,
    kerb_width: float,
    sidewalk_width: float,
    kerb_height: float,
    skirt_depth: float,
    max_step: float,
    tile_m: float,
    kerb_material: str,
    surface_material: str,
) -> Dict:
    """
    Args:
        points: (M, 3) carriageway edge at road height, sidewalk on the left of the run direction
        max_step: longest segment along the run, in meters

    Returns:
        {"vertices", "uvs", "normals", "faces": {kerb_material: [...], surface_material: [...]}}
    """
    pts = _densify(np.asarray(points, dtype=float), max_step)
    xy, base = pts[:, :2], pts[:, 2]
    kerb_out, _ = offset_points(xy, kerb_width, closed=False)
    outer, _ = offset_points(xy, kerb_width + sidewalk_width, closed=False)
    top, bottom = base + kerb_height, base - skirt_depth
    u = arc_lengths(xy) / tile_m
    face_v = (kerb_height + skirt_depth) / tile_m

    kerb, surface = MeshBuilder(), MeshBuilder()
    for i in range(len(pts) - 1):
        j = i + 1
        d = xy[j] - xy[i]
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        toward_road = [float(d[1]), float(-d[0]), 0.0]
        away = [float(-d[1]), float(d[0]), 0.0]
        uv_face = [[u[i], 0.0], [u[j], 0.0], [u[j], face_v], [u[i], face_v]]

        kerb.quad([point3(xy[i], bottom[i]), point3(xy[j], bottom[j]), point3(xy[j], top[j]), point3(xy[i], top[i])], uv_face, toward_road)
        corners = [point3(xy[i], top[i]), point3(xy[j], top[j]), point3(kerb_out[j], top[j]), point3(kerb_out[i], top[i])]
        kerb.quad(corners, [[u[i], 0.0], [u[j], 0.0], [u[j], kerb_width / tile_m], [u[i], kerb_width / tile_m]], _upward(corners))
        corners = [point3(kerb_out[i], top[i]), point3(kerb_out[j], top[j]), point3(outer[j], top[j]), point3(outer[i], top[i])]
        across = sidewalk_width / tile_m
        surface.quad(corners, [[u[i], 0.0], [u[j], 0.0], [u[j], across], [u[i], across]], _upward(corners))
        kerb.quad([point3(outer[i], bottom[i]), point3(outer[j], bottom[j]), point3(outer[j], top[j]), point3(outer[i], top[i])], uv_face, away)

    # End caps: at the start the outward direction points against the run, at the end along it
    for index, neighbour, sign in ((0, 1, -1.0), (len(pts) - 1, len(pts) - 2, 1.0)):
        d = xy[index] - xy[neighbour] if sign > 0 else xy[neighbour] - xy[index]
        d = sign * d / max(float(np.linalg.norm(d)), 1e-12)
        span = (kerb_width + sidewalk_width) / tile_m
        kerb.quad(
            [point3(xy[index], bottom[index]), point3(outer[index], bottom[index]), point3(outer[index], top[index]), point3(xy[index], top[index])],
            [[0.0, 0.0], [span, 0.0], [span, face_v], [0.0, face_v]],
            [float(d[0]), float(d[1]), 0.0],
        )

    offset = len(kerb.vertices)
    return {
        "vertices": np.array(kerb.vertices + surface.vertices, dtype=float),
        "uvs": np.array(kerb.uvs + surface.uvs, dtype=float),
        "normals": np.array(kerb.normals + surface.normals, dtype=float),
        "faces": {kerb_material: kerb.faces, surface_material: [[a + offset, b + offset, c + offset] for a, b, c in surface.faces]},
    }
