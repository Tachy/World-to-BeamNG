"""
Flat roofs: detection and the surrounding sheet-metal rim as geometry.

The gravel surface itself is a normal roof polygon (different material, see RoofUvMapper); only the sheet-metal
rim is created here: one narrow prism per roof edge, made of outer, inner and top faces.
"""

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .. import config
from .edge_topology import edge_is_exterior_over_wall, orient_counter_clockwise, roof_rings, wall_lines
from .ring_geometry import newell_normal, open_ring, unit_or_none


@dataclass
class RimMesh:
    """Sheet-metal rim geometry (untextured, UVs = 0)."""

    vertices: np.ndarray  # (N, 3)
    uvs: np.ndarray  # (N, 2)
    faces: List[List[int]]

    @staticmethod
    def empty() -> "RimMesh":
        return RimMesh(np.zeros((0, 3)), np.zeros((0, 2)), [])


def is_flat_roof(verts: np.ndarray, max_slope_deg: float = config.FLAT_ROOF_MAX_SLOPE_DEG) -> bool:
    """Slope of the roof polygon is at most `max_slope_deg` degrees."""
    normal = unit_or_none(newell_normal(open_ring(verts)))
    if normal is None:
        return False
    slope = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    return slope <= max_slope_deg


class FlatRoofRimBuilder:
    """Builds the sheet-metal rim of all flat roofs of a building (only on edges on the outside above a wall)."""

    def __init__(
        self,
        height_m: float = config.FLAT_ROOF_EDGE_HEIGHT_M,
        thickness_m: float = config.FLAT_ROOF_EDGE_THICKNESS_M,
    ):
        self._height = height_m
        self._thickness = thickness_m

    def build(self, building: Dict) -> RimMesh:
        roofs = roof_rings(building)
        lines = wall_lines(building)

        vertices: List[np.ndarray] = []
        faces: List[List[int]] = []
        count = 0
        for index, ring in enumerate(roofs):
            if not is_flat_roof(ring):
                continue
            ordered = orient_counter_clockwise(ring)
            if ordered is None:
                continue
            for i in range(len(ordered)):
                a, b = ordered[i], ordered[(i + 1) % len(ordered)]
                if not edge_is_exterior_over_wall(a, b, roofs, index, lines):
                    continue
                prism_vertices, prism_faces = self._prism(a, b)
                vertices.append(prism_vertices)
                faces.extend([[f + count for f in face] for face in prism_faces])
                count += len(prism_vertices)

        if not vertices:
            return RimMesh.empty()
        stacked = np.vstack(vertices)
        return RimMesh(stacked, np.zeros((len(stacked), 2)), faces)

    def _prism(self, a: np.ndarray, b: np.ndarray):
        """
        Prism over the edge a->b (outer side on the right). Ends extended by the thickness so that corners close.

        Returns:
            (12, 3) vertices (own ones per face: flat shaded) and 6 triangles, all normals pointing outward
        """
        direction = unit_or_none(np.array([b[0] - a[0], b[1] - a[1], 0.0]))
        if direction is None:
            return np.zeros((0, 3)), []
        outward = np.array([direction[1], -direction[0], 0.0])
        up = np.array([0.0, 0.0, self._height])
        inward_step = -outward * self._thickness

        start = a - direction * self._thickness
        end = b + direction * self._thickness
        start_inner, end_inner = start + inward_step, end + inward_step

        outer = [start, end, end + up, start + up]
        inner = [end_inner, start_inner, start_inner + up, end_inner + up]
        top = [start + up, end + up, end_inner + up, start_inner + up]

        vertices = np.array(outer + inner + top)
        faces = []
        for base in (0, 4, 8):
            faces.append([base, base + 1, base + 2])
            faces.append([base, base + 2, base + 3])
        return vertices, faces
