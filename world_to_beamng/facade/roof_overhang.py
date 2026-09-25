"""
Roof overhang of pitched roofs.

The LOD2 roofs end exactly at the wall. Here every roof edge that lies outside above a wall is moved outward within
the roof plane: eave (horizontal edge) by `eave_m`, gable edge (verge) by `verge_m`. Both dimensions are measured
HORIZONTALLY (as is customary in construction); at the eave the edge is therefore moved by `eave_m / cos(pitch)`
within the roof plane. Edges without a wall
below them (ridges, valleys, abutments) and firewalls remain unchanged. The overhang is a slab of
`thickness_m` thickness PERPENDICULAR to the roof surface (rectangular profile): the top face is the extended roof
surface, plus soffit and fascia board.

The displacement is a polygon offset with its own amount per edge; the corners are mitered. As a result, the
overhangs of two roof surfaces meet exactly on the (extended) ridge line at a ridge.
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from shapely.geometry import Polygon

from .. import config
from .edge_topology import WallLine, edge_is_exterior_over_wall, orient_counter_clockwise
from .ring_geometry import UP, newell_normal, unit_or_none

_HORIZONTAL_EDGE_SLOPE = 0.17  # below |dz| / length an edge counts as horizontal (~10 degrees)
_MIN_COS_SLOPE = 0.3  # very steep roofs: the eave overhang does not grow beyond 1/0.3
_MITER_LIMIT = 4.0  # the corner miter may be at most this many overhang widths away from the corner
_MIN_AREA_RATIO = 0.99  # the offset outline must never become smaller than the original


@dataclass
class SlopedRoof:
    """Roof surface including overhang (ring counter-clockwise seen from above) and its fascia/soffit."""

    ring: np.ndarray  # (M, 3)
    trim_vertices: np.ndarray  # (K, 3)
    trim_faces: List[List[int]]


class RoofOverhangBuilder:
    """Builds the extended roof surface and the thickness of the overhang for a pitched roof polygon."""

    def __init__(
        self,
        eave_m: float = config.ROOF_EAVE_OVERHANG_M,
        verge_m: float = config.ROOF_VERGE_OVERHANG_M,
        thickness_m: float = config.ROOF_OVERHANG_THICKNESS_M,
    ):
        self._eave = eave_m
        self._verge = verge_m
        self._thickness = thickness_m

    def build(self, ring: np.ndarray, index: int, roofs: Sequence[np.ndarray], lines: Sequence[WallLine]) -> SlopedRoof:
        """
        Args:
            ring: roof ring without closing point
            index: position of this ring in `roofs`
            roofs: all roof rings of the building (for shared edges)
            lines: footprint lines of the building's walls

        Returns:
            SlopedRoof; without overhang (no edge above a wall) `ring` is the input ring, rotated if necessary.
        """
        ordered = orient_counter_clockwise(ring)
        if ordered is None:
            return SlopedRoof(ring, np.zeros((0, 3)), [])
        empty = SlopedRoof(ordered, np.zeros((0, 3)), [])

        normal = unit_or_none(newell_normal(ordered))
        u_axis = unit_or_none(np.cross(UP, normal)) if normal is not None else None
        if normal is None or u_axis is None:
            return empty
        v_axis = unit_or_none(np.cross(normal, u_axis))

        points = np.column_stack([ordered @ u_axis, ordered @ v_axis])
        distances = self._edge_distances(ordered, points, index, roofs, lines, cos_slope=float(normal[2]))
        if not distances.any():
            return empty

        moved = self._offset(points, distances)
        if moved is None or not self._is_valid(points, moved):
            return empty

        plane_offset = float(ordered[0] @ normal)
        new_ring = np.outer(moved[:, 0], u_axis) + np.outer(moved[:, 1], v_axis) + plane_offset * normal
        trim_vertices, trim_faces = self._trim(ordered, new_ring, distances, normal)
        return SlopedRoof(new_ring, trim_vertices, trim_faces)

    # ------------------------------------------------------------------ Edges

    def _edge_distances(self, ring, points, index, roofs, lines, cos_slope: float) -> np.ndarray:
        """Overhang per edge i (from point i to i+1) in the roof plane: eave, gable or 0."""
        count = len(ring)
        distances = np.zeros(count)
        for i in range(count):
            a, b = ring[i], ring[(i + 1) % count]
            if not edge_is_exterior_over_wall(a, b, roofs, index, lines):
                continue
            length = float(np.linalg.norm(b - a))
            if length < 1e-6:
                continue
            horizontal = abs(float(b[2] - a[2])) / length < _HORIZONTAL_EDGE_SLOPE
            if not horizontal:
                distances[i] = self._verge
                continue
            direction = points[(i + 1) % count] - points[i]
            outward = np.array([direction[1], -direction[0]]) / max(float(np.linalg.norm(direction)), 1e-12)
            # Eave: the outward direction points downslope in the roof plane (-v); a free upper edge counts as gable
            distances[i] = self._eave / max(cos_slope, _MIN_COS_SLOPE) if outward[1] < 0 else self._verge
        return distances

    @staticmethod
    def _offset(points: np.ndarray, distances: np.ndarray):
        """Polygon offset in the roof plane with its own amount per edge; None if a corner cannot be computed."""
        count = len(points)
        tangents = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(tangents, axis=1)
        if (lengths < 1e-9).any():
            return None
        tangents = tangents / lengths[:, None]
        normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])  # right of the direction of travel = outside (counter-clockwise)
        limit = _MITER_LIMIT * float(distances.max())

        moved = np.empty_like(points)
        for j in range(count):
            prev, cur = (j - 1) % count, j
            a = points[j] + normals[prev] * distances[prev]
            b = points[j] + normals[cur] * distances[cur]
            cross = float(tangents[prev, 0] * tangents[cur, 1] - tangents[prev, 1] * tangents[cur, 0])
            if abs(cross) < 1e-6:  # (nearly) parallel neighboring edges: move by the average
                moved[j] = points[j] + (normals[prev] * distances[prev] + normals[cur] * distances[cur]) / 2
                continue
            step = float((b[0] - a[0]) * tangents[cur, 1] - (b[1] - a[1]) * tangents[cur, 0]) / cross
            corner = a + step * tangents[prev]
            if float(np.linalg.norm(corner - points[j])) > limit:  # very sharp corner: the miter would shoot out
                return None
            moved[j] = corner
        return moved

    @staticmethod
    def _is_valid(points: np.ndarray, moved: np.ndarray) -> bool:
        """The offset outline must be a valid polygon and must not reduce the area."""
        original, extended = Polygon(points), Polygon(moved)
        return bool(extended.is_valid and extended.area >= original.area * _MIN_AREA_RATIO)

    # ------------------------------------------------------------------ Thickness

    def _trim(self, ring: np.ndarray, new_ring: np.ndarray, distances: np.ndarray, normal: np.ndarray):
        """
        Soffit and fascia board per extended edge.

        The slab extends `thickness` downward, perpendicular to the roof surface. Soffit: strip wall -> outer edge
        (parallel to the roof surface, faces down). Fascia board: face at the outer edge (perpendicular to the roof
        surface, faces outward). (Ring counter-clockwise: outside is to the right of the direction of travel.)
        """
        drop = normal * self._thickness  # roof normal points up: the slab lies below
        count = len(ring)
        vertices, faces, base = [], [], 0
        for i in range(count):
            if distances[i] <= 0:
                continue
            j = (i + 1) % count
            a, b, a_new, b_new = ring[i], ring[j], new_ring[i], new_ring[j]
            soffit = [a - drop, b - drop, b_new - drop, a_new - drop]
            fascia = [a_new, a_new - drop, b_new - drop, b_new]
            vertices.extend(soffit + fascia)
            for start in (base, base + 4):
                faces.append([start, start + 1, start + 2])
                faces.append([start, start + 2, start + 3])
            base += 8
        if not vertices:
            return np.zeros((0, 3)), []
        return np.array(vertices), faces
