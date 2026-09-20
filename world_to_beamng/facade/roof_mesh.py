"""
Dachgeometrie eines Gebäudes: Schrägdächer (mit Überstand), Flachdächer (Kies) und die Dicke der Überstände.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .. import config
from .edge_topology import orient_counter_clockwise, roof_rings, wall_lines
from .flat_roof import is_flat_roof
from .roof_overhang import RoofOverhangBuilder
from .roof_uv import RoofUvMapper
from .triangulate import triangulate_ccw


@dataclass
class MeshPart:
    """Vertices, UVs und Dreiecke eines Materials."""

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    uvs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    faces: List[List[int]] = field(default_factory=list)


class _PartBuilder:
    def __init__(self):
        self._vertices: List[np.ndarray] = []
        self._uvs: List[np.ndarray] = []
        self._faces: List[List[int]] = []
        self._count = 0

    def add(self, vertices: np.ndarray, uvs: np.ndarray, faces: List[List[int]]) -> None:
        self._vertices.append(vertices)
        self._uvs.append(uvs)
        self._faces.extend([[i + self._count for i in face] for face in faces])
        self._count += len(vertices)

    def build(self) -> MeshPart:
        if not self._vertices:
            return MeshPart()
        return MeshPart(np.vstack(self._vertices), np.vstack(self._uvs), self._faces)


@dataclass
class RoofParts:
    sloped: MeshPart  # Biberschwanz, mit Überstand
    flat: MeshPart  # Kiesfläche
    trim: MeshPart  # Stirnbrett und Untersicht der Überstände


class RoofMeshBuilder:
    """Baut alle Dachflächen eines Gebäude-Dicts (`roofs`, `walls`)."""

    def __init__(self):
        self._overhang = RoofOverhangBuilder()
        self._tile_uv = RoofUvMapper()
        self._gravel_uv = RoofUvMapper(repeat_m=config.FLAT_ROOF_GRAVEL_REPEAT_M)

    def build(self, building) -> RoofParts:
        roofs = roof_rings(building)
        lines = wall_lines(building)
        sloped, flat, trim = _PartBuilder(), _PartBuilder(), _PartBuilder()

        for index, ring in enumerate(roofs):
            if is_flat_roof(ring):
                ordered = orient_counter_clockwise(ring)
                if ordered is None:
                    continue
                self._add_polygon(flat, ordered, self._gravel_uv)
                continue

            roof = self._overhang.build(ring, index, roofs, lines)
            self._add_polygon(sloped, roof.ring, self._tile_uv)
            if len(roof.trim_vertices):
                trim.add(roof.trim_vertices, np.zeros((len(roof.trim_vertices), 2)), roof.trim_faces)

        return RoofParts(sloped.build(), flat.build(), trim.build())

    @staticmethod
    def _add_polygon(builder: _PartBuilder, ring: np.ndarray, mapper: RoofUvMapper) -> None:
        """Ring (gegen den Uhrzeigersinn von oben) triangulieren; die längentreuen UVs dienen als ebene Koordinaten."""
        uvs = mapper.map_polygon(ring)
        faces = triangulate_ccw(uvs)
        if faces:
            builder.add(ring, uvs, faces)
