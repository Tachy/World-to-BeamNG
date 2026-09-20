"""
Dachüberstand von Schrägdächern.

Die LOD2-Dächer enden exakt an der Wand. Hier wird jede Dachkante, die außen über einer Wand liegt, in der Dachebene
nach außen verschoben: Traufe (waagerechte Kante) um `eave_m`, Giebelkante (Ortgang) um `verge_m`. Beide Maße sind
WAAGERECHT gemessen (wie am Bau üblich); an der Traufe wird deshalb in der Dachebene um `eave_m / cos(Neigung)`
verschoben. Kanten ohne Wand
darunter (Grate, Kehlen, Anschlüsse) und Brandwände bleiben unverändert. Der Überstand ist eine Platte von
`thickness_m` Dicke SENKRECHT zur Dachfläche (Rechteckprofil): die obere Fläche ist die verlängerte Dachfläche, dazu
kommen Untersicht und Stirnbrett.

Die Verschiebung ist ein Polygon-Offset mit eigenem Betrag je Kante; die Ecken werden verschnitten. Dadurch treffen sich
die Überstände zweier Dachflächen an einem Grat genau auf der (verlängerten) Gratlinie.
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from shapely.geometry import Polygon

from .. import config
from .edge_topology import WallLine, edge_is_exterior_over_wall, orient_counter_clockwise
from .ring_geometry import UP, newell_normal, unit_or_none

_HORIZONTAL_EDGE_SLOPE = 0.17  # |dz| / Länge darunter gilt eine Kante als waagerecht (~10 Grad)
_MIN_COS_SLOPE = 0.3  # sehr steile Dächer: der Traufüberstand wächst nicht über 1/0,3 hinaus
_MITER_LIMIT = 4.0  # Eckverschnitt darf höchstens so viele Überstandsbreiten von der Ecke entfernt liegen
_MIN_AREA_RATIO = 0.99  # der verschobene Umriss darf nie kleiner werden als der ursprüngliche


@dataclass
class SlopedRoof:
    """Dachfläche samt Überstand (Ring gegen den Uhrzeigersinn von oben) und dessen Stirnbrett/Untersicht."""

    ring: np.ndarray  # (M, 3)
    trim_vertices: np.ndarray  # (K, 3)
    trim_faces: List[List[int]]


class RoofOverhangBuilder:
    """Erzeugt für ein Schrägdachpolygon die verlängerte Dachfläche und die Dicke des Überstands."""

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
            ring: Dachring ohne Schlusspunkt
            index: Position dieses Rings in `roofs`
            roofs: alle Dachringe des Gebäudes (für gemeinsame Kanten)
            lines: Grundrisslinien der Wände des Gebäudes

        Returns:
            SlopedRoof; ohne Überstand (keine Kante über einer Wand) ist `ring` der Eingangsring, ggf. gedreht.
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

    # ------------------------------------------------------------------ Kanten

    def _edge_distances(self, ring, points, index, roofs, lines, cos_slope: float) -> np.ndarray:
        """Überstand je Kante i (von Punkt i zu i+1) in der Dachebene: Traufe, Giebel oder 0."""
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
            # Traufe: Außenrichtung zeigt in der Dachebene hangabwärts (-v); eine freie Oberkante zählt als Giebel
            distances[i] = self._eave / max(cos_slope, _MIN_COS_SLOPE) if outward[1] < 0 else self._verge
        return distances

    @staticmethod
    def _offset(points: np.ndarray, distances: np.ndarray):
        """Polygon-Offset in der Dachebene mit eigenem Betrag je Kante; None bei nicht berechenbarer Ecke."""
        count = len(points)
        tangents = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(tangents, axis=1)
        if (lengths < 1e-9).any():
            return None
        tangents = tangents / lengths[:, None]
        normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])  # rechts der Laufrichtung = außen (gegen den Uhrzeigersinn)
        limit = _MITER_LIMIT * float(distances.max())

        moved = np.empty_like(points)
        for j in range(count):
            prev, cur = (j - 1) % count, j
            a = points[j] + normals[prev] * distances[prev]
            b = points[j] + normals[cur] * distances[cur]
            cross = float(tangents[prev, 0] * tangents[cur, 1] - tangents[prev, 1] * tangents[cur, 0])
            if abs(cross) < 1e-6:  # (fast) parallele Nachbarkanten: gemittelt verschieben
                moved[j] = points[j] + (normals[prev] * distances[prev] + normals[cur] * distances[cur]) / 2
                continue
            step = float((b[0] - a[0]) * tangents[cur, 1] - (b[1] - a[1]) * tangents[cur, 0]) / cross
            corner = a + step * tangents[prev]
            if float(np.linalg.norm(corner - points[j])) > limit:  # sehr spitze Ecke: Verschnitt würde ausreißen
                return None
            moved[j] = corner
        return moved

    @staticmethod
    def _is_valid(points: np.ndarray, moved: np.ndarray) -> bool:
        """Der verschobene Umriss muss ein gültiges Polygon sein und die Fläche nicht verkleinern."""
        original, extended = Polygon(points), Polygon(moved)
        return bool(extended.is_valid and extended.area >= original.area * _MIN_AREA_RATIO)

    # ------------------------------------------------------------------ Dicke

    def _trim(self, ring: np.ndarray, new_ring: np.ndarray, distances: np.ndarray, normal: np.ndarray):
        """
        Untersicht und Stirnbrett je verlängerter Kante.

        Die Platte reicht `thickness` senkrecht zur Dachfläche nach unten. Untersicht: Streifen Wand -> Außenkante
        (parallel zur Dachfläche, zeigt nach unten). Stirnbrett: Fläche an der Außenkante (senkrecht zur Dachfläche,
        zeigt nach außen). (Ring gegen den Uhrzeigersinn: außen liegt rechts der Laufrichtung.)
        """
        drop = normal * self._thickness  # Dachnormale zeigt nach oben: die Platte liegt darunter
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
