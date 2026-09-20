"""
Kanten-Nachbarschaft von Dachpolygonen und Wänden eines Gebäudes.

Gemeinsam genutzt vom Blechrand der Flachdächer und vom Dachüberstand der Schrägdächer: beide gelten nur für Dachkanten,
die außen über einer Wand liegen (nicht für Grate, Kehlen, Dachstufen oder Brandwände).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .ring_geometry import open_ring

EDGE_MATCH_TOLERANCE_M = 0.05  # Kanten zweier Dachpolygone, die im Grundriss so nah liegen, sind "gemeinsam"
WALL_LINE_TOLERANCE_M = 0.10  # Dachkante liegt "über" einer Wand, wenn beide Endpunkte so nah an deren Grundrisslinie sind
WALL_ABOVE_MARGIN_M = 0.3  # Wand muss so weit über die Dachkante reichen, damit sie als Brandwand/Attika gilt


@dataclass(frozen=True)
class WallLine:
    """Grundrisslinie einer (senkrechten) Wand samt Höhe ihrer Oberkante."""

    origin: np.ndarray  # (2,) einer der beiden Endpunkte
    direction: np.ndarray  # (2,) Einheitsvektor
    length: float
    z_max: float

    def contains(self, point_xy: np.ndarray, tolerance: float = WALL_LINE_TOLERANCE_M) -> bool:
        offset = point_xy - self.origin
        along = float(offset @ self.direction)
        across = abs(float(offset[0] * self.direction[1] - offset[1] * self.direction[0]))
        return across <= tolerance and -tolerance <= along <= self.length + tolerance


def wall_lines(building: dict) -> List[WallLine]:
    """Grundrisslinien aller Wände (Strecke zwischen den beiden am weitesten entfernten Punkten der Wand)."""
    lines = []
    for verts, _ in building.get("walls", []):
        ring = open_ring(verts)
        xy = ring[:, :2]
        span = xy[:, None, :] - xy[None, :, :]
        i, j = np.unravel_index(np.argmax(np.einsum("ijk,ijk->ij", span, span)), (len(xy), len(xy)))
        vector = xy[j] - xy[i]
        length = float(np.linalg.norm(vector))
        if length < 1e-6:
            continue
        lines.append(WallLine(xy[i], vector / length, length, float(ring[:, 2].max())))
    return lines


def roof_rings(building: dict) -> List[np.ndarray]:
    """Dachringe ohne Schlusspunkt."""
    return [open_ring(verts) for verts, _ in building.get("roofs", [])]


def is_shared(a: np.ndarray, b: np.ndarray, roofs: Sequence[np.ndarray], own_index: int) -> bool:
    """Kante a->b fällt im Grundriss mit einer Kante eines ANDEREN Dachpolygons zusammen (Dachstufe, Anschluss)."""
    tolerance = EDGE_MATCH_TOLERANCE_M
    for index, other in enumerate(roofs):
        if index == own_index:
            continue
        for i in range(len(other)):
            p, q = other[i], other[(i + 1) % len(other)]
            same = np.linalg.norm(a[:2] - p[:2]) <= tolerance and np.linalg.norm(b[:2] - q[:2]) <= tolerance
            swapped = np.linalg.norm(a[:2] - q[:2]) <= tolerance and np.linalg.norm(b[:2] - p[:2]) <= tolerance
            if same or swapped:
                return True
    return False


def walls_under_edge(a: np.ndarray, b: np.ndarray, lines: Sequence[WallLine]) -> List[WallLine]:
    """Wände, deren Grundrisslinie die Dachkante a->b trägt (beide Endpunkte und Mittelpunkt liegen darauf)."""
    middle = (a[:2] + b[:2]) / 2
    return [line for line in lines if all(line.contains(point) for point in (a[:2], middle, b[:2]))]


def is_covered_by_higher_wall(a: np.ndarray, b: np.ndarray, lines: Sequence[WallLine]) -> bool:
    """Eine Wand läuft an dieser Kante deutlich über das Dach hinaus (Brandwand, Attika)."""
    top = max(float(a[2]), float(b[2])) + WALL_ABOVE_MARGIN_M
    return any(line.z_max > top for line in walls_under_edge(a, b, lines))


def edge_is_exterior_over_wall(
    a: np.ndarray, b: np.ndarray, roofs: Sequence[np.ndarray], own_index: int, lines: Sequence[WallLine]
) -> bool:
    """Dachkante liegt außen über einer Wand: keine Nachbarfläche, eine Wand darunter, keine höhere Wand."""
    if is_shared(a, b, roofs, own_index):
        return False
    under = walls_under_edge(a, b, lines)
    if not under:
        return False
    return not is_covered_by_higher_wall(a, b, lines)


def orient_counter_clockwise(ring: np.ndarray) -> Optional[np.ndarray]:
    """Ring gegen den Uhrzeigersinn von oben (Fläche im Grundriss > 0); None bei entartetem Ring."""
    xy = ring[:, :2]
    signed_area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    if abs(signed_area) < 1e-9:
        return None
    return ring if signed_area > 0 else ring[::-1]
