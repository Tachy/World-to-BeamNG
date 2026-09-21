"""
Höhe der nächsten Straßen-Centerline für Mauern.

Liegt eine Mauer höchstens `max_distance` Meter neben einer Centerline, steht sie auf Straßenhöhe (z. B. Stützmauer
am Straßenrand); sonst auf dem Gelände. Das Terrain ist innerhalb der Straße exakt auf die Centerline-Höhe gesetzt,
fällt daneben aber über die Böschung ab - dort weichen Centerline- und Geländehöhe voneinander ab.
"""

from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial import cKDTree

SAMPLE_STEP_M = 0.2  # Abstand der Stützpunkte auf der Centerline; Höhenfehler dadurch höchstens Steigung * 0.1 m


def centerlines_from_roads(roads: Sequence[Dict]) -> List[np.ndarray]:
    """Centerlines ((N, 3) x, y, z) aller Straßen-Dicts mit `trimmed_centerline` (mindestens 2 Punkte)."""
    lines = []
    for road in roads:
        line = road.get("trimmed_centerline")
        if line is not None and len(line) >= 2:
            lines.append(np.asarray(line, dtype=float))
    return lines


class RoadBaseHeight:
    """Callable (x, y) -> Höhe der nächsten Centerline; NaN, wo keine im Abstand `max_distance` liegt."""

    def __init__(self, centerlines: Sequence[np.ndarray], max_distance: float):
        self._max_distance = max_distance
        samples = [self._densify(np.asarray(line, dtype=float)) for line in centerlines]
        self._tree = cKDTree(np.vstack(samples)[:, :2]) if samples else None
        self._z = np.concatenate([s[:, 2] for s in samples]) if samples else np.empty(0)

    @staticmethod
    def _densify(line: np.ndarray) -> np.ndarray:
        """Stützpunkte im Abstand SAMPLE_STEP_M entlang der Centerline, Höhe linear zwischen ihren Punkten."""
        points = [line[:1]]
        for start, end in zip(line[:-1], line[1:]):
            steps = max(1, int(np.ceil(np.linalg.norm(end[:2] - start[:2]) / SAMPLE_STEP_M)))
            fractions = (np.arange(1, steps + 1) / steps)[:, None]
            points.append(start + (end - start) * fractions)
        return np.vstack(points)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if self._tree is None:
            return np.full(x.shape, np.nan)
        distance, index = self._tree.query(np.column_stack([x.ravel(), y.ravel()]), k=1)
        z = np.where(distance <= self._max_distance, self._z[index], np.nan)
        return z.reshape(x.shape)
