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
    """Callable (x, y) -> Höhe der nächsten Centerline; NaN, wo keine im Abstand `max_distance` liegt.

    Der KD-Baum (alle Centerlines in 0,2-m-Schritten, bei einem 4-km-Export Millionen Punkte) entsteht erst beim
    ersten Aufruf - ohne Mauern wird er nie gebraucht.
    """

    def __init__(self, centerlines: Sequence[np.ndarray], max_distance: float):
        self._max_distance = max_distance
        self._centerlines = list(centerlines)
        self._tree = None
        self._z = None

    def _build(self):
        samples = [self._densify(np.asarray(line, dtype=float)) for line in self._centerlines]
        self._tree = cKDTree(np.vstack(samples)[:, :2]) if samples else None
        self._z = np.concatenate([s[:, 2] for s in samples]) if samples else np.empty(0)
        self._centerlines = None

    @staticmethod
    def _densify(line: np.ndarray) -> np.ndarray:
        """Stützpunkte im Abstand SAMPLE_STEP_M entlang der Centerline, Höhe linear zwischen ihren Punkten.

        Je Segment steps = ceil(Länge / SAMPLE_STEP_M) (mindestens 1) Punkte bei den Anteilen 1/steps .. steps/steps -
        für alle Segmente auf einmal statt Segment für Segment (gleiche Rechenschritte, bitgleiches Ergebnis).
        """
        start, end = line[:-1], line[1:]
        lengths = np.linalg.norm(end[:, :2] - start[:, :2], axis=1)
        steps = np.maximum(1, np.ceil(lengths / SAMPLE_STEP_M).astype(int))
        segment = np.repeat(np.arange(len(steps)), steps)
        k = np.arange(len(segment)) - np.repeat(np.cumsum(steps) - steps, steps) + 1  # 1..steps je Segment
        fractions = (k / steps[segment])[:, None]
        points = start[segment] + (end[segment] - start[segment]) * fractions
        return np.vstack([line[:1], points])

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if self._centerlines is not None:
            self._build()
        if self._tree is None:
            return np.full(x.shape, np.nan)
        distance, index = self._tree.query(np.column_stack([x.ravel(), y.ravel()]), k=1)
        z = np.where(distance <= self._max_distance, self._z[index], np.nan)
        return z.reshape(x.shape)
