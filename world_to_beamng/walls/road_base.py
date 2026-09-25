"""
Height of the nearest road centerline for walls.

If a wall is at most `max_distance` meters from a centerline, it stands at road height (e.g. a retaining wall
at the roadside); otherwise on the terrain. The terrain is set exactly to the centerline height within the road,
but beside it drops off over the embankment - there the centerline and terrain heights differ.
"""

from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial import cKDTree

SAMPLE_STEP_M = 0.2  # spacing of the sample points on the centerline; height error thus at most gradient * 0.1 m


def centerlines_from_roads(roads: Sequence[Dict]) -> List[np.ndarray]:
    """Centerlines ((N, 3) x, y, z) of all road dicts with `trimmed_centerline` (at least 2 points)."""
    lines = []
    for road in roads:
        line = road.get("trimmed_centerline")
        if line is not None and len(line) >= 2:
            lines.append(np.asarray(line, dtype=float))
    return lines


class RoadBaseHeight:
    """Callable (x, y) -> height of the nearest centerline; NaN where none is within `max_distance`.

    The KD-tree (all centerlines in 0.2 m steps, millions of points for a 4 km export) is only built on the
    first call - without walls it is never needed.
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
        """Sample points spaced SAMPLE_STEP_M along the centerline, height linear between its points.

        Per segment steps = ceil(length / SAMPLE_STEP_M) (at least 1) points at the fractions 1/steps .. steps/steps -
        for all segments at once instead of segment by segment (same arithmetic steps, bit-identical result).
        """
        start, end = line[:-1], line[1:]
        lengths = np.linalg.norm(end[:, :2] - start[:, :2], axis=1)
        steps = np.maximum(1, np.ceil(lengths / SAMPLE_STEP_M).astype(int))
        segment = np.repeat(np.arange(len(steps)), steps)
        k = np.arange(len(segment)) - np.repeat(np.cumsum(steps) - steps, steps) + 1  # 1..steps per segment
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
