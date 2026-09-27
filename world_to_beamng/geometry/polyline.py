"""
Small numeric helpers shared by the road, structure, wall and texture code: arc length along a polyline and the
smoothstep blend curve.
"""

import numpy as np


def arc_lengths(points) -> np.ndarray:
    """
    Cumulative arc length per point (0.0 at the first point), measured over ALL columns of `points` - pass
    `points[:, :2]` for the plan length of a 3D line.
    """
    points = np.asarray(points, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite blend 3t² - 2t³ of `t` clipped to 0..1 (zero slope at both ends)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
