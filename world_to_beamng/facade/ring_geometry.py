"""
Geometry helpers for CityGML rings (wall and roof polygons).
"""

import numpy as np

UP = np.array([0.0, 0.0, 1.0])
_MIN_NORMAL_LENGTH = 1e-9


def open_ring(verts: np.ndarray) -> np.ndarray:
    """
    Removes the closing point of a GML ring (last point = first point).

    Args:
        verts: (N, 3) ring points, possibly with a duplicate closing point

    Returns:
        (M, 3) ring points without closing point
    """
    if len(verts) > 3 and float(np.abs(verts[0] - verts[-1]).max()) < 1e-8:
        return verts[:-1]
    return verts


def newell_normal(ring: np.ndarray) -> np.ndarray:
    """
    Face normal of a (possibly concave) planar ring according to Newell.

    Args:
        ring: (N, 3) ring points without closing point

    Returns:
        Normal, length = twice the polygon area; (0, 0, 0) for a degenerate ring
    """
    c = ring - ring.mean(axis=0)
    n = np.roll(c, -1, axis=0)
    # Cross products written out: np.cross is much slower for many small rings
    return np.array(
        [
            (c[:, 1] * n[:, 2] - c[:, 2] * n[:, 1]).sum(),
            (c[:, 2] * n[:, 0] - c[:, 0] * n[:, 2]).sum(),
            (c[:, 0] * n[:, 1] - c[:, 1] * n[:, 0]).sum(),
        ]
    )


def unit_or_none(vector: np.ndarray):
    """Normalized vector, None for a (near-)zero vector."""
    length = np.linalg.norm(vector)
    if length < _MIN_NORMAL_LENGTH:
        return None
    return vector / length
