"""
Triangulierung einfacher (auch konkaver) Polygone.
"""

from typing import List

import mapbox_earcut
import numpy as np


def triangulate_ccw(points: np.ndarray) -> List[List[int]]:
    """
    Dreiecke eines einfachen Polygons als Indizes in `points`, jedes gegen den Uhrzeigersinn.

    Args:
        points: (N, 2) Ring ohne Schlusspunkt

    Returns:
        Liste von [a, b, c]; entartete (flächenlose) Dreiecke fehlen. Leer bei weniger als 3 Punkten.
    """
    if len(points) < 3:
        return []
    if len(points) == 3:
        triangles = [[0, 1, 2]]
    else:
        indices = mapbox_earcut.triangulate_float64(points.astype(np.float64), np.array([len(points)], dtype=np.uint32))
        triangles = indices.reshape(-1, 3).tolist()

    result = []
    for a, b, c in triangles:
        cross = (points[b, 0] - points[a, 0]) * (points[c, 1] - points[a, 1]) - (points[b, 1] - points[a, 1]) * (
            points[c, 0] - points[a, 0]
        )
        if abs(cross) < 1e-12:
            continue
        result.append([a, b, c] if cross > 0 else [a, c, b])
    return result
