"""
Geometrie-Helfer für CityGML-Ringe (Wand- und Dachpolygone).
"""

import numpy as np

UP = np.array([0.0, 0.0, 1.0])
_MIN_NORMAL_LENGTH = 1e-9


def open_ring(verts: np.ndarray) -> np.ndarray:
    """
    Entfernt den Schlusspunkt eines GML-Rings (letzter Punkt = erster Punkt).

    Args:
        verts: (N, 3) Ringpunkte, ggf. mit doppeltem Schlusspunkt

    Returns:
        (M, 3) Ringpunkte ohne Schlusspunkt
    """
    if len(verts) > 3 and float(np.abs(verts[0] - verts[-1]).max()) < 1e-8:
        return verts[:-1]
    return verts


def newell_normal(ring: np.ndarray) -> np.ndarray:
    """
    Flächennormale eines (auch konkaven) planaren Rings nach Newell.

    Args:
        ring: (N, 3) Ringpunkte ohne Schlusspunkt

    Returns:
        Normale, Länge = doppelte Polygonfläche; (0, 0, 0) bei entartetem Ring
    """
    c = ring - ring.mean(axis=0)
    n = np.roll(c, -1, axis=0)
    # Kreuzprodukte ausgeschrieben: np.cross ist bei vielen kleinen Ringen deutlich langsamer
    return np.array(
        [
            (c[:, 1] * n[:, 2] - c[:, 2] * n[:, 1]).sum(),
            (c[:, 2] * n[:, 0] - c[:, 0] * n[:, 2]).sum(),
            (c[:, 0] * n[:, 1] - c[:, 1] * n[:, 0]).sum(),
        ]
    )


def unit_or_none(vector: np.ndarray):
    """Normierter Vektor, None bei (fast) Nullvektor."""
    length = np.linalg.norm(vector)
    if length < _MIN_NORMAL_LENGTH:
        return None
    return vector / length
