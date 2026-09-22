"""
Gemeinsame Portal-Geometrie für Tunnel: die Stirnfläche wird nicht rechtwinklig zur Tunnelachse abgeschnitten,
sondern an die natürliche Hangneigung angepasst (siehe Design-Spec Abschnitt 5). Die Heightmap selbst bleibt
dabei unverändert - nur der Portal-Rahmen (tunnels/tunnel_mesh.py::build_portal_frame_mesh()) wird entlang der
Achse verschoben, abhängig von der Höhe über dem Boden.
"""

from typing import Callable, Tuple

import numpy as np

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sample_slope_along_axis(ground_at: HeightAt, point_xy: Tuple[float, float], axis_direction: Tuple[float, float], sample_dist: float) -> float:
    """
    Hangneigung entlang `axis_direction` am Punkt (Steigung, positiv = Gelände wird in Achsrichtung höher).

    `axis_direction` ist ein Einheitsvektor; typischerweise zeigt er vom Portal INS Tunnelinnere.
    """
    ax, ay = axis_direction
    forward = (point_xy[0] + ax * sample_dist, point_xy[1] + ay * sample_dist)
    backward = (point_xy[0] - ax * sample_dist, point_xy[1] - ay * sample_dist)
    h_forward = float(ground_at(np.array([forward[0]]), np.array([forward[1]]))[0])
    h_backward = float(ground_at(np.array([backward[0]]), np.array([backward[1]]))[0])
    return (h_forward - h_backward) / (2.0 * sample_dist)


def portal_axial_shift(height_above_floor: float, slope_along_axis: float) -> float:
    """
    Achsversatz (in Metern, in Richtung `axis_direction`) eines Portal-Ring-Punkts je nach Höhe über dem Boden.

    Bei steigendem Gelände (slope_along_axis > 0, `axis_direction` zeigt ins Tunnelinnere) rückt die Decke
    weiter ins Tunnelinnere (positiver Versatz) als der Boden (Versatz 0) - der Rahmen wirkt, als wäre er schräg
    in den Hang gesetzt.
    """
    return slope_along_axis * height_above_floor
