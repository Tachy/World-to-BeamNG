"""
Metrisches UV-Mapping für Dachflächen.

Die UVs werden in der Dachebene selbst gemessen (3D-Länge), nicht im Grundriss. Dadurch hat jeder Ziegel auf jedem
Dach dieselbe Breite (config.ROOF_TILE_WIDTH_M) und wird auf steilen Dächern nicht gestreckt.
"""

import numpy as np

from .. import config
from .ring_geometry import UP, newell_normal, open_ring, unit_or_none

# Unterhalb dieser horizontalen Normalen-Komponente (sin der Neigung, ~0,06°) gilt ein Dach als flach
_FLAT_ROOF_HORIZONTAL_NORMAL = 1e-3


class RoofUvMapper:
    """
    Berechnet Dach-UVs, längentreu bis auf den Faktor `repeat_m`.

    U läuft horizontal entlang der Traufe, V die Fallinie hinauf. Der Ursprung liegt an der Traufecke, damit die
    Ziegelreihen an der Traufe beginnen.
    """

    def __init__(self, repeat_m: float = config.ROOF_REPEAT_M):
        """
        Args:
            repeat_m: Meter in der Dachebene je Wiederholung der Textur
        """
        self._repeat_m = repeat_m

    def map_polygon(self, verts: np.ndarray) -> np.ndarray:
        """
        Args:
            verts: (N, 3) Weltkoordinaten eines Dachpolygons (Ring, mit oder ohne Schlusspunkt)

        Returns:
            (N, 2) UVs, eine Zeile je Eingabepunkt; Nullen bei entartetem Polygon
        """
        normal = unit_or_none(newell_normal(open_ring(verts)))
        if normal is None:
            return np.zeros((len(verts), 2))

        if normal[2] < 0:  # Ringrichtung ist nicht garantiert: Dachnormale zeigt immer nach oben
            normal = -normal

        u_axis, v_axis = self._axes(normal)
        u = verts @ u_axis
        v = verts @ v_axis
        return np.column_stack([u - u.min(), v - v.min()]) / self._repeat_m

    @staticmethod
    def _axes(normal: np.ndarray):
        """U horizontal entlang der Traufe, V die Fallinie hinauf (beide in der Dachebene)."""
        eave = unit_or_none(np.cross(UP, normal))
        if eave is None or np.hypot(normal[0], normal[1]) < _FLAT_ROOF_HORIZONTAL_NORMAL:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        return eave, unit_or_none(np.cross(normal, eave))
