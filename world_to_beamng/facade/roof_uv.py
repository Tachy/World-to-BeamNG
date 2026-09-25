"""
Metric UV mapping for roof surfaces.

The UVs are measured in the roof plane itself (3D length), not in the footprint. As a result, every tile on every
roof has the same width (config.ROOF_TILE_WIDTH_M) and is not stretched on steep roofs.
"""

import numpy as np

from .. import config
from .ring_geometry import UP, newell_normal, open_ring, unit_or_none

# Below this horizontal normal component (sin of the pitch, ~0.06°) a roof counts as flat
_FLAT_ROOF_HORIZONTAL_NORMAL = 1e-3


class RoofUvMapper:
    """
    Computes roof UVs, length-preserving up to the factor `repeat_m`.

    U runs horizontally along the eave, V up the fall line. The origin lies at the eave corner so that the
    tile courses start at the eave.
    """

    def __init__(self, repeat_m: float = config.ROOF_REPEAT_M):
        """
        Args:
            repeat_m: meters in the roof plane per texture repeat
        """
        self._repeat_m = repeat_m

    def map_polygon(self, verts: np.ndarray) -> np.ndarray:
        """
        Args:
            verts: (N, 3) world coordinates of a roof polygon (ring, with or without closing point)

        Returns:
            (N, 2) UVs, one row per input point; zeros for a degenerate polygon
        """
        normal = unit_or_none(newell_normal(open_ring(verts)))
        if normal is None:
            return np.zeros((len(verts), 2))

        if normal[2] < 0:  # ring direction is not guaranteed: the roof normal always points up
            normal = -normal

        u_axis, v_axis = self._axes(normal)
        u = verts @ u_axis
        v = verts @ v_axis
        return np.column_stack([u - u.min(), v - v.min()]) / self._repeat_m

    @staticmethod
    def _axes(normal: np.ndarray):
        """U horizontal along the eave, V up the fall line (both in the roof plane)."""
        eave = unit_or_none(np.cross(UP, normal))
        if eave is None or np.hypot(normal[0], normal[1]) < _FLAT_ROOF_HORIZONTAL_NORMAL:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        return eave, unit_or_none(np.cross(normal, eave))
