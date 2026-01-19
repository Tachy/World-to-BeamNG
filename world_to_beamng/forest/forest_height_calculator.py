"""
Forest Height Calculator: Bilineare Interpolation für Baum-Höhen.

Interpoliert Z-Koordinaten (Höhe) für Baumpositionen (x, y) aus dem
Terrain-Elevation-Grid mittels bilinearer Interpolation.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class ForestHeightCalculator:
    """
    Berechnet Höhen für Baumpositionen mittels bilinearer Interpolation.

    Nutzt das Terrain-Elevation-Grid vom TerrainWorkflow.
    """

    def __init__(self):
        """Initialisiere HeightCalculator."""
        pass

    def calculate_heights(
        self,
        points: List[Tuple[float, float]],
        height_points: np.ndarray,
        height_elevations: np.ndarray,
        grid_info: Optional[Dict] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Berechne Z-Koordinaten für (x, y) Punkte.

        Mit 1m×1m Auflösung: Nearest-Neighbor statt teurer Bilinearinterpolation!

        Args:
            points: Liste von (x, y) Punkten
            height_points: numpy array (N, 2) mit (x, y) vom Terrain-Grid
            height_elevations: numpy array (N,) mit Z-Werten
            grid_info: Optional - ignoriert (1m Auflösung reicht für NN)

        Returns:
            Liste von (x, y, z) Punkten
        """
        if not points:
            return []

        if height_points is None or height_elevations is None:
            logger.warning("Keine Höhendaten verfügbar, nutze z=0")
            return [(x, y, 0.0) for x, y in points]

        # Konvertiere zu numpy für Performance
        points_array = np.array(points, dtype=float)

        # Nearest-Neighbor: Schnell und ausreichend für 1m×1m Grid!
        from scipy.spatial import cKDTree

        tree = cKDTree(height_points)
        _, indices = tree.query(points_array)

        heights = height_elevations[indices]

        # Kombiniere zu (x, y, z)
        result = [(points[i][0], points[i][1], float(heights[i])) for i in range(len(points))]

        logger.info(
            f"✓ Höhen interpoliert: {len(result)} Punkte (min={np.min(heights):.1f}m, max={np.max(heights):.1f}m)"
        )

        return result

    def _interpolate_bilinear(
        self, query_points: np.ndarray, grid_points: np.ndarray, grid_values: np.ndarray
    ) -> np.ndarray:
        """
        Bilineare Interpolation von Grid-Punkten zu Query-Punkten.

        Args:
            query_points: (N, 2) array mit (x, y) zu interpolierenden Punkten
            grid_points: (M, 2) array mit (x, y) Grid-Punkten
            grid_values: (M,) array mit Z-Werten

        Returns:
            (N,) array mit interpolierten Z-Werten
        """
        # Extrahiere Grid-Struktur
        unique_x = np.unique(grid_points[:, 0])
        unique_y = np.unique(grid_points[:, 1])

        # Grid muss regelmäßig sein
        if len(unique_x) < 2 or len(unique_y) < 2:
            logger.warning("Grid hat zu wenige Punkte für Interpolation")
            return np.zeros(len(query_points))

        # Sortiere Grid
        unique_x = np.sort(unique_x)
        unique_y = np.sort(unique_y)

        # Erstelle 2D-Grid aus Punkten
        grid_2d = self._create_grid_2d(grid_points, grid_values, unique_x, unique_y)

        # Interpoliere für jeden Query-Punkt
        result = np.zeros(len(query_points))

        for i, (qx, qy) in enumerate(query_points):
            result[i] = self._interpolate_point(qx, qy, unique_x, unique_y, grid_2d)

        return result

    def _create_grid_2d(
        self, grid_points: np.ndarray, grid_values: np.ndarray, unique_x: np.ndarray, unique_y: np.ndarray
    ) -> np.ndarray:
        """
        Erstelle 2D-Grid aus unstrukturierten Punkten.

        Args:
            grid_points: (M, 2) array
            grid_values: (M,) array
            unique_x: Sortierte unique X-Werte
            unique_y: Sortierte unique Y-Werte

        Returns:
            (len(unique_x), len(unique_y)) array
        """
        grid_2d = np.zeros((len(unique_x), len(unique_y)))

        # Erstelle Lookup-Dict für schnellen Zugriff
        point_dict = {}
        for i, (x, y) in enumerate(grid_points):
            point_dict[(x, y)] = grid_values[i]

        # Fülle Grid
        for ix, x in enumerate(unique_x):
            for iy, y in enumerate(unique_y):
                key = (x, y)
                if key in point_dict:
                    grid_2d[ix, iy] = point_dict[key]
                else:
                    # Nearest-Neighbor-Fallback
                    grid_2d[ix, iy] = self._find_nearest_value(x, y, grid_points, grid_values)

        return grid_2d

    def _interpolate_point(
        self, qx: float, qy: float, unique_x: np.ndarray, unique_y: np.ndarray, grid_2d: np.ndarray
    ) -> float:
        """
        Bilineare Interpolation für einen einzelnen Punkt.

        Args:
            qx, qy: Query-Punkt
            unique_x: Sortierte Grid X-Werte
            unique_y: Sortierte Grid Y-Werte
            grid_2d: 2D-Grid mit Z-Werten

        Returns:
            Interpolierter Z-Wert
        """
        # Finde umgebende Grid-Punkte
        ix = np.searchsorted(unique_x, qx, side="right") - 1
        iy = np.searchsorted(unique_y, qy, side="right") - 1

        # Clamp zu Grid-Grenzen
        ix = max(0, min(ix, len(unique_x) - 2))
        iy = max(0, min(iy, len(unique_y) - 2))

        # Grid-Zell-Ecken
        x0, x1 = unique_x[ix], unique_x[ix + 1]
        y0, y1 = unique_y[iy], unique_y[iy + 1]

        # Z-Werte an den 4 Ecken
        z00 = grid_2d[ix, iy]
        z10 = grid_2d[ix + 1, iy]
        z01 = grid_2d[ix, iy + 1]
        z11 = grid_2d[ix + 1, iy + 1]

        # Normalisierte Koordinaten (0-1)
        dx = (qx - x0) / (x1 - x0) if x1 != x0 else 0.0
        dy = (qy - y0) / (y1 - y0) if y1 != y0 else 0.0

        # Bilineare Interpolation
        z0 = z00 * (1 - dx) + z10 * dx
        z1 = z01 * (1 - dx) + z11 * dx
        z = z0 * (1 - dy) + z1 * dy

        return z

    def _find_nearest_value(self, x: float, y: float, grid_points: np.ndarray, grid_values: np.ndarray) -> float:
        """
        Finde nächsten Grid-Wert (Fallback für fehlende Grid-Punkte).

        Args:
            x, y: Query-Punkt
            grid_points: (M, 2) array
            grid_values: (M,) array

        Returns:
            Z-Wert des nächsten Punkts
        """
        distances = np.sqrt((grid_points[:, 0] - x) ** 2 + (grid_points[:, 1] - y) ** 2)
        nearest_idx = np.argmin(distances)
        return grid_values[nearest_idx]

    def calculate_heights_for_forest_points(
        self,
        forest_points: Dict[int, List[Tuple[float, float]]],
        height_points: np.ndarray,
        height_elevations: np.ndarray,
        grid_info: Optional[Dict] = None,
    ) -> Dict[int, List[Tuple[float, float, float]]]:
        """
        Berechne Höhen für mehrere Waldpolygone.

        Args:
            forest_points: Dict forest_index → Liste von (x, y) Punkten
            height_points: Terrain-Grid Punkte
            height_elevations: Terrain-Grid Z-Werte
            grid_info: Optional - Grid-Metadaten

        Returns:
            Dict forest_index → Liste von (x, y, z) Punkten
        """
        result = {}

        for forest_idx, points in forest_points.items():
            points_3d = self.calculate_heights(
                points=points, height_points=height_points, height_elevations=height_elevations, grid_info=grid_info
            )
            result[forest_idx] = points_3d

        total_points = sum(len(pts) for pts in result.values())
        logger.info(f"✓ Höhen für {total_points} Baumpositionen interpoliert")

        return result
