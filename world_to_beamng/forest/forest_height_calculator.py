"""
Forest Height Calculator: Optimierte Höhen-Interpolation.

Nutzt gecachte KD-Tree und vektorisierte NumPy-Operationen für Performance.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class ForestHeightCalculator:
    """
    Berechnet Höhen für Baumpositionen mittels optimierter Interpolation.

    OPTIMIERUNGEN:
    1. KD-Tree wird einmal gebaut und cached
    2. Query wird vektorisiert (alle Punkte auf einmal, nicht in Schleife)
    3. Nutzt NumPy für Maximum Performance
    """

    def __init__(self):
        """Initialisiere HeightCalculator."""
        self._kdtree_cache = {}  # {id(height_points) → cKDTree}
        self._grid_cache = {}  # Gecachte Grid-Strukturen

    def _get_or_build_kdtree(self, height_points: np.ndarray) -> cKDTree:
        """
        Baue KD-Tree einmal und cache ihn.

        Args:
            height_points: Terrain-Grid Punkte

        Returns:
            Gecachter oder neuer cKDTree
        """
        cache_key = id(height_points)

        if cache_key not in self._kdtree_cache:
            logger.debug(f"Baue KD-Tree für {len(height_points)} Punkte...")
            self._kdtree_cache[cache_key] = cKDTree(height_points)

        return self._kdtree_cache[cache_key]

    def calculate_heights(
        self,
        points: List[Tuple[float, float]],
        height_points: np.ndarray,
        height_elevations: np.ndarray,
        grid_info: Optional[Dict] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Berechne Z-Koordinaten für (x, y) Punkte - OPTIMIERT.

        Nutzt gecachte KD-Tree und vektorisierte Query.

        Args:
            points: Liste von (x, y) Punkten
            height_points: numpy array (N, 2) mit (x, y) vom Terrain-Grid
            height_elevations: numpy array (N,) mit Z-Werten
            grid_info: Optional - Grid-Metadaten (ignoriert bei NN)

        Returns:
            Liste von (x, y, z) Punkten
        """
        if not points:
            return []

        if height_points is None or height_elevations is None:
            logger.warning("Keine Höhendaten verfügbar, nutze z=0")
            return [(x, y, 0.0) for x, y in points]

        # Konvertiere zu numpy
        points_array = np.array(points, dtype=np.float32)

        # Nutze gecachten KD-Tree - KRITISCH FÜR PERFORMANCE!
        tree = self._get_or_build_kdtree(height_points)

        # Vektorisierte Query - alle Punkte auf einmal!
        # Das ist VIEL schneller als in Schleife
        _, indices = tree.query(points_array, workers=-1)  # -1 = nutze alle CPU-Kerne

        heights = height_elevations[indices]

        # Kombiniere zu (x, y, z)
        result = [(float(points[i][0]), float(points[i][1]), float(heights[i])) for i in range(len(points))]

        logger.info(f"✓ Höhen für {len(result)} Punkte (min={np.min(heights):.1f}m, max={np.max(heights):.1f}m)")

        return result

    def calculate_heights_for_forest_points(
        self,
        forest_points: Dict[int, List[Tuple[float, float]]],
        height_points: np.ndarray,
        height_elevations: np.ndarray,
        grid_info: Optional[Dict] = None,
    ) -> Dict[int, List[Tuple[float, float, float]]]:
        """
        Berechne Höhen für mehrere Waldpolygone - OPTIMIERT.

        Nutzt gecachten KD-Tree für alle Forests!

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
                points=points,
                height_points=height_points,
                height_elevations=height_elevations,
                grid_info=grid_info,
            )
            result[forest_idx] = points_3d

        total_points = sum(len(pts) for pts in result.values())
        logger.info(f"✓ Höhen für {total_points} Baumpositionen interpoliert (mit Cache)")

        return result
