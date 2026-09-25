"""
Forest Height Calculator: optimized height interpolation.

Uses a cached KD-tree and vectorized NumPy operations for performance.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class ForestHeightCalculator:
    """
    Computes heights for tree positions using optimized interpolation.

    OPTIMIZATIONS:
    1. The KD-tree is built once and cached
    2. The query is vectorized (all points at once, not in a loop)
    3. Uses NumPy for maximum performance
    """

    def __init__(self):
        """Initialize the HeightCalculator."""
        self._kdtree_cache = {}  # {id(height_points) → cKDTree}

    def _get_or_build_kdtree(self, height_points: np.ndarray) -> cKDTree:
        """
        Build the KD-tree once and cache it.

        Args:
            height_points: terrain grid points

        Returns:
            Cached or new cKDTree
        """
        cache_key = id(height_points)

        if cache_key not in self._kdtree_cache:
            logger.debug(f"Building KD-tree for {len(height_points)} points...")
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
        Compute Z coordinates for (x, y) points - OPTIMIZED.

        Uses a cached KD-tree and a vectorized query.

        Args:
            points: list of (x, y) points
            height_points: numpy array (N, 2) with (x, y) from the terrain grid
            height_elevations: numpy array (N,) with Z values
            grid_info: optional - grid metadata (ignored for NN)

        Returns:
            List of (x, y, z) points
        """
        if not points:
            return []

        if height_points is None or height_elevations is None:
            logger.warning("No height data available, using z=0")
            return [(x, y, 0.0) for x, y in points]

        # Convert to numpy
        points_array = np.array(points, dtype=np.float32)

        # Use the cached KD-tree - CRITICAL FOR PERFORMANCE!
        tree = self._get_or_build_kdtree(height_points)

        # Vectorized query - all points at once!
        # This is MUCH faster than a loop
        _, indices = tree.query(points_array, workers=-1)  # -1 = use all CPU cores

        heights = height_elevations[indices]

        # Combine into (x, y, z)
        result = [(float(points[i][0]), float(points[i][1]), float(heights[i])) for i in range(len(points))]

        logger.info(f"✓ Heights for {len(result)} points (min={np.min(heights):.1f}m, max={np.max(heights):.1f}m)")

        return result

    def calculate_heights_from_sampler(
        self,
        points: List[Tuple[float, float]],
        height_at,
    ) -> List[Tuple[float, float, float]]:
        """
        Z coordinates from a height query of the FINISHED terrain heightmap (bilinear, after road embedding).

        This way the trees stand exactly on what BeamNG renders - not on the raw DGM1 points
        (nearest neighbor deviated by more than 1 m on slopes, and likewise on embedded roads).

        Args:
            points: list of (x, y) points
            height_at: callable (x_array, y_array) -> z_array (see make_height_sampler)
        """
        if not points:
            return []
        xy = np.asarray(points, dtype=float)
        z = np.asarray(height_at(xy[:, 0], xy[:, 1]), dtype=float)
        return [(float(x), float(y), float(h)) for (x, y), h in zip(points, z)]

    def calculate_heights_for_forest_points(
        self,
        forest_points: Dict[int, List[Tuple[float, float]]],
        height_points: np.ndarray,
        height_elevations: np.ndarray,
        grid_info: Optional[Dict] = None,
        height_at=None,
    ) -> Dict[int, List[Tuple[float, float, float]]]:
        """
        Compute heights for multiple forest polygons - OPTIMIZED.

        Uses the cached KD-tree for all forests!

        Args:
            forest_points: dict forest_index → list of (x, y) points
            height_points: terrain grid points
            height_elevations: terrain grid Z values
            grid_info: optional - grid metadata
            height_at: optional - height query of the finished heightmap; takes precedence over the raw points

        Returns:
            Dict forest_index → list of (x, y, z) points
        """
        result = {}

        for forest_idx, points in forest_points.items():
            if height_at is not None:
                points_3d = self.calculate_heights_from_sampler(points, height_at)
            else:
                points_3d = self.calculate_heights(
                    points=points,
                    height_points=height_points,
                    height_elevations=height_elevations,
                    grid_info=grid_info,
                )
            result[forest_idx] = points_3d

        total_points = sum(len(pts) for pts in result.values())
        logger.info(f"✓ Heights interpolated for {total_points} tree positions (with cache)")

        return result
