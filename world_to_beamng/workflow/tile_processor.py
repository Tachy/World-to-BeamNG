"""
Tile processing logic.

Extracts the tile loading and processing logic from multitile.py.
"""

from world_to_beamng.logging_config import LoggerConfig
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from ..core.cache_manager import CacheManager
from ..terrain.elevation_io import read_elevation_tile_cached, reproject_points

logger = LoggerConfig.get_logger()


class TileProcessor:
    """
    Processes individual DGM tiles.

    Responsible for:
    - Loading elevation data
    - Caching
    - Coordinate transformation
    """

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def load_height_data(
        self, tile: Dict, tile_hash: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load the elevation data of a DGM1 tile (with cache).

        Args:
            tile: Tile metadata dict
            tile_hash: Optional - hash for the cache

        Returns:
            Tuple (height_points, height_elevations) or (None, None)
        """
        filepath = tile.get("filepath")
        if not filepath or not Path(filepath).exists():
            logger.error(f"  [!] DGM1 file missing: {filepath}")
            return None, None

        logger.info(f"  [→] Loading DGM1: {Path(filepath).name}")
        points, elevations, _crs_epsg, _bbox_utm = read_elevation_tile_cached(filepath, self.cache, tile_hash)

        if points is None or elevations is None:
            return None, None

        if tile.get("reproject_from") is not None:  # tile in a different CRS, see utils.tile_scanner.align_tiles_to_crs()
            points = reproject_points(points, tile["reproject_from"], tile["target_epsg"])

        return points, elevations

    def load_height_data_multi(self, tiles: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Loads and combines the elevation data of several tiles into a
        single point cloud (vstack/hstack) - the same combination logic as
        when loading several XYZ files within a single tile ZIP
        (elevation_io.read_elevation_tile), just one level higher for multiple files.

        Assumes that the tiles form a gapless, rectangular
        area (user responsibility, see utils.tile_scanner).

        Args:
            tiles: List of tile metadata dicts (as returned by scan_elevation_tiles())

        Returns:
            Tuple (points, elevations) or (None, None) if a tile fails
        """
        all_points = []
        all_elevations = []

        for tile in tiles:
            points, elevations = self.load_height_data(tile)
            if points is None:
                logger.error(f"  [!] Height data for {tile.get('filename')} missing - total area incomplete")
                return None, None
            all_points.append(points)
            all_elevations.append(elevations)

        if not all_points:
            return None, None

        return np.vstack(all_points), np.hstack(all_elevations)

    def ensure_local_offset(
        self, global_offset: Tuple[float, float], height_points: np.ndarray, height_elevations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform global coordinates to local ones (relative to global_offset).

        Args:
            global_offset: (origin_x, origin_y) global offset
            height_points: N×2 array of points
            height_elevations: N array of elevations

        Returns:
            Tuple (local_points, elevations)
        """
        origin_x, origin_y = global_offset

        local_points = height_points.copy()
        local_points[:, 0] -= origin_x
        local_points[:, 1] -= origin_y

        return local_points, height_elevations
