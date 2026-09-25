"""
Terrain grid generation.
"""

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from .. import config
from .elevation import get_height_data_hash
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def create_terrain_grid(height_points, height_elevations, grid_spacing=10.0, tile_hash=None):
    """Creates a regular grid from the elevation data (OPTIMIZED with caching).

    Args:
        height_points: XY coordinates (local coordinates)
        height_elevations: Z values
        grid_spacing: Grid spacing in meters
        tile_hash: Optional - tile_hash for cache consistency (multi-tile mode)
    """
    logger.info(f"  Creating terrain grid (spacing: {grid_spacing}m)...")

    # Grid bounds were already set in world_to_beamng.py (from height_points)
    # Only determined here for internal calculations
    min_x, max_x = height_points[:, 0].min(), height_points[:, 0].max()
    min_y, max_y = height_points[:, 1].min(), height_points[:, 1].max()

    # Check whether a cached grid exists (version 3 with correct bounds!)
    # Use the passed tile_hash or fall back to the global hash
    effective_hash = tile_hash or get_height_data_hash()
    if effective_hash:
        cache_file = config.CACHE_DIR / f"grid_v3_{effective_hash}_spacing{grid_spacing:.1f}m.npz"

        if cache_file.exists():
            logger.info(f"  [OK] Grid cache found: {cache_file.name}")
            data = np.load(cache_file)
            grid_points = data["grid_points"]
            grid_elevations = data["grid_elevations"]
            nx = int(data["nx"])
            ny = int(data["ny"])
            logger.info(f"  [OK] Grid loaded from cache: {nx} x {ny} = {len(grid_points)} vertices")
            # IMPORTANT: the grid was cached in UTM, transform to local!
            # (height_points were already transformed, min_x/min_y are local)
            # Nothing to do here - grid_points are already in the same system as height_points
            return grid_points, grid_elevations, nx, ny

    # Create grid points (inclusive of max_x and max_y!)
    # IMPORTANT: np.arange does not include max, hence + grid_spacing
    x_coords = np.arange(min_x, max_x + grid_spacing * 0.5, grid_spacing)
    y_coords = np.arange(min_y, max_y + grid_spacing * 0.5, grid_spacing)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Interpolate heights for grid points (CHUNKED for better performance)
    logger.info(f"  Creating interpolator...")
    interpolator = NearestNDInterpolator(height_points, height_elevations)

    logger.info(f"  Interpolating {len(grid_points)} grid points (in chunks)...")
    chunk_size = 500000  # 500k points per chunk
    grid_elevations = np.empty(len(grid_points), dtype=np.float64)

    num_chunks = (len(grid_points) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(grid_points))
        grid_elevations[start_idx:end_idx] = interpolator(grid_points[start_idx:end_idx])

        if (i + 1) % 5 == 0 or i == num_chunks - 1:
            progress = ((i + 1) / num_chunks) * 100
            logger.info(f"    {progress:.0f}% ({i + 1}/{num_chunks} Chunks)")

    nx = len(x_coords)
    ny = len(y_coords)
    logger.info(f"  Grid: {nx} x {ny} = {len(grid_points)} vertices")

    # Cache the grid for future use (version 3 with correct bounds!)
    if effective_hash:
        cache_file = config.CACHE_DIR / f"grid_v3_{effective_hash}_spacing{grid_spacing:.1f}m.npz"
        logger.info(f"  Saving grid cache: {cache_file.name}")
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            grid_points=grid_points,
            grid_elevations=grid_elevations,
            nx=nx,
            ny=ny,
        )
        logger.info(f"  [OK] Grid cache created")

    return grid_points, grid_elevations, nx, ny
