"""
Builds the square .ter heightmap array directly from the existing
elevation grid (terrain.grid.create_terrain_grid) - replaces the previous
triangulation in TerrainMeshBuilder.
"""

import numpy as np

VALID_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)


def next_power_of_two_size(min_size: int) -> int:
    """Smallest valid .ter size >= min_size (power of two, 128-8192)."""
    for size in VALID_SIZES:
        if size >= min_size:
            return size
    raise ValueError(
        f"Required terrain size ({min_size}px) exceeds the .ter maximum of 8192px. "
        f"Increase TERRAIN_SQUARE_SIZE or reduce the area."
    )


def build_heightmap(
    grid_points: np.ndarray,
    grid_elevations: np.ndarray,
    nx: int,
    ny: int,
    square_size: float,
) -> dict:
    """
    Builds a square heightmap array padded to a power of two.

    Args:
        grid_points: (N, 2) local XY coordinates, row-major (y-major) as returned by
            terrain.grid.create_terrain_grid (np.meshgrid with
            default indexing="xy", then .ravel() -> order is
            [y0x0, y0x1, ..., y0x(nx-1), y1x0, ...])
        grid_elevations: (N,) height values in meters, same order as grid_points
        nx, ny: grid dimensions (width, height) as returned by create_terrain_grid
        square_size: meters per raster cell (config.TERRAIN_SQUARE_SIZE)

    Returns:
        {
            "heights": (size, size) float64 array, absolute world-coordinate heights.
                       heights[row, col] corresponds to world position
                       (origin_x + col*square_size, origin_y + row*square_size).
            "size": int (power of two),
            "origin_x": float (world X of cell [*, 0]),
            "origin_y": float (world Y of cell [0, *]),
        }
    """
    if len(grid_elevations) != nx * ny:
        raise ValueError(f"grid_elevations has {len(grid_elevations)} values, expected nx*ny={nx * ny}")

    # square_size must match the actual grid spacing (see spec:
    # TERRAIN_SQUARE_SIZE must equal GRID_SPACING) - build_heightmap()
    # silently assumes this and would otherwise produce a heightmap that
    # spatially does not match the real elevation data.
    if nx >= 2:
        actual_spacing = float(grid_points[1, 0] - grid_points[0, 0])
        if not np.isclose(actual_spacing, square_size, rtol=0.01):
            raise ValueError(
                f"square_size ({square_size}) does not match the actual grid spacing "
                f"({actual_spacing:.3f}) - build_heightmap() assumes both are identical "
                f"(see spec: TERRAIN_SQUARE_SIZE must equal GRID_SPACING). If they "
                f"differ on purpose, build_heightmap() must be extended with real resampling."
            )

    source_heights = grid_elevations.reshape(ny, nx)

    size = next_power_of_two_size(max(nx, ny))

    origin_x = float(grid_points[:, 0].min())
    origin_y = float(grid_points[:, 1].min())

    heights = np.empty((size, size), dtype=np.float64)
    heights[:ny, :nx] = source_heights

    # Edge padding: continue the last real column/row instead of jumping to 0
    # (prevents a visible edge at the data border, see spec section 8)
    if size > nx:
        heights[:ny, nx:] = source_heights[:, -1:]
    if size > ny:
        heights[ny:, :] = heights[ny - 1 : ny, :]

    return {
        "heights": heights,
        "size": size,
        "origin_x": origin_x,
        "origin_y": origin_y,
    }
