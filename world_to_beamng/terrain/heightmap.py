"""
Baut das quadratische .ter-Heightmap-Array direkt aus dem bestehenden
Elevation-Grid (terrain.grid.create_terrain_grid) - ersetzt die bisherige
Triangulierung in TerrainMeshBuilder.
"""

import numpy as np

VALID_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)


def next_power_of_two_size(min_size: int) -> int:
    """Kleinste gültige .ter-Größe >= min_size (Zweierpotenz, 128-8192)."""
    for size in VALID_SIZES:
        if size >= min_size:
            return size
    raise ValueError(
        f"Benötigte Terrain-Größe ({min_size}px) übersteigt das .ter-Maximum von 8192px. "
        f"TERRAIN_SQUARE_SIZE erhöhen oder Gebiet verkleinern."
    )


def build_heightmap(
    grid_points: np.ndarray,
    grid_elevations: np.ndarray,
    nx: int,
    ny: int,
    square_size: float,
) -> dict:
    """
    Baut ein quadratisches, auf Zweierpotenz aufgefülltes Heightmap-Array.

    Args:
        grid_points: (N, 2) lokale XY-Koordinaten, row-major (y-major) wie von
            terrain.grid.create_terrain_grid zurückgegeben (np.meshgrid mit
            default indexing="xy", dann .ravel() -> Reihenfolge ist
            [y0x0, y0x1, ..., y0x(nx-1), y1x0, ...])
        grid_elevations: (N,) Höhenwerte in Metern, gleiche Reihenfolge wie grid_points
        nx, ny: Grid-Dimensionen (Breite, Höhe) wie von create_terrain_grid zurückgegeben
        square_size: Meter pro Rasterzelle (config.TERRAIN_SQUARE_SIZE)

    Returns:
        {
            "heights": (size, size) float64 Array, absolute Weltkoordinaten-Höhen.
                       heights[row, col] entspricht Weltposition
                       (origin_x + col*square_size, origin_y + row*square_size).
            "size": int (Zweierpotenz),
            "origin_x": float (Welt-X der Zelle [*, 0]),
            "origin_y": float (Welt-Y der Zelle [0, *]),
        }
    """
    if len(grid_elevations) != nx * ny:
        raise ValueError(f"grid_elevations hat {len(grid_elevations)} Werte, erwartet nx*ny={nx * ny}")

    source_heights = grid_elevations.reshape(ny, nx)

    size = next_power_of_two_size(max(nx, ny))

    origin_x = float(grid_points[:, 0].min())
    origin_y = float(grid_points[:, 1].min())

    heights = np.empty((size, size), dtype=np.float64)
    heights[:ny, :nx] = source_heights

    # Rand-Padding: letzte echte Spalte/Zeile fortsetzen statt auf 0 zu springen
    # (verhindert eine sichtbare Kante am Datenrand, siehe Spec Abschnitt 8)
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
