"""
Terrain-Grid Generierung.
"""

import os
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from .. import config
from .elevation import get_height_data_hash


def create_terrain_grid(height_points, height_elevations, grid_spacing=10.0):
    """Erstellt ein reguläres Grid aus den Hoehendaten (OPTIMIERT mit Caching)."""
    print(f"\nErstelle Terrain-Grid (Abstand: {grid_spacing}m)...")

    # Finde Bounds in lokalen Koordinaten (IMMER berechnen, auch fuer Cache-Fall!)
    min_x, max_x = height_points[:, 0].min(), height_points[:, 0].max()
    min_y, max_y = height_points[:, 1].min(), height_points[:, 1].max()
    config.GRID_BOUNDS_LOCAL = (min_x, min_y, max_x, max_y)

    # Pruefe ob gecachtes Grid existiert (Version 2 fuer lokale Koordinaten!)
    height_hash = get_height_data_hash()
    if height_hash:
        cache_file = os.path.join(
            config.CACHE_DIR, f"grid_v2_{height_hash}_spacing{grid_spacing:.1f}m.npz"
        )

        if os.path.exists(cache_file):
            print(f"  [OK] Grid-Cache gefunden: {os.path.basename(cache_file)}")
            data = np.load(cache_file)
            grid_points = data["grid_points"]
            grid_elevations = data["grid_elevations"]
            nx = int(data["nx"])
            ny = int(data["ny"])
            print(
                f"  [OK] Grid aus Cache geladen: {nx} x {ny} = {len(grid_points)} Vertices"
            )
            # WICHTIG: Grid wurde in UTM gecacht, transformiere zu lokal!
            # (height_points wurden bereits transformiert, min_x/min_y sind lokal)
            # Wir muessen hier nichts tun - grid_points sind schon im gleichen System wie height_points
            return grid_points, grid_elevations, nx, ny

    # Erstelle Grid-Punkte
    x_coords = np.arange(min_x, max_x, grid_spacing)
    y_coords = np.arange(min_y, max_y, grid_spacing)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Interpoliere Hoehen fuer Grid-Punkte (CHUNKED fuer bessere Performance)
    print(f"  Erstelle Interpolator...")
    interpolator = NearestNDInterpolator(height_points, height_elevations)

    print(f"  Interpoliere {len(grid_points)} Grid-Punkte (in Chunks)...")
    chunk_size = 500000  # 500k Punkte pro Chunk
    grid_elevations = np.empty(len(grid_points), dtype=np.float64)

    num_chunks = (len(grid_points) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(grid_points))
        grid_elevations[start_idx:end_idx] = interpolator(
            grid_points[start_idx:end_idx]
        )

        if (i + 1) % 5 == 0 or i == num_chunks - 1:
            progress = ((i + 1) / num_chunks) * 100
            print(f"    {progress:.0f}% ({i + 1}/{num_chunks} Chunks)")

    nx = len(x_coords)
    ny = len(y_coords)
    print(f"  Grid: {nx} x {ny} = {len(grid_points)} Vertices")

    # Cache das Grid fuer zukuenftige Verwendung (Version 2 fuer lokale Koordinaten!)
    if height_hash:
        cache_file = os.path.join(
            config.CACHE_DIR, f"grid_v2_{height_hash}_spacing{grid_spacing:.1f}m.npz"
        )
        print(f"  Speichere Grid-Cache: {os.path.basename(cache_file)}")
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez_compressed(
            cache_file,
            grid_points=grid_points,
            grid_elevations=grid_elevations,
            nx=nx,
            ny=ny,
        )
        print(f"  [OK] Grid-Cache erstellt")

    return grid_points, grid_elevations, nx, ny
