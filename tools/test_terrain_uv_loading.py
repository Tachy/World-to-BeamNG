"""
Test: Was wird vom DAE-Loader für Terrain-UVs geladen?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.dae_loader import load_dae_tile
import numpy as np

dae_path = r"C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\levels\world_to_beamng\art\shapes\terrain_-1000_-1000.dae"

print("Lade DAE...")
data = load_dae_tile(dae_path)

tiles_info = data.get("tiles", {})

# Prüfe tile_-2_-2 (entspricht tile_-1000_-1000)
tile_name = "tile_-2_-2"

if tile_name in tiles_info:
    tile_data = tiles_info[tile_name]
    uvs = tile_data["uvs"]
    vertices = tile_data["vertices"]
    
    print(f"\nTile: {tile_name}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  UVs: {len(uvs)}")
    
    # Erste 5 UVs
    print(f"\n  Erste 5 UVs (sollte (0,0), (0.004,0), (0.004,0.004), (0,0.004), (0.008,0) sein):")
    for i in range(min(5, len(uvs))):
        print(f"    [{i}] ({uvs[i][0]:.6f}, {uvs[i][1]:.6f})")
    
    # Erste 5 Vertices
    print(f"\n  Erste 5 Vertices:")
    for i in range(min(5, len(vertices))):
        print(f"    [{i}] ({vertices[i][0]:.2f}, {vertices[i][1]:.2f}, {vertices[i][2]:.2f})")
    
    # UV-Range
    print(f"\n  UV-Range:")
    print(f"    U: [{uvs[:, 0].min():.6f}, {uvs[:, 0].max():.6f}]")
    print(f"    V: [{uvs[:, 1].min():.6f}, {uvs[:, 1].max():.6f}]")
    
    # Vergleich mit erwartetem Wert
    expected_uvs = [
        (0.0, 0.0),
        (0.004, 0.0),
        (0.004, 0.004),
        (0.0, 0.004),
        (0.008, 0.0)
    ]
    
    print(f"\n  Vergleich mit erwarteten UVs:")
    all_match = True
    for i in range(min(5, len(uvs))):
        actual = (uvs[i][0], uvs[i][1])
        expected = expected_uvs[i]
        match = abs(actual[0] - expected[0]) < 0.0001 and abs(actual[1] - expected[1]) < 0.0001
        status = "✓" if match else "✗"
        print(f"    [{i}] {status} Actual: ({actual[0]:.6f}, {actual[1]:.6f}) vs Expected: ({expected[0]:.6f}, {expected[1]:.6f})")
        if not match:
            all_match = False
    
    if all_match:
        print(f"\n  ✅ ALLE UVs stimmen überein!")
    else:
        print(f"\n  ❌ UVs stimmen NICHT überein!")
else:
    print(f"Tile {tile_name} nicht gefunden!")
    print(f"Verfügbare Tiles: {list(tiles_info.keys())[:5]}")
