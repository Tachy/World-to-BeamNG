"""Tests für world_to_beamng.terrain.road_embedding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.terrain.road_embedding import (
    embed_roads_into_heightmap,
    road_mesh_to_arrays,
)


def test_road_mesh_to_arrays():
    road_mesh_data = [
        {"vertices": [0, 1, 2], "road_id": 1, "uvs": {}},
        {"vertices": [1, 2, 3], "road_id": 1, "uvs": {}},
    ]
    all_vertices = np.array(
        [[0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10]], dtype=np.float64
    )

    vertices, triangles = road_mesh_to_arrays(road_mesh_data, all_vertices)

    assert np.array_equal(vertices, all_vertices)
    assert triangles.shape == (2, 3)
    assert list(triangles[0]) == [0, 1, 2]
    assert list(triangles[1]) == [1, 2, 3]


def test_embed_roads_lowers_only_near_road():
    # 20x20 Heightmap, 1m/Zelle, überall 100m hoch
    size = 20
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # Eine flache Straße bei Z=95 (5m unter natürlichem Terrain), Fläche x=[5,15], y=[5,15]
    road_vertices = np.array(
        [[5, 5, 95], [15, 5, 95], [5, 15, 95], [15, 15, 95]], dtype=np.float64
    )
    road_triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    margin = 0.1
    result = embed_roads_into_heightmap(
        heights, origin_x, origin_y, square_size, road_vertices, road_triangles, margin
    )

    # Zellen unter der Straße müssen auf ~95 - 0.1 = 94.9 abgesenkt sein
    assert np.isclose(result[10, 10], 95.0 - margin, atol=0.5)

    # Zellen weit weg von der Straße müssen unverändert bei 100 bleiben
    assert result[1, 1] == 100.0
    assert result[18, 18] == 100.0

    # Original-Array darf nicht verändert worden sein (Funktion gibt Kopie zurück)
    assert heights[10, 10] == 100.0


def test_embed_roads_never_raises_terrain():
    # Straße LIEGT HÖHER als natürliches Terrain -> Terrain darf NICHT angehoben werden
    size = 10
    heights = np.full((size, size), 50.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road_vertices = np.array(
        [[2, 2, 200], [8, 2, 200], [2, 8, 200], [8, 8, 200]], dtype=np.float64
    )
    road_triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    result = embed_roads_into_heightmap(
        heights, origin_x, origin_y, square_size, road_vertices, road_triangles, margin=0.1
    )

    assert np.all(result <= 50.0)


if __name__ == "__main__":
    test_road_mesh_to_arrays()
    print("[OK] test_road_mesh_to_arrays")
    test_embed_roads_lowers_only_near_road()
    print("[OK] test_embed_roads_lowers_only_near_road")
    test_embed_roads_never_raises_terrain()
    print("[OK] test_embed_roads_never_raises_terrain")
    print("Alle Tests bestanden.")
