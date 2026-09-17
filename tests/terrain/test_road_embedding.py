"""Tests für world_to_beamng.terrain.road_embedding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.terrain.road_embedding import (
    road_mesh_to_arrays,
    embed_roads_into_heightmap,
    sample_heightmap_bilinear,
    build_road_embankment_profiles,
    apply_embankment_blend,
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


def test_sample_heightmap_bilinear_matches_grid_points():
    heights = np.array([[0.0, 10.0], [20.0, 30.0]])
    # Exact grid points should return exact values
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = sample_heightmap_bilinear(heights, origin_x=0.0, origin_y=0.0, square_size=1.0, points_xy=points)
    assert np.allclose(result, [0.0, 10.0, 20.0, 30.0])

    # Midpoint should be the average of all 4 corners
    midpoint = np.array([[0.5, 0.5]])
    result_mid = sample_heightmap_bilinear(heights, 0.0, 0.0, 1.0, midpoint)
    assert np.isclose(result_mid[0], 15.0)


def test_build_road_embankment_profiles_straight_road():
    # Flat terrain at 100.0, road at 95.0 (5m cut) -> slope_width should reflect that
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # A straight road along x=20, from y=5 to y=35, road_z=95.0, width=6.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)
    poly = {"trimmed_centerline": centerline, "osm_tags": {"highway": "residential"}}

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    assert len(roads) == 1
    road = roads[0]
    # Road is at x=20, half_width=3.0 -> left edge at x=23, right edge at x=17
    assert np.allclose(road["left_edge_xyz"][:, 0], 23.0)
    assert np.allclose(road["right_edge_xyz"][:, 0], 17.0)
    # Natural height at edges should be 100.0 (flat terrain)
    assert np.allclose(road["left_natural_z"], 100.0)
    # height_diff = |100 - 95| = 5, tan(45deg) = 1 -> slope_width = 5.0
    assert np.allclose(road["left_slope_width"], 5.0)


def test_apply_embankment_blend_interpolates_correctly():
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # One "road" edge segment at x=20, y from 10 to 30, road_z=95, slope_width=5
    n = 21
    y_vals = np.linspace(10, 30, n)
    edge_xyz = np.column_stack([np.full(n, 20.0), y_vals, np.full(n, 95.0)])
    slope_width = np.full(n, 5.0)
    natural_z = np.full(n, 100.0)

    roads = [{
        "left_edge_xyz": edge_xyz,
        "right_edge_xyz": np.empty((0, 3)),
        "left_slope_width": slope_width,
        "right_slope_width": np.empty(0),
        "left_natural_z": natural_z,
        "right_natural_z": np.empty(0),
    }]

    result = apply_embankment_blend(heights, origin_x, origin_y, square_size, roads)

    # At the edge itself (x=20, dist~0): should be close to road height (95), not exactly
    # touched since dist>0 required, but very close cells should approach it
    near_edge_col = 21  # x=21, dist=1 from edge at x=20
    mid_row = 20
    val_near_edge = result[mid_row, near_edge_col]
    assert val_near_edge < 100.0  # pulled toward road height
    assert val_near_edge > 95.0   # not fully at road height (partway through blend)

    # Far beyond slope_width (x=30, dist=10 > slope_width=5): untouched
    far_col = 30
    assert result[mid_row, far_col] == 100.0

    # Original heights array must not be mutated
    assert heights[mid_row, near_edge_col] == 100.0


def test_apply_embankment_blend_handles_fill_and_cut():
    # Cut: road lower than terrain (tested above, terrain=100, road=95)
    # Fill: road higher than terrain
    size = 20
    heights = np.full((size, size), 50.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    n = 11
    y_vals = np.linspace(3, 13, n)
    edge_xyz = np.column_stack([np.full(n, 10.0), y_vals, np.full(n, 60.0)])  # road at 60, terrain at 50 -> fill
    slope_width = np.full(n, 3.0)
    natural_z = np.full(n, 50.0)

    roads = [{
        "left_edge_xyz": edge_xyz, "right_edge_xyz": np.empty((0, 3)),
        "left_slope_width": slope_width, "right_slope_width": np.empty(0),
        "left_natural_z": natural_z, "right_natural_z": np.empty(0),
    }]

    result = apply_embankment_blend(heights, origin_x, origin_y, square_size, roads)

    mid_row = 8
    # Near edge (x=11, dist=1): pulled UP toward road height (60), i.e. > natural 50
    near_val = result[mid_row, 11]
    assert near_val > 50.0
    assert near_val < 60.0


if __name__ == "__main__":
    test_road_mesh_to_arrays()
    print("[OK] test_road_mesh_to_arrays")
    test_embed_roads_lowers_only_near_road()
    print("[OK] test_embed_roads_lowers_only_near_road")
    test_embed_roads_never_raises_terrain()
    print("[OK] test_embed_roads_never_raises_terrain")
    test_sample_heightmap_bilinear_matches_grid_points()
    print("[OK] test_sample_heightmap_bilinear_matches_grid_points")
    test_build_road_embankment_profiles_straight_road()
    print("[OK] test_build_road_embankment_profiles_straight_road")
    test_apply_embankment_blend_interpolates_correctly()
    print("[OK] test_apply_embankment_blend_interpolates_correctly")
    test_apply_embankment_blend_handles_fill_and_cut()
    print("[OK] test_apply_embankment_blend_handles_fill_and_cut")
    print("Alle Tests bestanden.")
