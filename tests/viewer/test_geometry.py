"""Tests for tools/level_viewer/geometry.py: ribbons, oriented boxes, water boxes, spawn direction, texture coords."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from tools.level_viewer.geometry import (
    forward_direction,
    heading_deg,
    height_grid_normals,
    merge_meshes,
    oriented_box,
    ribbon,
    road_z_offset,
    terrain_tcoords,
    tile_sample_range,
    to_vtk_faces,
    water_block_box,
)
from world_to_beamng.managers.item_manager import ItemManager


def test_ribbon_along_x_has_half_width_offsets_left_and_right():
    points, faces = ribbon([[0, 0, 5, 4], [10, 0, 5, 4], [20, 0, 6, 4]], z_offset=0.1)

    assert points[0::2, 1] == pytest.approx([2, 2, 2])  # left of +X travel = +Y
    assert points[1::2, 1] == pytest.approx([-2, -2, -2])
    assert points[:, 2] == pytest.approx([5.1, 5.1, 5.1, 5.1, 6.1, 6.1])
    assert faces.tolist() == [[0, 1, 3, 2], [2, 3, 5, 4]]


def test_ribbon_skips_duplicate_nodes_and_rejects_single_points():
    points, faces = ribbon([[0, 0, 0, 2], [0, 0, 0, 2], [0, 10, 0, 2]])

    assert len(points) == 4 and len(faces) == 1
    assert points[0, 0] == pytest.approx(-1)  # travel +Y -> left = -X
    assert len(ribbon([[0, 0, 0, 2]])[0]) == 0


def test_lower_render_priority_is_drawn_higher():
    assert road_z_offset(1) > road_z_offset(12) > 0
    assert road_z_offset(None) == pytest.approx(road_z_offset(20))


def test_oriented_box_uses_rotation_rows_as_local_axes():
    rotation = np.asarray(ItemManager._heading_rotation_matrix(0.0, 1.0)).reshape(3, 3)  # vehicle facing north
    points, faces = oriented_box([100, 200, 10], [10, 2, 4], rotation)

    extent = points.max(axis=0) - points.min(axis=0)
    # local +X/+Y/+Z (sizes 10/2/4) map onto the world directions of rows 0/1/2
    expected = np.abs(rotation[0]) * 10 + np.abs(rotation[1]) * 2 + np.abs(rotation[2]) * 4
    assert extent == pytest.approx(expected)
    assert extent == pytest.approx([10, 2, 4])  # facing north: local X (10 m) = world -X, local Y (2 m) = world -Y
    assert points.mean(axis=0) == pytest.approx([100, 200, 10])
    assert faces.shape == (6, 4)


def test_forward_direction_matches_the_heading_used_to_build_the_matrix():
    for dx, dy in [(1, 0), (0, 1), (-0.6, 0.8)]:
        rotation = np.asarray(ItemManager._heading_rotation_matrix(dx, dy)).reshape(3, 3)
        assert forward_direction(rotation)[:2] == pytest.approx([dx, dy])
    assert heading_deg([1, 0, 0]) == pytest.approx(90)
    assert heading_deg([0, 1, 0]) == pytest.approx(0)


def test_water_block_hangs_down_from_the_water_surface():
    points, _ = water_block_box([20, 30, 100], [10, 4, 2])

    assert points.min(axis=0) == pytest.approx([15, 28, 98])
    assert points.max(axis=0) == pytest.approx([25, 32, 100])


def test_terrain_tcoords_put_the_image_top_row_at_the_north_edge():
    uv = terrain_tcoords(np.array([0.0, 100.0]), np.array([100.0, 0.0]), x_min=0, y_max=100, size_x=100, size_y=100)

    assert uv.tolist() == [[0.0, 1.0], [1.0, 0.0]]  # NW corner -> v = 1 (top), SE corner -> v = 0


def test_merge_meshes_offsets_faces_and_keeps_item_ids():
    a = ribbon([[0, 0, 0, 2], [10, 0, 0, 2]])
    b = ribbon([[0, 5, 0, 2], [10, 5, 0, 2], [20, 5, 0, 2]])
    points, faces, ids = merge_meshes([(a[0], a[1], 7), (b[0], b[1], 9)])

    assert len(points) == 10 and faces.max() == 9
    assert ids.tolist() == [7, 9, 9]
    assert to_vtk_faces(faces[:1]).tolist() == [4, 0, 1, 3, 2]


def test_neighbouring_photo_tiles_share_exactly_one_sample_even_off_grid():
    samples = np.arange(-9.5, 10.0, 1.0)  # terrain samples on half meters, tile bounds on whole ones

    west = tile_sample_range(samples, -10.0, 0.0, 1.0)
    east = tile_sample_range(samples, 0.0, 10.0, 1.0)

    assert west[-1] == east[0]  # shared seam sample -> no gap between the two surfaces
    assert np.union1d(west, east).tolist() == list(range(len(samples)))
    assert tile_sample_range(np.arange(0.0, 21.0, 1.0), 0.0, 10.0, 1.0).tolist() == list(range(11))  # on-grid bounds


def test_height_grid_normals_of_a_tilted_plane():
    x = np.arange(0.0, 5.0)
    y = np.arange(0.0, 4.0)
    z = 0.5 * x[None, :] + 0.0 * y[:, None]  # rises toward +x

    normals = height_grid_normals(x, y, z)

    expected = np.array([-0.5, 0.0, 1.0]) / np.sqrt(1.25)
    assert normals.shape == (4, 5, 3)
    assert np.allclose(normals.reshape(-1, 3), expected, atol=1e-6)
