"""Tests for world_to_beamng.terrain.heightmap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.terrain.heightmap import build_heightmap, next_power_of_two_size


def test_next_power_of_two_size():
    assert next_power_of_two_size(100) == 128
    assert next_power_of_two_size(128) == 128
    assert next_power_of_two_size(129) == 256
    assert next_power_of_two_size(2049) == 4096


def test_next_power_of_two_size_exceeds_max():
    try:
        next_power_of_two_size(9000)
        assert False, "sollte ValueError werfen"
    except ValueError as e:
        assert "8192" in str(e)


def _make_regular_grid(nx, ny, spacing, base_height=100.0):
    """Builds a synthetic grid like terrain.grid.create_terrain_grid() returns it."""
    x_coords = np.arange(nx) * spacing
    y_coords = np.arange(ny) * spacing
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)  # shape (ny, nx)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    # Height = base_height + x*0.1 (linear gradient, to check the reshape order)
    grid_elevations = base_height + grid_x.ravel() * 0.1
    return grid_points, grid_elevations, nx, ny


def test_build_heightmap_preserves_real_data():
    nx, ny, spacing = 50, 40, 2.0
    grid_points, grid_elevations, nx, ny = _make_regular_grid(nx, ny, spacing)

    result = build_heightmap(grid_points, grid_elevations, nx, ny, square_size=spacing)

    assert result["size"] == 128  # next_power_of_two_size(max(50, 40)) == 128
    assert result["heights"].shape == (128, 128)
    assert result["origin_x"] == 0.0
    assert result["origin_y"] == 0.0

    # Real data in the range [0:ny, 0:nx] must be preserved exactly
    expected = grid_elevations.reshape(ny, nx)
    assert np.allclose(result["heights"][:ny, :nx], expected)


def test_build_heightmap_pads_edges_without_cliff():
    nx, ny, spacing = 10, 10, 2.0
    grid_points, grid_elevations, nx, ny = _make_regular_grid(nx, ny, spacing)

    result = build_heightmap(grid_points, grid_elevations, nx, ny, square_size=spacing)
    heights = result["heights"]

    # Padding area (right of column nx-1) must continue the last real column,
    # not jump to 0 (that would be a visible edge, see spec section 8)
    last_real_col = heights[:ny, nx - 1]
    first_padded_col = heights[:ny, nx]
    assert np.allclose(last_real_col, first_padded_col)


def test_build_heightmap_rejects_mismatched_square_size():
    nx, ny, spacing = 10, 10, 2.0
    grid_points, grid_elevations, nx, ny = _make_regular_grid(nx, ny, spacing)

    try:
        build_heightmap(grid_points, grid_elevations, nx, ny, square_size=5.0)  # wrong, doesn't match spacing=2.0
        assert False, "should raise ValueError (square_size != actual grid spacing)"
    except ValueError as e:
        assert "square_size" in str(e)


if __name__ == "__main__":
    test_next_power_of_two_size()
    print("[OK] test_next_power_of_two_size")
    test_next_power_of_two_size_exceeds_max()
    print("[OK] test_next_power_of_two_size_exceeds_max")
    test_build_heightmap_preserves_real_data()
    print("[OK] test_build_heightmap_preserves_real_data")
    test_build_heightmap_pads_edges_without_cliff()
    print("[OK] test_build_heightmap_pads_edges_without_cliff")
    test_build_heightmap_rejects_mismatched_square_size()
    print("[OK] test_build_heightmap_rejects_mismatched_square_size")
    print("Alle Tests bestanden.")
