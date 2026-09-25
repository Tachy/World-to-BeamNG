"""
Small end-to-end test: synthetic grid + a synthetic road through the
complete terrain export path (heightmap -> embankment -> road embedding -> .ter),
to secure the ordering invariant (pristine heights first) with real code instead of just
a comment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.terrain.heightmap import build_heightmap
from world_to_beamng.terrain.road_embedding import (
    build_road_embankment_profiles,
    apply_embankment_blend,
    embed_roads_into_heightmap,
)
from world_to_beamng.terrain.ter_writer import write_ter, read_ter, encode_heights_to_u16


class _FakeMapper:
    def get_road_properties(self, tags):
        return {"width": 6.0}


def test_road_to_ter_full_chain(tmp_path):
    # Flat 50x50 grid at 100m height, 1m spacing
    nx, ny, spacing = 50, 50, 1.0
    x_coords = np.arange(nx) * spacing
    y_coords = np.arange(ny) * spacing
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_elevations = np.full(nx * ny, 100.0)

    heightmap_result = build_heightmap(grid_points, grid_elevations, nx, ny, square_size=spacing)
    heights = heightmap_result["heights"]
    size = heightmap_result["size"]
    origin_x = heightmap_result["origin_x"]
    origin_y = heightmap_result["origin_y"]

    # One road at x=25, Z=95 (5m cut), width=6
    centerline = np.array([[25.0, y, 95.0] for y in range(5, 45)], dtype=float)
    # 2D road polygon (x=[22,28], y=[5,45]) - corresponds to width=6 around the centerline
    road_polygon = np.array([[22.0, 5.0], [28.0, 5.0], [28.0, 45.0], [22.0, 45.0]])
    road_slope_polygons_2d = [
        {"trimmed_centerline": centerline, "osm_tags": {}, "road_polygon": road_polygon}
    ]

    profiles = build_road_embankment_profiles(
        road_slope_polygons_2d, heights, origin_x, origin_y, spacing,
        _FakeMapper(), slope_angle_deg=45.0, min_slope_width=2.0,
    )
    heights = apply_embankment_blend(heights, origin_x, origin_y, spacing, profiles)

    # DecalRoad approach: terrain is set exactly to centerline height
    # (no safety margin anymore, see the road_embedding.py module docstring)
    heights = embed_roads_into_heightmap(heights, origin_x, origin_y, spacing, road_slope_polygons_2d)

    z_min = float(heights.min())
    max_height = float(heights.max() - z_min) + 10.0
    heights_u16 = encode_heights_to_u16(heights, z_min, max_height)
    layer_map = np.zeros((size, size), dtype=np.uint8)

    ter_path = tmp_path / "test.ter"
    write_ter(ter_path, heights_u16, layer_map, ["test_material"])

    read_heightmap, read_layer_map, read_names = read_ter(ter_path)
    assert read_heightmap.shape == (size, size)

    # Under the road the height must be clearly lower than far away (natural terrain)
    row_mid = 25
    col_under_road = 25
    col_far_away = 5
    assert heights[row_mid, col_under_road] < heights[row_mid, col_far_away] - 2.0
    # Far from the cut: unchanged at 100
    assert np.isclose(heights[row_mid, col_far_away], 100.0)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_road_to_ter_full_chain(Path(tmp))
        print("[OK] test_road_to_ter_full_chain")
        print("Alle Tests bestanden.")
