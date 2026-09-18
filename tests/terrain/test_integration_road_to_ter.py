"""
Kleiner End-to-End-Test: synthetisches Grid + eine synthetische Straße durch den
kompletten Terrain-Export-Pfad (heightmap -> embankment -> road embedding -> .ter),
um die Reihenfolge-Invariante (pristine heights zuerst) mit echtem Code statt nur
einem Kommentar abzusichern.
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
    # Flaches 50x50 Grid bei 100m Höhe, 1m Abstand
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

    # Eine Straße bei x=25, Z=95 (5m Einschnitt), width=6
    centerline = np.array([[25.0, y, 95.0] for y in range(5, 45)], dtype=float)
    # 2D-Straßenpolygon (x=[22,28], y=[5,45]) - entspricht width=6 um die Centerline
    road_polygon = np.array([[22.0, 5.0], [28.0, 5.0], [28.0, 45.0], [22.0, 45.0]])
    road_slope_polygons_2d = [
        {"trimmed_centerline": centerline, "osm_tags": {}, "road_polygon": road_polygon}
    ]

    profiles = build_road_embankment_profiles(
        road_slope_polygons_2d, heights, origin_x, origin_y, spacing,
        _FakeMapper(), slope_angle_deg=45.0, min_slope_width=2.0,
    )
    heights = apply_embankment_blend(heights, origin_x, origin_y, spacing, profiles)

    # DecalRoad-Ansatz: Terrain wird exakt auf Centerline-Höhe gesetzt
    # (kein Sicherheitsabstand mehr, siehe road_embedding.py-Moduldocstring)
    heights = embed_roads_into_heightmap(heights, origin_x, origin_y, spacing, road_slope_polygons_2d)

    z_min = float(heights.min())
    max_height = float(heights.max() - z_min) + 10.0
    heights_u16 = encode_heights_to_u16(heights, z_min, max_height)
    layer_map = np.zeros((size, size), dtype=np.uint8)

    ter_path = tmp_path / "test.ter"
    write_ter(ter_path, heights_u16, layer_map, ["test_material"])

    read_heightmap, read_layer_map, read_names = read_ter(ter_path)
    assert read_heightmap.shape == (size, size)

    # Unter der Straße muss die Höhe deutlich niedriger sein als weit entfernt (natürliches Terrain)
    row_mid = 25
    col_under_road = 25
    col_far_away = 5
    assert heights[row_mid, col_under_road] < heights[row_mid, col_far_away] - 2.0
    # Weit entfernt vom Einschnitt: unverändert bei 100
    assert np.isclose(heights[row_mid, col_far_away], 100.0)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_road_to_ter_full_chain(Path(tmp))
        print("[OK] test_road_to_ter_full_chain")
        print("Alle Tests bestanden.")
