"""Tests für world_to_beamng.terrain.terrain_materials."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from shapely.geometry import Polygon

from world_to_beamng.terrain.terrain_materials import (
    get_landuse_category,
    build_photo_fallback_layer,
    paint_landuse_materials,
    build_terrain_material_entries,
    build_terrain_material_texture_set,
)

LANDUSE_MAPPINGS_FIXTURE = {
    "forest": {"priority": 10, "internal_name": "mat_forest", "baseColorMap": "a/forest_b.png"},
    "meadow": {"priority": 4, "internal_name": "mat_grass", "baseColorMap": "a/grass_b.png"},
    "farmland": {"priority": 5, "internal_name": "mat_dirt", "baseColorMap": "a/dirt_b.png"},
    "water": {"priority": 15, "internal_name": "mat_water", "baseColorMap": "a/water_b.png"},
}


def test_get_landuse_category_matches_active_only():
    assert get_landuse_category({"landuse": "forest"}, LANDUSE_MAPPINGS_FIXTURE) == "forest"
    assert get_landuse_category({"natural": "wood"}, LANDUSE_MAPPINGS_FIXTURE) is None  # nicht in Fixture
    # "water" ist in landuse_mappings, aber NICHT in ACTIVE_LANDUSE_CATEGORIES -> None
    assert get_landuse_category({"natural": "water"}, LANDUSE_MAPPINGS_FIXTURE) is None
    assert get_landuse_category({}, LANDUSE_MAPPINGS_FIXTURE) is None


def test_photo_fallback_layer_one_material_per_tile():
    # 10x10 Raster, 1m/Zelle, tile_size=5m -> 2x2 Kacheln im Bereich
    # Kein Padding involviert: real_max_x/y = origin + (size-1)*square_size
    size, origin_x, origin_y, square_size = 10, 0.0, 0.0, 1.0
    real_max_x = origin_x + (size - 1) * square_size
    real_max_y = origin_y + (size - 1) * square_size
    layer_map, names = build_photo_fallback_layer(
        size=size, origin_x=origin_x, origin_y=origin_y, square_size=square_size, tile_size=5.0,
        real_max_x=real_max_x, real_max_y=real_max_y,
    )
    assert layer_map.shape == (10, 10)
    assert len(names) == 4  # 2x2 Kacheln
    assert all(n.startswith("tile_") for n in names)
    # Zelle (0,0) und Zelle (9,9) müssen unterschiedliche Kachel-Materialien haben
    assert layer_map[0, 0] != layer_map[9, 9]


def test_photo_fallback_layer_clamps_padding_to_real_bounds():
    # Heightmap ist größer als die echten Höhendaten (Zweierpotenz-Padding,
    # siehe heightmap.py:build_heightmap). Zellen jenseits der echten Bounds
    # müssen dieselbe Kachel wie der echte Rand bekommen, nicht eine neue.
    size, origin_x, origin_y, square_size, tile_size = 20, 0.0, 0.0, 1.0, 100.0
    real_nx = 12  # echte Daten reichen nur bis Spalte 11 (0-indiziert)
    real_max_x = origin_x + (real_nx - 1) * square_size  # = 11.0
    real_max_y = origin_y + (size - 1) * square_size  # keine Padding in Y

    layer_map, names = build_photo_fallback_layer(
        size=size, origin_x=origin_x, origin_y=origin_y, square_size=square_size, tile_size=tile_size,
        real_max_x=real_max_x, real_max_y=real_max_y,
    )

    # Alles liegt innerhalb einer einzigen 100m-Kachel -> nur ein Material,
    # auch für Spalten jenseits von real_max_x (Padding-Bereich)
    assert len(names) == 1
    real_edge_material = layer_map[0, real_nx - 1]
    padded_material = layer_map[0, size - 1]
    assert padded_material == real_edge_material


def test_paint_landuse_overwrites_photo_fallback():
    size = 20
    real_max_x = 0.0 + (size - 1) * 1.0
    real_max_y = 0.0 + (size - 1) * 1.0
    layer_map, names = build_photo_fallback_layer(
        size=size, origin_x=0.0, origin_y=0.0, square_size=1.0, tile_size=100.0,
        real_max_x=real_max_x, real_max_y=real_max_y,
    )
    assert len(names) == 1  # ein Foto-Tile deckt alles ab

    forest_polygon = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    landuse_polygons = [{"osm_tags": {"landuse": "forest"}, "geometry": forest_polygon}]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    assert "mat_forest" in new_names
    forest_index = new_names.index("mat_forest")

    # Zelle innerhalb des Wald-Polygons muss jetzt das Wald-Material haben
    assert new_layer_map[10, 10] == forest_index
    # Zelle außerhalb muss beim Foto-Fallback bleiben
    assert new_layer_map[0, 0] == names.index(names[0])
    assert new_layer_map[0, 0] != forest_index


def test_paint_landuse_priority_resolves_overlap():
    size = 20
    layer_map = np.zeros((size, size), dtype=np.uint8)
    names = ["tile_0_0"]

    # Zwei überlappende Polygone: farmland (priority=5) und forest (priority=10)
    farmland_poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])  # deckt alles ab
    forest_poly = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])  # kleinerer Ausschnitt

    landuse_polygons = [
        {"osm_tags": {"landuse": "farmland"}, "geometry": farmland_poly},
        {"osm_tags": {"landuse": "forest"}, "geometry": forest_poly},
    ]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    forest_index = new_names.index("mat_forest")
    farmland_index = new_names.index("mat_dirt")

    # Im Überlappungsbereich gewinnt forest (höhere priority)
    assert new_layer_map[10, 10] == forest_index
    # Außerhalb des Wald-Polygons, aber innerhalb des Farmland-Polygons: farmland
    assert new_layer_map[1, 1] == farmland_index


def test_build_terrain_material_entries():
    material_names = ["tile_0_0", "mat_forest"]
    photo_tile_names = ["tile_0_0"]

    entries = build_terrain_material_entries(
        material_names, photo_tile_names, LANDUSE_MAPPINGS_FIXTURE, "world_to_beamng", 500.0
    )

    assert "tile_0_0" in entries
    assert entries["tile_0_0"]["class"] == "TerrainMaterial"
    assert "levels/world_to_beamng" in entries["tile_0_0"]["baseColorBaseTex"]

    assert "mat_forest" in entries
    assert entries["mat_forest"]["baseColorBaseTex"] == "a/forest_b.png"


def test_build_terrain_material_texture_set():
    entries = build_terrain_material_texture_set("myTerrainTextureSet", base_tex_size=1024)
    assert "myTerrainTextureSet" in entries
    assert entries["myTerrainTextureSet"]["class"] == "TerrainMaterialTextureSet"
    assert entries["myTerrainTextureSet"]["baseTexSize"] == [1024, 1024]


if __name__ == "__main__":
    test_get_landuse_category_matches_active_only()
    print("[OK] test_get_landuse_category_matches_active_only")
    test_photo_fallback_layer_one_material_per_tile()
    print("[OK] test_photo_fallback_layer_one_material_per_tile")
    test_photo_fallback_layer_clamps_padding_to_real_bounds()
    print("[OK] test_photo_fallback_layer_clamps_padding_to_real_bounds")
    test_paint_landuse_overwrites_photo_fallback()
    print("[OK] test_paint_landuse_overwrites_photo_fallback")
    test_paint_landuse_priority_resolves_overlap()
    print("[OK] test_paint_landuse_priority_resolves_overlap")
    test_build_terrain_material_entries()
    print("[OK] test_build_terrain_material_entries")
    test_build_terrain_material_texture_set()
    print("[OK] test_build_terrain_material_texture_set")
    print("Alle Tests bestanden.")
