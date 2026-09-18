"""Tests für world_to_beamng.terrain.terrain_materials."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from shapely.geometry import Polygon

from PIL import Image

from world_to_beamng.terrain.terrain_materials import (
    get_landuse_category,
    build_photo_fallback_layer,
    paint_landuse_materials,
    build_terrain_material_entries,
    build_terrain_material_texture_set,
    ensure_flat_pbr_placeholders,
    ensure_landuse_base_textures_sized,
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


def test_photo_fallback_layer_single_material_for_whole_area():
    # Seit 2026-09-18: EIN Foto-Material für die gesamte Fläche statt vieler
    # 500m-Kachel-Materialien (siehe Docstring von build_photo_fallback_layer() -
    # BeamNGs Terrain-Atlas-Packer verdreht Kacheln sichtbar, wenn ihm zu viele
    # große, einzigartige Materialien übergeben werden).
    layer_map, names = build_photo_fallback_layer(size=20)

    assert layer_map.shape == (20, 20)
    assert names == ["aerial_photo"]
    assert (layer_map == 0).all()


def test_paint_landuse_overwrites_photo_fallback():
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)
    assert len(names) == 1  # ein Foto-Material deckt alles ab

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


def _fake_placeholders_for_tier(tier: str) -> dict:
    return {
        channel: f"/levels/world_to_beamng/art/shapes/textures/_flat_{channel}_{tier}.png"
        for channel in ("baseColor", "normal", "roughness", "ao", "height")
    }


_FAKE_PLACEHOLDERS = {tier: _fake_placeholders_for_tier(tier) for tier in ("base", "detail", "macro")}

# BeamNGs v1.5-Terrain-Material-Editor speichert kein TerrainMaterial mit
# leerem Texturslot (siehe terrain_materials.py::_add_required_pbr_slots());
# ohne einen davon rendert BeamNG die "warning texture" (grauer Boden).
_REQUIRED_TERRAIN_TEX_FIELDS = [
    f"{channel}{tier}Tex" for channel in ("baseColor", "normal", "roughness", "ao", "height") for tier in ("Base", "Detail", "Macro")
]


def test_build_terrain_material_entries():
    material_names = ["aerial_photo", "mat_forest"]
    photo_tile_names = ["aerial_photo"]

    entries = build_terrain_material_entries(
        material_names, photo_tile_names, LANDUSE_MAPPINGS_FIXTURE, "world_to_beamng", 2048.0, _FAKE_PLACEHOLDERS
    )

    assert "aerial_photo" in entries
    assert entries["aerial_photo"]["class"] == "TerrainMaterial"
    assert "levels/world_to_beamng" in entries["aerial_photo"]["baseColorBaseTex"]
    assert entries["aerial_photo"]["baseColorBaseTexSize"] == 2048.0

    assert "mat_forest" in entries
    assert entries["mat_forest"]["baseColorBaseTex"] == "a/forest_b.png"

    for name in ("aerial_photo", "mat_forest"):
        for field in _REQUIRED_TERRAIN_TEX_FIELDS:
            assert field in entries[name], f"{name} fehlt Pflichtfeld {field}"


def test_ensure_flat_pbr_placeholders_matches_declared_tex_sizes(tmp_path):
    # Regression: BeamNG meldet "dont have required size of W-H" und rendert
    # die "warning texture" (grauer Boden), wenn ein Texturslot nicht exakt
    # die in der TerrainMaterialTextureSet deklarierte Pixelgröße hat - ein
    # generisches 8x8-Platzhalterbild reichte NICHT (Recherche 2026-09-18).
    placeholders = ensure_flat_pbr_placeholders(tmp_path, "world_to_beamng", base_tex_size=4096, detail_tex_size=1024, macro_tex_size=1024)

    for tier, expected_size in (("base", 4096), ("detail", 1024), ("macro", 1024)):
        for channel, level_path in placeholders[tier].items():
            filename = level_path.rsplit("/", 1)[-1]
            with Image.open(tmp_path / filename) as img:
                assert img.size == (expected_size, expected_size), f"{tier}/{channel}: {img.size}"


def test_ensure_landuse_base_textures_sized_resizes_mismatched_textures(tmp_path):
    beamng_dir = tmp_path / "levels" / "world_to_beamng"
    textures_dir = beamng_dir / "art" / "shapes" / "textures"
    source_dir = beamng_dir / "art" / "shapes" / "assets"
    source_dir.mkdir(parents=True)
    small_texture = source_dir / "t_forest_ground_b.png"
    Image.new("RGB", (512, 512), (10, 80, 10)).save(small_texture)

    landuse_mappings = {
        "forest": {
            "internal_name": "mat_forest",
            "baseColorMap": f"levels/world_to_beamng/art/shapes/assets/{small_texture.name}",
        },
    }

    result = ensure_landuse_base_textures_sized(landuse_mappings, 4096, beamng_dir, textures_dir, "world_to_beamng")

    new_path = result["forest"]["baseColorMap"]
    assert new_path != landuse_mappings["forest"]["baseColorMap"]
    filename = new_path.rsplit("/", 1)[-1]
    with Image.open(textures_dir / filename) as img:
        assert img.size == (4096, 4096)


def test_build_terrain_material_texture_set():
    entries = build_terrain_material_texture_set("myTerrainTextureSet", base_tex_size=1024)
    assert "myTerrainTextureSet" in entries
    assert entries["myTerrainTextureSet"]["class"] == "TerrainMaterialTextureSet"
    assert entries["myTerrainTextureSet"]["baseTexSize"] == [1024, 1024]
    # TerrainBlock.materialTextureSet resolves this via the SimObject "name" field
    # (not "internalName" - that's only for TerrainMaterial .ter-layer lookups).
    # Missing "name" -> BeamNG logs "Failed to find TerrainMaterialTextureSet with
    # name: ..." and crashes on the first terrain draw with a D3D12 root-cbv assert.
    assert entries["myTerrainTextureSet"]["name"] == "myTerrainTextureSet"


if __name__ == "__main__":
    test_get_landuse_category_matches_active_only()
    print("[OK] test_get_landuse_category_matches_active_only")
    test_photo_fallback_layer_single_material_for_whole_area()
    print("[OK] test_photo_fallback_layer_single_material_for_whole_area")
    test_paint_landuse_overwrites_photo_fallback()
    print("[OK] test_paint_landuse_overwrites_photo_fallback")
    test_paint_landuse_priority_resolves_overlap()
    print("[OK] test_paint_landuse_priority_resolves_overlap")
    test_build_terrain_material_entries()
    print("[OK] test_build_terrain_material_entries")
    test_build_terrain_material_texture_set()
    print("[OK] test_build_terrain_material_texture_set")
    print("Alle Tests bestanden.")
