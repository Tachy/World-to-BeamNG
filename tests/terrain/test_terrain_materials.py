"""Tests for world_to_beamng.terrain.terrain_materials."""

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
    ensure_landuse_detail_textures_sized,
)

LANDUSE_MAPPINGS_FIXTURE = {
    "forest": {
        "osm_tags": {"landuse": ["forest"], "natural": ["wood"]},
        "priority": 10,
        "internal_name": "mat_forest",
        "groundModelName": "grass",
        "detailColorMap": "a/forest_b.png",
        "detailNormalMap": "a/forest_nm.png",
        "detailStrength": 0.3,
    },
    "meadow": {
        "osm_tags": {"landuse": ["meadow", "grass"], "natural": ["grassland"], "leisure": ["park"]},
        "priority": 4,
        "internal_name": "mat_grass",
        "groundModelName": "grass",
        "detailColorMap": "a/grass_b.png",
    },
    "farmland": {
        "osm_tags": {"landuse": ["farmland"]},
        "priority": 5,
        "internal_name": "mat_dirt",
        "groundModelName": "dirt",
        "detailColorMap": "a/dirt_b.png",
    },
    # Residential areas etc.: aerial photo stays, but covers layers beneath it
    "urban": {"osm_tags": {"landuse": ["residential"]}, "priority": 12, "keep_photo": True},
    # Water: own area, but the ground beneath it is meadow (material of the category "meadow")
    "water": {"osm_tags": {"natural": ["water"]}, "priority": 15, "use_material_of": "meadow"},
    "disabled": {
        "osm_tags": {"landuse": ["quarry"]},
        "priority": 3,
        "internal_name": "mat_quarry",
        "active": False,
        "detailColorMap": "a/quarry_b.png",
    },
    # Without osm_tags (e.g. the "base" fallback entry) it is never assigned
    "base": {"priority": 0, "internal_name": "mat_base_satellite"},
}


def test_get_landuse_category_matches_tag_values():
    assert get_landuse_category({"landuse": "forest"}, LANDUSE_MAPPINGS_FIXTURE) == "forest"
    # Alias tags: natural=wood belongs to the category "forest", not only landuse=forest
    assert get_landuse_category({"natural": "wood"}, LANDUSE_MAPPINGS_FIXTURE) == "forest"
    assert get_landuse_category({"landuse": "grass"}, LANDUSE_MAPPINGS_FIXTURE) == "meadow"
    assert get_landuse_category({"natural": "grassland"}, LANDUSE_MAPPINGS_FIXTURE) == "meadow"
    assert get_landuse_category({"leisure": "park"}, LANDUSE_MAPPINGS_FIXTURE) == "meadow"


def test_get_landuse_category_defaults_a_genuinely_unlisted_value_to_meadow():
    # Area HAS a landuse/natural/leisure tag, but NO category knows this value at all (not even an
    # inactive one) -> generic grass instead of staying unpainted (photo remainder).
    assert get_landuse_category({"landuse": "railway"}, LANDUSE_MAPPINGS_FIXTURE) == "meadow"


def test_get_landuse_category_active_false_excludes_from_the_default_too():
    # "active": False means "deliberately known, but never a terrain layer" (e.g. regions like
    # natural=mountain_range, see test_landuse_config.py) - this is different from "unknown" and
    # must NOT simply fall through to the generic grass default.
    assert get_landuse_category({"landuse": "quarry"}, LANDUSE_MAPPINGS_FIXTURE) is None


def test_get_landuse_category_stays_none_without_any_area_tag():
    # No landuse/natural/leisure tag at all (building, completely untagged element) -> stays None,
    # does NOT become grass (that would flatten all untagged terrain instead of staying with the aerial photo).
    assert get_landuse_category({"building": "yes"}, LANDUSE_MAPPINGS_FIXTURE) is None
    assert get_landuse_category({}, LANDUSE_MAPPINGS_FIXTURE) is None


def test_get_landuse_category_default_is_skipped_if_meadow_itself_is_missing():
    mappings_without_meadow = {k: v for k, v in LANDUSE_MAPPINGS_FIXTURE.items() if k != "meadow"}

    assert get_landuse_category({"landuse": "railway"}, mappings_without_meadow) is None


def test_get_landuse_category_returns_photo_category():
    assert get_landuse_category({"landuse": "residential"}, LANDUSE_MAPPINGS_FIXTURE) == "urban"


def test_get_landuse_category_prefers_higher_priority_on_multiple_tags():
    # forest (10) beats meadow (4), regardless of which tag is checked first
    tags = {"landuse": "meadow", "natural": "wood"}

    assert get_landuse_category(tags, LANDUSE_MAPPINGS_FIXTURE) == "forest"


def test_photo_fallback_layer_single_material_for_whole_area():
    # Since 2026-09-18: ONE photo material for the whole area instead of many
    # 500 m tile materials (see the docstring of build_photo_fallback_layer() -
    # BeamNG's terrain atlas packer visibly scrambles tiles when it is given too many
    # large, unique materials).
    layer_map, names = build_photo_fallback_layer(size=20)

    assert layer_map.shape == (20, 20)
    assert names == ["aerial_photo"]
    assert (layer_map == 0).all()


def test_paint_landuse_overwrites_photo_fallback():
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)
    assert len(names) == 1  # one photo material covers everything

    forest_polygon = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    landuse_polygons = [{"osm_tags": {"landuse": "forest"}, "geometry": forest_polygon}]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    assert "mat_forest" in new_names
    forest_index = new_names.index("mat_forest")

    # Cell inside the forest polygon must now have the forest material
    assert new_layer_map[10, 10] == forest_index
    # Cell outside must stay at the photo fallback
    assert new_layer_map[0, 0] == names.index(names[0])
    assert new_layer_map[0, 0] != forest_index


def test_paint_landuse_paints_an_unmapped_value_as_meadow_instead_of_leaving_the_photo():
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)

    railway_polygon = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    landuse_polygons = [{"osm_tags": {"landuse": "railway"}, "geometry": railway_polygon}]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    assert "mat_grass" in new_names
    assert new_layer_map[10, 10] == new_names.index("mat_grass")


def test_paint_landuse_priority_resolves_overlap():
    size = 20
    layer_map = np.zeros((size, size), dtype=np.uint8)
    names = ["tile_0_0"]

    # Two overlapping polygons: farmland (priority=5) and forest (priority=10)
    farmland_poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])  # covers everything
    forest_poly = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])  # smaller section

    landuse_polygons = [
        {"osm_tags": {"landuse": "farmland"}, "geometry": farmland_poly},
        {"osm_tags": {"landuse": "forest"}, "geometry": forest_poly},
    ]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    forest_index = new_names.index("mat_forest")
    farmland_index = new_names.index("mat_dirt")

    # In the overlap area forest wins (higher priority)
    assert new_layer_map[10, 10] == forest_index
    # Outside the forest polygon, but inside the farmland polygon: farmland
    assert new_layer_map[1, 1] == farmland_index


def test_paint_landuse_photo_category_restores_photo_over_other_layers():
    # Residential area (keep_photo, priority 12) lies over a meadow (4): there the
    # aerial photo (index 0) should be visible again, not the grass material.
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)
    meadow = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    residential = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    polygons = [
        {"osm_tags": {"landuse": "meadow"}, "geometry": meadow},
        {"osm_tags": {"landuse": "residential"}, "geometry": residential},
    ]

    new_map, new_names = paint_landuse_materials(layer_map, names, size, 0.0, 0.0, 1.0, polygons, LANDUSE_MAPPINGS_FIXTURE)

    assert new_names == ["aerial_photo", "mat_grass"]  # no material created for "urban"
    assert new_map[10, 10] == 0  # residential area: aerial photo
    assert new_map[1, 1] == new_names.index("mat_grass")  # meadow outside


def test_paint_landuse_accepts_multipolygon_geometry():
    from shapely.geometry import MultiPolygon

    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)
    two_parts = MultiPolygon([Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]), Polygon([(10, 10), (14, 10), (14, 14), (10, 14)])])

    new_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, [{"osm_tags": {"landuse": "forest"}, "geometry": two_parts}], LANDUSE_MAPPINGS_FIXTURE
    )

    forest = new_names.index("mat_forest")
    assert new_map[2, 2] == forest and new_map[12, 12] == forest and new_map[7, 7] == 0


def _fake_placeholders_for_tier(tier: str) -> dict:
    return {
        channel: f"/levels/world_to_beamng/art/shapes/textures/_flat_{channel}_{tier}.png"
        for channel in ("baseColor", "normal", "roughness", "ao", "height")
    }


_FAKE_PLACEHOLDERS = {tier: _fake_placeholders_for_tier(tier) for tier in ("base", "detail", "macro")}

# BeamNG's v1.5 terrain material editor does not save a TerrainMaterial with
# an empty texture slot (see terrain_materials.py::_add_required_pbr_slots());
# without one of them BeamNG renders the "warning texture" (gray ground).
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

    for name in ("aerial_photo", "mat_forest"):
        for field in _REQUIRED_TERRAIN_TEX_FIELDS:
            assert field in entries[name], f"{name} fehlt Pflichtfeld {field}"


def test_landuse_material_uses_aerial_photo_as_base_and_grey_texture_as_detail():
    # BeamNG's terrain textures (t_grass_01_b ...) are gray DETAIL textures
    # (mean RGB ~124, saturation < 15/255): as a base texture they produce
    # uniformly gray areas. The color therefore comes from the aerial photo,
    # the gray texture lies on top as detail (like in BeamNG's own levels).
    entries = build_terrain_material_entries(
        ["aerial_photo", "mat_forest"], ["aerial_photo"], LANDUSE_MAPPINGS_FIXTURE, "world_to_beamng", 2048.0, _FAKE_PLACEHOLDERS
    )
    forest, photo = entries["mat_forest"], entries["aerial_photo"]

    assert forest["baseColorBaseTex"] == photo["baseColorBaseTex"]
    assert forest["baseColorBaseTexSize"] == photo["baseColorBaseTexSize"] == 2048.0
    assert forest["baseColorDetailTex"] == "a/forest_b.png"
    assert forest["baseColorDetailStrength"] == [0.3, 0.0]
    assert forest["normalDetailTex"] == "a/forest_nm.png"
    assert forest["normalDetailStrength"] == [1.0, 0.0]


def test_landuse_material_placeholders_do_not_overwrite_real_detail_textures():
    entries = build_terrain_material_entries(
        ["aerial_photo", "mat_grass"], ["aerial_photo"], LANDUSE_MAPPINGS_FIXTURE, "world_to_beamng", 2048.0, _FAKE_PLACEHOLDERS
    )
    grass = entries["mat_grass"]

    assert grass["baseColorDetailTex"] == "a/grass_b.png"
    assert grass["baseColorDetailStrength"] != [0.0, 0.0]  # default strength, not muted
    # without detailNormalMap the normal detail slot stays a silent placeholder
    assert grass["normalDetailTex"] == _FAKE_PLACEHOLDERS["detail"]["normal"]
    assert grass["normalDetailStrength"] == [0.0, 0.0]


def test_landuse_material_sets_ground_model_uppercase():
    entries = build_terrain_material_entries(
        ["aerial_photo", "mat_dirt"], ["aerial_photo"], LANDUSE_MAPPINGS_FIXTURE, "world_to_beamng", 2048.0, _FAKE_PLACEHOLDERS
    )

    # BeamNG's groundmodels.json only knows UPPERCASE names; without
    # groundmodelName BeamNG logs "ground model not found ... using asphalt"
    assert entries["mat_dirt"]["groundmodelName"] == "DIRT"


def test_photo_layer_entry_keeps_the_asphalt_grip_explicitly():
    entries = build_terrain_material_entries(
        ["aerial_photo"], ["aerial_photo"], LANDUSE_MAPPINGS_FIXTURE, "world_to_beamng", 2048.0, _FAKE_PLACEHOLDERS
    )

    # Behavior unchanged (roads need the asphalt grip): it used to be the silent engine fallback,
    # now ASPHALT is stated explicitly (otherwise BeamNG logs "ground model not found ... using asphalt")
    assert entries["aerial_photo"]["groundmodelName"] == "ASPHALT"


def test_ensure_flat_pbr_placeholders_matches_declared_tex_sizes(tmp_path):
    # Regression: BeamNG reports "dont have required size of W-H" and renders
    # the "warning texture" (gray ground) if a texture slot does not have exactly
    # the pixel size declared in the TerrainMaterialTextureSet - a
    # generic 8x8 placeholder image was NOT enough (research 2026-09-18).
    placeholders = ensure_flat_pbr_placeholders(tmp_path, "world_to_beamng", base_tex_size=4096, detail_tex_size=1024, macro_tex_size=1024)

    for tier, expected_size in (("base", 4096), ("detail", 1024), ("macro", 1024)):
        for channel, level_path in placeholders[tier].items():
            filename = level_path.rsplit("/", 1)[-1]
            with Image.open(tmp_path / filename) as img:
                assert img.size == (expected_size, expected_size), f"{tier}/{channel}: {img.size}"


def _sized_fixture(tmp_path, size):
    beamng_dir = tmp_path / "levels" / "world_to_beamng"
    textures_dir = beamng_dir / "art" / "shapes" / "textures"
    source_dir = beamng_dir / "art" / "shapes" / "assets"
    source_dir.mkdir(parents=True)
    texture = source_dir / "t_forest_ground_b.png"
    Image.new("RGB", (size, size), (10, 80, 10)).save(texture)
    mappings = {
        "forest": {
            "internal_name": "mat_forest",
            "detailColorMap": f"levels/world_to_beamng/art/shapes/assets/{texture.name}",
        },
        "urban": {"keep_photo": True},
    }
    return mappings, beamng_dir, textures_dir


def test_ensure_landuse_detail_textures_sized_resizes_mismatched_textures(tmp_path):
    # Detail textures must have exactly the detailTexSize of the TerrainMaterialTextureSet,
    # otherwise BeamNG renders the "warning texture" (gray ground).
    mappings, beamng_dir, textures_dir = _sized_fixture(tmp_path, 512)

    result = ensure_landuse_detail_textures_sized(mappings, 1024, beamng_dir, textures_dir, "world_to_beamng")

    new_path = result["forest"]["detailColorMap"]
    assert new_path != mappings["forest"]["detailColorMap"]
    with Image.open(textures_dir / new_path.rsplit("/", 1)[-1]) as img:
        assert img.size == (1024, 1024)
    assert result["urban"] == {"keep_photo": True}


def test_ensure_landuse_detail_textures_sized_keeps_correctly_sized_textures(tmp_path):
    mappings, beamng_dir, textures_dir = _sized_fixture(tmp_path, 1024)

    result = ensure_landuse_detail_textures_sized(mappings, 1024, beamng_dir, textures_dir, "world_to_beamng")

    assert result["forest"]["detailColorMap"] == mappings["forest"]["detailColorMap"]


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
    test_get_landuse_category_matches_tag_values()
    print("[OK] test_get_landuse_category_matches_tag_values")
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


def test_mark_padding_as_holes_only_touches_cells_beyond_the_data():
    from world_to_beamng.terrain.terrain_materials import mark_padding_as_holes
    from world_to_beamng.terrain.ter_writer import EMPTY_LAYER_VALUE

    layer_map = np.full((8, 8), 3, dtype=np.uint8)  # 8x8 terrain, data only 5 columns x 6 rows

    result = mark_padding_as_holes(layer_map, data_cols=5, data_rows=6)

    assert (result[:6, :5] == 3).all()  # real data stays untouched
    assert (result[:, 5:] == EMPTY_LAYER_VALUE).all()  # columns beyond the data
    assert (result[6:, :] == EMPTY_LAYER_VALUE).all()  # rows beyond the data
    assert (layer_map == 3).all()  # input is not modified


def test_mark_padding_as_holes_without_padding_is_a_noop():
    from world_to_beamng.terrain.terrain_materials import mark_padding_as_holes

    layer_map = np.full((4, 4), 1, dtype=np.uint8)

    assert (mark_padding_as_holes(layer_map, data_cols=4, data_rows=4) == 1).all()


# --- Four-image mode: one photo and one variant of each land use layer per tile ----------------------


def _tile_entries():
    names = ["aerial_photo_0", "aerial_photo_1", "mat_forest_t0", "mat_grass_t1"]
    parents = {"mat_forest_t0": ("mat_forest", "aerial_photo_0"), "mat_grass_t1": ("mat_grass", "aerial_photo_1")}
    return build_terrain_material_entries(
        names,
        ["aerial_photo_0", "aerial_photo_1"],
        LANDUSE_MAPPINGS_FIXTURE,
        "world_to_beamng",
        4096.0,
        _FAKE_PLACEHOLDERS,
        variant_parents=parents,
        photo_extents={"aerial_photo_0": 2000.0, "aerial_photo_1": 2000.0},
    )


def test_each_tile_photo_is_its_own_material_with_the_tile_extent():
    entries = _tile_entries()

    for k in (0, 1):
        photo = entries[f"aerial_photo_{k}"]
        assert photo["internalName"] == f"aerial_photo_{k}"
        assert photo["baseColorBaseTex"] == f"/levels/world_to_beamng/art/shapes/textures/aerial_photo_{k}.png"
        assert photo["baseColorBaseTexSize"] == 2000.0  # tile, not the whole area (4096)


def test_landuse_variants_use_the_photo_of_their_own_tile_and_keep_the_detail_texture():
    entries = _tile_entries()

    forest, grass = entries["mat_forest_t0"], entries["mat_grass_t1"]
    assert forest["internalName"] == "mat_forest_t0" and grass["internalName"] == "mat_grass_t1"
    assert forest["baseColorBaseTex"].endswith("/aerial_photo_0.png")
    assert grass["baseColorBaseTex"].endswith("/aerial_photo_1.png")
    assert forest["baseColorBaseTexSize"] == grass["baseColorBaseTexSize"] == 2000.0
    assert forest["baseColorDetailTex"] == "a/forest_b.png"  # detail comes from the layer, not from the name
    assert grass["baseColorDetailTex"] == "a/grass_b.png"
    assert forest["groundmodelName"] == "GRASS"


def test_variants_have_unique_persistent_ids_and_all_required_slots():
    entries = _tile_entries()

    assert len({e["persistentId"] for e in entries.values()}) == len(entries)
    for entry in entries.values():
        for field in _REQUIRED_TERRAIN_TEX_FIELDS:
            assert field in entry


def test_single_photo_mode_is_unchanged_without_variants():
    entries = build_terrain_material_entries(
        ["aerial_photo", "mat_forest"], ["aerial_photo"], LANDUSE_MAPPINGS_FIXTURE, "world_to_beamng", 2048.0, _FAKE_PLACEHOLDERS
    )

    assert entries["mat_forest"]["baseColorBaseTex"].endswith("/aerial_photo.png")
    assert entries["mat_forest"]["baseColorBaseTexSize"] == 2048.0


def test_photo_materials_have_an_explicit_ground_model():
    # Without a ground model BeamNG logs "ground model not found for collision: 'AERIAL_PHOTO_0' - using asphalt"
    entries = build_terrain_material_entries(
        ["aerial_photo_0", "aerial_photo_1"], ["aerial_photo_0", "aerial_photo_1"], LANDUSE_MAPPINGS_FIXTURE,
        "world_to_beamng", 4096.0, _FAKE_PLACEHOLDERS,
    )

    assert entries["aerial_photo_0"]["groundmodelName"] == entries["aerial_photo_1"]["groundmodelName"] == "ASPHALT"


def _reference_paint(layer_map, size, origin_x, origin_y, square_size, shapes):
    """Previous method: rasterize every polygon over the full map."""
    from affine import Affine
    from rasterio.features import rasterize

    transform = Affine.translation(origin_x, origin_y) * Affine.scale(square_size, square_size)
    result = layer_map.copy()
    for geometry, index in shapes:
        mask = rasterize([(geometry, index)], out_shape=(size, size), transform=transform, fill=255, dtype="uint8")
        hit = mask != 255
        result[hit] = mask[hit]
    return result


def test_paint_landuse_matches_full_grid_rasterization_for_windowed_burning():
    from shapely.geometry import MultiPolygon, box

    size, origin_x, origin_y, square_size = 64, -30.0, 12.5, 2.0  # map covers x -30..98, y 12.5..140.5
    rng = np.random.RandomState(4)
    geometries = [
        box(0, 20, 40, 60),
        box(-60, -40, -10, 30),  # extends beyond the map on the left/bottom
        box(80, 120, 200, 300),  # extends beyond on the right/top
        box(500, 500, 520, 520),  # completely outside
        Polygon([(10, 30), (70, 40), (50, 100)], holes=[[(30, 45), (45, 48), (38, 60)]]),
        MultiPolygon([box(-20, 60, 0, 80), box(60, 60, 90, 90)]),
        box(20.3, 33.7, 20.9, 34.1),  # smaller than one cell
    ]
    for _ in range(12):
        x, y = rng.uniform(-40, 100), rng.uniform(0, 150)
        geometries.append(Polygon(np.column_stack([x + rng.uniform(0, 30, 5), y + rng.uniform(0, 30, 5)])).convex_hull)

    categories = ["forest", "meadow", "farmland", "urban"]
    polygons = [
        {"osm_tags": {"landuse": {"forest": "forest", "meadow": "meadow", "farmland": "farmland", "urban": "residential"}[categories[i % 4]]}, "geometry": g}
        for i, g in enumerate(geometries)
    ]
    layer_map = np.full((size, size), 7, dtype=np.uint8)
    names = ["aerial_photo"] * 1

    result, result_names = paint_landuse_materials(
        layer_map, names, size, origin_x, origin_y, square_size, polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    scored = []
    for polygon in polygons:
        category = get_landuse_category(polygon["osm_tags"], LANDUSE_MAPPINGS_FIXTURE)
        data = LANDUSE_MAPPINGS_FIXTURE[category]
        index = 0 if data.get("keep_photo") else result_names.index(data["internal_name"])
        scored.append((data["priority"], polygon["geometry"], index))
    scored.sort(key=lambda item: item[0])
    expected = _reference_paint(layer_map, size, origin_x, origin_y, square_size, [(g, i) for _, g, i in scored])

    assert (result == expected).all()
    assert (result != 7).any()


def test_paint_landuse_without_background_category_leaves_unmapped_area_as_photo():
    # Previous behavior (no background_category passed): areas without any land use polygon
    # stay with the aerial photo - this is exactly the gap reported by the user that background_category closes.
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)
    forest_polygon = Polygon([(5, 5), (10, 5), (10, 10), (5, 10)])
    landuse_polygons = [{"osm_tags": {"landuse": "forest"}, "geometry": forest_polygon}]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    assert new_layer_map[18, 18] == 0  # far from the forest polygon: stays aerial photo, no grass


def test_paint_landuse_background_category_fills_unmapped_area_with_meadow():
    # Regression: areas WITHOUT any land use polygon (no OSM element covers them) never used to get
    # grass - DEFAULT_LANDUSE_CATEGORY (get_landuse_category()) only applies to an EXISTING but
    # unknown landuse tag VALUE, not when no element exists at all. background_category first fills the
    # ENTIRE area with the given category, before real polygons are burned in on top.
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)
    forest_polygon = Polygon([(5, 5), (10, 5), (10, 10), (5, 10)])
    landuse_polygons = [{"osm_tags": {"landuse": "forest"}, "geometry": forest_polygon}]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE,
        background_category="meadow",
    )

    assert "mat_grass" in new_names
    assert new_layer_map[18, 18] == new_names.index("mat_grass")  # far from the forest: now meadow instead of photo
    assert new_layer_map[7, 7] == new_names.index("mat_forest")  # forest polygon still covers the background


def test_paint_landuse_background_category_is_overridden_by_any_real_polygon_regardless_of_priority():
    # The background is painted BEFORE the priority-sorted loop, not as a participant in it - a
    # low-priority real polygon (residential, priority 3 < meadow 4) must still cover it.
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)
    residential = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    landuse_polygons = [{"osm_tags": {"landuse": "residential"}, "geometry": residential}]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE,
        background_category="meadow",
    )

    assert new_layer_map[10, 10] == 0  # residential area (keep_photo) wins despite lower priority than meadow
    assert new_layer_map[1, 1] == new_names.index("mat_grass")  # outside: background meadow


def test_paint_landuse_background_category_with_keep_photo_is_a_noop():
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, [], LANDUSE_MAPPINGS_FIXTURE, background_category="urban",
    )

    assert new_names == names
    assert (new_layer_map == 0).all()


def test_paint_landuse_unknown_background_category_is_a_noop():
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, [], LANDUSE_MAPPINGS_FIXTURE, background_category="not_a_category",
    )

    assert new_names == names
    assert (new_layer_map == 0).all()


def test_ground_under_water_is_meadow_even_inside_a_forest():
    size = 20
    layer_map, names = build_photo_fallback_layer(size=size)
    forest = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    pond = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    landuse_polygons = [
        {"osm_tags": {"landuse": "forest"}, "geometry": forest},
        {"osm_tags": {"natural": "water"}, "geometry": pond},
    ]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    assert new_layer_map[10, 10] == new_names.index("mat_grass")  # under the water: meadow, not forest or aerial photo
    assert new_layer_map[1, 1] == new_names.index("mat_forest")
    assert "mat_water" not in new_names and len(new_names) == 3  # no water material of its own
