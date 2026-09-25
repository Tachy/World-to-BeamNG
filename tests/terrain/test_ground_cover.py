"""Tests for world_to_beamng.terrain.ground_cover and the photo masking of the layer.

Grass blades etc. are separate GroundCover objects in BeamNG (billboards from a
texture atlas), bound to a layer via the name of a terrain material (`layer`) -
they do NOT grow on their own from the terrain texture.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import Polygon

from world_to_beamng.terrain.ground_cover import build_billboard_material_entries, build_ground_cover_items
from world_to_beamng.terrain.terrain_materials import mask_layer_map_with_photo

TEMPLATES_DATA = {
    "billboard_materials": {
        "m_grass": {"name": "m_grass", "mapTo": "m_grass", "class": "Material", "Stages": [{"baseColorMap": "/assets/x.png"}]},
        "m_flowers": {"name": "m_flowers", "mapTo": "m_flowers", "class": "Material", "Stages": [{}]},
    },
    "templates": {
        "grass_short": {
            "material": "m_grass",
            "radius": 120,
            "gridSize": 6,
            "dissolveRadius": 80,
            "shapeCullRadius": 90,
            "maxBillboardTiltAngle": 40,
            "types": [
                {"billboardUVs": [0, 0, 1, 0.5], "sizeMin": 0.1, "sizeMax": 0.2},
                {"billboardUVs": [0, 0.5, 1, 0.5], "sizeMin": 0.1, "sizeMax": 0.3, "probability": 0.5},
            ],
        },
        "flowers": {"material": "m_flowers", "radius": 50, "types": [{"billboardUVs": [0, 0, 0.5, 0.5]}]},
    },
}

MAPPINGS = {
    "meadow": {"osm_tags": {"landuse": ["meadow"]}, "internal_name": "mat_grass", "groundCover": ["grass_short", "flowers"]},
    "forest": {"osm_tags": {"landuse": ["forest"]}, "internal_name": "mat_forest", "groundCover": ["grass_short"]},
    "urban": {"osm_tags": {"landuse": ["residential"]}, "keep_photo": True},
}


def _items(used_layers, **kwargs):
    return build_ground_cover_items(MAPPINGS, used_layers, TEMPLATES_DATA, max_elements=1000, max_radius=100, **kwargs)


def test_one_ground_cover_object_per_used_layer_and_template():
    items = _items(["aerial_photo", "mat_grass"])

    assert sorted(i["name"] for i in items) == ["gc_mat_grass_flowers", "gc_mat_grass_grass_short"]


def test_layers_not_painted_in_the_terrain_get_no_ground_cover():
    # A layer that does not occur in any cell of the layer map needs no objects
    items = _items(["aerial_photo"])

    assert items == []


def test_every_type_is_bound_to_the_terrain_layer():
    # Without layer, the vegetation would grow on ALL terrain materials
    items = _items(["aerial_photo", "mat_grass", "mat_forest"])

    for item in items:
        layers = {t["layer"] for t in item["Types"]}
        assert layers == {"mat_grass" if "mat_grass" in item["name"] else "mat_forest"}


def test_item_uses_template_material_types_and_object_fields():
    item = next(i for i in _items(["aerial_photo", "mat_grass"]) if i["name"].endswith("grass_short"))

    assert item["material"] == "m_grass"
    assert len(item["Types"]) == 2
    assert item["Types"][0]["billboardUVs"] == [0, 0, 1, 0.5]
    assert item["Types"][1]["probability"] == 0.5
    assert item["gridSize"] == 6
    assert item["maxBillboardTiltAngle"] == 40


def test_radius_is_capped_and_dependent_radii_stay_inside():
    item = next(i for i in _items(["aerial_photo", "mat_grass"]) if i["name"].endswith("grass_short"))

    assert item["radius"] == 100  # template 120 -> cap 100
    assert item["dissolveRadius"] <= item["radius"]
    assert item["shapeCullRadius"] <= item["radius"]


def test_max_elements_comes_from_the_argument():
    for item in _items(["aerial_photo", "mat_grass"]):
        assert item["maxElements"] == 1000


def test_photo_categories_and_unknown_templates_are_skipped():
    mappings = {**MAPPINGS, "orchard": {"internal_name": "mat_orchard", "groundCover": ["does_not_exist"]}}

    items = build_ground_cover_items(mappings, ["aerial_photo", "mat_orchard"], TEMPLATES_DATA, max_elements=10, max_radius=100)

    assert items == []


def test_item_names_are_unique():
    items = _items(["aerial_photo", "mat_grass", "mat_forest"])

    names = [i["name"] for i in items]
    assert len(names) == len(set(names))


def test_billboard_materials_only_for_used_templates_with_persistent_ids():
    items = _items(["aerial_photo", "mat_forest"])  # only grass_short

    materials = build_billboard_material_entries(items, TEMPLATES_DATA)

    assert set(materials) == {"m_grass"}
    assert materials["m_grass"]["class"] == "Material"
    assert materials["m_grass"]["persistentId"]  # BeamNG needs unique IDs
    assert materials["m_grass"]["Stages"][0]["baseColorMap"] == "/assets/x.png"


def test_billboard_material_ids_are_unique_per_call():
    items = _items(["aerial_photo", "mat_grass"])

    materials = build_billboard_material_entries(items, TEMPLATES_DATA)
    ids = [m["persistentId"] for m in materials.values()]

    assert len(ids) == len(set(ids)) == 2


def test_mask_layer_map_with_photo_resets_cells_under_geometry():
    layer_map = np.full((20, 20), 2, dtype=np.uint8)
    road = Polygon([(0, 8), (20, 8), (20, 12), (0, 12)])

    result = mask_layer_map_with_photo(layer_map, 20, 0.0, 0.0, 1.0, [road])

    assert result[10, 10] == 0  # under the road: aerial photo, nothing grows there
    assert result[2, 2] == 2  # next to it, the land use stays
    assert (layer_map == 2).all()  # input stays unchanged


def test_mask_layer_map_with_photo_buffer_widens_the_mask():
    layer_map = np.full((20, 20), 2, dtype=np.uint8)
    road = Polygon([(0, 9), (20, 9), (20, 11), (0, 11)])

    narrow = mask_layer_map_with_photo(layer_map, 20, 0.0, 0.0, 1.0, [road])
    wide = mask_layer_map_with_photo(layer_map, 20, 0.0, 0.0, 1.0, [road], buffer=2.0)

    assert (wide == 0).sum() > (narrow == 0).sum()


def test_mask_layer_map_with_photo_without_geometries_is_a_copy():
    layer_map = np.full((5, 5), 3, dtype=np.uint8)

    result = mask_layer_map_with_photo(layer_map, 5, 0.0, 0.0, 1.0, [])

    assert (result == 3).all()
    assert result is not layer_map


# --- Dense tall grass: the real templates and the meadow mapping -------------------------------
# In BeamNG's originals, a dense "close" preset (grid 4, ~1.3 elements/m², blades up to 1.2 m) sits on top of
# the thin "distant" far layer (grid 6-8, ~0.3-0.6 elements/m²). The far layer alone
# yields only single small blades.

REAL_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "data" / "ground_cover_templates.json"
REAL_MAPPINGS_PATH = Path(__file__).parent.parent.parent / "data" / "osm_to_beamng.json"


def _real_templates():
    import json

    return json.loads(REAL_TEMPLATES_PATH.read_text(encoding="utf-8"))["templates"]


def _clumps_per_m2(template):
    grid = template["gridSize"]
    total = sum(
        t.get("probability", 1.0) * (t.get("minClumpCount", 1) + t.get("maxClumpCount", 1)) / 2 for t in template["types"]
    )
    return total / (grid * grid)


@pytest.mark.parametrize("name", ["grass_medium_close", "dry_grass_medium_close"])
def test_dense_close_presets_are_present_and_really_dense(name):
    template = _real_templates()[name]

    assert template["gridSize"] <= 4  # tight grid = dense clumps
    assert template["radius"] <= 60  # "close": near range only, the thin layer covers the distance
    assert max(t.get("sizeMax", 1) for t in template["types"]) >= 1.0  # tall blades
    assert _clumps_per_m2(template) >= 1.0
    # 6 real billboard types (the two further types of the originals are empty placeholders without shape/UVs)
    assert len(template["types"]) >= 6


def test_meadow_gets_a_dense_tall_grass_preset_besides_the_distant_layers():
    import json

    meadow = json.loads(REAL_MAPPINGS_PATH.read_text(encoding="utf-8"))["landuse_mappings"]["meadow"]
    templates = _real_templates()

    dense = [n for n in meadow["groundCover"] if templates[n]["gridSize"] <= 4 and "grass" in n]
    assert dense, "meadow without dense grass preset"
    # the thin far layer is kept for the long view
    assert any(templates[n]["radius"] >= 100 for n in meadow["groundCover"] if "grass" in n)


def test_meadow_grass_is_much_denser_than_the_old_distant_only_setup():
    import json

    meadow = json.loads(REAL_MAPPINGS_PATH.read_text(encoding="utf-8"))["landuse_mappings"]["meadow"]
    templates = _real_templates()

    grass = [n for n in meadow["groundCover"] if "grass" in n]
    density = sum(_clumps_per_m2(templates[n]) for n in grass)
    old_density = _clumps_per_m2(templates["grass_short"]) + _clumps_per_m2(templates["grass_long"])
    assert density > 1.8 * old_density


# --- Four-image mode: one separate object per tile variant ---------------------------------------------
# A GroundCover object carries at most 8 types (all 229 objects in BeamNG's original levels have exactly 8;
# Torque: MAX_COVERTYPES = 8). If the types are multiplied for several tile layers (24-32 types), the
# grass is missing entirely. Therefore each variant gets its own object with the types of the template.

MAX_COVER_TYPES = 8


def test_each_tile_variant_gets_its_own_object_with_the_templates_own_types():
    variants = {"mat_grass": ["mat_grass_t0", "mat_grass_t1"]}

    items = build_ground_cover_items(
        MAPPINGS, ["mat_grass"], TEMPLATES_DATA, max_elements=1000, max_radius=100, layer_variants=variants
    )

    short = [i for i in items if i["name"].endswith("grass_short")]
    assert sorted(i["name"] for i in short) == ["gc_mat_grass_t0_grass_short", "gc_mat_grass_t1_grass_short"]
    template_types = len(TEMPLATES_DATA["templates"]["grass_short"]["types"])
    for item in short:
        assert len(item["Types"]) == template_types  # not multiplied
    assert {t["layer"] for t in short[0]["Types"]} == {"mat_grass_t0"}
    assert {t["layer"] for t in short[1]["Types"]} == {"mat_grass_t1"}


def test_variant_objects_share_material_and_settings_with_the_single_photo_object():
    variants = {"mat_grass": ["mat_grass_t0", "mat_grass_t1"]}

    tiled = build_ground_cover_items(
        MAPPINGS, ["mat_grass"], TEMPLATES_DATA, max_elements=1000, max_radius=100, layer_variants=variants
    )
    single = build_ground_cover_items(MAPPINGS, ["mat_grass"], TEMPLATES_DATA, max_elements=1000, max_radius=100)

    for item in tiled:
        reference = next(i for i in single if i["name"] == item["name"].replace("_t0_", "_").replace("_t1_", "_"))
        assert {k: v for k, v in item.items() if k not in ("name", "Types")} == {k: v for k, v in reference.items() if k not in ("name", "Types")}


def test_layers_without_variants_keep_their_name_and_bind_to_themselves():
    items = build_ground_cover_items(
        MAPPINGS, ["mat_grass", "mat_forest"], TEMPLATES_DATA, max_elements=1000, max_radius=100,
        layer_variants={"mat_grass": ["mat_grass_t0"]},
    )

    forest = next(i for i in items if i["name"].startswith("gc_mat_forest"))
    grass = next(i for i in items if i["name"].startswith("gc_mat_grass"))
    assert forest["name"] == "gc_mat_forest_grass_short" and {t["layer"] for t in forest["Types"]} == {"mat_forest"}
    assert {t["layer"] for t in grass["Types"]} == {"mat_grass_t0"}


def test_object_names_are_unique_with_variants():
    variants = {"mat_grass": ["mat_grass_t0", "mat_grass_t1", "mat_grass_t2", "mat_grass_t3"]}

    items = build_ground_cover_items(
        MAPPINGS, ["mat_grass", "mat_forest"], TEMPLATES_DATA, max_elements=1000, max_radius=100, layer_variants=variants
    )

    names = [i["name"] for i in items]
    assert len(names) == len(set(names))


def test_no_object_exceeds_the_engine_limit_of_8_types_in_any_mode():
    # with the REAL templates and land use categories, single-photo and four-image mode
    import json

    real = json.loads(REAL_TEMPLATES_PATH.read_text(encoding="utf-8"))
    mappings = json.loads(REAL_MAPPINGS_PATH.read_text(encoding="utf-8"))["landuse_mappings"]
    layers = [c["internal_name"] for c in mappings.values() if c.get("internal_name")]
    variants = {layer: [f"{layer}_t{k}" for k in range(4)] for layer in layers}

    for layer_variants in (None, variants):
        items = build_ground_cover_items(mappings, layers, real, max_elements=1000, max_radius=100, layer_variants=layer_variants)
        assert items
        assert max(len(i["Types"]) for i in items) <= MAX_COVER_TYPES
