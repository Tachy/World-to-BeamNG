"""
Tests for the mapping of OSM tags -> forest type.

Only landuse=forest gets the tall mixed forest; woods (natural=wood, landuse=wood) get the dense
deciduous forest of large deciduous trees with undergrowth; everything else with trees (natural=forest,
wetland, nature reserve, ...) gets only low deciduous trees.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import pytest

from world_to_beamng.forest.forest_normalizer import ForestNormalizer

CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "osm_to_beamng.json"
TALL_MIXED = "german_mixed_forest"
TALL_DECIDUOUS = "german_deciduous_dense"
LOW = "german_low_deciduous"
BROADLEAF = "german_broadleaf_undergrowth"


@pytest.fixture(scope="module")
def forest_config():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"forest_type_templates": data["forest_type_templates"], "forest_mappings": data["forest_mappings"]}


@pytest.fixture(scope="module")
def normalizer(forest_config):
    return ForestNormalizer(forest_config, osm_mapper=None)


def test_landuse_forest_gets_the_tall_mixed_forest(normalizer):
    assert normalizer._map_to_forest_type({"landuse": "forest"}) == TALL_MIXED


@pytest.mark.parametrize("tags", [{"natural": "wood"}, {"landuse": "wood"}, {"natural": "wood", "layer": "-2"}])
def test_woods_get_dense_broadleaf_with_undergrowth(normalizer, tags):
    assert normalizer._map_to_forest_type(tags) == BROADLEAF


@pytest.mark.parametrize("tags", [{"natural": "forest"}, {"natural": "wetland"}, {"leisure": "nature_reserve"}])
def test_everything_else_with_trees_gets_only_low_deciduous_trees(normalizer, tags):
    assert normalizer._map_to_forest_type(tags) == LOW


@pytest.mark.parametrize(
    "extra",
    [{"trees": "conifer"}, {"trees": "broadleaf"}, {"leaf_type": "needleleaf"}, {"leaf_type": "broadleaved"}],
)
def test_tag_overrides_never_redirect_a_wood_to_another_type(normalizer, extra):
    # natural=wood with a conifer/deciduous tree tag stays the dense deciduous forest (overrides only apply to landuse=forest)
    assert normalizer._map_to_forest_type({"natural": "wood", **extra}) == BROADLEAF
    assert normalizer._map_to_forest_type({"natural": "forest", **extra}) == LOW


def test_overrides_still_refine_landuse_forest(normalizer):
    # Conifer forest becomes mixed forest (no pure conifer forest type), deciduous forest the dense tall deciduous forest
    assert normalizer._map_to_forest_type({"landuse": "forest", "leaf_type": "needleleaf"}) == TALL_MIXED
    assert normalizer._map_to_forest_type({"landuse": "forest", "trees": "conifer"}) == TALL_MIXED
    assert normalizer._map_to_forest_type({"landuse": "forest", "leaf_type": "broadleaved"}) == TALL_DECIDUOUS


def test_low_area_types_are_unchanged(normalizer):
    assert normalizer._map_to_forest_type({"natural": "scrub"}) == "german_sparse_deciduous"
    assert normalizer._map_to_forest_type({"natural": "heath"}) == "german_sparse_deciduous"
    assert normalizer._map_to_forest_type({"leisure": "park"}) == "german_sparse_deciduous"
    assert normalizer._map_to_forest_type({"landuse": "orchard"}) == "orchard_area"
    assert normalizer._map_to_forest_type({"natural": "tree_row"}) == "tree_row"  # tree row (line), see test_forest_tree_rows


def test_overrides_without_a_scope_still_apply_everywhere():
    # Backward compatible: without "tag_overrides_only_for" the overrides apply to all as before
    config = {
        "forest_type_templates": {},
        "forest_mappings": {"natural": {"wood": "a"}, "tag_overrides": {"trees=conifer": "b"}},
    }
    assert ForestNormalizer(config, osm_mapper=None)._map_to_forest_type({"natural": "wood", "trees": "conifer"}) == "b"


def test_low_deciduous_template_has_only_low_deciduous_trees(forest_config):
    template = forest_config["forest_type_templates"][LOW]
    trees = template["preferred_trees"]

    assert len(trees) >= 8
    assert sum(trees.values()) == pytest.approx(1.0)
    # tall/conifer trees, groups and bushes are taboo (measured heights: forest/large/douglasfir >= 15 m)
    for name in trees:
        assert not any(bad in name for bad in ("douglasfir", "large", "forest", "dead", "bush", "wall", "blocker"))
    # only real, small deciduous trees
    assert all(any(kind in name for kind in ("aspen_small", "beech_small", "oak_sml")) for name in trees)


def test_low_deciduous_scale_keeps_trees_low(forest_config):
    # average_height acts as a scale (target height / 20 m): low forest must stay below the tall one
    low = forest_config["forest_type_templates"][LOW]["average_height"]
    tall = forest_config["forest_type_templates"][TALL_MIXED]["average_height"]

    assert max(low) <= max(tall) and min(low) <= min(tall)
    assert max(low) / 20.0 <= 1.2  # scale <= 1.2: small assets (max 12 m) stay below ~15 m
    assert min(low) / 20.0 >= 0.5  # do not fall below the scale clamp


def test_generator_produces_the_same_rules_as_the_committed_json(forest_config):
    import generate_forest_assets as gen

    trees_by_type = {k: [k] for k in json.loads(
        (Path(__file__).parent.parent.parent / "data" / "osm_to_beamng.json").read_text(encoding="utf-8")
    )["forest_type_templates"][LOW]["preferred_trees"]}
    trees_by_type.update({t: [t] for t in forest_config["forest_type_templates"][TALL_MIXED]["preferred_trees"]})
    trees_by_type.update({t: [t] for t in forest_config["forest_type_templates"][TALL_DECIDUOUS]["preferred_trees"]})
    trees_by_type.update({t: [t] for t in forest_config["forest_type_templates"]["german_sparse_deciduous"]["preferred_trees"]})
    trees_by_type.update({t: [t] for t in forest_config["forest_type_templates"][BROADLEAF]["preferred_trees"]})
    types = gen.generate_forest_types(trees_by_type)
    mappings = gen.generate_forest_mappings(types)

    assert LOW in types
    assert mappings["landuse"]["forest"] == TALL_MIXED
    assert mappings["landuse"]["wood"] == BROADLEAF
    assert mappings["natural"]["wood"] == BROADLEAF
    assert mappings["natural"]["forest"] == LOW
    assert mappings["tag_overrides_only_for"] == ["landuse=forest"]
    assert mappings["clearings"]["forest_type"] == LOW
    assert set(types[LOW]["preferred_trees"]) <= set(gen.LOW_DECIDUOUS_TREES)
    assert types[BROADLEAF] == forest_config["forest_type_templates"][BROADLEAF]
    assert "osm_id_overrides" not in mappings  # rule instead of a one-off exception


# --- Clearings (inner rings of forest relations) --------------------------------------------


class _AlwaysForest:
    """Minimal OSMMapper stand-in: every element passed in counts as forest."""

    forest_mappings = {}

    def is_forest(self, tags):
        return True


def _way(way_id, points):
    return {"type": "way", "id": way_id, "geometry": [{"x": x, "y": y} for x, y in points], "tags": {}}


def _relation(outer_ids, inner_ids=(), tags=None):
    members = [{"type": "way", "ref": i, "role": "outer"} for i in outer_ids]
    members += [{"type": "way", "ref": i, "role": "inner"} for i in inner_ids]
    return {"type": "relation", "id": 1, "members": members, "tags": tags or {"landuse": "forest", "type": "multipolygon"}}


def _normalize(forest_config, osm_data):
    normalizer = ForestNormalizer(forest_config, osm_mapper=_AlwaysForest())
    result = normalizer.normalize_tile((-500, -500, 500, 500), "t", osm_data=osm_data)
    assert result["status"] == "success"
    return result["forests"]


SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
HOLE_RING = [(40, 40), (60, 40), (60, 60), (40, 60), (40, 40)]


def test_clearing_inside_a_forest_relation_gets_only_low_deciduous_trees(forest_config):
    from shapely.geometry import Point

    forests = _normalize(forest_config, [_way(10, SQUARE), _way(11, HOLE_RING), _relation([10], [11])])

    by_type = {f["type"]: f for f in forests}
    assert set(by_type) == {TALL_MIXED, LOW}
    tall, clearing = by_type[TALL_MIXED]["geometry"], by_type[LOW]["geometry"]
    assert tall.area == pytest.approx(100 * 100 - 20 * 20)  # tall forest without the clearing
    assert not tall.contains(Point(50, 50))
    assert clearing.area == pytest.approx(20 * 20)  # the clearing itself
    assert clearing.contains(Point(50, 50))


def test_relation_without_inner_ring_stays_a_single_tall_forest(forest_config):
    forests = _normalize(forest_config, [_way(10, SQUARE), _relation([10])])

    assert [f["type"] for f in forests] == [TALL_MIXED]
    assert forests[0]["geometry"].area == pytest.approx(100 * 100)


def test_outer_ring_split_over_several_ways_is_assembled_not_chord_closed(forest_config):
    # Concave ring (U shape, area 8800) from two open ways, as usual in OSM: closing each way individually
    # with a chord would yield overlapping/missing areas
    first = _way(10, [(0, 0), (100, 0), (100, 100), (60, 100)])
    second = _way(12, [(60, 100), (60, 40), (40, 40), (40, 100), (0, 100), (0, 0)])

    forests = _normalize(forest_config, [first, second, _relation([10, 12])])

    assert len(forests) == 1
    assert forests[0]["geometry"].area == pytest.approx(100 * 100 - 20 * 60)


def test_clearing_type_is_configurable_and_falls_back_to_the_forest_type(forest_config):
    config = {
        "forest_type_templates": forest_config["forest_type_templates"],
        "forest_mappings": {k: v for k, v in forest_config["forest_mappings"].items() if k != "clearings"},
    }

    forests = _normalize(config, [_way(10, SQUARE), _way(11, HOLE_RING), _relation([10], [11])])

    # without a "clearings" entry the clearing is treated like the surrounding forest, but geometrically correct
    assert {f["type"] for f in forests} == {TALL_MIXED}
    assert sum(f["geometry"].area for f in forests) == pytest.approx(100 * 100)


# --- Gardens, allotments, residential areas: small trees and bushes -----------------------------

GARDEN = "garden_mixed"
RESIDENTIAL = "residential_green"
SINGLE = "single_tree"
BUSH_MARKERS = ("bush",)


@pytest.mark.parametrize("tags", [{"leisure": "garden"}, {"landuse": "allotments"}])
def test_gardens_and_allotments_get_small_trees_and_bushes(normalizer, tags):
    assert normalizer._map_to_forest_type(tags) == GARDEN


def test_residential_areas_get_bushes(normalizer):
    assert normalizer._map_to_forest_type({"landuse": "residential"}) == RESIDENTIAL


def test_garden_template_mixes_small_trees_and_bushes_and_stays_low(forest_config):
    trees = forest_config["forest_type_templates"][GARDEN]["preferred_trees"]

    assert sum(trees.values()) == pytest.approx(1.0)
    bushes = {n: w for n, w in trees.items() if "bush" in n}
    small_trees = {n: w for n, w in trees.items() if "bush" not in n}
    assert bushes and small_trees
    assert 0.3 <= sum(small_trees.values()) <= 0.6  # trees and bushes, bushes slightly predominant
    for name in small_trees:
        assert any(kind in name for kind in ("aspen_small", "beech_small", "oak_sml"))
    for name in trees:
        assert not any(bad in name for bad in ("douglasfir", "large", "forest", "dead", "group", "wall"))


def test_residential_template_has_only_bushes(forest_config):
    trees = forest_config["forest_type_templates"][RESIDENTIAL]["preferred_trees"]

    assert sum(trees.values()) == pytest.approx(1.0)
    assert len(trees) >= 5
    assert all("bush" in name for name in trees)


def test_garden_and_residential_are_sparse_enough_not_to_look_like_a_forest(forest_config):
    templates = forest_config["forest_type_templates"]

    # ForestWorkflow uses 5 m minimum spacing: spacing = 5 / sqrt(density); target about 7-14 m
    for name in (GARDEN, RESIDENTIAL):
        spacing = 5.0 / templates[name]["tree_density"] ** 0.5
        assert 7.0 <= spacing <= 14.0, f"{name}: Abstand {spacing:.1f} m"
    assert templates[GARDEN]["average_height"][1] / 20.0 <= 1.2  # no oversized scales


def test_single_tree_template_exists_and_is_configured(forest_config):
    template = forest_config["forest_type_templates"][SINGLE]

    assert forest_config["forest_mappings"]["single_trees"] == {"forest_type": SINGLE}
    assert sum(template["preferred_trees"].values()) == pytest.approx(1.0)


def test_single_trees_are_large_broad_deciduous_trees(forest_config):
    # OSM natural=tree: free-standing large deciduous trees (oak/beech with a broad crown), no small trees,
    # no slender forest trunks (*_forest_*), no aspens (only about 10 m) and no conifers
    trees = forest_config["forest_type_templates"][SINGLE]["preferred_trees"]

    assert len(trees) >= 4
    for name in trees:
        assert name.startswith(("tree_oak_large_", "tree_beech_large_")), name


def test_single_trees_are_scaled_up_not_down(forest_config):
    # average_height acts as a scale (target height / 20 m); the assets are 13-21 m tall
    low, high = forest_config["forest_type_templates"][SINGLE]["average_height"]

    assert low / 20.0 >= 0.8  # not below about 80 % of the original size
    assert high / 20.0 <= 1.3  # and not disproportionately large


def test_tree_rows_keep_their_small_trees(forest_config):
    trees = forest_config["forest_type_templates"]["tree_row"]["preferred_trees"]

    assert all(any(kind in n for kind in ("aspen_small", "beech_small", "oak_sml")) for n in trees)


def test_holes_of_non_forest_relations_are_not_planted_as_clearings(forest_config):
    # A hole in a residential area is something else (field, forest, pond): "clearings" does not plant there
    residential = {"landuse": "residential", "type": "multipolygon"}

    forests = _normalize(forest_config, [_way(10, SQUARE), _way(11, HOLE_RING), _relation([10], [11], tags=residential)])

    assert [f["type"] for f in forests] == [RESIDENTIAL]
    assert forests[0]["geometry"].area == pytest.approx(100 * 100 - 20 * 20)  # hole is left out anyway


def test_generator_produces_garden_rules_too(forest_config):
    import generate_forest_assets as gen

    every_tree = set()
    for name in (GARDEN, RESIDENTIAL, SINGLE, LOW, TALL_MIXED, TALL_DECIDUOUS, "german_sparse_deciduous"):
        every_tree |= set(forest_config["forest_type_templates"][name]["preferred_trees"])
    types = gen.generate_forest_types({t: [t] for t in every_tree})
    mappings = gen.generate_forest_mappings(types)

    assert {GARDEN, RESIDENTIAL, SINGLE} <= set(types)
    assert mappings["leisure"]["garden"] == GARDEN
    assert mappings["landuse"]["allotments"] == GARDEN
    assert mappings["landuse"]["residential"] == RESIDENTIAL
    assert mappings["single_trees"] == {"forest_type": SINGLE}
    assert set(types[SINGLE]["preferred_trees"]) <= set(gen.LARGE_DECIDUOUS_TREES)
    assert all("large" not in t for t in types["tree_row"]["preferred_trees"])  # rows stay small
    assert mappings["clearings"]["only_for"]  # clearings only for forest relations


# --- Orchards: about 5 m tall, round-crowned deciduous trees ----------------------------------------------

ORCHARD = "orchard_area"
# measured model heights of the assets (Z extent of the DAE) - broad-crowned small deciduous trees, crown about half as wide as tall
ORCHARD_MODEL_HEIGHTS = {"tree_oak_sml_a": 8.5, "tree_oak_sml_b": 9.5, "tree_beech_small_c": 9.9}


def test_orchard_maps_to_the_orchard_template(normalizer):
    assert normalizer._map_to_forest_type({"landuse": "orchard"}) == ORCHARD


def test_orchard_has_only_broad_crowned_deciduous_trees_no_bushes_no_slender_aspens(forest_config):
    trees = forest_config["forest_type_templates"][ORCHARD]["preferred_trees"]

    assert set(trees) <= set(ORCHARD_MODEL_HEIGHTS) and len(trees) >= 3
    assert sum(trees.values()) == pytest.approx(1.0)
    assert not any(bad in name for name in trees for bad in ("bush", "aspen", "douglasfir", "large", "forest"))


def test_orchard_trees_end_up_about_5_m_tall(forest_config):
    # Final height = model height x scale; scale = target height / 20 m, clamped to 0.5..2.0 (ForestInstanceGenerator)
    template = forest_config["forest_type_templates"][ORCHARD]
    low, high = template["average_height"]
    scale_low, scale_high = (min(2.0, max(0.5, v / 20.0)) for v in (low, high))
    heights = [ORCHARD_MODEL_HEIGHTS[n] * s for n in template["preferred_trees"] for s in (scale_low, scale_high)]
    mean = sum(ORCHARD_MODEL_HEIGHTS[n] * (scale_low + scale_high) / 2 for n in template["preferred_trees"]) / len(template["preferred_trees"])

    assert 4.6 <= mean <= 5.6  # about 5 m
    assert min(heights) >= 3.8 and max(heights) <= 6.5  # no dwarfs, no giants


def test_orchard_spacing_looks_like_an_orchard_not_a_forest(forest_config):
    spacing = 5.0 / forest_config["forest_type_templates"][ORCHARD]["tree_density"] ** 0.5  # ForestWorkflow: 5 m minimum spacing

    assert 7.0 <= spacing <= 11.0


def test_generator_emits_the_same_orchard_rules(forest_config):
    import generate_forest_assets as gen

    every_tree = set()
    for template in forest_config["forest_type_templates"].values():
        every_tree |= set(template["preferred_trees"])
    types = gen.generate_forest_types({t: [t] for t in every_tree})
    committed = forest_config["forest_type_templates"][ORCHARD]

    assert set(types[ORCHARD]["preferred_trees"]) == set(committed["preferred_trees"]) <= set(gen.ORCHARD_TREES)
    assert types[ORCHARD]["average_height"] == committed["average_height"]
    assert types[ORCHARD]["tree_density"] == committed["tree_density"]


# --- Dense forest of large deciduous trees with undergrowth (natural=wood / landuse=wood) ---------------------------------


def test_broadleaf_wood_is_all_large_deciduous_trees_plus_undergrowth(forest_config):
    template = forest_config["forest_type_templates"][BROADLEAF]
    trees = template["preferred_trees"]
    canopy = {n: w for n, w in trees.items() if "bush" not in n}
    undergrowth = {n: w for n, w in trees.items() if "bush" in n}

    assert sum(trees.values()) == pytest.approx(1.0)
    # all trees broad-crowned and large (oak/beech "large"), no conifers, no aspens/forest trunks/small trees
    assert canopy and all(any(kind in n for kind in ("oak_large", "beech_large")) for n in canopy)
    assert not any(bad in n for n in trees for bad in ("douglasfir", "fir", "dead", "aspen_small", "forest", "group"))
    # Undergrowth: bushes with a noticeable share
    assert undergrowth and 0.2 <= sum(undergrowth.values()) <= 0.5


def test_broadleaf_wood_is_much_denser_but_smaller_than_the_first_version(forest_config):
    templates = forest_config["forest_type_templates"]
    broadleaf, low = templates[BROADLEAF], templates[LOW]

    # Minimum spacing in ForestWorkflow = 5 m / sqrt(density): 2/3 of 5/sqrt(1.6) = 2.63 m -> density 3.6
    spacing = 5.0 / broadleaf["tree_density"] ** 0.5
    assert spacing == pytest.approx(2.0 / 3.0 * 5.0 / 1.6**0.5, rel=0.01)
    assert broadleaf["tree_density"] >= 3 * low["tree_density"]
    # half the tree size: scale (target height / 20 m) 0.5-0.65 instead of 1.0-1.3; not below the clamp of 0.5
    assert min(broadleaf["average_height"]) / 20.0 == pytest.approx(0.5)
    assert max(broadleaf["average_height"]) / 20.0 == pytest.approx(0.65)
