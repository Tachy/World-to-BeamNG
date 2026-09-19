"""
Tests für die Zuordnung OSM-Tags -> Waldtyp.

Nur landuse=forest bekommt den hohen Mischwald; alles andere mit Bäumen (natural=wood,
landuse=wood, Feuchtgebiet, Naturschutzgebiet, ...) bekommt nur niedrige Laubbäume.
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


@pytest.fixture(scope="module")
def forest_config():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"forest_type_templates": data["forest_type_templates"], "forest_mappings": data["forest_mappings"]}


@pytest.fixture(scope="module")
def normalizer(forest_config):
    return ForestNormalizer(forest_config, osm_mapper=None)


def test_landuse_forest_gets_the_tall_mixed_forest(normalizer):
    assert normalizer._map_to_forest_type({"landuse": "forest"}) == TALL_MIXED


@pytest.mark.parametrize(
    "tags",
    [
        {"natural": "wood"},
        {"landuse": "wood"},
        {"natural": "forest"},
        {"natural": "wetland"},
        {"leisure": "nature_reserve"},
    ],
)
def test_everything_else_with_trees_gets_only_low_deciduous_trees(normalizer, tags):
    assert normalizer._map_to_forest_type(tags) == LOW


@pytest.mark.parametrize(
    "extra",
    [{"trees": "conifer"}, {"trees": "broadleaf"}, {"leaf_type": "needleleaf"}, {"leaf_type": "broadleaved"}],
)
def test_tag_overrides_never_make_a_non_forest_polygon_tall(normalizer, extra):
    # natural=wood mit Nadel-/Laubbaum-Tag darf nicht in einen hohen Waldtyp umgeleitet werden
    assert normalizer._map_to_forest_type({"natural": "wood", **extra}) == LOW


def test_overrides_still_refine_landuse_forest(normalizer):
    # Nadelwald wird Mischwald (kein reiner Nadelwald-Typ), Laubwald der dichte hohe Laubwald
    assert normalizer._map_to_forest_type({"landuse": "forest", "leaf_type": "needleleaf"}) == TALL_MIXED
    assert normalizer._map_to_forest_type({"landuse": "forest", "trees": "conifer"}) == TALL_MIXED
    assert normalizer._map_to_forest_type({"landuse": "forest", "leaf_type": "broadleaved"}) == TALL_DECIDUOUS


def test_low_area_types_are_unchanged(normalizer):
    assert normalizer._map_to_forest_type({"natural": "scrub"}) == "german_sparse_deciduous"
    assert normalizer._map_to_forest_type({"natural": "heath"}) == "german_sparse_deciduous"
    assert normalizer._map_to_forest_type({"leisure": "park"}) == "german_sparse_deciduous"
    assert normalizer._map_to_forest_type({"landuse": "orchard"}) == "orchard_area"
    assert normalizer._map_to_forest_type({"natural": "tree_row"}) == "tree_row"  # Baumreihe (Linie), s. test_forest_tree_rows


def test_overrides_without_a_scope_still_apply_everywhere():
    # Abwärtskompatibel: ohne "tag_overrides_only_for" gelten die Overrides wie bisher für alle
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
    # hohe/Nadel-Bäume, Gruppen und Büsche sind tabu (gemessene Höhen: forest/large/douglasfir >= 15 m)
    for name in trees:
        assert not any(bad in name for bad in ("douglasfir", "large", "forest", "dead", "bush", "wall", "blocker"))
    # nur echte, kleine Laubbäume
    assert all(any(kind in name for kind in ("aspen_small", "beech_small", "oak_sml")) for name in trees)


def test_low_deciduous_scale_keeps_trees_low(forest_config):
    # average_height wirkt als Skalierung (Zielhöhe / 20 m): niedriger Wald muss unter dem hohen bleiben
    low = forest_config["forest_type_templates"][LOW]["average_height"]
    tall = forest_config["forest_type_templates"][TALL_MIXED]["average_height"]

    assert max(low) <= max(tall) and min(low) <= min(tall)
    assert max(low) / 20.0 <= 1.2  # Skalierung <= 1,2: kleine Assets (max 12 m) bleiben unter ~15 m
    assert min(low) / 20.0 >= 0.5  # nicht unter den Clamp der Skalierung fallen


def test_generator_produces_the_same_rules_as_the_committed_json(forest_config):
    import generate_forest_assets as gen

    trees_by_type = {k: [k] for k in json.loads(
        (Path(__file__).parent.parent.parent / "data" / "osm_to_beamng.json").read_text(encoding="utf-8")
    )["forest_type_templates"][LOW]["preferred_trees"]}
    trees_by_type.update({t: [t] for t in forest_config["forest_type_templates"][TALL_MIXED]["preferred_trees"]})
    trees_by_type.update({t: [t] for t in forest_config["forest_type_templates"][TALL_DECIDUOUS]["preferred_trees"]})
    trees_by_type.update({t: [t] for t in forest_config["forest_type_templates"]["german_sparse_deciduous"]["preferred_trees"]})
    types = gen.generate_forest_types(trees_by_type)
    mappings = gen.generate_forest_mappings(types)

    assert LOW in types
    assert mappings["landuse"]["forest"] == TALL_MIXED
    assert mappings["landuse"]["wood"] == LOW
    assert mappings["natural"]["wood"] == LOW
    assert mappings["tag_overrides_only_for"] == ["landuse=forest"]
    assert mappings["clearings"]["forest_type"] == LOW
    assert set(types[LOW]["preferred_trees"]) <= set(gen.LOW_DECIDUOUS_TREES)


# --- Lichtungen (innere Ringe von Wald-Relationen) ------------------------------------------


class _AlwaysForest:
    """Minimaler OSMMapper-Ersatz: jedes übergebene Element gilt als Wald."""

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
    assert tall.area == pytest.approx(100 * 100 - 20 * 20)  # hoher Wald ohne Lichtung
    assert not tall.contains(Point(50, 50))
    assert clearing.area == pytest.approx(20 * 20)  # die Lichtung selbst
    assert clearing.contains(Point(50, 50))


def test_relation_without_inner_ring_stays_a_single_tall_forest(forest_config):
    forests = _normalize(forest_config, [_way(10, SQUARE), _relation([10])])

    assert [f["type"] for f in forests] == [TALL_MIXED]
    assert forests[0]["geometry"].area == pytest.approx(100 * 100)


def test_outer_ring_split_over_several_ways_is_assembled_not_chord_closed(forest_config):
    # Konkaver Ring (U-Form, Fläche 8800) aus zwei offenen Ways, wie in OSM üblich: jeden Way einzeln
    # mit einer Sehne zu schließen ergäbe überlappende/fehlende Flächen
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

    # ohne "clearings"-Eintrag wird die Lichtung wie der umgebende Wald behandelt, aber geometrisch korrekt
    assert {f["type"] for f in forests} == {TALL_MIXED}
    assert sum(f["geometry"].area for f in forests) == pytest.approx(100 * 100)


# --- Gärten, Kleingärten, Wohngebiete: kleine Bäume und Büsche -----------------------------------

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
    assert 0.3 <= sum(small_trees.values()) <= 0.6  # Bäume und Büsche, Büsche leicht überwiegend
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

    # ForestWorkflow nutzt 5 m Mindestabstand: Abstand = 5 / sqrt(Dichte); Ziel ca. 7-14 m
    for name in (GARDEN, RESIDENTIAL):
        spacing = 5.0 / templates[name]["tree_density"] ** 0.5
        assert 7.0 <= spacing <= 14.0, f"{name}: Abstand {spacing:.1f} m"
    assert templates[GARDEN]["average_height"][1] / 20.0 <= 1.2  # keine übergroßen Skalierungen


def test_single_tree_template_exists_and_is_configured(forest_config):
    template = forest_config["forest_type_templates"][SINGLE]

    assert forest_config["forest_mappings"]["single_trees"] == {"forest_type": SINGLE}
    assert all(any(kind in n for kind in ("aspen_small", "beech_small", "oak_sml")) for n in template["preferred_trees"])


def test_holes_of_non_forest_relations_are_not_planted_as_clearings(forest_config):
    # Loch in einem Wohngebiet ist etwas anderes (Feld, Wald, Teich): dort pflanzt "clearings" nicht
    residential = {"landuse": "residential", "type": "multipolygon"}

    forests = _normalize(forest_config, [_way(10, SQUARE), _way(11, HOLE_RING), _relation([10], [11], tags=residential)])

    assert [f["type"] for f in forests] == [RESIDENTIAL]
    assert forests[0]["geometry"].area == pytest.approx(100 * 100 - 20 * 20)  # Loch trotzdem ausgespart


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
    assert mappings["clearings"]["only_for"]  # Lichtungen nur für Wald-Relationen
