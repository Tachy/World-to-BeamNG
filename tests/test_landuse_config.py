"""Konsistenz-Tests für data/osm_to_beamng.json["landuse_mappings"].

Ziel: Jeder relevante OSM-Landnutzungstyp wird in BeamNG irgendwie gerendert
(eigene Terrain-Schicht mit Bodenbewuchs) - außer Wohn-/Gewerbegebiete und
Wasser, wo bewusst das Luftbild bleibt.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.terrain.terrain_materials import get_landuse_category

DATA_DIR = Path(__file__).parent.parent / "data"
MAPPINGS = json.loads((DATA_DIR / "osm_to_beamng.json").read_text(encoding="utf-8"))["landuse_mappings"]
TEMPLATES = json.loads((DATA_DIR / "ground_cover_templates.json").read_text(encoding="utf-8"))["templates"]

LEVEL_TEXTURE_PREFIX = "levels/world_to_beamng/art/shapes/assets/materials/terrain/"


def _material_categories():
    return {name: data for name, data in MAPPINGS.items() if data.get("osm_tags") and not data.get("keep_photo")}


def _photo_categories():
    return {name: data for name, data in MAPPINGS.items() if data.get("keep_photo")}


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"landuse": "forest"}, "forest"),
        ({"natural": "wood"}, "forest"),
        ({"landuse": "meadow"}, "meadow"),
        ({"landuse": "grass"}, "meadow"),
        ({"landuse": "village_green"}, "meadow"),
        ({"landuse": "recreation_ground"}, "meadow"),
        ({"landuse": "cemetery"}, "meadow"),
        ({"natural": "grassland"}, "meadow"),
        ({"leisure": "park"}, "meadow"),
        ({"leisure": "garden"}, "meadow"),
        ({"leisure": "pitch"}, "meadow"),
        ({"landuse": "farmland"}, "farmland"),
        ({"landuse": "farmyard"}, "farmland"),
        ({"landuse": "allotments"}, "farmland"),
        ({"landuse": "orchard"}, "orchard"),
        ({"landuse": "vineyard"}, "vineyard"),
        ({"natural": "scrub"}, "scrub"),
        ({"natural": "wetland"}, "wetland"),
        # Wohngebiete: eigene Schicht mit Rasen (Straßen/Häuser werden per Maske ausgespart)
        ({"landuse": "residential"}, "residential"),
        # Luftbild bleibt (Häuser/Straßen/Wasser im Foto)
        ({"landuse": "commercial"}, "urban"),
        ({"landuse": "industrial"}, "urban"),
        ({"landuse": "greenhouse_horticulture"}, "urban"),
        ({"natural": "water"}, "water"),
        ({"landuse": "basin"}, "water"),
    ],
)
def test_osm_tag_maps_to_expected_category(tags, expected):
    assert get_landuse_category(tags, MAPPINGS) == expected


def test_region_relations_and_unrelated_tags_are_not_mapped():
    # natural=mountain_range ist eine Region, keine Bodenfläche
    assert get_landuse_category({"natural": "mountain_range"}, MAPPINGS) is None
    assert get_landuse_category({"building": "yes"}, MAPPINGS) is None


def test_commercial_and_water_keep_the_aerial_photo():
    assert MAPPINGS["urban"]["keep_photo"] is True
    assert MAPPINGS["water"]["keep_photo"] is True
    # Foto-Kategorien müssen andere Schichten überdecken können
    assert MAPPINGS["urban"]["priority"] > max(d["priority"] for d in _material_categories().values())
    # Wohngebiete sind NICHT mehr Foto-Kategorie, sondern haben Rasen
    assert "residential" not in MAPPINGS["urban"]["osm_tags"].get("landuse", [])


def test_residential_gets_lawn_but_yields_to_more_specific_areas():
    residential = MAPPINGS["residential"]

    assert residential["internal_name"] == "mat_residential"
    assert any("grass" in template for template in residential["groundCover"])
    # kurzer Rasen, kein hohes Gras zwischen den Häusern
    assert all(TEMPLATES[t]["gridSize"] > 4 or max(x.get("sizeMax", 1) for x in TEMPLATES[t]["types"]) < 1.0 for t in residential["groundCover"])
    # Gärten/Parks/Wiesen/Wald/Obst innerhalb eines Wohngebiets behalten ihre eigene Schicht
    assert residential["priority"] < min(d["priority"] for n, d in _material_categories().items() if n != "residential")


def test_no_osm_tag_value_is_claimed_by_two_categories():
    seen = {}
    for name, data in MAPPINGS.items():
        for key, values in data.get("osm_tags", {}).items():
            for value in values:
                assert (key, value) not in seen, f"{key}={value} in '{name}' und '{seen[(key, value)]}'"
                seen[(key, value)] = name


def test_material_categories_are_complete_and_unique():
    names = [d["internal_name"] for d in _material_categories().values()]
    assert len(names) == len(set(names)), "internal_name muss je Kategorie eindeutig sein (Terrain-Layer-Name)"
    for name, data in _material_categories().items():
        for field in ("priority", "internal_name", "groundModelName", "detailColorMap", "detailNormalMap", "groundCover"):
            assert field in data, f"{name}: Feld {field} fehlt"
        assert data["internal_name"].startswith("mat_")


def test_priorities_are_unique_among_material_categories():
    priorities = [d["priority"] for d in _material_categories().values()]
    assert len(priorities) == len(set(priorities)), "gleiche Priorität -> Überlappung nicht deterministisch"


def test_detail_textures_are_vendored_level_paths():
    # tools/vendor_shared_textures.py kopiert nur Pfade unter diesem Prefix in den Level
    for name, data in _material_categories().items():
        for key in ("detailColorMap", "detailNormalMap"):
            assert data[key].startswith(LEVEL_TEXTURE_PREFIX), f"{name}.{key}: {data[key]}"
            assert data[key].endswith(".png")


def test_ground_cover_references_existing_templates():
    for name, data in _material_categories().items():
        assert data["groundCover"], f"{name}: kein Bodenbewuchs"
        for template in data["groundCover"]:
            assert template in TEMPLATES, f"{name}: unbekannte Ground-Cover-Vorlage {template}"


def test_vineyard_has_row_settings_and_grass_undergrowth():
    vineyard = MAPPINGS["vineyard"]
    rows = vineyard["rows"]

    assert rows["orientation"] in ("gradient", "contour")
    assert 1.5 <= rows["row_spacing"] <= 4.0
    assert rows["segment_length"] > 0
    assert rows["item"] in ("grape_vine", "grape_vine_group")
    # Gras unter den Reben
    assert any("grass" in template for template in vineyard["groundCover"])
