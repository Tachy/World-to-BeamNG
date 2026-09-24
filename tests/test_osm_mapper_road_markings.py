"""Tests für die Markierungs-Linienmaterialien (data/osm_to_beamng.json -> road_markings) und ihren
materials.json-Eintrag. Vorbild: BeamNGs eigene line_white / line_dashed_long
(west_coast_usa/art/road/main.materials.json)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng import config
from world_to_beamng.osm.osm_mapper import OSMMapper

CONFIG_PATH = Path(__file__).parent.parent / "data" / "osm_to_beamng.json"
LINES_PREFIX = "levels/world_to_beamng/art/shapes/assets/materials/decalroad/lines/"


@pytest.fixture(scope="module")
def mapper():
    return OSMMapper(config_path=str(CONFIG_PATH))


def test_marking_materials_exist_for_configured_names(mapper):
    assert config.ROAD_MARKING_EDGE_MATERIAL in mapper.road_markings
    assert config.ROAD_MARKING_DIVIDER_MATERIAL in mapper.road_markings


def test_marking_materials_use_vanilla_line_textures(mapper):
    edge = mapper.road_markings[config.ROAD_MARKING_EDGE_MATERIAL]
    divider = mapper.road_markings[config.ROAD_MARKING_DIVIDER_MATERIAL]

    assert edge["textures"]["opacityMap"] == LINES_PREFIX + "line_white/t_line_white_o.data.dds"
    # line_white_dashed: Strich durchgehend weiß (line_dashed_long ist innerhalb des Strichs halb grau, halb weiß -
    # im Spiel sichtbar), zwei Striche je Kachel, Strich:Lücke 1:1
    assert divider["textures"]["baseColorMap"] == LINES_PREFIX + "line_white_dashed/t_line_white_dashed_b.color.dds"
    assert divider["textures"]["opacityMap"] == LINES_PREFIX + "line_white_dashed/t_line_white_dashed_o.data.dds"
    assert divider["textures"]["normalMap"] == LINES_PREFIX + "line_dashed_short/t_line_white_dashed_nm.normal.dds"
    assert divider["textureLength"] == pytest.approx(24.0)  # 6 m Strich, 6 m Lücke
    for marking in (edge, divider):
        assert set(marking["textures"]) == {"baseColorMap", "normalMap", "opacityMap"}
        assert marking["textureLength"] > 0
    assert edge["annotation"] == "SOLID_LINE"
    assert divider["annotation"] == "DASHED_LINE"


def test_marking_material_entry_follows_vanilla_line_schema(mapper):
    props = mapper.road_markings[config.ROAD_MARKING_DIVIDER_MATERIAL]
    entry = mapper.generate_marking_material_entry(config.ROAD_MARKING_DIVIDER_MATERIAL, props)

    assert entry["name"] == entry["mapTo"] == config.ROAD_MARKING_DIVIDER_MATERIAL
    assert entry["class"] == "Material"
    assert entry["annotation"] == "DASHED_LINE"
    assert entry["translucent"] is True
    assert entry["translucentZWrite"] is True
    assert entry["castShadows"] is False
    assert entry["materialTag0"] == "RoadAndPath"
    assert entry["Stages"][0]["opacityMap"].endswith("t_line_white_dashed_o.data.dds")
    assert "__name" not in entry


def test_marking_material_entries_get_unique_persistent_ids(mapper):
    props = mapper.road_markings[config.ROAD_MARKING_EDGE_MATERIAL]
    first = mapper.generate_marking_material_entry("a", props)
    second = mapper.generate_marking_material_entry("b", props)

    assert first["persistentId"] != second["persistentId"]


def test_markings_are_drawn_after_every_road_surface(mapper):
    # DecalRoads werden in ABSTEIGENDER renderPriority gezeichnet (kleinster Wert zuletzt = oben) - im Spiel lagen
    # Linien mit 20 unter dem Asphalt mit 8.
    road_priorities = [config.ROAD_RENDER_PRIORITY_BASE - s["priority"] for s in mapper.surface_types.values()]
    assert all(config.ROAD_MARKING_RENDER_PRIORITY < p for p in road_priorities)
    assert config.ROAD_MARKING_RENDER_PRIORITY >= 1
