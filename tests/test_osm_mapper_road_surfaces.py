"""Tests for the road surface mapping (4 surfaces) against the real
data/osm_to_beamng.json.

Background: The earlier `dirt_road` surface used `dirt_road_gravels`, a
gravel overlay with ~92 % transparent pixels - forest tracks were therefore
practically invisible. Now there are separate surfaces for dirt road (dirt_road)
and gravel road (gravel_road), both with opaque area textures from BeamNG's own
west_coast_usa level (road_dirt_02 / road_gravel).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.osm.osm_mapper import OSMMapper

CONFIG_PATH = Path(__file__).parent.parent / "data" / "osm_to_beamng.json"


@pytest.fixture(scope="module")
def mapper():
    return OSMMapper(config_path=str(CONFIG_PATH))


def _surface(mapper, **tags):
    return mapper.get_road_properties(tags)["internal_name"]


def test_road_surfaces_defined(mapper):
    assert set(mapper.surface_types) == {
        "asphalt_road_standard",
        "dirt_road",
        "gravel_road",
        "concrete",
        "cobblestone_road",
    }


@pytest.mark.parametrize(
    "tags, expected",
    [
        # highway defaults
        ({"highway": "track"}, "dirt_road"),
        ({"highway": "path"}, "dirt_road"),
        ({"highway": "residential"}, "asphalt_road_standard"),
        ({"highway": "secondary"}, "asphalt_road_standard"),
        ({"highway": "footway"}, "concrete"),
        ({"highway": "steps"}, "concrete"),
        # tracktype: grade1 -> asphalt, grade2 -> gravel, grade3..5 -> dirt road
        ({"highway": "track", "tracktype": "grade1"}, "asphalt_road_standard"),
        ({"highway": "track", "tracktype": "grade2"}, "gravel_road"),
        ({"highway": "track", "tracktype": "grade3"}, "dirt_road"),
        ({"highway": "track", "tracktype": "grade4"}, "dirt_road"),
        ({"highway": "track", "tracktype": "grade5"}, "dirt_road"),
        # surface tag: gravel-like surfaces
        ({"highway": "track", "surface": "gravel"}, "gravel_road"),
        ({"highway": "track", "surface": "fine_gravel"}, "gravel_road"),
        ({"highway": "track", "surface": "compacted"}, "gravel_road"),
        ({"highway": "service", "surface": "gravel"}, "gravel_road"),
        ({"highway": "path", "surface": "gravel"}, "gravel_road"),
        # surface tag: dirt-like surfaces
        ({"highway": "track", "surface": "ground"}, "dirt_road"),
        ({"highway": "track", "surface": "dirt"}, "dirt_road"),
        ({"highway": "track", "surface": "earth"}, "dirt_road"),
        # surface tag beats tracktype
        ({"highway": "track", "surface": "ground", "tracktype": "grade2"}, "dirt_road"),
        ({"highway": "track", "surface": "gravel", "tracktype": "grade5"}, "gravel_road"),
        # existing behavior: asphalt surface stays asphalt
        ({"highway": "track", "surface": "asphalt"}, "asphalt_road_standard"),
        ({"highway": "path", "surface": "paved"}, "asphalt_road_standard"),
        # paving (e.g. Tremola at the Gotthard): dedicated cobblestone surface
        ({"highway": "secondary", "surface": "sett"}, "cobblestone_road"),
        ({"highway": "residential", "surface": "cobblestone"}, "cobblestone_road"),
        ({"highway": "residential", "surface": "unhewn_cobblestone"}, "cobblestone_road"),
        ({"highway": "service", "surface": "paving_stones"}, "cobblestone_road"),
        # unpaved/natural: never asphalt, even if the highway type would be asphalt
        ({"highway": "service", "surface": "unpaved"}, "gravel_road"),
        ({"highway": "track", "surface": "unpaved"}, "gravel_road"),
        ({"highway": "path", "surface": "grass"}, "dirt_road"),
        ({"highway": "path", "surface": "rock"}, "dirt_road"),
    ],
)
def test_surface_mapping(mapper, tags, expected):
    assert _surface(mapper, **tags) == expected


def test_surface_specific_properties_follow_surface_not_highway(mapper):
    # A gravel road gets the gravel properties, even if the highway type
    # (service) would actually be asphalt - the width stays from the highway type.
    props = mapper.get_road_properties({"highway": "service", "surface": "gravel"})

    assert props["internal_name"] == "gravel_road"
    assert props["groundModelName"] == "gravel"
    assert props["width"] == 3.5


def test_gravel_road_has_own_priority_between_dirt_and_asphalt(mapper):
    dirt = mapper.surface_types["dirt_road"]["priority"]
    gravel = mapper.surface_types["gravel_road"]["priority"]
    asphalt = mapper.surface_types["asphalt_road_standard"]["priority"]

    assert dirt < gravel < asphalt


def test_dirt_and_gravel_do_not_use_sparse_gravels_overlay(mapper):
    # dirt_road_gravels is a scattered-gravel overlay (mean opacity 19/255)
    # and therefore practically invisible as the sole track surface.
    for name in ("dirt_road", "gravel_road"):
        for path in mapper.surface_types[name]["textures"].values():
            assert "dirt_road_gravels" not in path, f"{name} still uses the overlay: {path}"


def test_dirt_and_gravel_use_opaque_road_surface_textures(mapper):
    dirt = mapper.surface_types["dirt_road"]["textures"]
    gravel = mapper.surface_types["gravel_road"]["textures"]

    # Both share opacity/AO/roughness from m_dirt_road_01 (like BeamNG's
    # road_dirt_02 / road_gravel), but differ in the color image.
    assert dirt["opacityMap"].endswith("m_dirt_road_01/t_dirt_road_o.data.dds")
    assert gravel["opacityMap"] == dirt["opacityMap"]
    assert dirt["baseColorMap"].endswith("m_dirt_road_01/t_dirt_road_b.color.dds")
    assert gravel["baseColorMap"].endswith("road_gravel/t_dirt_road_02_b.color.dds")
    assert dirt["baseColorMap"] != gravel["baseColorMap"]
    # All maps complete (PBR stage like in BeamNG's own materials)
    for tex in (dirt, gravel):
        assert set(tex) >= {"baseColorMap", "normalMap", "roughnessMap", "ambientOcclusionMap", "opacityMap"}


def test_gravel_material_entry_is_translucent_gravel_ground_type(mapper):
    props = mapper.get_road_properties({"highway": "track", "surface": "gravel"})
    entry = mapper.generate_materials_json_entry("gravel_road", props)

    assert entry["groundType"] == "GRAVEL"
    assert entry["annotation"] == "NATURE"
    assert entry["translucent"] is True
    assert entry["Stages"][0]["opacityMap"].endswith("t_dirt_road_o.data.dds")


def test_gravel_material_entry_passes_opacity_factor(mapper):
    # BeamNG's road_gravel sets opacityFactor=0.721 so that the terrain below
    # shows through slightly - optionally configurable per surface here.
    entry = mapper.generate_materials_json_entry("x", {"textures": {"baseColorMap": "a.dds"}, "opacityFactor": 0.72})

    assert entry["Stages"][0]["opacityFactor"] == pytest.approx(0.72)


def test_material_entry_without_opacity_factor_has_none(mapper):
    entry = mapper.generate_materials_json_entry("x", {"textures": {"baseColorMap": "a.dds"}})

    assert "opacityFactor" not in entry["Stages"][0]


def test_cobblestone_surface_uses_vendored_cobblestone_textures(mapper):
    cobble = mapper.surface_types["cobblestone_road"]

    assert cobble["groundModelName"] == "cobblestone"
    assert cobble["textures"]["baseColorMap"].endswith("tileable/stone/italy_cobblestone/italy_cobblestone_d.dds")
    assert cobble["textures"]["normalMap"].endswith("tileable/stone/italy_cobblestone/italy_cobblestone_n.dds")
    # drawn between gravel and asphalt (renderPriority)
    assert (
        mapper.surface_types["gravel_road"]["priority"]
        < cobble["priority"]
        < mapper.surface_types["asphalt_road_standard"]["priority"]
    )
