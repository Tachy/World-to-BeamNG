"""Tests for the highway type defaults (width/material) against the real data/osm_to_beamng.json.

Background (docs/OSM_ROAD_ANALYSIS.md): motorway/trunk/primary and all *_link types were missing from
highway_defaults and fell back to the unclassified default (5 m); links were also truncated to their
base type ('primary_link' -> 'primary') and thus as wide as the main carriageway.
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


@pytest.mark.parametrize("highway", ["motorway", "trunk", "primary", "secondary", "tertiary"])
def test_major_roads_have_own_default(mapper, highway):
    assert highway in mapper.defaults


@pytest.mark.parametrize("highway", ["motorway", "trunk", "primary", "secondary", "tertiary"])
def test_link_roads_have_own_default(mapper, highway):
    assert f"{highway}_link" in mapper.defaults


@pytest.mark.parametrize("highway", ["motorway", "trunk", "primary"])
def test_major_roads_are_wider_than_unclassified(mapper, highway):
    width = mapper.get_road_properties({"highway": highway})["width"]

    assert width > mapper.get_road_properties({"highway": "unclassified"})["width"]


@pytest.mark.parametrize("highway", ["motorway", "trunk", "primary", "secondary", "tertiary"])
def test_link_without_lanes_is_single_lane_ramp(mapper, highway):
    # Without a lanes tag a ramp is single-lane - much narrower than its main carriageway.
    link = mapper.get_road_properties({"highway": f"{highway}_link"})
    main = mapper.get_road_properties({"highway": highway})

    assert link["width"] < main["width"]
    assert 3.5 <= link["width"] <= 4.5
    assert link["internal_name"] == "asphalt_road_standard"


def test_link_lanes_tag_still_wins(mapper):
    assert mapper.get_road_properties({"highway": "primary_link", "lanes": "2"})["width"] == pytest.approx(6.5)


def test_unknown_suffix_type_falls_back_to_base_type(mapper):
    # Unknown variants with an underscore still use their base type.
    assert mapper.get_road_properties({"highway": "primary_foo"})["width"] == pytest.approx(
        mapper.get_road_properties({"highway": "primary"})["width"]
    )


def test_living_street_keeps_own_default(mapper):
    assert mapper.get_road_properties({"highway": "living_street"})["width"] == pytest.approx(4.5)
