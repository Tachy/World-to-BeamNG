"""Tests for extract_roads_from_osm(): which highway ways are exported as a road."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.osm.parser import extract_roads_from_osm


def _way(way_id, **tags):
    return {"type": "way", "id": way_id, "nodes": [1, 2], "tags": tags}


def test_regular_roads_are_kept():
    elements = [_way(1, highway="primary"), _way(2, highway="primary_link"), _way(3, highway="track")]

    assert [w["id"] for w in extract_roads_from_osm(elements)] == [1, 2, 3]


def test_area_ways_are_dropped():
    assert extract_roads_from_osm([_way(1, highway="pedestrian", area="yes")]) == []


@pytest.mark.parametrize("lifecycle", ["construction", "proposed", "planned", "abandoned", "disused", "razed", "demolished"])
def test_lifecycle_highways_are_dropped(lifecycle):
    # e.g. the 2nd Gotthard tube under construction: highway=construction + construction=trunk + tunnel=yes
    elements = [_way(1, highway=lifecycle, construction="trunk", tunnel="yes"), _way(2, highway="secondary")]

    assert [w["id"] for w in extract_roads_from_osm(elements)] == [2]


def test_non_way_and_non_highway_elements_are_dropped():
    elements = [
        {"type": "node", "id": 1, "tags": {"highway": "stop"}},
        _way(2, building="yes"),
        {"type": "way", "id": 3, "nodes": [1, 2]},
    ]

    assert extract_roads_from_osm(elements) == []
