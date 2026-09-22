"""Tests für world_to_beamng.geometry.road_structures: Klassifizierung von Brücken/Tunneln/Galerien anhand OSM-Tags."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.road_structures import classify_structure, split_by_structure_type


def test_bridge_tag_is_classified_as_bridge():
    assert classify_structure({"highway": "primary", "bridge": "yes"}) == "bridge"
    assert classify_structure({"highway": "primary", "bridge": "viaduct"}) == "bridge"


def test_bridge_no_is_not_a_bridge():
    assert classify_structure({"highway": "primary", "bridge": "no"}) == "surface"


def test_avalanche_protector_is_a_gallery_not_a_tunnel():
    assert classify_structure({"highway": "primary", "tunnel": "avalanche_protector"}) == "gallery"


def test_other_tunnel_values_are_classified_as_tunnel():
    assert classify_structure({"highway": "trunk", "tunnel": "yes"}) == "tunnel"
    assert classify_structure({"waterway": "stream", "tunnel": "culvert"}) == "tunnel"
    assert classify_structure({"highway": "path", "tunnel": "building_passage"}) == "tunnel"


def test_tunnel_no_is_not_a_tunnel():
    assert classify_structure({"highway": "primary", "tunnel": "no"}) == "surface"


def test_missing_tags_are_surface():
    assert classify_structure({}) == "surface"
    assert classify_structure(None) == "surface"
    assert classify_structure({"highway": "residential"}) == "surface"


def test_bridge_takes_priority_over_tunnel_if_both_are_present():
    assert classify_structure({"bridge": "yes", "tunnel": "yes"}) == "bridge"


def test_split_by_structure_type_separates_surface_from_structures():
    roads = [
        {"road_id": 1, "structure_type": "surface"},
        {"road_id": 2, "structure_type": "bridge"},
        {"road_id": 3, "structure_type": "tunnel"},
        {"road_id": 4, "structure_type": "gallery"},
        {"road_id": 5},  # fehlendes Feld -> gilt als surface
    ]

    surface, structures = split_by_structure_type(roads)

    assert [r["road_id"] for r in surface] == [1, 5]
    assert [r["road_id"] for r in structures] == [2, 3, 4]
