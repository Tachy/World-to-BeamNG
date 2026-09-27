"""Tests for world_to_beamng.sidewalks.selection: which road sides get a sidewalk, and with which surface."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.sidewalks.selection import select_sidewalk_sides, sidewalk_sides

MAPPING = {
    "default_surface_material": "asphalt_road_standard",
    "surface_materials": {"asphalt": "asphalt_road_standard", "paving_stones": "cobblestone_road", "concrete": "concrete"},
}
EXCLUDED = frozenset({"footway", "path", "track"})


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"sidewalk": "both"}, {"left": True, "right": True}),
        ({"sidewalk": "yes"}, {"left": True, "right": True}),
        ({"sidewalk": "separate"}, {"left": True, "right": True}),
        ({"sidewalk": "left"}, {"left": True, "right": False}),
        ({"sidewalk": "right"}, {"left": False, "right": True}),
        ({"sidewalk": "no"}, {"left": False, "right": False}),
        ({}, {"left": False, "right": False}),
        ({"sidewalk:both": "separate"}, {"left": True, "right": True}),
        ({"sidewalk:left": "yes", "sidewalk:right": "no"}, {"left": True, "right": False}),
        ({"sidewalk": "both", "sidewalk:right": "no"}, {"left": True, "right": False}),
        ({"sidewalk:both": "no", "sidewalk:left": "separate"}, {"left": True, "right": False}),
    ],
)
def test_sides_follow_the_tags(tags, expected):
    assert sidewalk_sides(tags) == expected


def test_surface_per_side_with_fallback_chain():
    tags = {"highway": "residential", "sidewalk": "both", "sidewalk:left:surface": "paving_stones", "sidewalk:surface": "concrete"}
    assert select_sidewalk_sides(tags, EXCLUDED, MAPPING) == {"left": "cobblestone_road", "right": "concrete"}


def test_both_surface_applies_to_both_sides():
    tags = {"highway": "residential", "sidewalk": "both", "sidewalk:both:surface": "paving_stones"}
    assert select_sidewalk_sides(tags, EXCLUDED, MAPPING) == {"left": "cobblestone_road", "right": "cobblestone_road"}


def test_missing_or_unknown_surface_falls_back_to_the_default():
    tags = {"highway": "residential", "sidewalk": "both", "sidewalk:right:surface": "unhewn_cobblestone"}
    assert select_sidewalk_sides(tags, EXCLUDED, MAPPING) == {"left": "asphalt_road_standard", "right": "asphalt_road_standard"}


def test_excluded_highways_and_untagged_roads_get_nothing():
    assert select_sidewalk_sides({"highway": "footway", "sidewalk": "both"}, EXCLUDED, MAPPING) == {}
    assert select_sidewalk_sides({"highway": "residential"}, EXCLUDED, MAPPING) == {}
