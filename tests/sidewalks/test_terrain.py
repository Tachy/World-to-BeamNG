"""Tests for world_to_beamng.sidewalks.terrain: the road outline is widened on the sidewalk sides only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.sidewalks.terrain import attach_sidewalks

MAPPING = {"default_surface_material": "asphalt_road_standard", "surface_materials": {}}
EXCLUDED = frozenset({"footway"})


def _road(tags, structure="surface", **extra):
    centerline = np.array([[x, 0.0, 100.0] for x in np.arange(0.0, 20.5, 1.0)])
    outline = np.array([[0.0, -3.0], [20.0, -3.0], [20.0, 3.0], [0.0, 3.0]])
    return {"trimmed_centerline": centerline, "road_polygon": outline, "osm_tags": tags, "structure_type": structure, **extra}


def _props(poly):
    surface = (poly.get("osm_tags") or {}).get("surface")
    return {"width": 6.0, "internal_name": "concrete" if surface == "concrete" else "asphalt_road_standard"}


def _attach(roads):
    return attach_sidewalks(roads, _props, MAPPING, EXCLUDED, frozenset({"concrete"}), extra=1.15)


def test_left_sidewalk_widens_only_the_left_side():
    road = _road({"highway": "residential", "sidewalk": "left"})
    assert _attach([road]) == 1
    assert road["sidewalk_sides"] == {"left": "asphalt_road_standard"}
    assert road["sidewalk_extra"] == {"left": 1.15}
    assert road["road_polygon"][:, 1].max() == pytest.approx(4.15)
    assert road["road_polygon"][:, 1].min() == pytest.approx(-3.0)


def test_roads_without_sidewalk_and_structures_stay_untouched():
    plain = _road({"highway": "residential"})
    bridge = _road({"highway": "residential", "sidewalk": "both"}, structure="bridge")
    before = [plain["road_polygon"].copy(), bridge["road_polygon"].copy()]
    assert _attach([plain, bridge]) == 0
    assert "sidewalk_sides" not in plain and "sidewalk_sides" not in bridge
    np.testing.assert_array_equal(plain["road_polygon"], before[0])
    np.testing.assert_array_equal(bridge["road_polygon"], before[1])


def test_width_transition_nodes_are_followed():
    nodes = np.array([[0.0, 0.0, 100.0, 6.0], [10.0, 0.0, 100.0, 6.0], [20.0, 0.0, 100.0, 10.0]])
    road = _road({"highway": "residential", "sidewalk": "both"}, width_nodes=nodes)
    _attach([road])
    ys_at_end = road["road_polygon"][np.isclose(road["road_polygon"][:, 0], 20.0), 1]
    assert ys_at_end.max() == pytest.approx(5.0 + 1.15) and ys_at_end.min() == pytest.approx(-5.0 - 1.15)


def test_roads_without_a_decal_road_get_no_sidewalk():
    road = _road({"highway": "residential", "sidewalk": "both", "surface": "concrete"})
    before = road["road_polygon"].copy()
    assert _attach([road]) == 0
    assert "sidewalk_sides" not in road
    np.testing.assert_array_equal(road["road_polygon"], before)
