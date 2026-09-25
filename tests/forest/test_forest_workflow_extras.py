"""Tests for building distance and single trees (OSM points) in the ForestWorkflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from shapely.geometry import Point, box

from world_to_beamng import config
from world_to_beamng.osm.landuse_polygons import make_local_transform
from world_to_beamng.workflow.forest_workflow import ForestWorkflow

OFFSET = (412000.0, 5297000.0)  # UTM origin near Freiburg
TO_LOCAL = make_local_transform(OFFSET)


def _building(x0, y0, x1, y1, way_id=1):
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return {"type": "way", "id": way_id, "tags": {"building": "yes"}, "geometry": [{"x": x, "y": y} for x, y in ring]}


def _tree_node(lat, lon, node_id=1, tags=None):
    return {"type": "node", "id": node_id, "lat": lat, "lon": lon, "tags": tags or {"natural": "tree"}}


def test_building_buffer_keeps_trees_away_from_walls():
    workflow = ForestWorkflow(config)

    buffer = workflow._create_building_buffer([_building(0, 0, 10, 10)], margin=2.0)

    assert buffer.contains(Point(-1.5, 5))  # 1.5 m in front of the wall
    assert buffer.contains(Point(5, 5))  # inside the building
    assert not buffer.contains(Point(-3.0, 5))  # further away is free


def test_building_buffer_ignores_non_buildings_and_returns_none_without_buildings():
    workflow = ForestWorkflow(config)
    road = {"type": "way", "id": 2, "tags": {"highway": "residential"}, "geometry": [{"x": 0, "y": 0}, {"x": 5, "y": 5}]}

    assert workflow._create_building_buffer([road], margin=2.0) is None
    assert workflow._create_building_buffer([], margin=2.0) is None


def test_single_trees_come_from_natural_tree_nodes_inside_the_tile():
    workflow = ForestWorkflow(config)
    inside = _tree_node(47.83, 7.68, 1)
    far_away = _tree_node(47.90, 7.80, 2)  # outside the tile
    other_node = _tree_node(47.8301, 7.6801, 3, tags={"amenity": "bench"})
    ax, ay = TO_LOCAL([{"lat": 47.83, "lon": 7.68}])[0]
    tile = (ax - 100, ay - 100, ax + 100, ay + 100)

    points = workflow._single_tree_points([inside, far_away, other_node], tile, OFFSET)

    assert len(points) == 1
    assert points[0] == pytest.approx((ax, ay), abs=1e-6)


def test_single_trees_are_not_placed_inside_the_exclusion_area():
    workflow = ForestWorkflow(config)
    node = _tree_node(47.83, 7.68)
    ax, ay = TO_LOCAL([{"lat": 47.83, "lon": 7.68}])[0]
    tile = (ax - 100, ay - 100, ax + 100, ay + 100)

    assert workflow._single_tree_points([node], tile, OFFSET, exclusion=box(ax - 3, ay - 3, ax + 3, ay + 3)) == []
    assert len(workflow._single_tree_points([node], tile, OFFSET, exclusion=box(ax + 50, ay + 50, ax + 60, ay + 60))) == 1
