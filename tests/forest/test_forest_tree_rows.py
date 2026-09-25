"""
Tests for tree rows (OSM natural=tree_row): a LINE along which trees stand at even spacing.

Previously the lines were treated like forest areas (polygon from the line points, the chord closes the
ring) - the trees then stood in a random strip instead of in a row.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import pytest
from shapely.geometry import LineString, MultiLineString, box

from world_to_beamng import config
from world_to_beamng.forest.forest_normalizer import ForestNormalizer
from world_to_beamng.forest.forest_point_generator import ForestPointGenerator
from world_to_beamng.workflow.forest_workflow import ForestWorkflow

CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "osm_to_beamng.json"
ROW = "tree_row"


@pytest.fixture(scope="module")
def forest_config():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"forest_type_templates": data["forest_type_templates"], "forest_mappings": data["forest_mappings"]}


class _AlwaysForest:
    forest_mappings = {}

    def is_forest(self, tags):
        return True


def _way(way_id, points, **tags):
    return {"type": "way", "id": way_id, "tags": tags, "geometry": [{"x": x, "y": y} for x, y in points]}


def _normalize(forest_config, osm_data):
    normalizer = ForestNormalizer(forest_config, osm_mapper=_AlwaysForest())
    result = normalizer.normalize_tile((-500, -500, 500, 500), "t", osm_data=osm_data)
    assert result["status"] == "success"
    return result["forests"]


# --- Configuration and normalizer ------------------------------------------------------------------


def test_tree_row_template_is_low_deciduous_trees_with_a_row_spacing(forest_config):
    template = forest_config["forest_type_templates"][ROW]

    assert 5.0 <= template["row_spacing"] <= 15.0
    assert sum(template["preferred_trees"].values()) == pytest.approx(1.0)
    for name in template["preferred_trees"]:
        assert any(kind in name for kind in ("aspen_small", "beech_small", "oak_sml")), name
    assert forest_config["forest_mappings"]["natural"]["tree_row"] == ROW


def test_tree_row_way_stays_a_line_and_is_not_turned_into_a_polygon(forest_config):
    points = [(0, 0), (50, 0), (100, 30)]

    forests = _normalize(forest_config, [_way(1, points, natural="tree_row")])

    assert len(forests) == 1
    assert forests[0]["type"] == ROW
    assert forests[0]["geometry"].geom_type == "LineString"
    assert forests[0]["geometry"].length == pytest.approx(LineString(points).length)


def test_area_ways_are_still_polygons(forest_config):
    square = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]

    forests = _normalize(forest_config, [_way(1, square, natural="wood")])

    assert forests[0]["geometry"].geom_type == "Polygon"


def test_tree_row_overrides_do_not_turn_it_into_a_tall_forest(forest_config):
    forests = _normalize(forest_config, [_way(1, [(0, 0), (50, 0)], natural="tree_row", leaf_type="needleleaf")])

    assert forests[0]["type"] == ROW


# --- Points along the line ------------------------------------------------------------------------


def _points(line, spacing=8.0, tile=(-500, -500, 500, 500), **generator_args):
    generator = ForestPointGenerator(**generator_args)
    forests = [{"type": ROW, "geometry": line, "tile_box": box(*tile)}]
    return generator, generator.generate_points_for_forests(forests, {ROW: {"row_spacing": spacing, "tree_density": 0.5}})[0]


def test_trees_stand_in_a_row_along_the_line_with_the_configured_spacing():
    _, points = _points(LineString([(0, 0), (100, 0)]))

    xs = sorted(x for x, _ in points)
    assert 11 <= len(points) <= 13  # 100 m / 8 m
    assert all(y == pytest.approx(0.0) for _, y in points)  # exactly on the line
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    assert all(5.5 <= g <= 10.5 for g in gaps)  # evenly spaced, with a slight offset


def test_row_follows_the_bends_of_the_line():
    line = LineString([(0, 0), (40, 0), (40, 40)])

    _, points = _points(line)

    assert len(points) >= 9
    assert all(line.distance(__import__("shapely").geometry.Point(p)) < 1e-6 for p in points)
    assert any(x == pytest.approx(40.0) and y > 5 for x, y in points)  # continues beyond the bend as well


def test_row_is_clipped_to_the_tile():
    _, points = _points(LineString([(0, 0), (200, 0)]), tile=(-10, -10, 100, 10))

    assert points
    assert all(x <= 100 for x, _ in points)


def test_multiline_geometries_are_supported():
    _, points = _points(MultiLineString([[(0, 0), (50, 0)], [(0, 20), (50, 20)]]))

    assert {round(y) for _, y in points} == {0, 20}


def test_row_trees_respect_the_row_exclusion_e_g_buildings():
    generator = ForestPointGenerator()
    generator.set_row_exclusion(box(40, -5, 60, 5))
    forests = [{"type": ROW, "geometry": LineString([(0, 0), (100, 0)]), "tile_box": box(-500, -500, 500, 500)}]

    points = generator.generate_points_for_forests(forests, {ROW: {"row_spacing": 8.0}})[0]

    assert points
    assert not any(40 <= x <= 60 for x, _ in points)


def test_tree_rows_along_a_road_are_not_removed_by_the_wide_forest_road_buffer():
    # Avenues stand a few meters beside the road: the 5 m forest buffer would otherwise delete them completely
    generator = ForestPointGenerator()
    generator.set_road_buffer(box(-10, -5, 110, 5))
    forests = [{"type": ROW, "geometry": LineString([(0, 0), (100, 0)]), "tile_box": box(-500, -500, 500, 500)}]

    points = generator.generate_points_for_forests(forests, {ROW: {"row_spacing": 8.0}})[0]

    assert len(points) >= 11


def test_a_line_shorter_than_the_spacing_still_gets_a_tree():
    _, points = _points(LineString([(0, 0), (3, 0)]))

    assert len(points) == 1


# --- Workflow: exclusion for rows ---------------------------------------------------------------


def test_rows_use_a_smaller_road_margin_than_forests():
    # the distance to the carriageway edge is what matters; the centerline buffers are only the fallback
    assert 0 < config.FOREST_ROW_SURFACE_MARGIN < config.FOREST_ROAD_SURFACE_MARGIN


def test_row_exclusion_contains_buildings_and_is_none_without_anything():
    workflow = ForestWorkflow(config)
    building = box(0, 0, 10, 10)

    assert workflow._create_row_exclusion([], None) is None
    assert workflow._create_row_exclusion([], building).contains(building.centroid)
