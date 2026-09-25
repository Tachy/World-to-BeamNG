"""Trees follow the finished terrain heightmap (after road embedding) and avoid the embedded road surfaces."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from shapely import intersects_xy

from world_to_beamng import config
from world_to_beamng.forest.forest_height_calculator import ForestHeightCalculator
from world_to_beamng.forest.vineyard_generator import make_height_sampler
from world_to_beamng.workflow.forest_workflow import ForestWorkflow


def _ramp_sampler():
    """Heightmap 20x20 m with z = x (slope 100 %) in a 1 m cell grid."""
    heights = np.tile(np.arange(20, dtype=float), (20, 1))
    return make_height_sampler(heights, 0.0, 0.0, 1.0)


def test_heights_from_sampler_are_bilinear_not_nearest_neighbor():
    calculator = ForestHeightCalculator()
    result = calculator.calculate_heights_from_sampler([(5.4, 3.0), (7.5, 8.25)], _ramp_sampler())
    assert [round(z, 3) for _, _, z in result] == [5.4, 7.5]  # nearest neighbor on whole cells would yield 5.0 / 8.0
    assert [(x, y) for x, y, _ in result] == [(5.4, 3.0), (7.5, 8.25)]


def test_heights_from_sampler_handles_empty_input():
    assert ForestHeightCalculator().calculate_heights_from_sampler([], _ramp_sampler()) == []


def _road(points):
    return {"road_polygon": np.array(points, dtype=float)}


def test_road_surface_exclusion_covers_the_embedded_road_plus_margin():
    workflow = ForestWorkflow(config)
    # 10 m long, 6 m wide carriageway along the x axis (y from -3 to 3)
    exclusion = workflow._create_road_surface_exclusion([_road([(0, -3), (10, -3), (10, 3), (0, 3)])], margin=4.0)
    assert intersects_xy(exclusion, 5.0, 0.0)
    assert intersects_xy(exclusion, 5.0, 6.5)  # 3.5 m beside the edge: still within the buffer
    assert not intersects_xy(exclusion, 5.0, 7.5)  # 4.5 m beside the edge: free


def test_road_surface_exclusion_ignores_degenerate_polygons_and_returns_none_without_roads():
    workflow = ForestWorkflow(config)
    assert workflow._create_road_surface_exclusion([], margin=4.0) is None
    assert workflow._create_road_surface_exclusion(None, margin=4.0) is None
    assert workflow._create_road_surface_exclusion([_road([(0, 0), (1, 1)])], margin=4.0) is None


def test_road_surface_exclusion_accepts_an_already_merged_surface():
    from shapely.geometry import box

    workflow = ForestWorkflow(config)
    from_list = workflow._create_road_surface_exclusion([_road([(0, -3), (10, -3), (10, 3), (0, 3)])], margin=4.0)
    from_union = workflow._create_road_surface_exclusion(box(0, -3, 10, 3), margin=4.0)

    assert from_union.symmetric_difference(from_list).area < 0.5
