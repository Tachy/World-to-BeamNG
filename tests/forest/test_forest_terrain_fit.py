"""Bäume folgen der fertigen Terrain-Heightmap (nach Straßen-Einbettung) und meiden die eingebetteten Straßenflächen."""

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
    """Heightmap 20x20 m mit z = x (Hang 100 %) im Zellraster 1 m."""
    heights = np.tile(np.arange(20, dtype=float), (20, 1))
    return make_height_sampler(heights, 0.0, 0.0, 1.0)


def test_heights_from_sampler_are_bilinear_not_nearest_neighbor():
    calculator = ForestHeightCalculator()
    result = calculator.calculate_heights_from_sampler([(5.4, 3.0), (7.5, 8.25)], _ramp_sampler())
    assert [round(z, 3) for _, _, z in result] == [5.4, 7.5]  # NN auf ganzen Zellen würde 5.0 / 8.0 liefern
    assert [(x, y) for x, y, _ in result] == [(5.4, 3.0), (7.5, 8.25)]


def test_heights_from_sampler_handles_empty_input():
    assert ForestHeightCalculator().calculate_heights_from_sampler([], _ramp_sampler()) == []


def _road(points):
    return {"road_polygon": np.array(points, dtype=float)}


def test_road_surface_exclusion_covers_the_embedded_road_plus_margin():
    workflow = ForestWorkflow(config)
    # 10 m lange, 6 m breite Fahrbahn entlang der x-Achse (y von -3 bis 3)
    exclusion = workflow._create_road_surface_exclusion([_road([(0, -3), (10, -3), (10, 3), (0, 3)])], margin=4.0)
    assert intersects_xy(exclusion, 5.0, 0.0)
    assert intersects_xy(exclusion, 5.0, 6.5)  # 3.5 m neben der Kante: noch im Puffer
    assert not intersects_xy(exclusion, 5.0, 7.5)  # 4.5 m neben der Kante: frei


def test_road_surface_exclusion_ignores_degenerate_polygons_and_returns_none_without_roads():
    workflow = ForestWorkflow(config)
    assert workflow._create_road_surface_exclusion([], margin=4.0) is None
    assert workflow._create_road_surface_exclusion(None, margin=4.0) is None
    assert workflow._create_road_surface_exclusion([_road([(0, 0), (1, 1)])], margin=4.0) is None
