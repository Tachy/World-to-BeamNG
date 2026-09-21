"""Tests für world_to_beamng.walls.road_base: Höhe der nächsten Straßen-Centerline in höchstens N Metern Abstand."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.walls.road_base import RoadBaseHeight, centerlines_from_roads

ROAD = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 110.0]])  # 10 % Steigung entlang x


def _z(centerlines, xs, ys, max_distance=5.0):
    return RoadBaseHeight(centerlines, max_distance)(np.asarray(xs, float), np.asarray(ys, float))


def test_height_is_interpolated_along_the_centerline():
    z = _z([ROAD], [0.0, 25.0, 50.0, 100.0], [1.0, 1.0, 1.0, 1.0])

    assert z == pytest.approx([100.0, 102.5, 105.0, 110.0], abs=0.05)


def test_height_between_the_centerline_points_does_not_snap_to_the_vertices():
    z = _z([ROAD], [33.3], [0.0])

    assert z[0] == pytest.approx(103.33, abs=0.03)  # linear, nicht der nächste Stützpunkt (100 oder 110)


def test_points_beyond_the_snap_distance_have_no_road_height():
    z = _z([ROAD], [50.0, 50.0, 50.0, 50.0], [4.9, 5.1, -4.9, -5.1])

    assert not np.isnan(z[0]) and np.isnan(z[1]) and not np.isnan(z[2]) and np.isnan(z[3])  # beidseitig, Grenze 5 m


def test_points_beyond_the_ends_of_the_road_use_the_end_and_the_distance_to_it():
    z = _z([ROAD], [-3.0, -6.0, 104.0, 106.0], [0.0, 0.0, 0.0, 0.0])

    assert z[0] == pytest.approx(100.0, abs=0.05) and np.isnan(z[1])
    assert z[2] == pytest.approx(110.0, abs=0.05) and np.isnan(z[3])


def test_the_nearest_of_several_roads_wins():
    upper = np.array([[0.0, 4.0, 200.0], [100.0, 4.0, 200.0]])
    z = _z([ROAD, upper], [50.0], [3.0])  # 3 m von ROAD (z=105), 1 m von der oberen Straße

    assert z[0] == pytest.approx(200.0)


def test_no_centerlines_means_no_road_height():
    assert np.isnan(_z([], [1.0, 2.0], [1.0, 2.0])).all()


def test_centerlines_are_taken_from_the_road_dicts_and_short_ones_are_ignored():
    roads = [
        {"trimmed_centerline": ROAD, "road_polygon": None},
        {"trimmed_centerline": np.array([[0.0, 0.0, 1.0]])},  # nur ein Punkt
        {"road_polygon": None},  # ohne Centerline
    ]

    lines = centerlines_from_roads(roads)

    assert len(lines) == 1 and lines[0].shape == (2, 3)
