"""Tests for world_to_beamng.walls.road_base: height of the nearest road centerline within at most N meters distance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.walls.road_base import SAMPLE_STEP_M, RoadBaseHeight, centerlines_from_roads

ROAD = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 110.0]])  # 10 % gradient along x


def _z(centerlines, xs, ys, max_distance=5.0):
    return RoadBaseHeight(centerlines, max_distance)(np.asarray(xs, float), np.asarray(ys, float))


def test_height_is_interpolated_along_the_centerline():
    z = _z([ROAD], [0.0, 25.0, 50.0, 100.0], [1.0, 1.0, 1.0, 1.0])

    assert z == pytest.approx([100.0, 102.5, 105.0, 110.0], abs=0.05)


def test_height_between_the_centerline_points_does_not_snap_to_the_vertices():
    z = _z([ROAD], [33.3], [0.0])

    assert z[0] == pytest.approx(103.33, abs=0.03)  # linear, not the nearest support point (100 or 110)


def test_points_beyond_the_snap_distance_have_no_road_height():
    z = _z([ROAD], [50.0, 50.0, 50.0, 50.0], [4.9, 5.1, -4.9, -5.1])

    assert not np.isnan(z[0]) and np.isnan(z[1]) and not np.isnan(z[2]) and np.isnan(z[3])  # on both sides, limit 5 m


def test_points_beyond_the_ends_of_the_road_use_the_end_and_the_distance_to_it():
    z = _z([ROAD], [-3.0, -6.0, 104.0, 106.0], [0.0, 0.0, 0.0, 0.0])

    assert z[0] == pytest.approx(100.0, abs=0.05) and np.isnan(z[1])
    assert z[2] == pytest.approx(110.0, abs=0.05) and np.isnan(z[3])


def test_the_nearest_of_several_roads_wins():
    upper = np.array([[0.0, 4.0, 200.0], [100.0, 4.0, 200.0]])
    z = _z([ROAD, upper], [50.0], [3.0])  # 3 m from ROAD (z=105), 1 m from the upper road

    assert z[0] == pytest.approx(200.0)


def test_no_centerlines_means_no_road_height():
    assert np.isnan(_z([], [1.0, 2.0], [1.0, 2.0])).all()


def test_centerlines_are_taken_from_the_road_dicts_and_short_ones_are_ignored():
    roads = [
        {"trimmed_centerline": ROAD, "road_polygon": None},
        {"trimmed_centerline": np.array([[0.0, 0.0, 1.0]])},  # only one point
        {"road_polygon": None},  # without centerline
    ]

    lines = centerlines_from_roads(roads)

    assert len(lines) == 1 and lines[0].shape == (2, 3)


def _reference_densify(line):
    """Earlier segment-by-segment loop - reference for the vectorized RoadBaseHeight._densify()."""
    points = [line[:1]]
    for start, end in zip(line[:-1], line[1:]):
        steps = max(1, int(np.ceil(np.linalg.norm(end[:2] - start[:2]) / SAMPLE_STEP_M)))
        fractions = (np.arange(1, steps + 1) / steps)[:, None]
        points.append(start + (end - start) * fractions)
    return np.vstack(points)


def test_densify_is_bit_identical_to_the_segment_loop():
    rng = np.random.default_rng(11)
    for n in (2, 3, 17, 60):
        line = np.cumsum(rng.normal(0.0, 8.0, size=(n, 3)), axis=0)
        line[1] = line[0]  # segment of length 0 -> exactly one point
        np.testing.assert_array_equal(RoadBaseHeight._densify(line), _reference_densify(line))


def test_the_kd_tree_is_only_built_on_the_first_query():
    base = RoadBaseHeight([ROAD], 5.0)
    assert base._tree is None  # without walls (no call) it is never created

    base(np.array([50.0]), np.array([0.0]))

    assert base._tree is not None
