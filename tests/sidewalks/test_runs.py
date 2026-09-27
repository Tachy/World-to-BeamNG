"""Tests for world_to_beamng.sidewalks.runs: kerb lines at the carriageway edge, cut where other roads join."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.sidewalks.runs import plan_sidewalk_runs

MAIN = [[x, 0.0, 100.0 + 0.01 * x, 6.0] for x in np.arange(0.0, 60.5, 5.0)]  # along +x, left = +y
SIDE_STREET = [[30.0, y, 100.3, 5.0] for y in np.arange(0.0, 30.5, 5.0)]  # joins MAIN from the left at x = 30


def _plan(roads, sides, blocking=None):
    return plan_sidewalk_runs(roads, sides, blocking or [True] * len(roads), clearance=1.15, min_length=2.0,
                              endpoint_tol=0.5, max_angle_deg=30.0)


def test_kerb_line_lies_on_the_carriageway_edge_at_road_height():
    runs = _plan([MAIN], [{"left": "asphalt_road_standard"}])
    assert len(runs) == 1
    points = runs[0]["points"]
    assert np.allclose(points[:, 1], 3.0)
    assert np.allclose(points[:, 2], 100.0 + 0.01 * points[:, 0])
    assert runs[0]["side"] == "left" and runs[0]["surface"] == "asphalt_road_standard"


def test_sidewalk_is_left_of_the_run_direction_on_both_sides():
    runs = {r["side"]: r["points"] for r in _plan([MAIN], [{"left": "a", "right": "a"}])}
    assert runs["left"][-1, 0] > runs["left"][0, 0]  # left side runs with the road (+x), sidewalk at +y is on its left
    assert runs["right"][-1, 0] < runs["right"][0, 0]  # right side runs against it, sidewalk at -y is on its left


def test_joining_road_cuts_a_gap_of_its_width_plus_both_sidewalks():
    runs = [r for r in _plan([MAIN, SIDE_STREET], [{"left": "a"}, {}]) if r["road_index"] == 0]
    assert len(runs) == 2
    before, after = sorted(runs, key=lambda r: r["points"][0, 0])
    assert before["points"][:, 0].max() == pytest.approx(30.0 - 2.5 - 1.15, abs=0.01)
    assert after["points"][:, 0].min() == pytest.approx(30.0 + 2.5 + 1.15, abs=0.01)


def test_non_blocking_ways_do_not_cut_the_sidewalk():
    footway = [[30.0, y, 100.3, 2.0] for y in np.arange(0.0, 30.5, 5.0)]
    runs = _plan([MAIN, footway], [{"left": "a"}, {}], blocking=[True, False])
    assert len(runs) == 1 and runs[0]["points"][:, 0].max() == pytest.approx(60.0)


def test_pieces_shorter_than_min_length_are_dropped():
    short = [[0.0, 0.0, 100.0, 6.0], [1.5, 0.0, 100.0, 6.0]]
    assert _plan([short], [{"left": "a"}]) == []


def test_straight_through_continuation_does_not_cut_the_sidewalk():
    # two pieces of the same street split at a node with a 15 degree kink (inner side of the bend = left)
    kink = np.radians(15.0)
    first = [[x, 0.0, 100.0, 6.0] for x in np.arange(0.0, 50.5, 5.0)]
    second = [[50.0 + d * np.cos(kink), d * np.sin(kink), 100.0, 6.0] for d in np.arange(0.0, 50.5, 5.0)]
    runs = plan_sidewalk_runs([first, second], [{"left": "a"}, {"left": "a"}], [True, True], clearance=1.15, min_length=2.0,
                              endpoint_tol=0.5, max_angle_deg=30.0)
    assert len(runs) == 2
    for run in runs:
        assert np.linalg.norm(np.diff(run["points"][:, :2], axis=0), axis=1).sum() == pytest.approx(50.0, abs=0.01)
