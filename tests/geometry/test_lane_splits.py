"""Tests for world_to_beamng.geometry.lane_splits: nodes where the lanes of one road split exactly onto several
branches (motorway exits/entrances, turn lanes)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.lane_splits import directional_lanes, find_lane_splits

LANE = 3.25


def _road(road_id, coords, **tags):
    return {"id": road_id, "coords": [(x, y, 100.0) for x, y in coords], "osm_tags": tags}


def _width(road):
    return LANE * directional_lanes(road["osm_tags"]).total


def _motto_bartola():
    """The four ways at OSM node 3688461068 in local coordinates, node = (0, 0) (docs/OSM_ROAD_ANALYSIS.md)."""
    trunk = _road(44220547, [(0.0, 0.0), (9.0, 8.5), (16.3, 14.8), (22.7, 19.9)], highway="primary", lanes="4")
    main = _road(129718739, [(-9.1, -10.5), (0.0, 0.0)], highway="primary", lanes="2",
                 **{"lanes:forward": "1", "lanes:backward": "1"})
    exit_ramp = _road(1036670761, [(0.0, 0.0), (-5.3, -1.8), (-8.2, -3.1), (-11.0, -5.1), (-13.5, -7.4)],
                      highway="primary_link", lanes="1", oneway="yes")
    entrance = _road(1036670763, [(-9.3, -19.6), (-3.4, -10.6), (-1.7, -7.2), (-0.8, -4.3), (0.0, 0.0)],
                     highway="primary_link", lanes="1", oneway="yes")
    return [trunk, main, exit_ramp, entrance]


def _by_id(split):
    return {branch.road["id"]: branch for branch in split.branches}


def test_directional_lanes_from_the_tags():
    assert directional_lanes({"lanes": "4"}) == (2, 2)
    assert directional_lanes({"lanes": "3", "lanes:backward": "1"}) == (2, 1)
    assert directional_lanes({"lanes": "2", "oneway": "yes"}) == (2, 0)
    assert directional_lanes({"lanes": "2", "oneway": "-1"}) == (0, 2)
    assert directional_lanes({"highway": "primary_link"}) == (1, 0)  # a ramp without lanes tag has one lane
    assert directional_lanes({"lanes": "3"}) == (2, 1)  # larger half forward, like the road markings


def test_motto_bartola_outer_lanes_become_the_ramps():
    roads = _motto_bartola()

    (split,) = find_lane_splits(roads, _width)

    assert split.trunk["id"] == 44220547
    branches = _by_id(split)
    assert set(branches) == {129718739, 1036670761, 1036670763}
    normal = np.asarray(split.left_normal)
    # looking from the node along the branches (south-west): entrance on the left, main road centred, exit on the right
    assert np.dot(branches[1036670763].slot_offset, normal) == pytest.approx(1.5 * LANE)
    assert np.linalg.norm(branches[129718739].slot_offset) == pytest.approx(0.0, abs=1e-9)
    assert np.dot(branches[1036670761].slot_offset, normal) == pytest.approx(-1.5 * LANE)
    assert branches[129718739].slot_width == pytest.approx(2 * LANE)
    assert branches[1036670761].slot_width == pytest.approx(LANE)


def test_the_slot_widths_add_up_to_the_trunk_width():
    (split,) = find_lane_splits(_motto_bartola(), _width)

    assert sum(branch.slot_width for branch in split.branches) == pytest.approx(split.trunk_width)


def test_a_oneway_road_splitting_into_two_straight_and_two_turn_lanes():
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (40.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn = _road(3, [(0.0, 0.0), (20.0, -10.0), (30.0, -30.0)], highway="primary_link", lanes="2", oneway="yes")

    (split,) = find_lane_splits([trunk, straight, turn], _width)

    branches = _by_id(split)
    normal = np.asarray(split.left_normal)
    assert np.dot(branches[2].slot_offset, normal) == pytest.approx(LANE)  # straight lanes on the left
    assert np.dot(branches[3].slot_offset, normal) == pytest.approx(-LANE)  # turn lanes on the right


def test_no_split_when_the_lanes_do_not_add_up():
    roads = _motto_bartola()
    roads[0]["osm_tags"]["lanes"] = "3"
    roads[0]["osm_tags"]["lanes:backward"] = "1"

    assert find_lane_splits(roads, _width) == []


def test_no_split_when_the_lanes_would_cross():
    roads = _motto_bartola()
    # swap the ramp directions: the exit now lies where traffic enters and vice versa
    for road in roads[2:]:
        road["coords"] = road["coords"][::-1]

    assert find_lane_splits(roads, _width) == []


def test_an_ordinary_t_junction_is_no_split():
    main_a = _road(1, [(-30.0, 0.0), (0.0, 0.0)], highway="secondary", lanes="2")
    main_b = _road(2, [(0.0, 0.0), (30.0, 0.0)], highway="secondary", lanes="2")
    side = _road(3, [(0.0, 0.0), (0.0, 30.0)], highway="residential", lanes="2")

    assert find_lane_splits([main_a, main_b, side], _width) == []


# --- Shifting the branches into their slots ---------------------------------------------------------------------------

from world_to_beamng.geometry.lane_splits import shift_branches_into_slots


def test_branch_ends_lie_side_by_side_across_the_trunk_after_the_shift():
    roads = _motto_bartola()
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, length=40.0)

    (split,) = splits
    normal = np.asarray(split.left_normal)
    ends = {}
    for branch in split.branches:
        end = np.asarray(branch.road["coords"][0 if branch.at_start else -1][:2])
        ends[branch.road["id"]] = float(np.dot(end, normal))
    assert ends[1036670763] == pytest.approx(1.5 * LANE)
    assert ends[129718739] == pytest.approx(0.0, abs=1e-9)
    assert ends[1036670761] == pytest.approx(-1.5 * LANE)
    assert roads[0]["coords"][0][:2] == (0.0, 0.0)  # the trunk stays


def test_the_shift_fades_out_along_the_branch_and_keeps_the_height():
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (40.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn_xy = [(0.0, 0.0)] + [(float(x), -0.5 * x) for x in range(5, 80, 5)]
    turn = _road(3, turn_xy, highway="primary_link", lanes="2", oneway="yes")
    roads = [trunk, straight, turn]
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, length=30.0)

    shifted = np.asarray(turn["coords"], dtype=float)
    original = np.asarray([(x, y, 100.0) for x, y in turn_xy])
    moved = np.linalg.norm(shifted[:, :2] - original[:, :2], axis=1)
    assert moved[0] == pytest.approx(LANE)
    assert np.all(np.diff(moved) <= 1e-9)  # fades out monotonically
    assert np.allclose(moved[np.hypot(original[:, 0], original[:, 1]) >= 30.0], 0.0)
    assert np.allclose(shifted[:, 2], 100.0)


def test_the_shift_continues_onto_the_next_piece_of_a_short_branch():
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (40.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn_a = _road(3, [(0.0, 0.0), (6.0, -8.0)], highway="primary_link", lanes="2", oneway="yes")  # 10 m long
    turn_b = _road(4, [(6.0, -8.0), (12.0, -16.0), (18.0, -24.0), (24.0, -32.0)], highway="primary_link", lanes="2",
                   oneway="yes")
    roads = [trunk, straight, turn_a, turn_b]
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, length=30.0)

    assert turn_a["coords"][-1] == turn_b["coords"][0]  # the joint stays closed
    assert turn_b["coords"][1][:2] != (12.0, -16.0)  # 20 m from the node: still shifted
    assert turn_b["coords"][-1][:2] == pytest.approx((24.0, -32.0))  # 40 m: done
