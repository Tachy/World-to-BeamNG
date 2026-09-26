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

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=30.0)

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


TURN_XY = [(0.0, 0.0), (5.0, -2.5), (10.0, -3.0), (20.0, -3.0), (30.0, -3.0), (40.0, -5.0), (60.0, -15.0), (80.0, -25.0)]


def _turn_split(max_connector=30.0, length=20.0, turn_xy=TURN_XY):
    """A oneway 4-lane road splitting into 2 straight lanes and 2 turn lanes; OSM draws the turn lanes from the node to
    their lane centre (3.0 m right of the axis, reached after 10 m), then along it, then turning away."""
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (80.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn = _road(3, turn_xy, highway="primary_link", lanes="2", oneway="yes")
    roads = [trunk, straight, turn]
    splits = find_lane_splits(roads, _width)
    shift_branches_into_slots(splits, roads, max_connector=max_connector, length=length)
    return turn


def _y_at(coords, x):
    coords = np.asarray(coords)
    return float(np.interp(x, coords[:, 0], coords[:, 1]))


def test_the_osm_connector_to_the_lane_centre_becomes_a_straight_run_in_the_slot():
    turn = _turn_split()

    coords = np.asarray(turn["coords"])
    assert len([p for p in coords if p[0] <= 9.0]) >= 4  # densified along the connector
    assert all(p[1] == pytest.approx(-LANE) for p in coords if p[0] <= 9.0)  # in its slot, not on OSM's diagonal
    assert np.allclose(coords[:, 2], 100.0)


def test_after_reaching_its_lane_the_branch_follows_the_osm_course():
    turn = _turn_split(length=20.0)

    # from the lane centre on OSM's course counts; only the 0.25 m between OSM's lane centre (3.0 m) and the slot
    # (3.25 m) fades out over 20 m
    assert _y_at(turn["coords"], 20.0) == pytest.approx(-3.0 - 0.25 * (1.0 - 0.5), abs=0.03)
    assert _y_at(turn["coords"], 40.0) == pytest.approx(-5.0)
    assert _y_at(turn["coords"], 60.0) == pytest.approx(-15.0)


def test_a_branch_turning_away_at_once_takes_its_course_where_it_passes_its_lane_centre():
    turn = _turn_split(turn_xy=[(0.0, 0.0), (20.0, -10.0), (40.0, -20.0)])  # no stretch along its lane at all

    assert _y_at(turn["coords"], 3.0) == pytest.approx(-LANE)  # connector in the slot
    assert _y_at(turn["coords"], 40.0) == pytest.approx(-20.0)  # then OSM's course


def test_the_shift_continues_onto_the_next_piece_of_a_short_branch():
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (80.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn_a = _road(3, [(0.0, 0.0), (4.0, -3.0)], highway="primary_link", lanes="2", oneway="yes")  # the connector
    turn_b = _road(4, [(4.0, -3.0), (20.0, -3.0), (40.0, -3.0), (60.0, -10.0)], highway="primary_link", lanes="2",
                   oneway="yes")
    roads = [trunk, straight, turn_a, turn_b]
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=20.0)

    assert turn_a["coords"][-1] == turn_b["coords"][0]  # the joint stays closed
    assert turn_b["coords"][0][1] == pytest.approx(-LANE)  # the lane is reached here, still in the slot
    assert turn_b["coords"][-1][:2] == pytest.approx((60.0, -10.0))  # on its own course


def test_trunk_and_branches_are_marked_with_the_split_node():
    roads = _motto_bartola()
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=30.0)

    node = splits[0].node
    assert roads[0]["lane_split_trunk_nodes"] == [node]
    assert all(road["lane_split_branch"]["node"] == node for road in roads[1:])
