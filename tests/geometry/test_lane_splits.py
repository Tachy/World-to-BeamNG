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

    shift_branches_into_slots(splits, roads, hold=30.0, length=30.0)

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


def _turn_split(hold=30.0, length=20.0, turn_xy=TURN_XY):
    """A oneway 4-lane road splitting into 2 straight lanes and 2 turn lanes; OSM draws the turn lanes from the node to
    their lane centre, along it and then turning away."""
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (80.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn = _road(3, turn_xy, highway="primary_link", lanes="2", oneway="yes")
    roads = [trunk, straight, turn]
    splits = find_lane_splits(roads, _width)
    shift_branches_into_slots(splits, roads, hold=hold, length=length)
    return turn


def _y_at(coords, x):
    coords = np.asarray(coords)
    return float(np.interp(x, coords[:, 0], coords[:, 1]))


def test_the_branch_runs_straight_on_in_its_lanes_of_the_trunk_for_the_hold_length():
    turn = _turn_split(hold=30.0)

    held = np.asarray([p for p in turn["coords"] if p[0] <= 30.0 + 1e-6])
    assert len(held) >= 10  # densified: the straight run does not depend on OSM's few points
    assert np.allclose(held[:, 1], -LANE)  # in its slot, OSM's connector to the lane centre is ignored
    assert np.allclose(held[:, 2], 100.0)


def test_after_the_hold_length_the_branch_moves_over_to_its_osm_course():
    turn = _turn_split(hold=30.0, length=20.0)

    assert -5.0 < _y_at(turn["coords"], 40.0) < -LANE  # between the held line and OSM's course
    assert _y_at(turn["coords"], 60.0) == pytest.approx(-15.0)  # 30 + 20 m: on OSM's course
    assert _y_at(turn["coords"], 80.0) == pytest.approx(-25.0)


def test_the_shift_continues_onto_the_next_piece_of_a_short_branch():
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (80.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn_a = _road(3, [(0.0, 0.0), (4.0, -3.0)], highway="primary_link", lanes="2", oneway="yes")  # OSM's connector
    turn_b = _road(4, [(4.0, -3.0), (20.0, -3.0), (40.0, -3.0), (60.0, -10.0)], highway="primary_link", lanes="2",
                   oneway="yes")
    roads = [trunk, straight, turn_a, turn_b]
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, hold=15.0, length=15.0)

    assert turn_a["coords"][-1] == turn_b["coords"][0]  # the joint stays closed
    assert turn_b["coords"][0][1] == pytest.approx(-LANE)  # still held in the slot
    assert turn_b["coords"][-1][:2] == pytest.approx((60.0, -10.0))  # on its own course


def test_the_branch_is_marked_with_the_geometry_of_its_slot():
    roads = _motto_bartola()
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, hold=10.0, length=10.0)

    mark = roads[3]["lane_split_branch"]  # the entrance
    assert mark["hold"] == pytest.approx(10.0)
    assert mark["trunk_width"] == pytest.approx(4 * LANE)
    assert np.dot(mark["slot_offset"], mark["left_normal"]) == pytest.approx(1.5 * LANE)
    assert np.dot(mark["axis"], mark["left_normal"]) == pytest.approx(0.0)


def test_trunk_and_branches_are_marked_with_the_split_node():
    roads = _motto_bartola()
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, hold=30.0, length=30.0)

    node = splits[0].node
    assert roads[0]["lane_split_trunk_nodes"] == [node]
    assert all(road["lane_split_branch"]["node"] == node for road in roads[1:])


# --- Markings on the stem ---------------------------------------------------------------------------------------------

from world_to_beamng.geometry.lane_splits import stem_marking_masks

STEM_MARK = {"node": (0.0, 0.0), "axis": (1.0, 0.0), "left_normal": (0.0, 1.0), "hold": 30.0, "trunk_width": 13.0}


def _nodes(points, width):
    return [[float(x), float(y), 100.0, width] for x, y in points]


def test_on_the_stem_the_edge_lines_between_branches_become_one_block_marking():
    xs = range(0, 61, 5)
    main = _nodes([(x, 0.0) for x in xs], 6.5)  # digitized away from the node
    ramp = _nodes([(x, 4.875 + max(0.0, x - 30.0) * 0.2) for x in xs][::-1], 3.25)  # digitized towards the node
    trunk = _nodes([(-40.0, 0.0), (0.0, 0.0)], 13.0)

    masks = stem_marking_masks([trunk, main, ramp], [STEM_MARK])

    main_mask, ramp_mask = masks[1], masks[2]
    on_stem = np.array([x < 30 for x in xs])
    # the main road's left edge (towards the ramp) is only drawn from the end of the stem on, its right edge everywhere
    assert np.array_equal(main_mask["edge_keep"][1.0], ~on_stem)
    assert -1.0 not in main_mask["edge_keep"]  # nothing lies beside its right edge
    assert not main_mask["blocks"]  # the ramp (the outer lane) draws the block marking
    # the ramp lies on the left and runs towards the node (-x): its left edge faces the main road
    assert np.array_equal(ramp_mask["edge_keep"][1.0], ~on_stem[::-1])
    assert -1.0 not in ramp_mask["edge_keep"]
    ((block_mask, sign, lane_width),) = ramp_mask["blocks"]
    assert sign == 1.0 and lane_width == 0.0 and np.array_equal(block_mask, on_stem[::-1])
    # split members do not cut each other's lines
    assert masks[1]["siblings"] == {0, 2} and masks[2]["siblings"] == {0, 1}
    assert 0 not in masks or not masks[0]["blocks"]


def test_on_the_stem_all_branches_share_one_height_and_blend_back_into_their_own_profile():
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (80.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn = _road(3, [(0.0, 0.0), (10.0, -3.0), (80.0, -20.0)], highway="primary_link", lanes="2", oneway="yes")
    straight["coords"] = [(x, y, 102.0) for x, y, _ in straight["coords"]]  # the branches lie at different heights
    roads = [trunk, straight, turn]
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, hold=30.0, length=20.0)

    for road in (straight, turn):
        coords = np.asarray(road["coords"])
        assert np.allclose(coords[coords[:, 0] < 29.0, 2], 101.0)  # one cross-section on the stem
    assert np.asarray(straight["coords"])[-1, 2] == pytest.approx(102.0)
    assert np.asarray(turn["coords"])[-1, 2] == pytest.approx(100.0)
