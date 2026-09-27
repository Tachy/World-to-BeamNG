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
    # across the main axis, which leaves the node a few degrees off the trunk's direction
    assert ends[1036670763] == pytest.approx(1.5 * LANE, abs=0.05)
    assert ends[129718739] == pytest.approx(0.0, abs=1e-9)
    assert ends[1036670761] == pytest.approx(-1.5 * LANE, abs=0.05)
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
    turn = _turn_split(turn_xy=[(0.0, 0.0), (20.0, -10.0), (40.0, -20.0)])  # a loop ramp leaving at once

    assert _y_at(turn["coords"], 3.0) == pytest.approx(-LANE)  # connector in the slot
    assert _y_at(turn["coords"], 20.0) == pytest.approx(-10.0, abs=0.15)  # then OSM's course (rounded), no long straight run
    assert _y_at(turn["coords"], 40.0) == pytest.approx(-20.0)


def test_the_shift_continues_onto_the_next_piece_of_a_short_branch():
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (80.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn_a = _road(3, [(0.0, 0.0), (4.0, -3.0)], highway="primary_link", lanes="2", oneway="yes")  # OSM's connector
    turn_b = _road(4, [(4.0, -3.0), (20.0, -3.0), (40.0, -3.0), (60.0, -10.0)], highway="primary_link", lanes="2",
                   oneway="yes")
    roads = [trunk, straight, turn_a, turn_b]
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=20.0)

    assert turn_a["coords"][-1] == turn_b["coords"][0]  # the joint stays closed
    assert turn_b["coords"][0][1] == pytest.approx(-LANE)  # the lane is reached here, still in the slot
    assert turn_b["coords"][-1][:2] == pytest.approx((60.0, -10.0))  # on its own course


def test_the_stem_is_as_long_as_the_shortest_connector_of_the_side_branches():
    turn = _turn_split()

    # the straight lanes lie centred-left (slot 1.625 m) and are drawn straight: no connector of their own
    assert turn["lane_split_branch"]["stem_length"] == pytest.approx(10.0, abs=0.6)


def test_the_branch_is_marked_with_the_geometry_of_its_slot():
    roads = _motto_bartola()
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=10.0)

    mark = roads[3]["lane_split_branch"]  # the entrance
    assert 0.0 < mark["hold"] <= 30.0
    assert mark["trunk_width"] == pytest.approx(4 * LANE)
    assert np.dot(mark["slot_offset"], mark["left_normal"]) == pytest.approx(1.5 * LANE)
    assert len(mark["stem_path"]) >= 2 and mark["stem_path"][0][:2] == pytest.approx(splits[0].node)


def test_trunk_and_branches_are_marked_with_the_split_node():
    roads = _motto_bartola()
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=30.0)

    node = splits[0].node
    assert roads[0]["lane_split_trunk_nodes"] == [node]
    assert all(road["lane_split_branch"]["node"] == node for road in roads[1:])


# --- Markings on the stem ---------------------------------------------------------------------------------------------

from world_to_beamng.geometry.lane_splits import stem_marking_masks

STEM_MARK = {"stem_path": [(float(x), 0.0, 100.0) for x in range(0, 31)], "stem_length": 30.0, "trunk_width": 13.0}


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


def test_on_the_ground_the_branches_take_the_main_road_height_on_the_stem_and_blend_back():
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (80.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    turn = _road(3, [(0.0, 0.0), (10.0, -3.0), (80.0, -20.0)], highway="primary_link", lanes="2", oneway="yes")
    straight["coords"] = [(x, y, 102.0) for x, y, _ in straight["coords"]]  # the branches lie at different heights
    roads = [trunk, straight, turn]
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=20.0)

    stem = turn["lane_split_branch"]["stem_length"]
    for road in (straight, turn):
        coords = np.asarray(road["coords"])
        assert np.allclose(coords[coords[:, 0] < stem - 0.5, 2], 102.0)  # one cross-section: the main road's height
    assert np.asarray(straight["coords"])[-1, 2] == pytest.approx(102.0)
    assert np.asarray(turn["coords"])[-1, 2] == pytest.approx(100.0)


# --- The main axis follows OSM, the side branches wait beside it ------------------------------------------------------


def _curved_split():
    """A 4-lane two-way trunk ending at the node; the main road (2 lanes) curves left on a 100 m radius, the ramps are
    drawn by OSM from the node to their lane centre within 8 m and then along it."""
    angles = np.linspace(0.0, 0.8, 41)
    main_xy = [(100.0 * np.sin(a), 100.0 * (1.0 - np.cos(a))) for a in angles]

    def beside(offset, reach_at=8.0):
        points = [(0.0, 0.0)]
        for a in angles[1:]:
            s = 100.0 * a
            x, y = 100.0 * np.sin(a), 100.0 * (1.0 - np.cos(a))
            nx, ny = -np.sin(a), np.cos(a)  # left normal of the curve
            k = min(1.0, s / reach_at)
            points.append((x + nx * offset * k, y + ny * offset * k))
        return points

    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4")
    main = _road(2, main_xy, highway="primary", lanes="2", **{"lanes:forward": "1", "lanes:backward": "1"})
    exit_ramp = _road(3, beside(-4.0), highway="primary_link", lanes="1", oneway="yes")
    entrance = _road(4, beside(4.0)[::-1], highway="primary_link", lanes="1", oneway="yes")
    return [trunk, main, exit_ramp, entrance], main_xy


def test_the_main_axis_keeps_its_osm_course_everywhere():
    roads, main_xy = _curved_split()
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=30.0)

    main = np.asarray(roads[1]["coords"])[:, :2]
    from shapely.geometry import LineString, Point

    osm = LineString(main_xy)
    assert max(osm.distance(Point(*p)) for p in main) < 1e-6  # not straightened, not moved


def test_side_branches_wait_parallel_to_the_curved_main_axis():
    roads, main_xy = _curved_split()
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=30.0)

    from shapely.geometry import LineString, Point

    osm_main = LineString(main_xy)
    exit_xy = np.asarray(roads[2]["coords"])[:, :2]
    near_node = [p for p in exit_xy if Point(*p).distance(Point(0.0, 0.0)) < 7.0]
    assert near_node and all(abs(osm_main.distance(Point(*p)) - 1.5 * LANE) < 0.05 for p in near_node)
    stem = np.asarray(roads[2]["lane_split_branch"]["stem_path"])
    assert max(osm_main.distance(Point(*p[:2])) for p in stem) < 1e-6  # the stem follows the main axis


def _max_kink_deg(coords, until):
    xy = np.asarray(coords)[:, :2]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    xy = xy[arc <= until]
    d = np.diff(xy, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-6]
    angles = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.max(np.abs((np.diff(angles) + 180.0) % 360.0 - 180.0)))


def test_the_side_branch_leaves_its_slot_without_a_kink():
    # OSM: to the lane centre within 8 m, along it, then turning away by 6 degrees at a sharp vertex at 20 m
    turn_xy = [(0.0, 0.0), (8.0, -3.0), (20.0, -3.0), (40.0, -5.1), (80.0, -9.3)]
    turn = _turn_split(turn_xy=turn_xy, length=20.0)

    assert _max_kink_deg(turn["coords"], until=60.0) < 1.5  # smooth, the densified points bend a little each
    assert _y_at(turn["coords"], 80.0) == pytest.approx(-9.3)  # back on OSM's course behind the zone



def test_a_side_bridge_stays_on_the_main_deck_height_until_it_leaves_and_then_runs_linear_to_its_abutment():
    def bridge(road_id, points, z_of, **tags):
        return {"id": road_id, "coords": [(x, y, z_of(x)) for x, y in points], "osm_tags": {"bridge": "yes", **tags}}

    main_z = lambda x: 100.0 + 0.05 * x  # the through span climbs
    trunk = bridge(1, [(-40.0, 0.0), (0.0, 0.0)], main_z, highway="primary", lanes="4", oneway="yes")
    straight = bridge(2, [(0.0, 0.0), (80.0, 0.0)], main_z, highway="primary", lanes="2", oneway="yes")
    # the turn lanes' own abutment is low (90 m at x = 60): OSM drew them from the node to their lane, then away
    turn_xy = [(0.0, 0.0), (8.0, -3.0), (20.0, -3.0), (60.0, -20.0)]
    turn = bridge(3, turn_xy, lambda x: 100.0 - x / 6.0, highway="primary_link", lanes="2", oneway="yes")
    ground = _road(4, [(60.0, -20.0), (90.0, -35.0)], highway="primary_link", lanes="2", oneway="yes")
    ground["coords"] = [(x, y, 90.0) for x, y, _ in ground["coords"]]
    roads = [trunk, straight, turn, ground]
    splits = find_lane_splits(roads, _width)

    shift_branches_into_slots(splits, roads, max_connector=30.0, length=20.0)

    coords = np.asarray(turn["coords"])
    attached = coords[coords[:, 0] < 10.0]
    assert np.allclose(attached[:, 2], main_z(attached[:, 0]), atol=0.02)  # on the main deck: flat cross-section
    leave = turn["lane_split_branch"]["leaves_at"]
    beyond = coords[np.hypot(coords[:, 0], coords[:, 1]) > leave + 1.0]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(coords[:, :2], axis=0), axis=1))])
    tail = arc >= leave
    slope = np.diff(coords[tail, 2]) / np.maximum(np.diff(arc[tail]), 1e-9)
    assert np.allclose(slope, slope[0], atol=1e-6)  # linear from where it leaves the deck ...
    assert coords[-1, 2] == pytest.approx(90.0)  # ... to its abutment
    assert np.allclose(np.asarray(ground["coords"])[:, 2], 90.0)  # the road behind the abutment is not touched


def _link_between_two_splits():
    """A short oneway link (about 32 m) that is a side branch at BOTH ends: it leaves split A at (0, 0) in A's right
    slot and joins split B at (30, -10) in B's right slot (the real case: a two-way ramp splitting into its two
    directions, each of which merges into a four-lane road a few dozen meters further on)."""
    from world_to_beamng.geometry.lane_splits import Branch, LaneSplit

    trunk_a = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight_a = _road(2, [(0.0, 0.0), (80.0, 0.0)], highway="primary", lanes="2", oneway="yes")
    link = _road(3, [(0.0, 0.0), (10.0, -5.0), (20.0, -8.0), (30.0, -10.0)], highway="primary_link", lanes="2",
                 oneway="yes")
    trunk_b = _road(4, [(30.0, -10.0), (70.0, -10.0)], highway="primary", lanes="4", oneway="yes")
    other_b = _road(5, [(0.0, -25.0), (15.0, -17.0), (30.0, -10.0)], highway="primary_link", lanes="2", oneway="yes")
    split_a = LaneSplit((0.0, 0.0), trunk_a, False, 4 * LANE, (0.0, 1.0), [
        Branch(straight_a, True, np.array([0.0, LANE]), 2 * LANE),
        Branch(link, True, np.array([0.0, -LANE]), 2 * LANE),
    ])
    # B: from the node into the branches is -x, its left normal is -y; the link comes in on B's right (+y)
    split_b = LaneSplit((30.0, -10.0), trunk_b, True, 4 * LANE, (0.0, -1.0), [
        Branch(other_b, False, np.array([0.0, -LANE]), 2 * LANE),
        Branch(link, False, np.array([0.0, LANE]), 2 * LANE),
    ])
    return [trunk_a, straight_a, link, trunk_b, other_b], [split_a, split_b], link


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_short_link_between_two_splits_ends_in_both_slots_without_folding(order):
    roads, splits, link = _link_between_two_splits()

    shift_branches_into_slots([splits[i] for i in order], roads, max_connector=30.0, length=30.0)

    xy = np.asarray(link["coords"])[:, :2]
    segments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert segments.min() > 1e-3  # no zero-length segment
    # both ends one slot offset from their node (across the main axis there, see the Motto Bartola test)
    assert np.linalg.norm(xy[0] - (0.0, 0.0)) == pytest.approx(LANE, abs=0.05)
    assert np.linalg.norm(xy[-1] - (30.0, -10.0)) == pytest.approx(LANE, abs=0.05)
    assert _max_kink_deg(link["coords"], until=float("inf")) < 20.0  # no fold, no sharp corner
    assert segments.sum() == pytest.approx(32.0, abs=4.0)  # about its own length, not doubled back


# the corner of the main axis itself still leaves a slight bend (the lateral profile is smoothed in its frame)
@pytest.mark.parametrize("main_turn_deg, max_kink_deg", [(20.0, 8.0), (30.0, 12.0)])
def test_a_side_branch_outside_a_corner_of_the_main_axis_stays_smooth(main_turn_deg, max_kink_deg):
    # the main axis turns left at x = 20; the side branch runs outside that corner, where the nearest point of the main
    # axis is the corner itself for a whole wedge of branch points
    turn = np.radians(main_turn_deg)
    trunk = _road(1, [(-40.0, 0.0), (0.0, 0.0)], highway="primary", lanes="4", oneway="yes")
    straight = _road(2, [(0.0, 0.0), (20.0, 0.0), (20.0 + 60.0 * np.cos(turn), 60.0 * np.sin(turn))],
                     highway="primary", lanes="2", oneway="yes")
    turn_xy = [(0.0, 0.0), (8.0, -3.0), (20.0, -3.0), (40.0, -12.0), (60.0, -24.0), (80.0, -36.0), (100.0, -48.0)]
    side = _road(3, turn_xy, highway="primary_link", lanes="2", oneway="yes")
    roads = [trunk, straight, side]

    shift_branches_into_slots(find_lane_splits(roads, _width), roads, max_connector=30.0, length=30.0)

    assert _max_kink_deg(side["coords"], until=float("inf")) < max_kink_deg  # no zigzag at the corner
    assert side["coords"][-1][:2] == pytest.approx((100.0, -48.0))
