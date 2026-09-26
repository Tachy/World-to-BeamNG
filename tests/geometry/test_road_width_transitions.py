"""Tests for the smooth width transitions between DecalRoads (geometry/road_width_transitions.py).

Requirement: width changes at a joint are smoothed with a spline over 10 m (5 m before, 5 m after).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.road_width_transitions import (
    apply_width_transitions,
    continuation_partners,
    find_continuations,
    smoothstep,
)

KW = dict(transition_length=10.0, step=1.0, endpoint_tol=0.5, max_angle_deg=30.0, min_delta=0.05, min_spacing=0.5)


def _road(points, width):
    return [[float(x), float(y), 100.0, float(width)] for x, y in points]


def _width_at(nodes, x):
    for n in nodes:
        if abs(n[0] - x) < 1e-6:
            return n[3]
    raise AssertionError(f"no node at x={x}")


def test_smoothstep_is_cubic_hermite():
    assert smoothstep(0.0) == 0.0 and smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == pytest.approx(0.5)
    assert smoothstep(0.25) == pytest.approx(0.15625)
    assert smoothstep(-1.0) == 0.0 and smoothstep(2.0) == 1.0


def test_straight_continuation_is_paired():
    a = _road([(0, 0), (10, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (30, 0), (40, 0)], 9.75)
    assert find_continuations([a, b], 0.5, 30.0) == [((0, "end"), (1, "start"))]


def test_right_angle_corner_is_not_paired():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (20, 20)], 9.75)
    assert find_continuations([a, b], 0.5, 30.0) == []


def test_link_branching_off_at_shallow_angle_loses_against_straight_main_road():
    a = _road([(-20, 0), (0, 0)], 6.5)
    b = _road([(0, 0), (20, 0)], 13.0)
    angle = np.radians(20.0)
    c = _road([(0, 0), (20 * np.cos(angle), 20 * np.sin(angle))], 4.0)
    pairs = find_continuations([a, b, c], 0.5, 30.0)
    assert pairs == [((0, "end"), (1, "start"))]
    assert continuation_partners(pairs) == {0: {1}, 1: {0}}


def test_ring_road_is_not_paired_with_itself():
    ring = _road([(0, 0), (10, 0), (10, 10), (0.2, 0.0)], 6.5)
    assert find_continuations([ring], 0.5, 180.0) == []


def test_width_blends_over_five_metres_on_each_side():
    a = _road([(0, 0), (10, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (30, 0), (40, 0)], 9.75)
    new_a, new_b = apply_width_transitions([a, b], **KW)
    assert _width_at(new_a, 20.0) == pytest.approx(8.125)
    assert _width_at(new_b, 20.0) == pytest.approx(8.125)
    assert _width_at(new_a, 15.0) == pytest.approx(6.5)
    assert _width_at(new_b, 25.0) == pytest.approx(9.75)
    assert _width_at(new_a, 17.0) == pytest.approx(6.5 + 3.25 * smoothstep(0.2))
    assert _width_at(new_b, 23.0) == pytest.approx(6.5 + 3.25 * smoothstep(0.8))
    assert _width_at(new_a, 10.0) == pytest.approx(6.5)
    assert _width_at(new_b, 40.0) == pytest.approx(9.75)


def test_blend_is_monotonic_and_inserts_nodes_every_metre():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 9.75)
    new_a, new_b = apply_width_transitions([a, b], **KW)
    xs = [n[0] for n in new_a] + [n[0] for n in new_b[1:]]
    assert {15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 22.0, 23.0, 24.0, 25.0} <= set(xs)
    widths = [n[3] for n in new_a] + [n[3] for n in new_b[1:]]
    assert all(w2 >= w1 - 1e-12 for w1, w2 in zip(widths, widths[1:]))


def test_inserted_nodes_keep_min_spacing():
    a = _road([(0, 0), (16.8, 0), (19.7, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 9.75)
    new_a, _ = apply_width_transitions([a, b], **KW)
    gaps = np.diff([n[0] for n in new_a])
    # the existing 0.3 m gap (19.7 -> 20) stays, nodes are only inserted with >= 0.5 m spacing
    assert sorted(gaps)[0] == pytest.approx(0.3)
    assert sum(g < 0.5 for g in gaps) == 1


def test_equal_widths_are_left_untouched():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 6.5)
    assert apply_width_transitions([a, b], **KW) == [a, b]


def test_short_road_shrinks_zone_symmetrically():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (24, 0)], 9.75)
    new_a, new_b = apply_width_transitions([a, b], **KW)
    assert _width_at(new_a, 18.0) == pytest.approx(6.5)
    assert _width_at(new_b, 22.0) == pytest.approx(9.75)
    assert _width_at(new_b, 24.0) == pytest.approx(9.75)
    assert _width_at(new_a, 20.0) == pytest.approx(8.125)


def test_both_ends_of_a_road_can_blend():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 9.75)
    c = _road([(40, 0), (60, 0)], 6.5)
    _, new_b, _ = apply_width_transitions([a, b, c], **KW)
    assert _width_at(new_b, 20.0) == pytest.approx(8.125)
    assert _width_at(new_b, 25.0) == pytest.approx(9.75)
    assert _width_at(new_b, 35.0) == pytest.approx(9.75)
    assert _width_at(new_b, 40.0) == pytest.approx(8.125)


def test_z_is_interpolated_for_inserted_nodes():
    a = [[0.0, 0.0, 100.0, 6.5], [20.0, 0.0, 120.0, 6.5]]
    b = _road([(20, 0), (40, 0)], 9.75)
    new_a, _ = apply_width_transitions([a, b], **KW)
    node = next(n for n in new_a if abs(n[0] - 17.0) < 1e-6)
    assert node[2] == pytest.approx(117.0)


def test_input_lists_are_not_modified():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (40, 0)], 9.75)
    before = [list(map(list, a)), list(map(list, b))]
    apply_width_transitions([a, b], **KW)
    assert [a, b] == before


# --- Close the wedge gap at kinked joints ---

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from world_to_beamng.geometry.road_width_transitions import close_continuation_gaps


def _surface(nodes):
    a = np.asarray(nodes, dtype=float)
    return LineString(a[:, :2]).buffer(float(a[0, 3]) / 2.0, cap_style="flat")


def _kinked_pair(kink_deg, reverse_second=False, width=6.5):
    kink = np.radians(kink_deg)
    first = _road([(-40, 0), (-20, 0), (0, 0)], width)
    second = _road([(0, 0), (20 * np.cos(kink), 20 * np.sin(kink)), (40 * np.cos(kink), 40 * np.sin(kink))], width)
    if reverse_second:
        second = second[::-1]
    return first, second


def _outer_wedge_point(kink_deg, width=6.5):
    # The outside of a left kink is on the right; point just before the road edge on the angle bisector
    half = np.radians(kink_deg) / 2.0
    normal_right = np.array([np.sin(half), -np.cos(half)])
    return Point(*(normal_right * (width / 2.0 - 0.1)))


@pytest.mark.parametrize("reverse_second", [False, True])
def test_kinked_continuation_leaves_no_outer_wedge_gap(reverse_second):
    first, second = _kinked_pair(15.0, reverse_second)
    assert not unary_union([_surface(first), _surface(second)]).contains(_outer_wedge_point(15.0))  # initial situation

    new_first, new_second = close_continuation_gaps([first, second], endpoint_tol=0.5, max_angle_deg=30.0)

    assert unary_union([_surface(new_first), _surface(new_second)]).contains(_outer_wedge_point(15.0))


def test_gap_closing_moves_only_the_joint_ends_along_their_own_direction():
    first, second = _kinked_pair(15.0)
    new_first, new_second = close_continuation_gaps([first, second], endpoint_tol=0.5, max_angle_deg=30.0)

    extension = 6.5 / 2.0 * np.tan(np.radians(15.0)) + 0.05
    assert new_first[-1][:2] == pytest.approx([extension, 0.0])  # continues straight in its own direction
    assert new_first[:-1] == first[:-1] and new_second[1:] == second[1:]
    assert new_first[-1][3] == first[-1][3]  # width stays the same
    start_dir = np.array(second[1][:2]) - np.array(second[0][:2])
    moved = np.array(new_second[0][:2]) - np.array(second[0][:2])
    assert float(np.dot(moved, start_dir)) < 0.0 and np.linalg.norm(moved) == pytest.approx(extension)


def test_straight_continuation_and_unpaired_ends_stay_untouched():
    a = _road([(0, 0), (10, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (30, 0), (40, 0)], 6.5)
    corner = _road([(40, 0), (40, 20)], 6.5)  # right-angled, no straight continuation

    assert close_continuation_gaps([a, b, corner], endpoint_tol=0.5, max_angle_deg=30.0) == [a, b, corner]


# --- lane-count changes and structures -------------------------------------------------------------------------------
LANE_KW = dict(KW, lane_change_length=100.0, fixed_transition_length=100.0)


def _long_pair(wa, wb):
    return _road([(x, 0.0) for x in range(-200, 1)], wa), _road([(x, 0.0) for x in range(0, 201)], wb)


def test_two_to_three_lanes_blends_over_50_m_on_each_side():
    a, b = _long_pair(6.5, 9.75)

    ra, rb = apply_width_transitions([a, b], lanes=[2, 3], **LANE_KW)

    assert _width_at(ra, -50.0) == pytest.approx(6.5)  # 50 m before the joint: still two lanes
    assert _width_at(ra, 0.0) == pytest.approx((6.5 + 9.75) / 2.0)
    assert _width_at(rb, 50.0) == pytest.approx(9.75)  # 50 m after it: three lanes
    assert 6.5 < _width_at(ra, -25.0) < _width_at(rb, 25.0) < 9.75


def test_width_change_without_more_than_two_lanes_keeps_the_short_transition():
    a, b = _long_pair(6.5, 7.0)

    ra, _ = apply_width_transitions([a, b], lanes=[2, 2], **LANE_KW)

    assert _width_at(ra, -5.0) == pytest.approx(6.5)
    assert _width_at(ra, -4.0) != pytest.approx(6.5)


def test_transition_to_a_structure_lies_entirely_on_the_road_over_100_m():
    road, bridge = _long_pair(9.75, 6.5)

    rr, rbridge = apply_width_transitions([road, bridge], lanes=[3, 2], fixed=[False, True], **LANE_KW)

    assert _width_at(rr, 0.0) == pytest.approx(6.5)  # already the bridge width at the joint
    assert _width_at(rr, -100.0) == pytest.approx(9.75)
    assert 6.5 < _width_at(rr, -50.0) < 9.75
    assert rbridge == [list(map(float, n)) for n in bridge]  # the structure keeps its width everywhere


def test_widening_toward_a_structure_also_lies_on_the_road():
    road, tunnel = _long_pair(6.5, 7.5)

    rr, rt = apply_width_transitions([road, tunnel], lanes=[2, 2], fixed=[False, True], **LANE_KW)

    assert _width_at(rr, 0.0) == pytest.approx(7.5) and _width_at(rr, -100.0) == pytest.approx(6.5)
    assert rt == [list(map(float, n)) for n in tunnel]


def test_structure_transition_is_limited_to_the_road_length():
    road = _road([(x, 0.0) for x in range(-40, 1)], 9.75)  # only 40 m to the next junction
    bridge = _road([(x, 0.0) for x in range(0, 201)], 6.5)

    rr, _ = apply_width_transitions([road, bridge], lanes=[3, 2], fixed=[False, True], **LANE_KW)

    assert _width_at(rr, 0.0) == pytest.approx(6.5) and _width_at(rr, -40.0) == pytest.approx(9.75)


def test_joint_between_two_structures_is_left_alone():
    tunnel, gallery = _long_pair(7.5, 6.5)

    rt, rg = apply_width_transitions([tunnel, gallery], lanes=[2, 2], fixed=[True, True], **LANE_KW)

    assert rt == [list(map(float, n)) for n in tunnel] and rg == [list(map(float, n)) for n in gallery]


# --- transitions reach across straight continuations -----------------------------------------------------------------
def test_lane_change_zone_continues_into_the_previous_piece():
    # The 2-lane road is split at a junction 30 m before the lane change: the 50 m zone still reaches 50 m back
    a1 = _road([(x, 0.0) for x in range(-200, -29)], 6.5)
    a2 = _road([(x, 0.0) for x in range(-30, 1)], 6.5)
    b = _road([(x, 0.0) for x in range(0, 201)], 9.75)

    r1, r2, rb = apply_width_transitions([a1, a2, b], lanes=[2, 2, 3], **LANE_KW)

    assert _width_at(r1, -50.0) == pytest.approx(6.5)
    assert 6.5 < _width_at(r1, -40.0) < _width_at(r2, -30.0) < _width_at(r2, 0.0)
    assert _width_at(r2, -30.0) == pytest.approx(_width_at(r1, -30.0))  # continuous across the piece joint
    assert _width_at(rb, 50.0) == pytest.approx(9.75)


def test_structure_transition_continues_into_the_previous_piece():
    r1 = _road([(x, 0.0) for x in range(-200, -39)], 9.75)
    r2 = _road([(x, 0.0) for x in range(-40, 1)], 9.75)
    bridge = _road([(x, 0.0) for x in range(0, 201)], 6.5)

    n1, n2, _ = apply_width_transitions([r1, r2, bridge], lanes=[3, 3, 2], fixed=[False, False, True], **LANE_KW)

    assert _width_at(n1, -100.0) == pytest.approx(9.75)
    assert 6.5 < _width_at(n1, -60.0) < 9.75
    assert _width_at(n2, 0.0) == pytest.approx(6.5)


def test_zone_stops_halfway_along_a_piece_whose_far_end_has_another_width_change():
    a = _road([(x, 0.0) for x in range(-40, 1)], 6.5)  # 40 m, another width change at its start (x = -40)
    before = _road([(x, 0.0) for x in range(-200, -39)], 4.0)
    b = _road([(x, 0.0) for x in range(0, 201)], 9.75)

    _, ra, _ = apply_width_transitions([before, a, b], lanes=[1, 2, 3], **LANE_KW)

    assert _width_at(ra, -20.0) == pytest.approx(6.5)  # the lane-change zone takes only half of this piece


# --- two joints close together: the stretch between them is shared, the outer sides keep their 50 m ------------------------
def _profile(roads):
    widths = {}
    for nodes in roads:
        for n in nodes:
            widths.setdefault(round(n[0]), n[3])
    return widths


def test_two_lane_changes_within_100_m_share_the_stretch_and_keep_50_m_on_their_outer_sides():
    # A2 at the Tremola ramp: 4 lanes -> (bridge) 3 lanes for only 66 m -> 2 lanes. Each zone takes half of the stretch
    # between the joints (33 m); on the far side of each joint the full 50 m stay.
    four = _road([(x, 0.0) for x in range(-200, 1)], 13.0)
    three_a = _road([(x, 0.0) for x in range(0, 29)], 9.75)
    three_b = _road([(x, 0.0) for x in range(28, 67)], 9.75)
    two = _road([(x, 0.0) for x in range(66, 267)], 6.5)

    widths = _profile(apply_width_transitions([four, three_a, three_b, two], lanes=[4, 3, 3, 2], **LANE_KW))

    assert widths[-51] == pytest.approx(13.0) and widths[-50] == pytest.approx(13.0)  # outer side of joint 1: 50 m
    assert widths[-25] < 13.0 - 0.1  # ... and it really blends there
    assert widths[33] == pytest.approx(9.75)  # the shared stretch: each zone ends in its middle
    assert widths[66 + 50] == pytest.approx(6.5) and widths[66 + 40] > 6.5 + 0.01  # outer side of joint 2: 50 m
    steps = np.abs(np.diff([widths[x] for x in sorted(widths)]))
    assert steps.max() < 0.3  # no jump where two zones used to overwrite each other
    inside = [widths[x] for x in range(0, 34)]
    assert all(b <= a + 1e-9 for a, b in zip(inside, inside[1:]))  # narrowing monotonically toward the middle


def test_zone_next_to_a_free_end_still_shrinks_symmetrically():
    a = _road([(0, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (24, 0)], 9.75)

    new_a, new_b = apply_width_transitions([a, b], **KW)

    assert _width_at(new_a, 18.0) == pytest.approx(6.5) and _width_at(new_b, 22.0) == pytest.approx(9.75)
