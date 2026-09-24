"""Tests für die weichen Breitenübergänge zwischen DecalRoads (geometry/road_width_transitions.py).

Anforderung: Breitenwechsel an einem Stoß werden über 10 m (5 m davor, 5 m danach) mit einem Spline geglättet.
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
    raise AssertionError(f"kein Knoten bei x={x}")


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
    # die vorhandene 0,3-m-Lücke (19.7 -> 20) bleibt, eingefügt wird nur mit >= 0,5 m Abstand
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


# --- Keil-Lücke an geknickten Stößen schließen ---

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
    # Außenseite eines Linksknicks ist rechts; Punkt kurz vor der Fahrbahnkante auf der Winkelhalbierenden
    half = np.radians(kink_deg) / 2.0
    normal_right = np.array([np.sin(half), -np.cos(half)])
    return Point(*(normal_right * (width / 2.0 - 0.1)))


@pytest.mark.parametrize("reverse_second", [False, True])
def test_kinked_continuation_leaves_no_outer_wedge_gap(reverse_second):
    first, second = _kinked_pair(15.0, reverse_second)
    assert not unary_union([_surface(first), _surface(second)]).contains(_outer_wedge_point(15.0))  # Ausgangslage

    new_first, new_second = close_continuation_gaps([first, second], endpoint_tol=0.5, max_angle_deg=30.0)

    assert unary_union([_surface(new_first), _surface(new_second)]).contains(_outer_wedge_point(15.0))


def test_gap_closing_moves_only_the_joint_ends_along_their_own_direction():
    first, second = _kinked_pair(15.0)
    new_first, new_second = close_continuation_gaps([first, second], endpoint_tol=0.5, max_angle_deg=30.0)

    extension = 6.5 / 2.0 * np.tan(np.radians(15.0)) + 0.05
    assert new_first[-1][:2] == pytest.approx([extension, 0.0])  # geradeaus weiter in eigener Richtung
    assert new_first[:-1] == first[:-1] and new_second[1:] == second[1:]
    assert new_first[-1][3] == first[-1][3]  # Breite bleibt
    start_dir = np.array(second[1][:2]) - np.array(second[0][:2])
    moved = np.array(new_second[0][:2]) - np.array(second[0][:2])
    assert float(np.dot(moved, start_dir)) < 0.0 and np.linalg.norm(moved) == pytest.approx(extension)


def test_straight_continuation_and_unpaired_ends_stay_untouched():
    a = _road([(0, 0), (10, 0), (20, 0)], 6.5)
    b = _road([(20, 0), (30, 0), (40, 0)], 6.5)
    corner = _road([(40, 0), (40, 20)], 6.5)  # rechtwinklig, keine Geradeaus-Fortsetzung

    assert close_continuation_gaps([a, b, corner], endpoint_tol=0.5, max_angle_deg=30.0) == [a, b, corner]
