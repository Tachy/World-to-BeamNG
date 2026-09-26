"""Tests for world_to_beamng.geometry.guardrails: where roads get guard rails and how the rail segments are placed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.guardrails import place_guardrail_items, plan_guardrail_runs

W = 6.5  # carriageway width
KW = dict(probe_offset=4.0, min_drop=2.0, edge_gap=0.2, extension=20.0, junction_clearance=0.5, endpoint_tol=0.5,
          max_angle_deg=30.0, min_length=3.0)


def _road(x0, x1, y=0.0, z=100.0, width=W):
    return [[float(x), y, z, width] for x in np.arange(x0, x1 + 0.5, 1.0)]


def _drop_right(x_from, x_to, depth=5.0):
    """Terrain at 100, but `depth` lower right of the road (y < -4) between x_from and x_to."""
    def height_at(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return np.where((y < -4.0) & (x >= x_from) & (x <= x_to), 100.0 - depth, 100.0)
    return height_at


def _x_range(run):
    return float(run[:, 0].min()), float(run[:, 0].max())


def test_a_drop_beside_the_road_gets_a_rail_extended_20_m_both_ways():
    runs = plan_guardrail_runs([_road(0, 200)], [True], _drop_right(90, 110), **KW)

    assert len(runs) == 1
    assert _x_range(runs[0]) == pytest.approx((70.0, 130.0))
    assert np.allclose(runs[0][:, 1], -(W / 2 + 0.2))  # traffic face 20 cm beside the carriageway edge
    assert np.allclose(runs[0][:, 2], 100.0)  # at road level


def test_run_points_have_the_road_on_their_right():
    # Placement convention: the rail's back (away from the road) is LEFT of the run direction
    run = plan_guardrail_runs([_road(0, 200)], [True], _drop_right(90, 110), **KW)[0]

    assert run[0, 0] > run[-1, 0]  # right-hand rail runs against the road direction


def test_small_drops_and_the_other_side_get_no_rail():
    assert plan_guardrail_runs([_road(0, 200)], [True], _drop_right(90, 110, depth=1.9), **KW) == []


def test_extension_stops_at_the_end_of_the_road_chain():
    # The road ends at x=100 (e.g. a bridge starts there): the rail ends flush, no 20 m beyond
    runs = plan_guardrail_runs([_road(0, 100)], [True], _drop_right(95, 100), **KW)

    assert _x_range(runs[0]) == pytest.approx((75.0, 100.0))


def test_extension_continues_into_the_straight_continuation():
    runs = plan_guardrail_runs([_road(0, 50), _road(50, 100)], [True, True], _drop_right(40, 45), **KW)

    assert len(runs) == 1
    assert _x_range(runs[0]) == pytest.approx((20.0, 65.0))


def test_rail_is_interrupted_where_another_road_joins():
    side_road = [[50.0, float(y), 100.0, 4.0] for y in np.arange(-40.0, 0.5, 1.0)]  # ends at the junction node on the centerline
    runs = plan_guardrail_runs([_road(0, 200), side_road], [True, True], _drop_right(20, 180), **KW)

    main = [r for r in runs if np.allclose(r[:, 1], -(W / 2 + 0.2))]  # the side road gets its own rails as well
    ranges = sorted(_x_range(r) for r in main)
    assert len(ranges) == 2
    assert ranges[0][1] < 50.0 - 2.0 - 0.5 + 0.1 and ranges[1][0] > 50.0 + 2.0 + 0.5 - 0.1


def test_ineligible_roads_get_no_rail_but_still_interrupt_others():
    footway = [[50.0, float(y), 100.0, 2.0] for y in np.arange(-40.0, 0.5, 1.0)]  # ends at the junction node on the centerline
    runs = plan_guardrail_runs([_road(0, 200), footway], [True, False], _drop_right(20, 180), **KW)

    assert len(runs) == 2
    assert all(np.all(r[:, 1] > -6.0) for r in runs)  # only rails along the main road


def test_placement_fills_a_run_with_3_m_segments_and_end_caps():
    run = np.array([[x, 0.0, 100.0] for x in np.arange(30.0, -0.5, -1.0)])  # 30 m against +x

    items = place_guardrail_items([run], segment_length=3.0, beam_offset=0.065, segment_item="rail",
                                  start_item="cap_start", end_item="cap_end")
    segments = [i for i in items if i["type"] == "rail"]

    assert len(segments) == 10
    assert sorted(round(i["pos"][0], 6) for i in segments) == pytest.approx([1.5 + 3 * k for k in range(10)])
    assert [i["type"] for i in items if i["type"] != "rail"] == ["cap_start", "cap_end"]
    caps = {i["type"]: i["pos"][0] for i in items if i["type"] != "rail"}
    assert caps == pytest.approx({"cap_start": 30.0, "cap_end": 0.0})


def test_segment_axes_follow_the_run_and_the_back_faces_away_from_the_road():
    run = np.array([[x, 0.0, 100.0 + 0.1 * x] for x in np.arange(0.0, 30.5, 1.0)])  # 10 % grade, road on the right

    items = place_guardrail_items([run], segment_length=3.0, beam_offset=0.065, segment_item="rail",
                                  start_item="cap_start", end_item="cap_end")
    rail = next(i for i in items if i["type"] == "rail")
    m = np.array(rail["rotationMatrix"]).reshape(3, 3)  # rows = model axes (BeamNG forest4 convention)

    assert m[0] == pytest.approx(np.array([1.0, 0.0, 0.1]) / np.hypot(1.0, 0.1))  # x along the run, with the grade
    assert m[1] == pytest.approx([0.0, 1.0, 0.0])  # y (rail back) left of the run = away from the road
    assert m[2][2] > 0.99
    assert rail["pos"][1] == pytest.approx(0.065)  # origin behind the traffic face by the beam depth
    assert rail["scale"] == 1.0


def test_runs_shorter_than_one_segment_are_dropped():
    run = np.array([[x, 0.0, 100.0] for x in (0.0, 1.0, 2.0)])

    assert place_guardrail_items([run], 3.0, 0.065, "rail", "a", "b") == []


def test_rail_height_follows_the_grade_also_at_junction_cuts():
    road = [[float(x), 0.0, 100.0 + 0.08 * x, W] for x in np.arange(0.0, 200.5, 1.0)]
    side_road = [[50.0, float(y), 104.0, 4.0] for y in np.arange(-40.0, 0.5, 1.0)]

    def height_at(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return np.where(y < -4.0, 0.0, 1000.0)  # always a deep drop on the right

    runs = plan_guardrail_runs([road, side_road], [True, False], height_at, **KW)

    for run in runs:
        assert np.all(np.isfinite(run[:, 2]))
        assert run[:, 2] == pytest.approx(100.0 + 0.08 * run[:, 0])
