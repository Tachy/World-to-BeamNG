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


from world_to_beamng.junctions.corners import find_junction_corners


def _t_roads():
    east = [[x, 0.0, 100.0, 6.0] for x in np.arange(0.0, 60.5, 5.0)]
    west = [[x, 0.0, 100.0, 6.0] for x in np.arange(-60.0, 0.5, 5.0)]
    north = [[0.0, y, 100.0, 5.0] for y in np.arange(0.0, 60.5, 5.0)]
    return [east, west, north]


def _t_corners(roads):
    inputs = [{"road_id": rid, "coords": np.asarray(r)[:, :3], "half_widths": np.asarray(r)[:, 3] / 2.0,
               "highway": "residential", "surface": "asphalt_road_standard"} for rid, r in zip(("east", "west", "north"), roads)]
    return find_junction_corners(inputs, {"default_radius": 6.0}, 0.5, 160.0, rank={})


def _plan_t(sides):
    roads = _t_roads()
    return plan_sidewalk_runs(roads, sides, [True] * 3, clearance=1.15, min_length=2.0, endpoint_tol=0.5, max_angle_deg=30.0,
                              corners=_t_corners(roads), road_ids=["east", "west", "north"])


def test_both_sided_corner_joins_the_runs_through_the_arc():
    runs = _plan_t([{"left": "a"}, {}, {"right": "a"}])  # east +y side and north +x side face the NE corner
    assert len(runs) == 1
    points = runs[0]["points"]
    center = np.array([2.5 + 6.0, 3.0 + 6.0])
    distances = np.linalg.norm(points[:, :2] - center, axis=1)
    assert distances.min() == pytest.approx(6.0, abs=0.05)  # the kerb follows the arc
    assert points[:, 0].max() == pytest.approx(60.0) and points[:, 1].max() == pytest.approx(60.0)
    # sidewalk (towards the fillet centre) is on the LEFT of the run direction on the arc
    k = int(np.argmin(np.abs(distances - 6.0) + np.abs(points[:, 0] - points[:, 1] + 0.5)))
    direction = points[min(k + 1, len(points) - 1), :2] - points[max(k - 1, 0), :2]
    to_center = center - points[k, :2]
    assert direction[0] * to_center[1] - direction[1] * to_center[0] > 0


def test_one_sided_corner_ends_at_the_tangent_point():
    runs = _plan_t([{"left": "a"}, {}, {}])
    assert len(runs) == 1
    assert runs[0]["points"][:, 0].min() == pytest.approx(2.5 + 6.0, abs=0.01)


def test_runs_of_other_roads_are_never_joined_into_a_corner():
    roads = _t_roads()
    # unrelated road whose left kerb (y = 9) heads for the north tangent point (2.5, 9); it cuts nothing itself
    passing = [[x, 11.0, 100.0, 4.0] for x in np.arange(30.0, 2.4, -2.5)]  # 30.0 ... 2.5
    runs = plan_sidewalk_runs(roads + [passing], [{"left": "a"}, {}, {"right": "a"}, {"left": "a"}], [True, True, True, False],
                              clearance=1.15, min_length=2.0, endpoint_tol=0.5, max_angle_deg=30.0,
                              corners=_t_corners(roads), road_ids=["east", "west", "north", "passing"])
    assert sorted(r["road_index"] for r in runs) == [2, 3]  # east + north joined (kept as the north run), passing alone
    passing_run = next(r for r in runs if r["road_index"] == 3)
    assert np.allclose(passing_run["points"][:, 1], 9.0) and passing_run["points"][:, 0].min() > 3.0  # cut by the north arm


def test_without_corners_runs_are_unchanged():
    roads = _t_roads()
    plain = plan_sidewalk_runs(roads, [{"left": "a"}, {}, {}], [True] * 3, clearance=1.15, min_length=2.0,
                               endpoint_tol=0.5, max_angle_deg=30.0)
    assert plain[0]["points"][:, 0].min() == pytest.approx(2.5 + 1.15, abs=0.01)


def test_joining_keeps_unrelated_pieces_of_the_same_road_side():
    from world_to_beamng.sidewalks.runs import _join_through_corners

    east, west, north = _t_roads()
    corner = next(c for c in _t_corners([east, west, north]) if {a["road_id"] for a in c["arms"]} == {"east", "north"})
    far_piece = {"road_index": 0, "side": "left", "surface": "a", "points": np.array([[40.0, 3.0, 100.0], [60.0, 3.0, 100.0]])}
    outgoing = {"road_index": 0, "side": "left", "surface": "a", "points": np.array([[8.5, 3.0, 100.0], [30.0, 3.0, 100.0]])}
    incoming = {"road_index": 2, "side": "right", "surface": "a", "points": np.array([[2.5, 60.0, 100.0], [2.5, 9.0, 100.0]])}
    runs = _join_through_corners([far_piece, outgoing, incoming], [(corner, 0, 2)])
    assert len(runs) == 2
    joined = next(r for r in runs if r["road_index"] == 2)
    assert joined["points"][0, 1] == pytest.approx(60.0) and joined["points"][-1, 0] == pytest.approx(30.0)
    assert any(r["points"][0, 0] == pytest.approx(40.0) for r in runs)


def test_block_with_sidewalks_all_around_joins_into_one_run_in_any_corner_order():
    rng = np.random.default_rng(1)
    step = lambda a, b: [[a[0] + (b[0] - a[0]) * k / 10, a[1] + (b[1] - a[1]) * k / 10, 100.0, 6.0] for k in range(11)]
    block = [step((0, 0), (50, 0)), step((50, 0), (50, 50)), step((50, 50), (0, 50)), step((0, 50), (0, 0))]
    stubs = [step((0, 0), (-30, -30)), step((50, 0), (80, -30)), step((50, 50), (80, 80)), step((0, 50), (-30, 80))]
    roads = block + stubs
    ids = [f"r{k}" for k in range(len(roads))]
    inputs = [{"road_id": rid, "coords": np.asarray(r)[:, :3], "half_widths": np.full(len(r), 3.0),
               "highway": "residential", "surface": "a"} for rid, r in zip(ids, roads)]
    corners = find_junction_corners(inputs, {"default_radius": 6.0}, 0.5, 160.0, rank={})
    sides = [{"left": "a"}] * 4 + [{}] * 4  # all four block roads have their sidewalk inside the block
    for _ in range(12):
        order = [corners[k] for k in rng.permutation(len(corners))]
        runs = plan_sidewalk_runs(roads, sides, [True] * len(roads), clearance=1.15, min_length=2.0, endpoint_tol=0.5,
                                  max_angle_deg=30.0, corners=order, road_ids=ids)
        assert len(runs) == 1


def test_corners_with_a_non_blocking_arm_do_not_trim_sidewalks():
    roads = _t_roads()  # east, west, north - here the north arm is a track that does not interrupt sidewalks
    runs = plan_sidewalk_runs(roads, [{"left": "a"}, {"left": "a"}, {}], [True, True, False], clearance=1.15, min_length=2.0,
                              endpoint_tol=0.5, max_angle_deg=30.0, corners=_t_corners(roads), road_ids=["east", "west", "north"])
    east = next(r for r in runs if r["road_index"] == 0)
    west = next(r for r in runs if r["road_index"] == 1)
    assert east["points"][:, 0].min() == pytest.approx(0.0)  # not trimmed at the track's tangent point (8.5)
    assert west["points"][:, 0].max() == pytest.approx(0.0)
