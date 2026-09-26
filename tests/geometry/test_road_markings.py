"""Tests for the road marking geometry (geometry/road_markings.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, box

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.road_markings import (
    DIVIDER,
    EDGE,
    MarkingLayout,
    build_marking_lines,
    forward_indices,
    line_offsets,
    marking_layout,
    offset_polyline,
    parse_lanes,
)

MARKED = {"primary", "secondary", "primary_link"}


def _layout(tags, width=6.5, surface="asphalt_road_standard"):
    return marking_layout(tags, width, surface, MARKED, "asphalt_road_standard", 5.5)


@pytest.mark.parametrize(
    "value, expected", [("2", 2), (" 3 ", 3), (1, 1), ("2;3", None), ("", None), ("0", None), (None, None)]
)
def test_parse_lanes(value, expected):
    assert parse_lanes(value) == expected


def test_layout_uses_lanes_tag():
    assert _layout({"highway": "primary", "lanes": "3"}) == MarkingLayout(lanes=3)


def test_layout_without_lanes_two_lanes_for_wide_road_one_for_narrow_or_link():
    assert _layout({"highway": "secondary"}, width=7.0) == MarkingLayout(lanes=2)
    assert _layout({"highway": "secondary"}, width=5.0) == MarkingLayout(lanes=1)
    assert _layout({"highway": "primary_link"}, width=7.0) == MarkingLayout(lanes=1)


def test_layout_none_for_unmarked_types_surfaces_and_lane_markings_no():
    assert _layout({"highway": "track"}) is None
    assert _layout({"highway": "secondary", "lane_markings": "no"}) is None
    assert _layout({"highway": "secondary"}, surface="cobblestone_road") is None


def test_line_offsets_single_lane_has_two_edges_no_divider():
    lines = line_offsets(np.array([4.0, 4.0]), 1, 0.25)
    assert [k for k, _ in lines] == [EDGE, EDGE]
    assert lines[0][1] == pytest.approx([1.75, 1.75])
    assert lines[1][1] == pytest.approx([-1.75, -1.75])


def test_line_offsets_two_lanes_have_centre_divider_following_width():
    lines = line_offsets(np.array([6.5, 9.75]), 2, 0.25)
    assert [k for k, _ in lines] == [EDGE, EDGE, DIVIDER]
    assert lines[0][1] == pytest.approx([3.0, 4.625])
    assert lines[2][1] == pytest.approx([0.0, 0.0])


def test_offset_polyline_straight_and_left_is_positive():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    assert offset_polyline(xy, np.array([2.0, 2.0, 2.0])) == pytest.approx(np.array([[0, 2], [10, 2], [20, 2]]))


def test_offset_polyline_survives_duplicate_nodes():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    result = offset_polyline(xy, np.full(4, -1.0))
    assert np.isfinite(result).all()
    assert result[:, 1] == pytest.approx([-1.0] * 4)


def _hairpin():
    # 10 m straight east, right-hand U-turn with 2 m radius, 10 m back west
    straight_in = [(x, 2.0) for x in np.arange(-10.0, 0.0, 1.0)]
    arc = [(2.0 * np.cos(a), 2.0 * np.sin(a)) for a in np.linspace(np.pi / 2, -np.pi / 2, 13)]
    straight_out = [(x, -2.0) for x in np.arange(-1.0, -11.0, -1.0)]
    return np.array(straight_in + arc + straight_out)


def _backward_steps(line, center, indices):
    """Number of line segments that run against the direction of travel (tangent of the centerline)."""
    count = 0
    for a, b in zip(indices, indices[1:]):
        tangent = center[min(b + 1, len(center) - 1)] - center[max(b - 1, 0)]
        count += float(np.dot(line[b] - line[a], tangent)) <= 0.0
    return count


def test_forward_indices_remove_backward_running_inner_line_in_tight_hairpin():
    center = _hairpin()
    inner = offset_polyline(center, np.full(len(center), -3.0))  # right = inside, offset > radius
    everything = np.arange(len(center))
    assert _backward_steps(inner, center, everything) > 0  # baseline: runs backward in the U-turn
    kept = forward_indices(inner, center)
    assert _backward_steps(inner, center, kept) == 0
    assert LineString(inner[kept]).is_simple
    assert kept[0] == 0 and kept[-1] == len(center) - 1


def test_forward_indices_keep_outer_line_of_hairpin_complete():
    center = _hairpin()
    outer = offset_polyline(center, np.full(len(center), 3.0))
    assert len(forward_indices(outer, center)) == len(center)


def test_build_marking_lines_two_lane_road():
    nodes = [[0.0, 0.0, 100.0, 6.5], [10.0, 0.0, 101.0, 6.5], [20.0, 0.0, 102.0, 6.5]]
    lines = build_marking_lines(nodes, MarkingLayout(lanes=2), 0.25)
    assert [k for k, _ in lines] == [EDGE, EDGE, DIVIDER]
    left = lines[0][1]
    assert left[:, 1] == pytest.approx([3.0, 3.0, 3.0])
    assert left[:, 2] == pytest.approx([100.0, 101.0, 102.0])


from shapely import STRtree

from world_to_beamng.geometry.road_markings import clip_line, junction_obstacles, road_surface_polygon


def test_clip_line_cuts_out_junction_area_and_interpolates_z():
    line = np.array([[0.0, 3.0, 100.0], [40.0, 3.0, 104.0]])
    pieces = clip_line(line, box(18.0, -10.0, 22.0, 10.0), 1.0)
    assert len(pieces) == 2
    assert pieces[0][-1, 0] == pytest.approx(18.0)
    assert pieces[0][-1, 2] == pytest.approx(101.8)
    assert pieces[1][0, 0] == pytest.approx(22.0)


def test_clip_line_drops_short_pieces_and_handles_no_obstacles():
    line = np.array([[0.0, 3.0, 100.0], [40.0, 3.0, 100.0]])
    assert len(clip_line(line, box(0.5, -10.0, 39.5, 10.0), 1.0)) == 0
    assert len(clip_line(line, None, 1.0)) == 1


def test_junction_obstacles_skip_self_and_excluded_and_far_roads():
    main_a = [[-20.0, 0.0, 0.0, 6.5], [0.0, 0.0, 0.0, 6.5]]
    main_b = [[0.0, 0.0, 0.0, 6.5], [20.0, 0.0, 0.0, 6.5]]
    side = [[0.0, 0.0, 0.0, 5.0], [0.0, 20.0, 0.0, 5.0]]
    far = [[100.0, 100.0, 0.0, 5.0], [120.0, 100.0, 0.0, 5.0]]
    polygons = [road_surface_polygon(n, 0.5) for n in (main_a, main_b, side, far)]
    tree = STRtree(polygons)

    obstacles = junction_obstacles(0, polygons, tree, excluded={1})
    assert obstacles.symmetric_difference(polygons[2]).area < 1.0  # except for the 1 cm border (perimeter 52 m)
    assert junction_obstacles(3, polygons, tree, excluded=set()) is None
    assert polygons[2].bounds == pytest.approx((-3.0, 0.0, 3.0, 20.0))


def test_side_road_cuts_gap_into_main_road_edge_line_only_on_its_side():
    main = [[-20.0, 0.0, 0.0, 6.5], [0.0, 0.0, 0.0, 6.5], [20.0, 0.0, 0.0, 6.5]]
    side = [[0.0, 0.0, 0.0, 5.0], [0.0, 20.0, 0.0, 5.0]]  # starts at the shared node on the centerline
    polygons = [road_surface_polygon(n, 0.5) for n in (main, side)]
    obstacles = junction_obstacles(0, polygons, STRtree(polygons), excluded=set())
    left, right, divider = [line for _, line in build_marking_lines(main, MarkingLayout(lanes=2), 0.25)]
    assert len(clip_line(left, obstacles, 1.0)) == 2  # gap in the edge line on the T-junction side
    assert len(clip_line(right, obstacles, 1.0)) == 1  # continuous on the opposite side
    assert len(clip_line(divider, obstacles, 1.0)) == 1  # divider line runs through at the T-junction


def test_crossing_road_interrupts_divider():
    main = [[-20.0, 0.0, 0.0, 6.5], [0.0, 0.0, 0.0, 6.5], [20.0, 0.0, 0.0, 6.5]]
    crossing = [[0.0, -20.0, 0.0, 5.0], [0.0, 20.0, 0.0, 5.0]]
    polygons = [road_surface_polygon(n, 0.5) for n in (main, crossing)]
    obstacles = junction_obstacles(0, polygons, STRtree(polygons), excluded=set())
    divider = build_marking_lines(main, MarkingLayout(lanes=2), 0.25)[2][1]
    pieces = clip_line(divider, obstacles, 1.0)
    assert len(pieces) == 2
    assert pieces[0][-1, 0] == pytest.approx(-3.0, abs=0.02)


def _polyline(points, width):
    return [[float(x), float(y), 0.0, float(width)] for x, y in points]


@pytest.mark.parametrize("angle_deg", [45.0, 30.0])
def test_oblique_side_road_keeps_divider_and_far_edge_continuous(angle_deg):
    # Side road (6.5 m) joins the main road at an angle: its flat end is not perpendicular to the main road and
    # otherwise extends past its centerline - divider line and opposite edge line must not suffer.
    main = _polyline([(-40, 0), (0, 0), (40, 0)], 6.5)
    a = np.radians(angle_deg)
    side = _polyline([(0, 0), (40 * np.cos(a), 40 * np.sin(a))], 6.5)
    nodes = [main, side]
    polygons = [road_surface_polygon(n, 0.5) for n in nodes]
    obstacles = junction_obstacles(0, polygons, STRtree(polygons), excluded=set(), centerlines=[np.array(n)[:, :2] for n in nodes])
    left, right, divider = [line for _, line in build_marking_lines(main, MarkingLayout(lanes=2), 0.25)]
    assert len(clip_line(left, obstacles, 1.0)) == 2  # T-junction side: gap
    assert len(clip_line(right, obstacles, 1.0)) == 1
    assert clip_line(right, obstacles, 1.0)[0][:, 0] == pytest.approx([-40.0, 0.0, 40.0])
    assert len(clip_line(divider, obstacles, 1.0)) == 1


def test_crossing_split_into_two_side_roads_still_interrupts_divider():
    main = _polyline([(-40, 0), (0, 0), (40, 0)], 6.5)
    north = _polyline([(0, 0), (10, 40)], 5.0)
    south = _polyline([(0, 0), (-10, -40)], 5.0)
    nodes = [main, north, south]
    polygons = [road_surface_polygon(n, 0.5) for n in nodes]
    obstacles = junction_obstacles(0, polygons, STRtree(polygons), excluded=set(), centerlines=[np.array(n)[:, :2] for n in nodes])
    divider = build_marking_lines(main, MarkingLayout(lanes=2), 0.25)[2][1]
    assert len(clip_line(divider, obstacles, 1.0)) == 2


@pytest.mark.parametrize("reverse_second", [False, True])
def test_edge_lines_of_kinked_partners_meet_at_the_joint(reverse_second):
    from world_to_beamng.geometry.road_markings import joint_normals

    kink = np.radians(12.0)
    first = _polyline([(-40, 0), (-20, 0), (0, 0)], 6.5)
    second = _polyline([(0, 0), (20 * np.cos(kink), 20 * np.sin(kink)), (40 * np.cos(kink), 40 * np.sin(kink))], 6.5)
    pair = ((0, "end"), (1, "start"))
    if reverse_second:
        second = second[::-1]
        pair = ((0, "end"), (1, "end"))
    normals = joint_normals([first, second], [pair])
    lines_a = build_marking_lines(first, MarkingLayout(lanes=2), 0.25, end_normal=normals.get((0, "end")))
    end_key = (1, "end") if reverse_second else (1, "start")
    kwargs = {"end_normal" if reverse_second else "start_normal": normals.get(end_key)}
    lines_b = build_marking_lines(second, MarkingLayout(lanes=2), 0.25, **kwargs)
    a_ends = sorted(tuple(np.round(line[-1, :2], 6)) for _, line in lines_a)
    b_ends = sorted(tuple(np.round(line[-1 if reverse_second else 0, :2], 6)) for _, line in lines_b)
    assert a_ends == b_ends


# --- solid double centre line from three lanes on ----------------------------------------------------------------------
from world_to_beamng.geometry.road_markings import CENTER  # noqa: E402


def _center_layout(tags, width=9.75):
    return marking_layout(tags, width, "asphalt_road_standard", MARKED, "asphalt_road_standard", 5.5,
                          double_center_min_lanes=3)


def test_three_lane_two_way_road_gets_the_direction_split_from_lanes_forward():
    assert _center_layout({"highway": "primary", "lanes": "3", "lanes:forward": "1", "lanes:backward": "2"}) == \
        MarkingLayout(lanes=3, forward=1)
    assert _center_layout({"highway": "primary", "lanes": "3", "lanes:backward": "1"}) == MarkingLayout(lanes=3, forward=2)


def test_four_lanes_without_direction_tags_split_in_the_middle():
    assert _center_layout({"highway": "primary", "lanes": "4"}) == MarkingLayout(lanes=4, forward=2)


def test_oneway_and_two_lane_roads_get_no_double_centre_line():
    assert _center_layout({"highway": "primary", "lanes": "3", "oneway": "yes"}) == MarkingLayout(lanes=3)
    assert _center_layout({"highway": "primary", "lanes": "2"}, width=6.5) == MarkingLayout(lanes=2)


def test_double_centre_line_replaces_the_divider_between_the_directions():
    # 9.75 m, 3 lanes, 1 forward: forward lane on the right (-4.875 .. -1.625), the two backward lanes on the left
    lines = line_offsets(np.array([9.75, 9.75]), 3, 0.25, forward=1, center_gap=0.1, line_width=0.15)

    kinds = [k for k, _ in lines]
    assert kinds.count(CENTER) == 2 and kinds.count(DIVIDER) == 1
    centers = sorted(float(o[0]) for k, o in lines if k == CENTER)
    assert centers == pytest.approx([-1.625 - 0.125, -1.625 + 0.125])  # 0.1 m gap between two 0.15 m lines
    divider = next(o for k, o in lines if k == DIVIDER)
    assert divider == pytest.approx([1.625, 1.625])  # between the two backward lanes, still dashed


def test_build_marking_lines_draws_the_double_centre_line():
    nodes = [[x, 0.0, 100.0, 9.75] for x in (0.0, 10.0, 20.0)]

    lines = build_marking_lines(nodes, MarkingLayout(lanes=4, forward=2), 0.25, center_gap=0.1, line_width=0.15)

    centers = [line for kind, line in lines if kind == CENTER]
    assert sorted(float(c[0, 1]) for c in centers) == pytest.approx([-0.125, 0.125])
