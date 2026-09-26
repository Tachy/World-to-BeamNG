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


# --- the centre line of a narrower road runs onto the double line of the wider road ------------------------------------
from world_to_beamng.geometry.road_markings import boundary_shifts, direction_boundary_offset  # noqa: E402


def test_direction_boundary_offset_is_the_lane_boundary_between_the_directions():
    assert direction_boundary_offset(9.75, 3, 1) == pytest.approx(-1.625)  # one forward lane on the right
    assert direction_boundary_offset(9.75, 3, 2) == pytest.approx(1.625)
    assert direction_boundary_offset(6.5, 2, 1) == pytest.approx(0.0)


def test_boundary_shift_moves_only_the_direction_boundary_lines():
    widths = np.array([9.75, 9.75])
    shift = np.array([0.0, 0.5])

    plain = dict(line_offsets(widths, 3, 0.25, forward=1, center_gap=0.1, line_width=0.15))
    shifted = line_offsets(widths, 3, 0.25, forward=1, center_gap=0.1, line_width=0.15, boundary_shift=shift)

    centers = [o for k, o in shifted if k == CENTER]
    assert all(c[1] - c[0] == pytest.approx(0.5) for c in centers)
    assert next(o for k, o in shifted if k == DIVIDER) == pytest.approx(plain[DIVIDER])  # dashed divider stays


def test_boundary_shift_moves_the_single_divider_of_a_two_lane_road():
    lines = line_offsets(np.array([6.5, 6.5]), 2, 0.25, boundary_shift=np.array([0.0, -1.0]))

    assert next(o for k, o in lines if k == DIVIDER) == pytest.approx([0.0, -1.0])


def _road_nodes(xs, width):
    return [[float(x), 0.0, 100.0, float(w)] for x, w in zip(xs, width if hasattr(width, "__len__") else [width] * len(xs))]


def test_two_lane_centre_line_shifts_onto_the_double_line_of_the_three_lane_road():
    # 2-lane road ends at x=0, 3-lane road (1 forward lane) continues; widths blend over 50 m each side
    from world_to_beamng.geometry.road_width_transitions import smoothstep

    xs = np.arange(-100.0, 1.0, 10.0)
    widths = [6.5 + 3.25 * smoothstep((50.0 - -x) / 100.0) if -x <= 50.0 else 6.5 for x in xs]
    narrow = _road_nodes(xs, widths)
    wide = _road_nodes(np.arange(0.0, 101.0, 10.0), [8.125] * 6 + [9.75] * 5)

    shifts = boundary_shifts([narrow, wide], [MarkingLayout(lanes=2), MarkingLayout(lanes=3, forward=1)],
                             [6.5, 9.75], [((0, "end"), (1, "start"))])

    joint_target = direction_boundary_offset(8.125, 3, 1)
    assert shifts[0][-1] == pytest.approx(joint_target)  # at the joint the lines meet
    assert shifts[0][0] == pytest.approx(0.0) and shifts[0][5] == pytest.approx(0.0, abs=1e-9)  # 50 m before it: none
    assert 1 not in shifts  # the wider road keeps its own double line


def test_opposite_digitization_mirrors_the_shift():
    xs = np.arange(0.0, 101.0, 10.0)
    narrow = _road_nodes(xs, [8.125] + [6.5] * 10)  # 2-lane road STARTS at the joint
    wide = _road_nodes(np.arange(-100.0, 1.0, 10.0), [9.75] * 5 + [8.125] * 6)  # digitized toward the joint as well

    same = boundary_shifts([wide, narrow], [MarkingLayout(lanes=3, forward=1), MarkingLayout(lanes=2)],
                           [9.75, 6.5], [((0, "end"), (1, "start"))])
    opposite = boundary_shifts([wide[::-1], narrow], [MarkingLayout(lanes=3, forward=1), MarkingLayout(lanes=2)],
                               [9.75, 6.5], [((0, "start"), (1, "start"))])

    assert same[1][0] == pytest.approx(direction_boundary_offset(8.125, 3, 1))
    assert opposite[1][0] == pytest.approx(-direction_boundary_offset(8.125, 3, 1))


def test_no_shift_without_a_double_line_on_the_wider_road_or_for_equal_lanes():
    narrow = _road_nodes(np.arange(-20.0, 1.0, 10.0), [6.5, 6.5, 8.0])
    wide = _road_nodes(np.arange(0.0, 21.0, 10.0), [8.0, 9.75, 9.75])

    assert boundary_shifts([narrow, wide], [MarkingLayout(lanes=2), MarkingLayout(lanes=3)], [6.5, 9.75],
                           [((0, "end"), (1, "start"))]) == {}  # oneway 3-lane road: no double line
    assert boundary_shifts([narrow, wide], [MarkingLayout(lanes=3, forward=1), MarkingLayout(lanes=3, forward=1)],
                           [6.5, 9.75], [((0, "end"), (1, "start"))]) == {}


def test_forced_double_centre_line_for_two_lane_roads():
    # Two-lane tunnels and galleries: layout asked to force the double line (structure rule from the workflow)
    two_way = marking_layout({"highway": "primary", "lanes": "2"}, 6.5, "asphalt_road_standard", MARKED,
                             "asphalt_road_standard", 5.5, double_center_min_lanes=3, force_double_center=True)
    oneway = marking_layout({"highway": "primary", "lanes": "2", "oneway": "yes"}, 6.5, "asphalt_road_standard", MARKED,
                            "asphalt_road_standard", 5.5, double_center_min_lanes=3, force_double_center=True)
    single = marking_layout({"highway": "primary_link"}, 4.0, "asphalt_road_standard", MARKED,
                            "asphalt_road_standard", 5.5, double_center_min_lanes=3, force_double_center=True)

    assert two_way == MarkingLayout(lanes=2, forward=1)
    assert oneway == MarkingLayout(lanes=2) and single == MarkingLayout(lanes=1)


# --- at structures the lines are aligned 50 m before the structure --------------------------------------------------------
from world_to_beamng.geometry.road_markings import structure_boundary_shifts  # noqa: E402
from world_to_beamng.geometry.road_width_transitions import smoothstep  # noqa: E402

SPAN, DONE = 100.0, 50.0


def _approach(x0, width_fn, step=10.0):
    xs = np.arange(x0, 0.001, step)
    return [[float(x), 0.0, 100.0, float(width_fn(-x))] for x in xs]  # width_fn(distance to the structure)


def _struct(x1, width, step=10.0):
    return [[float(x), 0.0, 100.0, float(width)] for x in np.arange(0.0, x1 + 0.001, step)]


def _wide_to_narrow(d):  # 3-lane width 9.75 -> 6.5 over 100 m before the structure
    return 6.5 + 3.25 * smoothstep(min(d, 100.0) / 100.0)


def _boundary(width, lanes, forward):
    return direction_boundary_offset(width, lanes, forward)


def test_road_lines_are_aligned_with_the_structure_50_m_before_it():
    road, tunnel = _approach(-200.0, _wide_to_narrow), _struct(100.0, 6.5)
    layouts = [MarkingLayout(lanes=3, forward=1), MarkingLayout(lanes=2, forward=1)]

    shifts = structure_boundary_shifts([road, tunnel], layouts, [False, True], [((0, "end"), (1, "start"))], SPAN, DONE)

    widths = np.array([n[3] for n in road])
    own = np.array([_boundary(w, 3, 1) for w in widths])
    distance = -np.array([n[0] for n in road])
    aligned = own + shifts[0]
    assert aligned[distance <= DONE + 1e-9] == pytest.approx(_boundary(6.5, 2, 1))  # exactly the structure's boundary
    assert shifts[0][distance >= SPAN - 1e-9] == pytest.approx(0.0)  # untouched from 100 m before
    assert 1 not in shifts  # the structure keeps its lines


def test_shift_is_mirrored_for_opposite_digitization():
    road = _approach(-200.0, _wide_to_narrow)
    tunnel = _struct(100.0, 9.75)[::-1]  # digitized toward the joint
    layouts = [MarkingLayout(lanes=2, forward=1), MarkingLayout(lanes=3, forward=1)]

    same = structure_boundary_shifts([road, _struct(100.0, 9.75)], layouts, [False, True], [((0, "end"), (1, "start"))], SPAN, DONE)
    opposite = structure_boundary_shifts([road, tunnel], layouts, [False, True], [((0, "end"), (1, "end"))], SPAN, DONE)

    assert opposite[0][-1] == pytest.approx(-_boundary(9.75, 3, 1) - _boundary(6.5, 2, 1))  # mirrored target
    assert same[0][-1] == pytest.approx(_boundary(9.75, 3, 1) - _boundary(6.5, 2, 1))


def test_shift_continues_across_road_pieces():
    first = [[float(x), 0.0, 100.0, 9.75] for x in np.arange(-200.0, -59.0, 10.0)]
    second = _approach(-60.0, _wide_to_narrow)
    tunnel = _struct(100.0, 6.5)
    layouts = [MarkingLayout(lanes=3, forward=1)] * 2 + [MarkingLayout(lanes=2, forward=1)]

    shifts = structure_boundary_shifts([first, second, tunnel], layouts, [False, False, True],
                                       [((0, "end"), (1, "start")), ((1, "end"), (2, "start"))], SPAN, DONE)

    assert shifts[1][-1] > 0.0 and shifts[1][0] > 0.0  # the second piece lies within 100 m of the tunnel
    assert shifts[0][-1] > 0.0  # 60 m before the tunnel: in the first piece, still within the zone
    assert shifts[0][0] == pytest.approx(0.0)  # 200 m before it: outside


def test_no_structure_shift_when_the_boundary_is_already_at_the_structures():
    road = [[float(x), 0.0, 100.0, 6.5] for x in np.arange(-100.0, 0.1, 10.0)]
    layouts = [MarkingLayout(lanes=2), MarkingLayout(lanes=2, forward=1)]

    shifts = structure_boundary_shifts([road, _struct(50.0, 6.5)], layouts, [False, True], [((0, "end"), (1, "start"))], SPAN, DONE)

    assert shifts == {} or np.allclose(shifts[0], 0.0)


def test_boundary_shifts_leave_structure_joints_to_the_structure_rule():
    narrow = _road_nodes(np.arange(-20.0, 1.0, 10.0), [6.5, 7.0, 8.125])
    wide = _road_nodes(np.arange(0.0, 21.0, 10.0), [8.125, 9.75, 9.75])

    assert boundary_shifts([narrow, wide], [MarkingLayout(lanes=2), MarkingLayout(lanes=3, forward=1)], [6.5, 9.75],
                           [((0, "end"), (1, "start"))], fixed=[False, True]) == {}


# --- block stripes replace the dashed divider of a tapering lane (a lane is dropped or added over 100 m) -----------------
from world_to_beamng.geometry.road_markings import BLOCK, block_inputs, taper_zones, zone_divider_masks  # noqa: E402

LANE_W = 3.25


def _blend(distance, half=50.0):  # 0 at 50 m before the joint ... 1 at 50 m after it (smoothstep)
    return smoothstep(min(max(distance + half, 0.0), 2 * half) / (2 * half))


def _symmetric_pair():
    """3-lane road (1 forward, 2 backward) ending at x=0, 2-lane road from x=0; widths blend over +-50 m."""
    wide = [[float(x), 0.0, 100.0, 9.75 + (6.5 - 9.75) * _blend(x)] for x in np.arange(-120.0, 1.0, 10.0)]
    narrow = [[float(x), 0.0, 100.0, 9.75 + (6.5 - 9.75) * _blend(x)] for x in np.arange(0.0, 121.0, 10.0)]
    layouts = [MarkingLayout(lanes=3, forward=1), MarkingLayout(lanes=2, forward=1)]
    return [wide, narrow], layouts, [9.75, 6.5], [False, False], [((0, "end"), (1, "start"))]


def test_zone_covers_the_widths_that_change_and_the_first_node_at_its_end():
    roads, layouts, own, fixed, pairs = _symmetric_pair()

    zones = taper_zones(roads, layouts, own, fixed, pairs)

    assert len(zones) == 1
    xs_wide = [n[0] for n, keep in zip(roads[0], zones[0]["pieces"][0][1]) if keep]
    xs_narrow = [n[0] for n, keep in zip(roads[1], zones[0]["pieces"][1][1]) if keep]
    assert xs_wide == [-50.0, -40.0, -30.0, -20.0, -10.0, 0.0]  # from the first node at the zone end to the joint
    assert xs_narrow == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert zones[0]["lane_width"] == pytest.approx(LANE_W)


def test_extra_lane_side_follows_the_directions_of_the_wider_road():
    roads, layouts, own, fixed, pairs = _symmetric_pair()
    backward_extra = taper_zones(roads, layouts, own, fixed, pairs)[0]
    layouts[0] = MarkingLayout(lanes=3, forward=2)  # two forward lanes: the extra lane is on the right

    forward_extra = taper_zones(roads, layouts, own, fixed, pairs)[0]

    assert [sign for _, _, sign in backward_extra["pieces"]] == [1.0, 1.0]  # left of the digitization direction
    assert [sign for _, _, sign in forward_extra["pieces"]] == [-1.0, -1.0]


def test_block_line_keeps_the_outer_lane_at_full_width_and_runs_into_the_centre_line():
    roads, layouts, own, fixed, pairs = _symmetric_pair()
    zone = taper_zones(roads, layouts, own, fixed, pairs)

    blocks = block_inputs(zone)
    lines = build_marking_lines(roads[0], layouts[0], 0.25, center_gap=0.1, line_width=0.15, blocks=blocks.get(0))
    narrow = build_marking_lines(roads[1], layouts[1], 0.25, blocks=blocks.get(1))

    block = next(line for kind, line in lines if kind == BLOCK)
    block_narrow = next(line for kind, line in narrow if kind == BLOCK)
    assert block[:, 0].min() == pytest.approx(-50.0) and block_narrow[:, 0].max() == pytest.approx(50.0)
    assert block[0, 1] == pytest.approx(4.875 - LANE_W)  # at the start of the zone: the ordinary divider position
    assert block[-1, 1] == pytest.approx(block_narrow[0, 1])  # continuous over the joint
    assert block_narrow[-1, 1] == pytest.approx(0.0)  # the inner lane is gone: the line meets the centre line
    edge = next(line for kind, line in lines if kind == EDGE and line[0, 1] > 0)  # left edge line of the wide road
    for x, y, _ in block[::2]:
        edge_y = edge[np.argmin(np.abs(edge[:, 0] - x)), 1]
        assert edge_y - y == pytest.approx(LANE_W - 0.25)  # outer lane: 3.25 m minus the edge inset, all along the zone


def test_dashed_divider_of_the_tapering_side_ends_where_the_zone_starts():
    roads, layouts, own, fixed, pairs = _symmetric_pair()
    zone = taper_zones(roads, layouts, own, fixed, pairs)

    masks = zone_divider_masks(zone)

    kept = [n[0] for n, keep in zip(roads[0], masks[0][1.0]) if keep]
    assert max(kept) == pytest.approx(-60.0)  # the zone starts at x = -50, the node before it is the last with the dashed line
    assert 1 not in masks  # the narrow road has no divider on that side

    lines = build_marking_lines(roads[0], layouts[0], 0.25, center_gap=0.1, line_width=0.15, divider_keep=masks[0])
    divider = next(line for kind, line in lines if kind == DIVIDER)
    assert divider[:, 0].max() == pytest.approx(-60.0)


def test_zone_on_the_road_before_a_structure_lies_on_the_road_over_100_m():
    def blend100(d):
        return smoothstep(min(max(d, 0.0), 100.0) / 100.0)  # d = distance to the structure

    road = [[float(x), 0.0, 100.0, 6.5 + 3.25 * blend100(-x)] for x in np.arange(-150.0, 1.0, 10.0)]
    tunnel = [[float(x), 0.0, 100.0, 6.5] for x in np.arange(0.0, 101.0, 10.0)]
    layouts = [MarkingLayout(lanes=3, forward=1), MarkingLayout(lanes=2, forward=1)]

    zone = taper_zones([road, tunnel], layouts, [9.75, 6.5], [False, True], [((0, "end"), (1, "start"))])

    assert len(zone) == 1 and [i for i, _, _ in zone[0]["pieces"]] == [0]  # only the road side
    xs = [n[0] for n, keep in zip(road, zone[0]["pieces"][0][1]) if keep]
    assert xs[0] == pytest.approx(-100.0) and xs[-1] == pytest.approx(0.0)


def test_lane_gain_toward_a_structure_grows_the_block_line_out_of_the_centre_line():
    road = [[float(x), 0.0, 100.0, 6.5 + 3.25 * smoothstep(min(-x, 100.0) / 100.0 * -1 + 1.0)] for x in np.arange(-150.0, 1.0, 10.0)]
    tunnel = [[float(x), 0.0, 100.0, 9.75] for x in np.arange(0.0, 101.0, 10.0)]
    layouts = [MarkingLayout(lanes=2, forward=1), MarkingLayout(lanes=3, forward=1)]

    zone = taper_zones([road, tunnel], layouts, [6.5, 9.75], [False, True], [((0, "end"), (1, "start"))])
    lines = build_marking_lines(road, layouts[0], 0.25, blocks=block_inputs(zone).get(0))

    block = next(line for kind, line in lines if kind == BLOCK)
    assert block[0, 1] == pytest.approx(0.0)  # starts in the centre line ...
    assert block[-1, 1] == pytest.approx(4.875 - LANE_W)  # ... and ends at the divider position of the tunnel


def test_opposite_digitization_mirrors_the_block_side():
    roads, layouts, own, fixed, _ = _symmetric_pair()
    narrow_reversed = roads[1][::-1]

    zone = taper_zones([roads[0], narrow_reversed], layouts, own, fixed, [((0, "end"), (1, "end"))])

    assert [sign for _, _, sign in zone[0]["pieces"]] == [1.0, -1.0]


def test_no_zone_for_equal_lanes_oneway_roads_or_more_than_one_extra_lane():
    roads, layouts, own, fixed, pairs = _symmetric_pair()

    assert taper_zones(roads, [MarkingLayout(lanes=3, forward=1)] * 2, own, fixed, pairs) == []
    assert taper_zones(roads, [MarkingLayout(lanes=3), layouts[1]], own, fixed, pairs) == []  # oneway: no forward count
    assert taper_zones(roads, [MarkingLayout(lanes=4, forward=2), layouts[1]], own, fixed, pairs) == []  # two extra lanes


# --- the uninvolved lane keeps its width in the taper zone ------------------------------------------------------------------
def _uninvolved_lane_widths(roads, layouts, zone, piece_index):
    """Distance from the right edge to the boundary line between the directions, per zone node of a piece."""
    piece, mask, _ = next(p for p in zone["pieces"] if p[0] == piece_index)
    shift = zone["shifts"][piece]
    widths = np.array([n[3] for n in roads[piece]])
    layout = layouts[piece]
    lane = layout.forward if layout.forward is not None else layout.lanes // 2
    boundary = -widths / 2.0 + lane * widths / layout.lanes + shift
    return (boundary + widths / 2.0)[mask]


def test_uninvolved_lane_keeps_its_width_through_the_zone_on_both_sides_of_the_joint():
    roads, layouts, own, fixed, pairs = _symmetric_pair()
    zone = taper_zones(roads, layouts, own, fixed, pairs)[0]

    wide = _uninvolved_lane_widths(roads, layouts, zone, 0)
    narrow = _uninvolved_lane_widths(roads, layouts, zone, 1)

    assert wide == pytest.approx(3.25) and narrow == pytest.approx(3.25)  # the single forward lane stays 3.25 m wide


def test_uninvolved_lane_blends_only_from_its_old_to_its_new_single_width():
    roads, layouts, own, fixed, pairs = _symmetric_pair()
    for node in roads[1]:  # the 2-lane road is 7.0 m wide: 3.5 m per lane
        node[3] = 7.0 + (node[3] - 6.5) * (9.75 - 7.0) / (9.75 - 6.5)
    for node in roads[0]:
        node[3] = 7.0 + (node[3] - 6.5) * (9.75 - 7.0) / (9.75 - 6.5)
    zone = taper_zones(roads, layouts, [9.75, 7.0], fixed, pairs)[0]

    widths = np.concatenate([_uninvolved_lane_widths(roads, layouts, zone, 0), _uninvolved_lane_widths(roads, layouts, zone, 1)])

    assert widths[0] == pytest.approx(3.25) and widths[-1] == pytest.approx(3.5)
    assert np.all(np.diff(widths) >= -1e-9)  # monotone: nothing narrower than the old lane, nothing wider than the new


def test_shift_is_zero_outside_the_zone_and_mirrored_for_a_forward_extra_lane():
    roads, layouts, own, fixed, pairs = _symmetric_pair()
    backward = taper_zones(roads, layouts, own, fixed, pairs)[0]
    layouts[0] = MarkingLayout(lanes=3, forward=2)  # the extra lane is on the right: the uninvolved lane is the left one

    forward = taper_zones(roads, layouts, own, fixed, pairs)[0]

    assert np.all(backward["shifts"][0][~backward["pieces"][0][1]] == 0.0)
    left = -(_uninvolved_lane_widths(roads, layouts, forward, 0) - roads[0][0][3] / 2.0 * 0)  # not used
    boundary_from_left = (roads[0][-1][3] / 2.0) - (
        -roads[0][-1][3] / 2.0 + 2 * roads[0][-1][3] / 3 + forward["shifts"][0][-1]
    )
    assert boundary_from_left == pytest.approx(3.25)  # the single backward lane on the left keeps 3.25 m


# --- two zones on one stretch (A2 ramp: 4 -> 3 -> 2 lanes within 66 m) -------------------------------------------------------
from world_to_beamng.geometry.road_width_transitions import apply_width_transitions, find_continuations  # noqa: E402


def _ramp_sequence(spacing=1.0):
    def road(x0, x1, width):
        xs = np.linspace(x0, x1, int(round((x1 - x0) / spacing)) + 1)  # irregular spacing: no node at the exact zone ends
        return [[float(x), 0.0, 100.0, float(width)] for x in xs]

    raw = [road(-200, 0, 13.0), road(0, 28, 9.75), road(28, 66, 9.75), road(66, 266, 6.5)]
    lanes = [4, 3, 3, 2]
    blended = apply_width_transitions(raw, transition_length=10.0, step=1.0, endpoint_tol=0.5, max_angle_deg=30.0,
                                      min_delta=0.05, min_spacing=0.5, lanes=lanes, lane_change_length=100.0)
    layouts = [MarkingLayout(lanes=4, forward=2), MarkingLayout(lanes=3, forward=1), MarkingLayout(lanes=3, forward=1),
               MarkingLayout(lanes=2, forward=1)]
    pairs = find_continuations(blended, 0.5, 30.0)
    return blended, layouts, [13.0, 9.75, 9.75, 6.5], [False] * 4, pairs


@pytest.mark.parametrize("spacing", [1.0, 1.3, 2.7])
def test_each_of_two_zones_on_a_shared_stretch_stays_on_its_own_half(spacing):
    roads, layouts, own, fixed, pairs = _ramp_sequence(spacing)

    zones = taper_zones(roads, layouts, own, fixed, pairs)

    assert len(zones) == 2
    four_to_three = next(z for z in zones if 0 in [p for p, _, _ in z["pieces"]])
    three_to_two = next(z for z in zones if 3 in [p for p, _, _ in z["pieces"]])
    def xs(zone):
        return sorted(round(n[0]) for piece, mask, _ in zone["pieces"] for n, keep in zip(roads[piece], mask) if keep)
    assert min(xs(four_to_three)) == -50 and max(xs(four_to_three)) <= 36  # 50 m on the 4-lane side, half of the stretch
    assert max(xs(three_to_two)) == 116 and min(xs(three_to_two)) >= 30  # half of the stretch, 50 m on the 2-lane side
    assert 3 not in [p for p, _, _ in four_to_three["pieces"]] and 0 not in [p for p, _, _ in three_to_two["pieces"]]


def test_a_zone_does_not_continue_into_a_piece_of_another_road_width():
    roads, layouts, own, fixed, pairs = _ramp_sequence()

    for zone in taper_zones(roads, layouts, own, fixed, pairs):
        widths = {own[p] for p, _, _ in zone["pieces"]}
        assert len(widths) == 2  # the wide and the narrow road of this joint, nothing else


# --- no-overtaking double line on the narrow road after a change to more lanes -------------------------------------
from world_to_beamng.geometry.road_markings import no_overtaking_masks  # noqa: E402


def _long_symmetric_pair():
    """Like _symmetric_pair(), but the narrow 2-lane road (no direction split of its own) runs to x=250."""
    roads, layouts, own, fixed, pairs = _symmetric_pair()
    roads[1] = [[float(x), 0.0, 100.0, 9.75 + (6.5 - 9.75) * _blend(x)] for x in np.arange(0.0, 251.0, 10.0)]
    layouts[1] = MarkingLayout(lanes=2)
    return roads, layouts, own, fixed, pairs


def test_double_line_continues_over_the_transition_and_100_m_beyond_it_on_the_narrow_road():
    roads, layouts, own, fixed, pairs = _long_symmetric_pair()

    masks = no_overtaking_masks(roads, layouts, own, fixed, pairs, extra=100.0)

    xs = [n[0] for n, keep in zip(roads[1], masks[1]) if keep]
    assert xs[0] == 0.0 and xs[-1] == pytest.approx(150.0)  # the transition ends at x=50, plus 100 m
    assert 0 not in masks  # the wider road has the double line anyway


def test_no_double_line_extension_for_equal_lanes_or_a_narrow_road_with_its_own_double_line():
    roads, layouts, own, fixed, pairs = _long_symmetric_pair()
    layouts[1] = MarkingLayout(lanes=2, forward=1)
    assert no_overtaking_masks(roads, layouts, own, fixed, pairs, extra=100.0) == {}
    layouts[1] = MarkingLayout(lanes=3)
    assert no_overtaking_masks(roads, layouts, own, fixed, pairs, extra=100.0) == {}


def test_double_line_replaces_the_dashed_centre_line_only_where_the_mask_says():
    roads, layouts, own, fixed, pairs = _long_symmetric_pair()
    masks = no_overtaking_masks(roads, layouts, own, fixed, pairs, extra=100.0)

    lines = build_marking_lines(roads[1], layouts[1], 0.25, center_gap=0.1, line_width=0.15, double_keep=masks[1])

    centers = [line for kind, line in lines if kind == CENTER]
    dividers = [line for kind, line in lines if kind == DIVIDER]
    assert len(centers) == 2 and len(dividers) == 1
    assert all(line[:, 0].min() == 0.0 and line[:, 0].max() == pytest.approx(150.0) for line in centers)
    assert dividers[0][:, 0].min() <= 150.0 and dividers[0][:, 0].max() == pytest.approx(250.0)  # takes over, no gap
    assert abs(centers[0][0, 1] - centers[1][0, 1]) == pytest.approx(0.25)  # the two lines of the double line
