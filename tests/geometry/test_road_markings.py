"""Tests für die Fahrbahnmarkierungs-Geometrie (geometry/road_markings.py)."""

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
    # 10 m geradeaus nach Osten, Rechtskehre mit 2 m Radius, 10 m zurück nach Westen
    straight_in = [(x, 2.0) for x in np.arange(-10.0, 0.0, 1.0)]
    arc = [(2.0 * np.cos(a), 2.0 * np.sin(a)) for a in np.linspace(np.pi / 2, -np.pi / 2, 13)]
    straight_out = [(x, -2.0) for x in np.arange(-1.0, -11.0, -1.0)]
    return np.array(straight_in + arc + straight_out)


def _backward_steps(line, center, indices):
    """Anzahl Liniensegmente, die entgegen der Fahrtrichtung (Tangente der Mittellinie) laufen."""
    count = 0
    for a, b in zip(indices, indices[1:]):
        tangent = center[min(b + 1, len(center) - 1)] - center[max(b - 1, 0)]
        count += float(np.dot(line[b] - line[a], tangent)) <= 0.0
    return count


def test_forward_indices_remove_backward_running_inner_line_in_tight_hairpin():
    center = _hairpin()
    inner = offset_polyline(center, np.full(len(center), -3.0))  # rechts = innen, Versatz > Radius
    everything = np.arange(len(center))
    assert _backward_steps(inner, center, everything) > 0  # Ausgangslage: läuft in der Kehre rückwärts
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
    assert obstacles.symmetric_difference(polygons[2]).area < 1.0  # bis auf den 1-cm-Rand (Umfang 52 m)
    assert junction_obstacles(3, polygons, tree, excluded=set()) is None
    assert polygons[2].bounds == pytest.approx((-3.0, 0.0, 3.0, 20.0))


def test_side_road_cuts_gap_into_main_road_edge_line_only_on_its_side():
    main = [[-20.0, 0.0, 0.0, 6.5], [0.0, 0.0, 0.0, 6.5], [20.0, 0.0, 0.0, 6.5]]
    side = [[0.0, 0.0, 0.0, 5.0], [0.0, 20.0, 0.0, 5.0]]  # beginnt am gemeinsamen Knoten auf der Mittellinie
    polygons = [road_surface_polygon(n, 0.5) for n in (main, side)]
    obstacles = junction_obstacles(0, polygons, STRtree(polygons), excluded=set())
    left, right, divider = [line for _, line in build_marking_lines(main, MarkingLayout(lanes=2), 0.25)]
    assert len(clip_line(left, obstacles, 1.0)) == 2  # Lücke in der Randlinie auf der Einmündungsseite
    assert len(clip_line(right, obstacles, 1.0)) == 1  # gegenüber durchgehend
    assert len(clip_line(divider, obstacles, 1.0)) == 1  # Leitlinie läuft an der T-Einmündung durch


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
    # Nebenstraße (6,5 m) mündet schräg in die Hauptstraße: ihr flaches Ende steht nicht senkrecht zur Hauptstraße und
    # reicht sonst über deren Mittellinie hinaus - Leitlinie und gegenüberliegende Randlinie dürfen nicht leiden.
    main = _polyline([(-40, 0), (0, 0), (40, 0)], 6.5)
    a = np.radians(angle_deg)
    side = _polyline([(0, 0), (40 * np.cos(a), 40 * np.sin(a))], 6.5)
    nodes = [main, side]
    polygons = [road_surface_polygon(n, 0.5) for n in nodes]
    obstacles = junction_obstacles(0, polygons, STRtree(polygons), excluded=set(), centerlines=[np.array(n)[:, :2] for n in nodes])
    left, right, divider = [line for _, line in build_marking_lines(main, MarkingLayout(lanes=2), 0.25)]
    assert len(clip_line(left, obstacles, 1.0)) == 2  # Einmündungsseite: Lücke
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
