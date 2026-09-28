"""Tests for world_to_beamng.junctions.corners: junction corners with a tangent-circle fillet."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.junctions.corners import corner_radius

TABLE = {"default_radius": 6.0, "radius_by_highway": {"service": 4.0, "residential": 6.0, "secondary": 10.0}}


def test_corner_radius_is_the_smaller_of_both_arms_with_default_for_unknown():
    assert corner_radius("secondary", "residential", TABLE) == 6.0
    assert corner_radius("secondary", "service", TABLE) == 4.0
    assert corner_radius("secondary", "secondary", TABLE) == 10.0
    assert corner_radius("motorway", "secondary", TABLE) == 6.0


from world_to_beamng.junctions.corners import find_junction_corners

RANK = {"secondary": 4, "residential": 2, "service": 1}


def _road(road_id, points, width, highway="residential", surface="asphalt_road_standard"):
    coords = np.array([[x, y, 100.0] for x, y in points])
    return {"road_id": road_id, "coords": coords, "half_widths": np.full(len(coords), width / 2.0), "highway": highway, "surface": surface}


def _line(a, b, step=5.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    count = int(np.ceil(np.linalg.norm(b - a) / step))
    return [tuple(a + (b - a) * k / count) for k in range(count + 1)]


def _corners(roads, max_angle=160.0):
    return find_junction_corners(roads, TABLE, endpoint_tol=0.5, max_angle_deg=max_angle, rank=RANK)


T_JUNCTION = [
    _road("east", _line((0, 0), (60, 0)), 6.0),
    _road("west", _line((-60, 0), (0, 0)), 6.0),
    _road("north", _line((0, 0), (0, 60)), 5.0),
]


def test_t_junction_gets_two_corners_and_the_straight_side_none():
    corners = _corners(T_JUNCTION)
    assert len(corners) == 2
    assert {tuple(sorted(a["road_id"] for a in c["arms"])) for c in corners} == {("east", "north"), ("north", "west")}


def test_cross_gets_four_corners_and_two_arm_nodes_none():
    cross = T_JUNCTION + [_road("south", _line((0, -60), (0, 0)), 5.0)]
    assert len(_corners(cross)) == 4
    assert _corners(T_JUNCTION[:2]) == []  # a plain continuation is no junction


def test_right_angle_fillet_tangent_points_center_and_area():
    corner = next(c for c in _corners(T_JUNCTION) if {a["road_id"] for a in c["arms"]} == {"east", "north"})
    r = corner["radius"]
    assert r == 6.0
    assert corner["corner_point"][:2] == pytest.approx([2.5, 3.0])
    ends = {tuple(np.round(corner["arc"][0, :2], 6)), tuple(np.round(corner["arc"][-1, :2], 6))}
    assert ends == {(2.5 + r, 3.0), (2.5, 3.0 + r)}
    assert corner["center"] == pytest.approx([2.5 + r, 3.0 + r])
    polygon = np.vstack([corner["corner_point"][None, :2], corner["arc"][:, :2]])
    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(r * r * (1 - math.pi / 4), rel=0.01)


def test_arm_sides_and_trims_follow_the_drawing_direction():
    corner = next(c for c in _corners(T_JUNCTION) if {a["road_id"] for a in c["arms"]} == {"east", "north"})
    arms = {a["road_id"]: a for a in corner["arms"]}
    assert arms["east"]["side"] == "left" and arms["east"]["end"] == "start"  # corner at +y of an eastbound road
    assert arms["north"]["side"] == "right" and arms["north"]["end"] == "start"  # corner at +x of a northbound road
    assert arms["east"]["trim"] == pytest.approx(2.5 + 6.0) and arms["north"]["trim"] == pytest.approx(3.0 + 6.0)
    other = next(c for c in _corners(T_JUNCTION) if {a["road_id"] for a in c["arms"]} == {"north", "west"})
    west = next(a for a in other["arms"] if a["road_id"] == "west")
    assert west["end"] == "end" and west["side"] == "left"  # west road is drawn towards the node (+x): the NW corner at +y is its left


def test_corner_heights_follow_the_arm_centerlines():
    roads = [_road("east", _line((0, 0), (60, 0)), 6.0), _road("west", _line((-60, 0), (0, 0)), 6.0),
             _road("north", _line((0, 0), (0, 60)), 5.0)]
    roads[0]["coords"][:, 2] = 100.0 + 0.1 * roads[0]["coords"][:, 0]  # east arm climbs 10 %
    corner = next(c for c in _corners(roads) if {a["road_id"] for a in c["arms"]} == {"east", "north"})
    east_end = next(p for p in corner["arc"] if p[1] == pytest.approx(3.0))
    assert east_end[2] == pytest.approx(100.0 + 0.1 * east_end[0], abs=0.05)


def test_short_arm_skips_the_corner():
    roads = [_road("east", _line((0, 0), (5, 0)), 6.0), _road("west", _line((-60, 0), (0, 0)), 6.0),
             _road("north", _line((0, 0), (0, 60)), 5.0)]
    ids = [{a["road_id"] for a in c["arms"]} for c in _corners(roads)]
    assert {"east", "north"} not in ids and {"north", "west"} in ids


def test_acute_y_stays_bounded_or_is_skipped():
    angle = math.radians(20.0)
    roads = [_road("trunk", _line((-60, 0), (0, 0)), 6.0),
             _road("a", _line((0, 0), (60 * math.cos(angle / 2), 60 * math.sin(angle / 2))), 6.0),
             _road("b", _line((0, 0), (60 * math.cos(angle / 2), -60 * math.sin(angle / 2))), 6.0)]
    for corner in _corners(roads):
        pts = np.vstack([corner["corner_point"][None, :2], corner["arc"][:, :2]])
        assert np.linalg.norm(pts - corner["node"][:2], axis=1).max() < 60.0


def test_opening_angle_above_the_limit_gets_nothing():
    kink = math.radians(170.0)
    roads = [_road("east", _line((0, 0), (60, 0)), 6.0),
             _road("other", _line((0, 0), (60 * math.cos(kink), 60 * math.sin(kink))), 6.0),
             _road("north", _line((0, 0), (0, -60)), 5.0)]
    for corner in _corners(roads):
        assert {a["road_id"] for a in corner["arms"]} != {"east", "other"}


def test_surface_of_the_higher_ranked_arm_fills_the_corner():
    roads = [_road("main_e", _line((0, 0), (60, 0)), 6.0, highway="secondary", surface="asphalt_road_standard"),
             _road("main_w", _line((-60, 0), (0, 0)), 6.0, highway="secondary", surface="asphalt_road_standard"),
             _road("lane", _line((0, 0), (0, 60)), 4.0, highway="service", surface="cobblestone_road")]
    assert {c["surface"] for c in _corners(roads)} == {"asphalt_road_standard"}


def test_heights_blend_smoothly_between_arms_of_opposite_grade():
    roads = [_road("east", _line((0, 0), (60, 0), 1.0), 6.0), _road("west", _line((-60, 0), (0, 0), 1.0), 6.0),
             _road("north", _line((0, 0), (0, 60), 1.0), 5.0)]
    roads[0]["coords"][:, 2] = 100.0 + 0.1 * roads[0]["coords"][:, 0]  # east climbs 10 %
    roads[2]["coords"][:, 2] = 100.0 - 0.1 * roads[2]["coords"][:, 1]  # north falls 10 %
    corner = next(c for c in _corners(roads) if {a["road_id"] for a in c["arms"]} == {"east", "north"})
    z = corner["arc"][:, 2]
    assert z[0] == pytest.approx(100.0 + 0.1 * corner["arc"][0, 0], abs=0.02)  # tangent A at arm A's height
    assert z[-1] == pytest.approx(100.0 - 0.1 * corner["arc"][-1, 1], abs=0.02)  # tangent B at arm B's height
    mean_step = abs(z[-1] - z[0]) / (len(z) - 1)
    assert np.abs(np.diff(z)).max() < 2.0 * mean_step + 0.01  # no jump at the bisector


def test_fillet_follows_a_curved_arm():
    from shapely.geometry import LineString, Point

    radius = 50.0  # the north arm bends to the east with a 50 m radius
    angles = np.linspace(np.pi, np.pi - 1.2, 61)
    north = [(radius + radius * np.cos(a), radius * np.sin(a)) for a in angles]
    roads = [_road("east", _line((0, 0), (60, 0), 1.0), 6.0), _road("west", _line((-60, 0), (0, 0), 1.0), 6.0),
             _road("north", north, 5.0)]
    corner = next(c for c in _corners(roads) if {a["road_id"] for a in c["arms"]} == {"east", "north"})
    kerb = LineString(np.array(north)).offset_curve(-2.5)  # right side of the north arm = the corner side
    tangent_b = corner["arc"][-1, :2]
    assert kerb.distance(Point(tangent_b)) < 0.05
    trim = next(a for a in corner["arms"] if a["road_id"] == "north")["trim"]
    assert kerb.project(Point(tangent_b)) == pytest.approx(trim, abs=0.05)
    assert np.linalg.norm(tangent_b - corner["center"]) == pytest.approx(corner["radius"], abs=0.05)
