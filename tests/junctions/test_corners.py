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


def test_surface_of_the_joining_road_fills_the_corner():
    roads = [_road("main_e", _line((0, 0), (60, 0)), 6.0, highway="secondary", surface="asphalt_road_standard"),
             _road("main_w", _line((-60, 0), (0, 0)), 6.0, highway="secondary", surface="asphalt_road_standard"),
             _road("lane", _line((0, 0), (0, 60)), 4.0, highway="service", surface="gravel_road")]
    assert {c["surface"] for c in _corners(roads)} == {"gravel_road"}


def test_same_rank_the_narrower_arm_decides_the_surface():
    roads = [_road("main_e", _line((0, 0), (60, 0)), 6.0, surface="asphalt_road_standard"),
             _road("main_w", _line((-60, 0), (0, 0)), 6.0, surface="asphalt_road_standard"),
             _road("side", _line((0, 0), (0, 60)), 4.0, surface="gravel_road")]
    assert {c["surface"] for c in _corners(roads)} == {"gravel_road"}


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


FACTORS = (1.0, 0.75, 0.5, 0.33, 0.2, 0.1)


def _short_t(east_length):
    return [_road("east", _line((0, 0), (east_length, 0)), 6.0), _road("west", _line((-60, 0), (0, 0)), 6.0),
            _road("north", _line((0, 0), (0, 60)), 5.0)]


def _ne_corner(roads, sidewalks=None):
    corners = find_junction_corners(roads, TABLE, 0.5, 160.0, rank=RANK, radius_factors=FACTORS, min_radius=0.5,
                                    kerb_min_radius=2.0, sidewalk_sides_by_id=sidewalks or {})
    return next((c for c in corners if {a["road_id"] for a in c["arms"]} == {"east", "north"}), None)


def test_short_arm_gets_a_smaller_radius_instead_of_no_corner():
    corner = _ne_corner(_short_t(5.0))
    assert corner["radius"] == pytest.approx(6.0 * 0.33)  # 3 m would not fit the 5 m arm (trim 2.5 + r)


def test_without_kerb_the_radius_may_drop_below_the_kerb_minimum():
    corner = _ne_corner(_short_t(4.0))  # trim 2.5 + r <= 4 -> r <= 1.5
    assert corner["radius"] == pytest.approx(6.0 * 0.2)


def test_a_corner_with_a_kerb_keeps_the_kerb_minimum_radius():
    kerb = {"east": {"left": "a"}, "north": {"right": "a"}}  # both arms have a sidewalk facing the corner
    assert _ne_corner(_short_t(4.0), kerb) is None  # 2 m does not fit, smaller is not allowed with a kerb
    assert _ne_corner(_short_t(5.0), kerb)["radius"] == pytest.approx(2.0)
    one_sided = {"east": {"left": "a"}}  # only one arm: no kerb in the arc
    assert _ne_corner(_short_t(4.0), one_sided)["radius"] == pytest.approx(6.0 * 0.2)


def test_both_ends_of_a_short_piece_share_it_without_overlapping():
    from shapely.geometry import Polygon

    table = {"default_radius": 10.0}
    roads = [_road("west", _line((-60, 0), (0, 0)), 6.0), _road("north0", _line((0, 0), (0, 60)), 6.0),
             _road("mid", _line((0, 0), (20, 0), 1.0), 6.0), _road("north20", _line((20, 0), (20, 60)), 6.0),
             _road("east", _line((20, 0), (80, 0)), 6.0)]
    corners = find_junction_corners(roads, table, 0.5, 160.0, rank={}, radius_factors=FACTORS, min_radius=0.5)
    on_mid = [c for c in corners if {a["road_id"] for a in c["arms"]} in ({"mid", "north0"}, {"mid", "north20"})]
    assert len(on_mid) == 2
    trims = [next(a["trim"] for a in c["arms"] if a["road_id"] == "mid") for c in on_mid]
    assert sum(trims) <= 20.0
    fills = [Polygon(np.vstack([c["corner_point"][None, :2], c["rim"][:, :2]])) for c in on_mid]
    assert fills[0].intersection(fills[1]).area < 1e-6


def test_a_hairpin_far_along_an_arm_does_not_cost_the_corner():
    # the east arm runs straight for 60 m, then turns back in a hairpin tighter than its half width
    hairpin = _line((0, 0), (60, 0), 1.0) + [(60.5, 0.5), (60.0, 1.0), (40.0, 1.0)]
    roads = [_road("east", hairpin, 6.0), _road("west", _line((-60, 0), (0, 0)), 6.0), _road("north", _line((0, 0), (0, 60)), 5.0)]
    ids = [{a["road_id"] for a in c["arms"]} for c in _corners(roads)]
    assert {"east", "north"} in ids


def test_offset_split_into_touching_parts_by_geos_does_not_cost_the_corner():
    # real arm from the Baden-Wuerttemberg map: straight, but GEOS offsets it by -1.25 m into two touching pieces
    b = [[-0.071, -0.018], [-0.615, -0.499], [-1.16, -0.979], [-1.704, -1.459], [-2.249, -1.94], [-2.793, -2.42],
         [-3.338, -2.901], [-3.882, -3.381], [-4.426, -3.861], [-4.971, -4.342], [-5.515, -4.822], [-6.06, -5.303],
         [-6.604, -5.783], [-7.148, -6.263], [-7.693, -6.744], [-8.237, -7.224], [-8.782, -7.705], [-9.326, -8.185],
         [-9.87, -8.665]]
    roads = [_road("north", _line((0, 0), (0, 60)), 6.5), _road("east", _line((0, 0), (60, 0)), 6.5),
             _road("lane", [tuple(p) for p in b], 2.5)]
    ids = [{a["road_id"] for a in c["arms"]} for c in find_junction_corners(roads, TABLE, 0.5, 160.0, rank=RANK,
                                                                           radius_factors=FACTORS, min_radius=0.5)]
    assert {"north", "lane"} in ids


def test_acute_corner_gets_only_a_small_tip_radius():
    angle = math.radians(12.0)  # two roads leave the node almost in parallel
    roads = [_road("trunk", _line((0, 60), (0, 0), 1.0), 6.0),
             _road("left", _line((0, 0), (-80 * math.sin(angle / 2), -80 * math.cos(angle / 2)), 1.0), 6.0),
             _road("right", _line((0, 0), (80 * math.sin(angle / 2), -80 * math.cos(angle / 2)), 1.0), 6.0)]
    corners = find_junction_corners(roads, TABLE, 0.5, 160.0, rank=RANK, radius_factors=FACTORS, min_radius=0.5,
                                    acute_angle_deg=45.0)
    tip = next(c for c in corners if {a["road_id"] for a in c["arms"]} == {"left", "right"})
    assert tip["radius"] == pytest.approx(0.5)
    outline = np.vstack([tip["corner_point"][None, :2], tip["rim"][:, :2]])
    assert np.linalg.norm(outline - tip["corner_point"][:2], axis=1).max() < 6.0  # a small tip, no long wedge


def test_a_corner_of_exactly_the_limit_angle_gets_no_fill():
    corner_angle = math.radians(160.0)  # "from 160 degrees on" nothing is drawn
    roads = [_road("east", _line((0, 0), (60, 0)), 6.0),
             _road("other", _line((0, 0), (60 * math.cos(corner_angle), 60 * math.sin(corner_angle))), 6.0),
             _road("south", _line((0, 0), (0, -60)), 5.0)]
    assert all({a["road_id"] for a in c["arms"]} != {"east", "other"} for c in _corners(roads, max_angle=160.0))


def _bent_t(grades=False):
    """T with an arm leaving at 60 deg that curves back towards the straight east arm (radius 40 m)."""
    radius, start = 40.0, math.radians(60.0)
    centre = np.array([math.cos(start - math.pi / 2), math.sin(start - math.pi / 2)]) * radius
    angles = np.linspace(start + math.pi / 2, start + math.pi / 2 - 0.9, 40)
    bend = [tuple(centre + radius * np.array([math.cos(a), math.sin(a)])) for a in angles]
    roads = [_road("east", _line((0, 0), (60, 0), 1.0), 6.0), _road("west", _line((-60, 0), (0, 0)), 6.0), _road("bend", bend, 6.0)]
    if grades:  # the two corner arms run away from the node with opposite grades
        roads[0]["coords"][:, 2] = 100.0 - 0.12 * roads[0]["coords"][:, 0]
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(roads[2]["coords"][:, :2], axis=0), axis=1))])
        roads[2]["coords"][:, 2] = 100.0 + 0.12 * along
    return roads


def _bent_corner(grades=False):
    corners = find_junction_corners(_bent_t(grades), TABLE, 0.5, 140.0, rank=RANK, radius_factors=FACTORS, min_radius=0.5,
                                    acute_angle_deg=45.0)
    return next(c for c in corners if {a["road_id"] for a in c["arms"]} == {"east", "bend"})


def test_arms_curving_towards_each_other_do_not_stretch_the_fill():
    from shapely.geometry import Polygon

    corner = _bent_corner()
    fill = Polygon(np.vstack([corner["corner_point"][None, :2], corner["rim"][:, :2]]))
    assert max(a["trim"] for a in corner["arms"]) < 15.6  # straight arms at 60 deg with r = 6 need 15.6 m
    assert fill.area < 24.7  # ... and 24.7 m^2 of fill


def test_fill_height_matches_each_road_along_its_kerb():
    from shapely.geometry import LineString

    from world_to_beamng.junctions.corners import _offset, corner_height
    from world_to_beamng.terrain.road_embedding import _project_onto_polyline

    corner = _bent_corner(grades=True)
    for arm, line, sign, half in zip(corner["arms"], corner["arm_lines"], (1.0, -1.0), corner["halves"]):
        kerb = _offset(LineString(line[:, :2]), sign * half)
        points = np.array([kerb.interpolate(s).coords[0] for s in np.linspace(0.0, arm["trim"], 15)])
        road = _project_onto_polyline(points[:, 0], points[:, 1], line[:, 0], line[:, 1], line[:, 2])
        assert np.abs(corner_height(corner, points[:, 0], points[:, 1]) - road).max() < 0.02
