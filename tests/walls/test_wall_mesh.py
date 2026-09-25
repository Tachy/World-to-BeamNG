"""Tests for world_to_beamng.walls.wall_mesh: rubble stone walls from OSM lines (barrier=wall) with a height value.

Only walls with a `height` tag are built. The wall is a box band of ~50 cm thickness along the line that follows the
terrain: top edge = ground at the centerline + height, bottom edge below the ground on both sides (no gap).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from world_to_beamng.walls.wall_mesh import build_wall_mesh, build_walls, parse_wall_height, select_walls

MATERIAL = "rubble_stone_wall"


def _flat(z=100.0):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _way(way_id, points, **tags):
    return {"type": "way", "id": way_id, "tags": {"barrier": "wall", **tags}, "geometry": [{"lon": x, "lat": y} for x, y in points]}


def _to_local(points):
    return [(p["lon"], p["lat"]) for p in points]


# --- Height value ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [("1.50", 1.5), ("3", 3.0), ("1.8", 1.8), ("2 m", 2.0), ("2,5", 2.5), (" 1.2m", 1.2)],
)
def test_wall_height_is_parsed_in_metres(value, expected):
    assert parse_wall_height(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "abc", "0", "-1", "50", "1..2"])
def test_missing_or_implausible_heights_are_rejected(value):
    assert parse_wall_height(value) is None


# --- Selection -------------------------------------------------------------------------------------------------------


def test_only_walls_with_a_height_are_selected():
    ways = [
        _way(1, [(0, 0), (10, 0)], height="1.5", material="stone"),
        _way(2, [(0, 5), (10, 5)]),  # no height
        _way(3, [(0, 8), (10, 8)], height="kaputt"),
        _way(4, [(0, 9), (10, 9)], height="2"),  # no material tag: stone assumed
    ]

    selected = select_walls(ways, _to_local)

    assert [w["osm_id"] for w in selected] == [1, 4]
    assert selected[0]["height"] == 1.5 and selected[0]["coords"] == [(0, 0), (10, 0)]


def test_other_materials_and_other_barriers_are_not_rubble_walls():
    ways = [
        _way(1, [(0, 0), (10, 0)], height="2", material="brick"),
        _way(2, [(0, 0), (10, 0)], height="2", material="concrete"),
        {**_way(3, [(0, 0), (10, 0)], height="2"), "tags": {"barrier": "fence", "height": "2"}},
        {**_way(4, [(0, 0), (10, 0)], height="2"), "type": "node"},
        _way(5, [(0, 0)], height="2"),  # too short
        _way(6, [(0, 0), (10, 0)], height="2", material="stone"),
        {**_way(7, [(0, 0), (10, 0)], height="2"), "tags": {"barrier": "retaining_wall", "height": "3"}},
    ]

    assert [w["osm_id"] for w in select_walls(ways, _to_local)] == [6, 7]


# --- Geometry -----------------------------------------------------------------------------------------------------


def _mesh(coords, height=2.0, ground=None, **kwargs):
    kwargs.setdefault("thickness", 0.5)
    kwargs.setdefault("sink", 0.3)
    kwargs.setdefault("max_step", 1.0)
    kwargs.setdefault("tile_m", 1.2)
    kwargs.setdefault("cap_thickness", 0.0)  # these tests check the wall body; the cap slabs: test_wall_cap.py
    return build_wall_mesh(coords, height, ground or _flat(), **kwargs)


def _triangles(mesh):
    return np.array([[mesh["vertices"][i] for i in face] for face in mesh["faces"]])


def test_straight_wall_is_half_a_metre_thick_and_as_high_as_tagged():
    mesh = _mesh([(0.0, 0.0), (10.0, 0.0)], height=2.0)
    v = mesh["vertices"]

    assert v[:, 1].min() == pytest.approx(-0.25) and v[:, 1].max() == pytest.approx(0.25)  # 50 cm thick
    assert v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(10.0)
    assert v[:, 2].max() == pytest.approx(102.0)  # ground 100 + height 2
    assert v[:, 2].min() == pytest.approx(99.7)  # 30 cm below the ground, no gap at the foot


def test_faces_point_outwards_and_match_the_stored_normals():
    mesh = _mesh([(0.0, 0.0), (10.0, 0.0)])
    tris, normals = _triangles(mesh), mesh["normals"]

    for face, tri in zip(mesh["faces"], tris):
        geometric = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        assert np.dot(geometric, normals[face[0]]) > 0  # winding order matches the normal (outward)
    # there are faces on both sides, on top and at the ends
    directions = {tuple(np.round(n, 2)) for n in normals}
    assert (0.0, 1.0, 0.0) in directions and (0.0, -1.0, 0.0) in directions and (0.0, 0.0, 1.0) in directions
    assert (1.0, 0.0, 0.0) in directions and (-1.0, 0.0, 0.0) in directions


def test_wall_follows_the_terrain_and_sinks_below_the_ground_on_both_sides():
    slope = lambda x, y: 100.0 + 0.1 * np.asarray(x, float) + 0.2 * np.asarray(y, float)  # also inclined across
    mesh = _mesh([(0.0, 0.0), (10.0, 0.0)], height=2.0, ground=slope)
    v = mesh["vertices"]

    top = v[v[:, 2] > 100.0 + 1.5]
    assert np.allclose(top[:, 2], slope(top[:, 0], 0.0) + 2.0, atol=1e-6)  # top edge = ground at center + height
    bottom = v[v[:, 2] < slope(v[:, 0], v[:, 1]) + 0.05]
    assert np.allclose(bottom[:, 2], slope(bottom[:, 0], bottom[:, 1]) - 0.3, atol=1e-6)  # each side below its ground


def test_long_walls_are_split_into_short_segments_so_they_can_follow_the_ground():
    mesh = _mesh([(0.0, 0.0), (10.0, 0.0)], max_step=1.0)

    xs = np.unique(np.round(mesh["vertices"][:, 0], 6))
    assert len(xs) >= 11 and np.diff(xs).max() <= 1.0 + 1e-6


def test_corner_is_mitred_without_gap_or_overlap():
    coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    mesh = _mesh(coords)
    top = [t for t, n in zip(_triangles(mesh), mesh["normals"][[f[0] for f in mesh["faces"]]]) if n[2] > 0.99]

    area = unary_union([Polygon(t[:, :2]).buffer(1e-6) for t in top if Polygon(t[:, :2]).area > 1e-9])
    expected = LineString(coords).buffer(0.25, cap_style="flat", join_style="mitre")
    assert area.symmetric_difference(expected).area < 0.02 * expected.area


def test_closed_ring_has_no_end_caps():
    ring = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0), (0.0, 0.0)]
    open_line = ring[:-1]

    closed_mesh, open_mesh = _mesh(ring), _mesh(open_line)

    # 24 segments of 1 m, per segment 3 quads (left, right, top) = 6 triangles; the open line additionally has 2 end
    # faces
    assert len(closed_mesh["faces"]) == 6 * 24
    assert len(open_mesh["faces"]) == 6 * 18 + 4
    v = closed_mesh["vertices"]
    assert v[:, 0].min() == pytest.approx(-0.25) and v[:, 0].max() == pytest.approx(6.25)  # outer edge of the corners (miter)


def test_uvs_are_metric_in_texture_tiles():
    mesh = _mesh([(0.0, 0.0), (12.0, 0.0)], tile_m=1.2)

    assert mesh["uvs"].shape == (len(mesh["vertices"]), 2)
    assert mesh["uvs"][:, 0].max() - mesh["uvs"][:, 0].min() == pytest.approx(12.0 / 1.2)  # 12 m = 10 texture tiles
    side = np.abs(mesh["normals"][:, 1]) > 0.99  # long sides (top/end have their own UVs)
    heights = mesh["uvs"][side, 1]
    assert heights.max() - heights.min() == pytest.approx((102.0 - 99.7) / 1.2)  # height in tiles


def test_build_walls_returns_one_mesh_per_wall_with_the_rubble_material():
    ways = [_way(1, [(0, 0), (10, 0)], height="1.5", material="stone"), _way(2, [(0, 5), (10, 5)]), _way(3, [(0, 8), (10, 8)], height="3")]

    meshes, stats = build_walls(ways, _to_local, _flat(), MATERIAL)

    assert [m["id"] for m in meshes] == ["wall_1", "wall_3"]
    assert all(set(m["faces"]) == {MATERIAL} and len(m["faces"][MATERIAL]) > 0 for m in meshes)
    assert stats["built"] == 2 and stats["length"] == pytest.approx(20.0) and stats["without_height"] == 1


# --- Height base at roads -------------------------------------------------------------------------------------------

from world_to_beamng.walls.road_base import RoadBaseHeight

ROAD_Z = 100.0
ROAD_ALONG_X = [np.array([[-50.0, 0.0, ROAD_Z], [50.0, 0.0, ROAD_Z]])]


def _road_base(max_distance=5.0):
    return RoadBaseHeight(ROAD_ALONG_X, max_distance)


def test_a_wall_next_to_a_road_stands_on_the_road_height_not_on_the_embankment():
    mesh = build_wall_mesh([(-10, 3), (10, 3)], 1.5, _flat(90.0), road_base_at=_road_base())  # embankment 10 m below the road

    z = mesh["vertices"][:, 2]
    assert z.max() == pytest.approx(ROAD_Z + 1.5)
    assert z.min() == pytest.approx(90.0 - 0.3)  # the foot reaches below the terrain, no gap


def test_a_wall_beyond_the_snap_distance_stands_on_the_terrain():
    mesh = build_wall_mesh([(-10, 8), (10, 8)], 1.5, _flat(90.0), road_base_at=_road_base())

    z = mesh["vertices"][:, 2]
    assert z.max() == pytest.approx(91.5) and z.min() == pytest.approx(89.7)


def test_terrain_above_the_road_does_not_lift_the_foot_above_the_road_base():
    mesh = build_wall_mesh([(-10, 3), (10, 3)], 1.5, _flat(105.0), road_base_at=_road_base())  # slope above the road

    z = mesh["vertices"][:, 2]
    assert z.max() == pytest.approx(ROAD_Z + 1.5)
    assert z.min() == pytest.approx(ROAD_Z - 0.3)


def test_a_wall_leaving_the_road_changes_over_to_the_terrain():
    mesh = build_wall_mesh([(0, 2), (0, 12)], 1.5, _flat(90.0), road_base_at=_road_base())  # from 2 m to 12 m road distance

    vertices = mesh["vertices"]
    near = vertices[np.abs(vertices[:, 1] - 2.0) < 0.3]
    far = vertices[np.abs(vertices[:, 1] - 12.0) < 0.3]
    assert near[:, 2].max() == pytest.approx(ROAD_Z + 1.5)
    assert far[:, 2].max() == pytest.approx(91.5)


def test_without_roads_the_mesh_is_unchanged():
    coords = [(0, 0), (10, 0), (10, 8)]
    plain = build_wall_mesh(coords, 1.5, _flat(90.0))
    with_none = build_wall_mesh(coords, 1.5, _flat(90.0), road_base_at=RoadBaseHeight([], 5.0))

    np.testing.assert_array_equal(plain["vertices"], with_none["vertices"])


def test_build_walls_passes_the_road_base_through():
    ways = [_way(1, [(-10, 3), (10, 3)], height="1.5")]

    meshes, _ = build_walls(ways, _to_local, _flat(90.0), MATERIAL, road_base_at=_road_base())

    assert meshes[0]["vertices"][:, 2].max() == pytest.approx(ROAD_Z + 1.5)
