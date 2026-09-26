"""Tests for world_to_beamng.bridges.bridge_mesh: bridge deck (carriageway + curb + railing) + support piers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.bridges.bridge_mesh import build_bridge_mesh, build_bridges

DECK, PIER, RAIL = "asphalt_road_standard", "bridge_concrete", "bridge_railing"


def _flat_ground(z=150.0):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _coords(length=60.0, z=200.0, n=7):
    return [(x, 5.0, z) for x in np.linspace(0.0, length, n)]


def _zs(mesh, material):
    faces = mesh["faces"][material]
    return np.array([mesh["vertices"][i][2] for face in faces for i in face])


def test_deck_top_is_flat_at_the_given_height_and_carriageway_width():
    # pier_spacing larger than the span: isolates the test to the pure deck geometry (no pier vertex in
    # "vertices" that would falsify the min() assumption below - see test_piers_reach_down_... for the pier
    # cases with the default pier_spacing).
    mesh = build_bridge_mesh(
        _coords(z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER,
        railing_material=RAIL, deck_thickness=0.6, pier_spacing=1000.0,
    )
    deck_zs = _zs(mesh, DECK)

    assert deck_zs.max() == pytest.approx(200.0)  # carriageway top edge = elevation profile, does NOT follow the terrain
    assert deck_zs.min() == pytest.approx(200.0 - 0.6)  # deck bottom edge


def test_deck_faces_use_the_road_material_not_the_pier_material():
    mesh = build_bridge_mesh(_coords(n=3), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, railing_material=RAIL)

    assert DECK in mesh["faces"] and len(mesh["faces"][DECK]) > 0


def _ys(mesh, material, z=None):
    v = mesh["vertices"]
    return np.array([v[i][1] - 5.0 for face in mesh["faces"][material] for i in face if z is None or v[i][2] == pytest.approx(z)])


def test_curb_sits_on_top_of_the_deck_outside_the_carriageway():
    mesh = build_bridge_mesh(
        _coords(n=3, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER,
        railing_material=RAIL, curb_width=0.25, curb_height=0.15, pier_spacing=1000.0,
    )
    deck_zs, pier_zs = _zs(mesh, DECK), _zs(mesh, PIER)

    assert deck_zs.max() == pytest.approx(200.0)  # carriageway stays at deck level
    assert pier_zs.max() == pytest.approx(200.0 + 0.15)  # curb top edge = deck + curb_height
    # The curb stands OUTSIDE the carriageway: inner face at width / 2 = 4 m, outer edge 0.25 m further out.
    curb_y = np.abs(_ys(mesh, PIER, z=200.0 + 0.15))
    assert curb_y.min() == pytest.approx(4.0)
    assert curb_y.max() == pytest.approx(4.25)


def test_carriageway_keeps_the_full_width_and_the_deck_is_wider_by_the_curbs():
    mesh = build_bridge_mesh(
        _coords(n=3, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER,
        railing_material=RAIL, curb_width=0.25, deck_thickness=0.6, pier_spacing=1000.0,
    )

    # carriageway = the deck faces looking up (the fascia also reaches z=200 but faces sideways)
    up = [f for f in mesh["faces"][DECK] if mesh["normals"][f[0]][2] > 0.99]
    carriageway_y = np.array([mesh["vertices"][i][1] - 5.0 for f in up for i in f])
    assert np.abs(carriageway_y).max() == pytest.approx(4.0)  # road material = full width
    assert np.abs(_ys(mesh, DECK, z=200.0 - 0.6)).max() == pytest.approx(4.25)  # slab carries the curbs


def test_railing_stands_centered_on_the_curb():
    mesh = build_bridge_mesh(
        _coords(n=3, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER,
        railing_material=RAIL, curb_width=0.4, railing_post_size=0.08, pier_spacing=1000.0,
    )

    rail_y = np.abs(_ys(mesh, RAIL))
    # curb spans 4.0 .. 4.4 m, its centerline is at 4.2 m; posts and handrail are 0.08 m wide
    assert rail_y.min() == pytest.approx(4.2 - 0.04)
    assert rail_y.max() == pytest.approx(4.2 + 0.04)


def test_railing_posts_and_handrail_sit_above_the_curb():
    mesh = build_bridge_mesh(
        _coords(n=3, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER,
        railing_material=RAIL, curb_height=0.15, railing_height=0.9, railing_post_size=0.08, pier_spacing=1000.0,
    )

    assert RAIL in mesh["faces"] and len(mesh["faces"][RAIL]) > 0
    rail_zs = _zs(mesh, RAIL)
    assert rail_zs.min() == pytest.approx(200.0 + 0.15)  # posts start at the curb top edge
    assert rail_zs.max() == pytest.approx(200.0 + 0.15 + 0.9 + 0.08 / 2.0)  # handrail top edge


def test_piers_reach_down_into_the_ground_below_a_deep_span():
    mesh = build_bridge_mesh(_coords(length=60.0, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, railing_material=RAIL, pier_spacing=25.0, pier_burial=5.0)

    assert len(mesh["faces"][PIER]) > 0
    assert _zs(mesh, PIER).min() == pytest.approx(145.0)  # (at least) one pier stands 5 m below the natural ground


def test_piers_are_half_the_bridge_width_wide_and_half_of_that_thick():
    mesh = build_bridge_mesh(_coords(length=60.0, z=200.0), width=9.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, railing_material=RAIL, pier_spacing=25.0)

    vertices = np.array([mesh["vertices"][i] for face in mesh["faces"][PIER] for i in face])
    pier = vertices[vertices[:, 2] < 190.0]  # far below the deck: pier vertices only
    at_first_pier = pier[np.abs(pier[:, 0] - 25.0) < 5.0]
    assert np.ptp(at_first_pier[:, 1]) == pytest.approx(4.5)  # across the road: 1/2 of 9 m
    assert np.ptp(at_first_pier[:, 0]) == pytest.approx(2.25)  # along the road: 1/2 of that


def test_no_piers_when_clearance_is_too_small():
    mesh = build_bridge_mesh(_coords(length=60.0, z=151.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, railing_material=RAIL, deck_thickness=0.2, min_pier_clearance=1.0)

    # PIER material still contains the curb faces, but no pier that reaches the terrain
    assert _zs(mesh, PIER).min() == pytest.approx(151.0)


def test_build_bridges_returns_one_mesh_per_bridge_with_its_own_deck_material():
    bridges = [
        {"id": 1, "coords": _coords(z=200.0), "width": 8.0, "deck_material": "asphalt_road_standard"},
        {"id": 2, "coords": _coords(z=210.0), "width": 6.0, "deck_material": "concrete"},
    ]

    meshes = build_bridges(bridges, _flat_ground(150.0), pier_material=PIER, railing_material=RAIL)

    assert [m["id"] for m in meshes] == ["bridge_1", "bridge_2"]
    assert "asphalt_road_standard" in meshes[0]["faces"] and "concrete" in meshes[1]["faces"]


def test_build_bridges_skips_degenerate_bridges():
    bridges = [{"id": 1, "coords": [(0.0, 0.0, 200.0)], "width": 8.0, "deck_material": DECK}]

    assert build_bridges(bridges, _flat_ground(150.0), pier_material=PIER, railing_material=RAIL) == []


def test_carriageway_uvs_follow_the_decal_road_layout():
    # Same texture placement as the DecalRoad on the approach: u across the carriageway 0..1, v along in repeats of
    # road_texture_length meters
    mesh = build_bridge_mesh(
        _coords(length=60.0, n=7, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK,
        pier_material=PIER, railing_material=RAIL, pier_spacing=1000.0, road_texture_length=5.0,
    )
    up = [f for f in mesh["faces"][DECK] if mesh["normals"][f[0]][2] > 0.99]
    uv = np.array([mesh["uvs"][i] for f in up for i in f])

    assert set(np.round(uv[:, 0], 6)) == {0.0, 1.0}
    assert uv[:, 1].min() == pytest.approx(0.0) and uv[:, 1].max() == pytest.approx(60.0 / 5.0)


def test_bridge_deck_curbs_and_railing_follow_a_width_that_changes_along_the_bridge():
    coords = [(x, 5.0, 200.0) for x in (0.0, 20.0, 40.0, 60.0)]
    widths = [6.5, 6.5, 9.75, 9.75]  # widens on the second half (lane change)

    mesh = build_bridge_mesh(coords, width=6.5, widths=widths, ground_at=_flat_ground(150.0), deck_material=DECK,
                             pier_material=PIER, railing_material=RAIL, curb_width=0.4, pier_spacing=1000.0)
    v = mesh["vertices"]

    def half_at(x, material, z):
        ys = [abs(v[i][1] - 5.0) for f in mesh["faces"][material] for i in f
              if abs(v[i][0] - x) < 1e-6 and abs(v[i][2] - z) < 1e-6]
        return max(ys)

    assert half_at(0.0, PIER, 200.2) == pytest.approx(3.25 + 0.4)  # curb outer edge, narrow end
    assert half_at(60.0, PIER, 200.2) == pytest.approx(4.875 + 0.4)  # and at the wide end
    up = [f for f in mesh["faces"][DECK] if mesh["normals"][f[0]][2] > 0.99]
    road_half = {round(v[i][0]): abs(v[i][1] - 5.0) for f in up for i in f}
    assert road_half[0] == pytest.approx(3.25) and road_half[60] == pytest.approx(4.875)  # carriageway = the width
    assert road_half[40] == pytest.approx(4.875)


def test_bridge_without_widths_keeps_the_constant_width():
    a = build_bridge_mesh(_coords(n=3), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK,
                          pier_material=PIER, railing_material=RAIL, pier_spacing=1000.0)
    b = build_bridge_mesh(_coords(n=3), width=8.0, widths=[8.0, 8.0, 8.0], ground_at=_flat_ground(150.0),
                          deck_material=DECK, pier_material=PIER, railing_material=RAIL, pier_spacing=1000.0)

    assert np.allclose(a["vertices"], b["vertices"])


# --- Shared deck of a bridge group (lane split on a bridge) -----------------------------------------------------------

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from world_to_beamng.bridges.bridge_mesh import build_bridge_group_mesh

LINK = "asphalt_link"


HOLD = 30.0


def _split_group(z=200.0):
    """A 4-lane trunk (13 m) ending at the node x=0; its branches as lane_splits.py leaves them: straight on in their
    lanes for HOLD meters, then moving apart - main road (6.5 m, centred) and a ramp (3.25 m) on each side."""
    def line(points):
        return [(x, y, z) for x, y in points]

    ramp = lambda sign: line([(0.0, sign * 4.875), (HOLD, sign * 4.875), (40.0, sign * 6.0), (60.0, sign * 12.0)])
    return [
        {"id": 1, "coords": line([(-40.0, 0.0), (-20.0, 0.0), (0.0, 0.0)]), "width": 13.0, "deck_material": DECK},
        {"id": 2, "coords": line([(0.0, 0.0), (HOLD, 0.0), (60.0, 0.0)]), "width": 6.5, "deck_material": DECK},
        {"id": 3, "coords": ramp(1.0), "width": 3.25, "deck_material": LINK},
        {"id": 4, "coords": ramp(-1.0)[::-1], "width": 3.25, "deck_material": LINK},  # digitized towards the node
    ]


STEM = {"path": [(float(x), 0.0, 200.0) for x in range(0, int(HOLD) + 1)], "width": 13.0, "deck_material": DECK}


def _group_mesh(**kwargs):
    return build_bridge_group_mesh(
        _split_group(), ground_at=_flat_ground(150.0), pier_material=PIER, railing_material=RAIL, stem=STEM, **kwargs
    )


def _carriageways(members):
    return unary_union([
        LineString([c[:2] for c in m["coords"]]).buffer(m["width"] / 2.0, cap_style="flat") for m in members
    ])


def _points(mesh, material):
    v = np.asarray(mesh["vertices"])
    return v[sorted({i for face in mesh["faces"][material] for i in face})]


def test_the_trunk_goes_on_as_one_deck_with_railings_only_outside():
    rail = _points(_group_mesh(), RAIL)

    stem_rail = rail[(rail[:, 0] > 1.0) & (rail[:, 0] < HOLD - 1.0)]
    assert len(stem_rail) > 0
    assert np.all(np.abs(np.abs(stem_rail[:, 1]) - 6.7) < 0.1)  # only on the outer curbs of the 13 m deck


def _up_faces_cover(mesh, point, z, materials):
    v, n = np.asarray(mesh["vertices"]), np.asarray(mesh["normals"])
    for material in materials:
        for face in mesh["faces"][material]:
            tri = v[face]
            if n[face[0]][2] > 0.5 and np.all(np.abs(tri[:, 2] - z) < 0.02) and Polygon(tri[:, :2]).buffer(1e-6).contains(point):
                return True
    return False


# In _split_group() the ramps' inner edges (3.25 m beside the axis up to x = 30) move away from the main road's edge:
# the gap is 1.125 m at x = 40 - it reaches 2 x 40 cm at about x = 37.1, where the cut begins.


def test_before_the_gap_reaches_two_curb_widths_the_branches_meet_in_the_middle_of_it():
    mesh = _group_mesh()
    concrete = _points(mesh, PIER)

    # x = 34: gap 0.45 m - no curb yet, the deck is closed across the gap
    assert _up_faces_cover(mesh, Point(34.0, 3.25 + 0.2), 200.0, (DECK, LINK))
    assert _up_faces_cover(mesh, Point(34.0, -3.25 - 0.2), 200.0, (DECK, LINK))
    raised = (concrete[:, 0] > HOLD) & (concrete[:, 0] < 36.5) & (np.abs(concrete[:, 1]) < 4.5) & (concrete[:, 2] > 200.0 + 1e-6)
    assert not np.any(raised)


def test_the_cut_begins_where_each_branch_has_room_for_its_curb():
    mesh = _group_mesh()
    concrete = _points(mesh, PIER)

    # beyond x = 37.1 the main road has its 40 cm curb along the cut (3.25 .. 3.65 m beside the axis)
    curb = concrete[(concrete[:, 0] > 38.0) & (concrete[:, 0] < 55.0) & (np.abs(concrete[:, 2] - 200.2) < 1e-6)]
    assert np.any(np.abs(np.abs(curb[:, 1]) - 3.65) < 0.05)
    # no railing along the cut for now, the outer sides keep theirs
    rail = _points(mesh, RAIL)
    assert not np.any((rail[:, 0] > HOLD) & (np.abs(rail[:, 1]) < 4.5))
    assert np.any((rail[:, 0] > 45.0) & (np.abs(rail[:, 1]) > 8.0))


def test_no_railing_crosses_a_carriageway():
    mesh = _group_mesh()
    inside = _carriageways(_split_group()).buffer(-0.1)

    assert not any(inside.contains(Point(p[0], p[1])) for p in _points(mesh, RAIL))


def test_the_stem_covers_the_whole_trunk_width_up_to_the_cut():
    mesh = _group_mesh()
    v, n = np.asarray(mesh["vertices"]), np.asarray(mesh["normals"])
    top = v[sorted({i for face in mesh["faces"][DECK] for i in face if n[i][2] > 0.5})]  # carriageway surface
    stem_top = top[(top[:, 0] > 0.5) & (top[:, 0] < HOLD - 0.5)]

    assert stem_top[:, 1].max() == pytest.approx(6.5) and stem_top[:, 1].min() == pytest.approx(-6.5)


def test_group_carriageways_keep_their_road_materials():
    mesh = _group_mesh()

    assert len(mesh["faces"][DECK]) > 0 and len(mesh["faces"][LINK]) > 0


def test_the_group_slab_underside_stays_below_a_steep_carriageway():
    members = _split_group()
    members[0]["coords"] = [(x, y, 200.0 - 0.1 * x) for x, y, _ in members[0]["coords"]]  # trunk climbs to x=-40

    mesh = build_bridge_group_mesh(members, ground_at=_flat_ground(150.0), pier_material=PIER, railing_material=RAIL,
                                   deck_thickness=0.6, pier_spacing=1000.0, stem=STEM)

    v, n = np.asarray(mesh["vertices"]), np.asarray(mesh["normals"])
    for material in (DECK, LINK, PIER):
        for face in mesh["faces"][material]:
            if n[face[0]][2] > -0.5:
                continue
            for x, y, z in v[face]:
                deck = 200.0 - 0.1 * x if x < 0.0 else 200.0
                assert z <= deck - 0.6 + 1e-6


def test_build_bridges_merges_a_group_into_one_mesh():
    members = [{**m, "group": 7} for m in _split_group()]
    single = {"id": 9, "coords": [(100.0, 0.0, 200.0), (140.0, 0.0, 200.0)], "width": 6.5, "deck_material": DECK}

    meshes = build_bridges(members + [single], _flat_ground(150.0), PIER, RAIL)

    assert sorted(m["id"] for m in meshes) == ["bridge_9", "bridge_group_7"]


def test_a_branch_piece_lying_completely_in_the_stem_adds_nothing():
    # the main road's first piece ends 14 m behind the node, still inside the stem; digitized towards the node
    members = _split_group()
    main = members[1]
    members[1:2] = [
        {**main, "id": 21, "coords": [(14.0, 0.0, 200.0), (0.0, 0.0, 200.0)]},
        {**main, "id": 22, "coords": [(14.0, 0.0, 200.0), (HOLD, 0.0, 200.0), (60.0, 0.0, 200.0)]},
    ]

    mesh = build_bridge_group_mesh(members, ground_at=_flat_ground(150.0), pier_material=PIER, railing_material=RAIL, stem=STEM)

    rail = _points(mesh, RAIL)
    stem_rail = rail[(rail[:, 0] > 1.0) & (rail[:, 0] < HOLD - 1.0)]
    assert np.all(np.abs(np.abs(stem_rail[:, 1]) - 6.7) < 0.1)  # nothing on the stem between its outer railings


def _covered(mesh, points, z_tol=0.05):
    """Which of the (x, y) `points` lie under an upward face of the deck (at road level)."""
    v, n = np.asarray(mesh["vertices"]), np.asarray(mesh["normals"])
    polygons = []
    for faces in mesh["faces"].values():
        for face in faces:
            if n[face[0]][2] > 0.5 and np.all(np.abs(v[face][:, 2] - 200.0) < z_tol + 0.2):
                polygons.append(Polygon(v[face][:, :2]).buffer(1e-4))
    union = unary_union(polygons)
    return [union.contains(Point(p)) for p in points]


def test_the_trunk_runs_into_the_stem_without_a_gap_where_the_main_axis_turns_off():
    members = _split_group()
    members[0]["coords"] = [(-40.0, 3.5, 200.0), (-20.0, 1.75, 200.0), (0.0, 0.0, 200.0)]  # trunk 5 degrees off the axis

    mesh = build_bridge_group_mesh(members, ground_at=_flat_ground(150.0), pier_material=PIER, railing_material=RAIL,
                                   pier_spacing=1000.0, stem=STEM)

    grid = [(x, y) for x in np.arange(-1.0, 1.01, 0.25) for y in np.arange(-6.2, 6.21, 0.4)]
    assert all(_covered(mesh, grid))


def test_the_branch_decks_start_exactly_on_the_end_edge_of_the_stem():
    mesh = _group_mesh(pier_spacing=1000.0)

    # the ramps turn away right behind the stem end (x = 30): no wedge between their decks and the stem
    grid = [(x, y) for x in np.arange(29.5, 30.51, 0.1) for y in list(np.arange(-6.2, -3.3, 0.3)) + list(np.arange(3.4, 6.21, 0.3))]
    assert all(_covered(mesh, grid))
