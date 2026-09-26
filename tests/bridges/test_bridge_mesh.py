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


def test_piers_reach_down_to_the_natural_ground_below_a_deep_span():
    mesh = build_bridge_mesh(_coords(length=60.0, z=200.0), width=8.0, ground_at=_flat_ground(150.0), deck_material=DECK, pier_material=PIER, railing_material=RAIL, pier_spacing=25.0)

    assert len(mesh["faces"][PIER]) > 0
    assert _zs(mesh, PIER).min() == pytest.approx(150.0)  # (at least) one pier reaches down to the natural ground


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
