"""Transition tunnel <-> gallery (docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md): round portal
as at the tunnel entrance (end ring of the tube or collar) instead of a portal box, plus the faces between the tube arc
and the gallery cross-section."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from world_to_beamng.tunnels.tunnel_mesh import arc_cross_section, build_tunnels, shell_cross_section
from world_to_beamng.tunnels.tunnel_portal import build_portal_block_mesh, plan_tunnels

FLOOR = "asphalt_road_standard"
TUNNEL = [(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)]


def _tunnel(coords=TUNNEL, width=6.5):
    return {"id": 1, "coords": coords, "width": width, "floor_material": FLOOR}


ROOF, WALL = 0.5, 5.0  # roof thickness, mountain wall thickness of the gallery


def _gallery(coords, width=6.5, gallery_id=2, open_side=None):
    return {"id": gallery_id, "coords": coords, "width": width, "floor_material": FLOOR, "osm_tags": {"covered": "yes"},
            "open_side": open_side}


def _plans(tunnels, galleries=None, collar_ratio=0.0):
    return plan_tunnels(
        tunnels, width_margin=1.5, segment_step=10.0, collar_ratio=collar_ratio, flat_depth=1.5, length=3.5,
        galleries=galleries, gallery_height=5.0, gallery_roof_thickness=ROOF, gallery_wall_thickness=WALL,
        transition_tol=0.5, shell_ratio=1.0 / 15.0,
    )


def test_portals_are_open_without_galleries():
    start, end = _plans([_tunnel()])[0]["portals"]
    assert start["kind"] == "open" and end["kind"] == "open"


def test_portal_on_a_gallery_endpoint_becomes_a_gallery_transition():
    start, end = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)])])[0]["portals"]

    assert start["kind"] == "gallery" and end["kind"] == "open"
    assert start["gallery_half_width"] == pytest.approx(3.25)
    assert start["gallery_height"] == pytest.approx(5.0)


def test_transition_portal_has_the_same_round_size_as_the_tunnel_entrance():
    # No portal box anymore: same dimensions as the open portal at the other end
    start, end = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)])], collar_ratio=0.1)[0]["portals"]

    for key in ("half_width", "shell", "collar"):
        assert start[key] == pytest.approx(end[key])
    assert start["top_z"] - start["floor_z"] == pytest.approx(end["top_z"] - end["floor_z"])
    assert start["bottom_z"] - start["floor_z"] == pytest.approx(end["bottom_z"] - end["floor_z"])


def test_gallery_digitised_away_from_the_tunnel_is_also_a_transition():
    start, _ = _plans([_tunnel()], [_gallery([(0.0, 0.0, 500.0), (-50.0, 0.0, 500.0)])])[0]["portals"]
    assert start["kind"] == "gallery"


def test_both_ends_can_be_gallery_transitions():
    galleries = [
        _gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], gallery_id=2),
        _gallery([(100.0, 0.0, 500.0), (150.0, 0.0, 500.0)], gallery_id=3),
    ]
    start, end = _plans([_tunnel()], galleries)[0]["portals"]
    assert start["kind"] == "gallery" and end["kind"] == "gallery"


def test_gallery_touching_the_portal_with_its_middle_is_no_transition():
    start, _ = _plans([_tunnel()], [_gallery([(0.0, -20.0, 500.0), (0.0, 20.0, 500.0)])])[0]["portals"]
    assert start["kind"] == "open"


COVER = 0.2  # thickness of the cover slabs at the transition


def _plane_faces(mesh, material, normal_x, x=0.0):
    """Triangles (as (3, 3) arrays) in the plane x = `x` (portal plane: 0) with normal (normal_x, 0, 0)."""
    v, n = mesh["vertices"], mesh["normals"]
    result = []
    for face in mesh["faces"].get(material, []):
        pts = v[face]
        if np.allclose(pts[:, 0], x) and np.allclose(n[face[0]], [normal_x, 0.0, 0.0]):
            result.append(pts)
    return result


def _area(tri):
    (_, y0, z0), (_, y1, z1), (_, y2, z2) = tri
    return abs((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)) / 2.0


@pytest.mark.parametrize("tunnel_width, gallery_width", [(6.5, 6.5), (6.5, 9.75), (4.0, 4.0)])
def test_transition_closes_exactly_the_gap_between_tube_arc_and_gallery_section(tunnel_width, gallery_width):
    # Toward the tube: tube cross-section minus passage (otherwise one could look out of the tunnel past the
    # gallery into the open).
    # Toward the gallery: gallery cross-section minus tube (wider gallery or gallery higher than the crown).
    plans = _plans([_tunnel(width=tunnel_width)], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], width=gallery_width)])
    portal = plans[0]["portals"][0]
    mesh = build_portal_block_mesh(portal, "concrete", arc_segments=12)
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    arc = Polygon(arc_cross_section(portal["radius"], 12))
    shell = Polygon(shell_cross_section(portal["radius"], 12, portal["shell"]))
    # Without a known valley side: leave out the mountain wall on both sides (the gallery end face covers it)
    body = box(-g - WALL, 0.0, g + WALL, gh + ROOF)

    step_region = arc.difference(body)
    gallery_region = box(-g, 0.0, g, gh).difference(shell)

    # Solid slabs, 20 cm thick: the step extends into the tube, the gallery cover into the gallery - both
    # sides of each slab are visible faces
    assert sum(_area(t) for t in _plane_faces(mesh, "concrete", 1.0, x=COVER)) == pytest.approx(step_region.area, rel=1e-6, abs=1e-9)
    assert sum(_area(t) for t in _plane_faces(mesh, "concrete", -1.0, x=0.0)) == pytest.approx(step_region.area, rel=1e-6, abs=1e-9)
    assert sum(_area(t) for t in _plane_faces(mesh, "concrete", -1.0, x=-COVER)) == pytest.approx(gallery_region.area, rel=1e-6, abs=1e-9)
    assert sum(_area(t) for t in _plane_faces(mesh, "concrete", 1.0, x=0.0)) == pytest.approx(gallery_region.area, rel=1e-6, abs=1e-9)
    assert mesh["vertices"][:, 0].min() >= -COVER - 1e-9 and mesh["vertices"][:, 0].max() <= COVER + 1e-9


def test_transition_without_collar_is_capped_by_the_tube_ring_and_has_no_block():
    plans = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)])])
    portal = plans[0]["portals"][0]
    meshes = build_tunnels(plans, "wall", "concrete", arc_segments=12)

    tube = meshes[0]
    ring = _plane_faces(tube, "concrete", -1.0)
    assert ring, "tube without end ring at the transition"
    transition = next(m for m in meshes if m["id"] == "tunnel_1_portal_start")
    v = transition["vertices"]
    assert np.all(np.abs(v[:, 1]) <= portal["radius"] + 1e-6)  # nothing beyond the tube arc: no box
    assert np.all(np.abs(v[:, 0]) <= COVER + 1e-9)  # only the 20 cm cover slabs at the portal plane


@pytest.mark.parametrize("coords, open_side", [
    ([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], "right"),  # digitized toward the tunnel: right = portal right
    ([(0.0, 0.0, 500.0), (-50.0, 0.0, 500.0)], "left"),  # digitized away from the tunnel: sides mirrored
])
def test_transition_step_leaves_out_the_gallery_roof_and_mountain_wall(coords, open_side):
    # The gallery's own end face closes roof and mountain wall - if the step face lay above it (same plane,
    # same direction), it would flicker. On the valley side (open) the step face extends to the tube arc.
    plans = _plans([_tunnel()], [_gallery(coords, open_side=open_side)])
    portal = plans[0]["portals"][0]  # axis +x: portal right = -y; both galleries are open toward -y
    mesh = build_portal_block_mesh(portal, "concrete", arc_segments=12)
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    arc = Polygon(arc_cross_section(portal["radius"], 12))
    body = box(-g, 0.0, g, gh + ROOF).union(box(-g - WALL, 0.0, -g, gh + ROOF))  # wall left (+y), valley right

    step = sum(_area(t) for t in _plane_faces(mesh, "concrete", 1.0, x=COVER))

    assert portal["gallery_wall_side"] == -1
    assert step == pytest.approx(arc.difference(body).area, rel=1e-6)


def test_transition_collar_is_the_same_rectangle_and_its_front_closes_the_gallery_side():
    # Collar at the transition as at the entrance (but perpendicular): end face toward the gallery = rectangle minus
    # tube arc; an additional gallery side face in the same plane would only exist outside the rectangle
    plans = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], width=9.75)], collar_ratio=0.1)
    start, end = plans[0]["portals"]
    mesh = build_portal_block_mesh(start, "concrete", arc_segments=12)
    radius, crown, wall = start["radius"], start["crown"], start["collar"]
    g, gh = start["gallery_half_width"], start["gallery_height"]
    arc = Polygon(arc_cross_section(radius, 12))
    shell = Polygon(shell_cross_section(radius, 12, start["shell"]))
    frame = box(-radius - wall, -wall, radius + wall, crown + wall)

    body = box(-g - WALL, 0.0, g + WALL, gh + ROOF)
    front = sum(_area(t) for t in _plane_faces(mesh, "concrete", -1.0, x=0.0))
    gallery_cover = sum(_area(t) for t in _plane_faces(mesh, "concrete", -1.0, x=-COVER))

    assert start["tilt"] == 0.0 and start["half_width"] == pytest.approx(end["half_width"])
    # In the portal plane toward the gallery: collar end face (rectangle minus arc) + front side of the step slab
    assert front == pytest.approx(frame.difference(arc).area + arc.difference(body).area, rel=1e-6)
    # gallery cover only outside of collar and tube
    assert gallery_cover == pytest.approx(box(-g, 0.0, g, gh).difference(frame.union(shell)).area, rel=1e-6, abs=1e-9)
