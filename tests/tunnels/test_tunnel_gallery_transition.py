"""Übergang Tunnel <-> Galerie (docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.tunnel_portal import plan_tunnels

FLOOR = "asphalt_road_standard"
TUNNEL = [(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)]


def _tunnel(coords=TUNNEL, width=6.5):
    return {"id": 1, "coords": coords, "width": width, "floor_material": FLOOR}


def _gallery(coords, width=6.5, gallery_id=2):
    return {"id": gallery_id, "coords": coords, "width": width, "floor_material": FLOOR, "osm_tags": {"covered": "yes"}}


def _plans(tunnels, galleries=None):
    return plan_tunnels(
        tunnels, width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=1.0,
        galleries=galleries, gallery_height=5.0, gallery_roof_thickness=0.5, gallery_floor_thickness=5.0,
        gallery_wall_thickness=5.0, transition_tol=0.5,
    )


def test_portals_are_open_without_galleries():
    start, end = _plans([_tunnel()])[0]["portals"]
    assert start["kind"] == "open" and end["kind"] == "open"


def test_portal_on_a_gallery_endpoint_becomes_a_gallery_transition():
    start, end = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)])])[0]["portals"]

    assert start["kind"] == "gallery" and end["kind"] == "open"
    assert start["gallery_half_width"] == pytest.approx(3.25)
    assert start["gallery_height"] == pytest.approx(5.0)
    # Block deckt den ganzen Galerie-Querschnitt ab: Bergwand (3,25 + 5 m), Dach (5,5 m), Bodenquader (5 m)
    assert start["half_width"] == pytest.approx(max(start["radius"] + 2.0, 3.25 + 5.0))
    assert start["top_z"] >= 500.0 + 5.0 + 0.5 + 0.2 - 1e-9
    assert start["bottom_z"] == pytest.approx(500.0 - 5.0)


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


def test_wide_gallery_widens_the_block():
    start, _ = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], width=9.75)])[0]["portals"]
    assert start["gallery_half_width"] == pytest.approx(4.875)
    assert start["half_width"] == pytest.approx(4.875 + 5.0)


from shapely.geometry import Polygon, box

from world_to_beamng.tunnels.tunnel_mesh import arc_cross_section
from world_to_beamng.tunnels.tunnel_portal import build_portal_block_mesh


def _transition_block():
    plans = _plans([_tunnel()], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)])])
    portal = plans[0]["portals"][0]  # Portalebene x = 0, Achse +x ins Tunnelinnere
    return portal, build_portal_block_mesh(portal, "concrete", arc_segments=12)


def _plane_faces(mesh, normal_x):
    """(Dreiecke als (3, 3)-Array) in der Portalebene x = 0 mit Normale (normal_x, 0, 0)."""
    v, n = mesh["vertices"], mesh["normals"]
    result = []
    for face in mesh["faces"]["concrete"]:
        pts = v[face]
        if np.allclose(pts[:, 0], 0.0) and np.allclose(n[face[0]], [normal_x, 0.0, 0.0]):
            result.append(pts)
    return result


def _area(tri):
    (_, y0, z0), (_, y1, z1), (_, y2, z2) = tri
    return abs((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)) / 2.0


def test_transition_front_wall_leaves_exactly_the_gallery_opening_free():
    portal, mesh = _transition_block()
    front = _plane_faces(mesh, -1.0)  # zur Galerie
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    hw = portal["half_width"]
    top, bottom = portal["top_z"] - 500.0, portal["bottom_z"] - 500.0

    assert sum(_area(t) for t in front) == pytest.approx(2 * hw * (top - bottom) - 2 * g * gh)
    for tri in front:  # kein Stirn-Dreieck liegt in der Öffnung
        cy, cz = tri[:, 1].mean(), tri[:, 2].mean() - 500.0
        assert not (abs(cy) < g and 0.0 < cz < gh)


def test_transition_step_faces_into_the_tunnel_between_arc_and_opening():
    portal, mesh = _transition_block()
    step = _plane_faces(mesh, 1.0)  # ins Tunnelinnere
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    ring = Polygon(arc_cross_section(portal["radius"], 12))

    assert step, "keine Stufenfläche"
    assert sum(_area(t) for t in step) == pytest.approx(ring.area - 2 * g * gh, rel=1e-6)


def test_open_portal_keeps_the_round_opening():
    plans = _plans([_tunnel()])
    mesh = build_portal_block_mesh(plans[0]["portals"][0], "concrete", arc_segments=12)
    assert _plane_faces(mesh, 1.0) == []  # keine Stufenfläche beim offenen Portal


@pytest.mark.parametrize("tunnel_width, gallery_width", [(6.5, 9.75), (4.0, 4.0)])
def test_transition_wall_has_no_hole_where_the_opening_exceeds_the_tube(tunnel_width, gallery_width):
    # Galerie breiter als der Tunnel bzw. schmale Straße (Krone 4,76 m < Galeriehöhe 5 m): der Teil der Rechteck-
    # Öffnung außerhalb des Röhrenbogens muss von der Stirnwand geschlossen werden, sonst sieht man in den hohlen Block.
    plans = _plans([_tunnel(width=tunnel_width)], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], width=gallery_width)])
    portal = plans[0]["portals"][0]
    mesh = build_portal_block_mesh(portal, "concrete", arc_segments=12)
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    hw, top, bottom = portal["half_width"], portal["top_z"] - 500.0, portal["bottom_z"] - 500.0
    arc = Polygon(arc_cross_section(portal["radius"], 12))
    passage = arc.intersection(box(-g, 0.0, g, gh))  # tatsächlich befahrbarer Durchgang

    front = sum(_area(t) for t in _plane_faces(mesh, -1.0))
    step = sum(_area(t) for t in _plane_faces(mesh, 1.0))

    assert front == pytest.approx(2 * hw * (top - bottom) - passage.area, rel=1e-6)
    assert step == pytest.approx(arc.area - passage.area, rel=1e-6)
