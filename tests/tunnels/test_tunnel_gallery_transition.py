"""Übergang Tunnel <-> Galerie (docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md): rundes Portal
wie am Tunneleingang (Stirnring der Röhre bzw. Kragen) statt Portalquader, dazu die Flächen zwischen Röhrenbogen und
Galerie-Querschnitt."""

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


ROOF, WALL = 0.5, 5.0  # Dachdicke, Bergwanddicke der Galerie


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
    # Kein Portalquader mehr: dieselben Maße wie das offene Portal am anderen Ende
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


COVER = 0.2  # Dicke der Abdeckplatten am Übergang


def _plane_faces(mesh, material, normal_x, x=0.0):
    """(Dreiecke als (3, 3)-Array) in der Ebene x = `x` (Portalebene: 0) mit Normale (normal_x, 0, 0)."""
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
    # Zur Röhre hin: Röhrenquerschnitt minus Durchgang (sonst sähe man aus dem Tunnel neben der Galerie ins Freie).
    # Zur Galerie hin: Galerie-Querschnitt minus Röhre (breitere Galerie bzw. Galerie höher als die Krone).
    plans = _plans([_tunnel(width=tunnel_width)], [_gallery([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], width=gallery_width)])
    portal = plans[0]["portals"][0]
    mesh = build_portal_block_mesh(portal, "concrete", arc_segments=12)
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    arc = Polygon(arc_cross_section(portal["radius"], 12))
    shell = Polygon(shell_cross_section(portal["radius"], 12, portal["shell"]))
    # Ohne bekannte Talseite: Bergwand auf beiden Seiten aussparen (die Galerie-Stirnfläche deckt sie ab)
    body = box(-g - WALL, 0.0, g + WALL, gh + ROOF)

    step_region = arc.difference(body)
    gallery_region = box(-g, 0.0, g, gh).difference(shell)

    # Massive Platten, 20 cm dick: die Stufe reicht in die Röhre, die Galerie-Abdeckung in die Galerie - beide
    # Seiten jeder Platte sind sichtbare Flächen
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
    assert ring, "Röhre ohne Stirnring am Übergang"
    transition = next(m for m in meshes if m["id"] == "tunnel_1_portal_start")
    v = transition["vertices"]
    assert np.all(np.abs(v[:, 1]) <= portal["radius"] + 1e-6)  # nichts über den Röhrenbogen hinaus: kein Quader
    assert np.all(np.abs(v[:, 0]) <= COVER + 1e-9)  # nur die 20-cm-Abdeckplatten an der Portalebene


@pytest.mark.parametrize("coords, open_side", [
    ([(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], "right"),  # zum Tunnel hin digitalisiert: rechts = Portal-rechts
    ([(0.0, 0.0, 500.0), (-50.0, 0.0, 500.0)], "left"),  # vom Tunnel weg digitalisiert: Seiten gespiegelt
])
def test_transition_step_leaves_out_the_gallery_roof_and_mountain_wall(coords, open_side):
    # Dach und Bergwand der Galerie schließt deren eigene Stirnfläche - läge die Stufenfläche darüber (gleiche Ebene,
    # gleiche Richtung), flackerte es. Talseitig (offen) schließt die Stufenfläche bis zum Röhrenbogen.
    plans = _plans([_tunnel()], [_gallery(coords, open_side=open_side)])
    portal = plans[0]["portals"][0]  # Achse +x: Portal-rechts = -y; beide Galerien sind nach -y offen
    mesh = build_portal_block_mesh(portal, "concrete", arc_segments=12)
    g, gh = portal["gallery_half_width"], portal["gallery_height"]
    arc = Polygon(arc_cross_section(portal["radius"], 12))
    body = box(-g, 0.0, g, gh + ROOF).union(box(-g - WALL, 0.0, -g, gh + ROOF))  # Wand links (+y), Tal rechts

    step = sum(_area(t) for t in _plane_faces(mesh, "concrete", 1.0, x=COVER))

    assert portal["gallery_wall_side"] == -1
    assert step == pytest.approx(arc.difference(body).area, rel=1e-6)


def test_transition_collar_is_the_same_rectangle_and_its_front_closes_the_gallery_side():
    # Kragen am Übergang wie am Eingang (aber senkrecht): Stirnseite zur Galerie = Rechteck minus Röhrenbogen; eine
    # zusätzliche Galerie-Seitenfläche in derselben Ebene gäbe es nur außerhalb des Rechtecks
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
    # In der Portalebene zur Galerie: Kragen-Stirnseite (Rechteck minus Bogen) + Vorderseite der Stufenplatte
    assert front == pytest.approx(frame.difference(arc).area + arc.difference(body).area, rel=1e-6)
    # Galerie-Abdeckung nur außerhalb von Kragen und Röhre
    assert gallery_cover == pytest.approx(box(-g, 0.0, g, gh).difference(frame.union(shell)).area, rel=1e-6, abs=1e-9)
