"""Tests für world_to_beamng.tunnels.tunnel_mesh: kreisrunde Tunnelröhre (Standardprofil: 240° Bogen über der
Fahrbahn, Boden als Sehne, Radius/Höhe aus der Breite abgeleitet), Verkettung der Tunnel-Stücke."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math

import numpy as np
import pytest

from world_to_beamng.tunnels.tunnel_mesh import (
    build_tunnel_mesh,
    build_tunnels,
    chain_tunnel_pieces,
    resample_tunnel_coords,
    tunnel_crown_height,
    tunnel_radius,
)
from world_to_beamng.tunnels.tunnel_portal import plan_tunnels

WALL, FRAME, FLOOR = "tunnel_concrete", "tunnel_concrete", "asphalt_road_standard"


def _straight_coords(length=100.0, z=500.0, n=11):
    return [(x, 0.0, z) for x in np.linspace(0.0, length, n)]


def test_tunnel_radius_and_crown_height_follow_the_floor_width():
    # Bodensehne = sqrt(3) * R bei 240°/120°-Aufteilung -> R = Breite / sqrt(3); exakt gewählt, damit R = 8
    width = 8.0 * math.sqrt(3.0)
    assert tunnel_radius(width) == pytest.approx(8.0)
    assert tunnel_crown_height(width) == pytest.approx(12.0)  # 1.5 * R


def test_tube_floor_is_flat_and_matches_the_road_width():
    # Nur die Boden-Vertices prüfen (nicht mesh["vertices"] insgesamt): der Kreisbogen ist breiter als die
    # Bodensehne (er wölbt sich bei arc_segments=12 bis auf ~R*cos(10°) > width/2 nach außen) - das ist korrekt
    # und kein Fehler, siehe Design-Spec Abschnitt 5 (240°-Bogen über einer schmaleren Bodensehne).
    mesh = build_tunnel_mesh(_straight_coords(z=500.0), width=8.0, floor_material=FLOOR, wall_material=WALL)
    floor_idx = sorted({i for tri in mesh["faces"][FLOOR] for i in tri})
    floor_v = mesh["vertices"][floor_idx]

    assert floor_v[:, 2].min() == pytest.approx(500.0)  # Boden = Höhenprofil
    assert floor_v[:, 1].min() == pytest.approx(-4.0) and floor_v[:, 1].max() == pytest.approx(4.0)  # Bodenbreite = 8 m


def test_crown_reaches_the_derived_height_above_the_floor():
    width = 8.0
    mesh = build_tunnel_mesh(_straight_coords(z=500.0), width=width, floor_material=FLOOR, wall_material=WALL, arc_segments=12)

    assert mesh["vertices"][:, 2].max() == pytest.approx(500.0 + tunnel_crown_height(width), abs=1e-6)


def test_tube_faces_are_split_by_material():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    assert set(mesh["faces"]) == {FLOOR, WALL}
    assert len(mesh["faces"][FLOOR]) == 2 * 2  # 2 Segmente, Boden = 1 Quad = 2 Dreiecke je Segment
    assert len(mesh["faces"][WALL]) == 2 * 6 * 2  # 6 Bogen-Streifen je Segment, je 2 Dreiecke


def test_floor_normal_points_up_into_the_tube():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    assert (0.0, 0.0, 1.0) in {tuple(np.round(n, 2)) for n in mesh["normals"]}


def test_arc_normals_are_unit_length_and_do_not_point_straight_up():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    wall_normals = np.array([mesh["normals"][face[0]] for face in mesh["faces"][WALL]])
    assert np.allclose(np.linalg.norm(wall_normals, axis=1), 1.0, atol=1e-6)
    assert not np.any(np.all(np.isclose(wall_normals, [0.0, 0.0, 1.0], atol=1e-3), axis=1))


def test_floor_and_arc_share_exact_edge_vertices_even_on_a_curve():
    # arc_ring() verwendet an den Bodenrändern exakt right[i]/left[i] wie das Boden-Mesh - sonst entstünde bei
    # einer Kurve (unterschiedliche Segment-Richtungen) ein Spalt zwischen Boden und Bogen.
    coords = [(0.0, 0.0, 500.0), (20.0, 2.0, 500.0), (40.0, 0.0, 500.0)]
    mesh = build_tunnel_mesh(coords, width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    floor_idx = {i for tri in mesh["faces"][FLOOR] for i in tri}
    wall_idx = {i for tri in mesh["faces"][WALL] for i in tri}
    floor_points = {tuple(np.round(mesh["vertices"][i], 3)) for i in floor_idx}
    wall_points = {tuple(np.round(mesh["vertices"][i], 3)) for i in wall_idx}

    assert len(floor_points & wall_points) >= 4  # beide Bodenrand-Ringe (Anfang+Ende) sind gemeinsame Punkte


def test_resample_tunnel_coords_reduces_point_count_for_a_long_tunnel():
    coords = _straight_coords(length=17000.0, z=500.0, n=17001)  # 1 m Abstand wie aus der normalen Pipeline

    resampled = resample_tunnel_coords(coords, step=10.0)

    assert len(resampled) < len(coords) / 5
    assert resampled[0] == pytest.approx(coords[0])
    assert resampled[-1][0] == pytest.approx(coords[-1][0], abs=1e-6)


def test_resample_tunnel_coords_leaves_short_tunnels_unchanged():
    coords = _straight_coords(length=5.0, z=500.0, n=6)

    assert resample_tunnel_coords(coords, step=10.0) == coords


def test_adjacent_segments_share_the_exact_same_ring_on_a_curve():
    # Die Ringe stehen auf Gehrung an den Centerline-Punkten: beide Segmente an einem Knick verwenden exakt
    # dieselben Ringpunkte - sonst klafft in Kurven ein Spalt zwischen den Röhrensegmenten.
    coords = [(0.0, 0.0, 500.0), (20.0, 0.0, 500.0), (40.0, 8.0, 500.0)]
    mesh = build_tunnel_mesh(coords, width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    wall_idx = sorted({i for tri in mesh["faces"][WALL] for i in tri})
    wall_v = np.round(mesh["vertices"][wall_idx], 6)
    # Vertices am Knick (Projektion auf die Achse nahe x=20) - dort gibt es nur EINEN Ring aus 7 Punkten
    at_joint = {tuple(v) for v in wall_v if abs(v[0] - 20.0) < 2.0 and abs(v[1]) < 6.0}
    assert len(at_joint) == 7


def _piece(piece_id, coords, width=7.0):
    return {"id": piece_id, "coords": coords, "width": width, "floor_material": FLOOR}


def test_chain_tunnel_pieces_joins_split_pieces_into_one_tube_regardless_of_direction():
    a = _piece(1, [(0.0, 0.0, 500.0), (10.0, 0.0, 501.0)])
    b = _piece(2, [(20.0, 0.0, 502.0), (10.0, 0.0, 501.0)])  # umgekehrt digitalisiert
    c = _piece(3, [(20.0, 0.0, 502.0), (30.0, 0.0, 503.0)])

    chains = chain_tunnel_pieces([b, c, a])

    assert len(chains) == 1
    xs = [p[0] for p in chains[0]["coords"]]
    assert xs in ([0.0, 10.0, 20.0, 30.0], [30.0, 20.0, 10.0, 0.0])


def test_chain_tunnel_pieces_ignores_a_different_tunnel_crossing_at_the_joint():
    # Fußweg-Tunnel kreuzt den Straßentunnel in 2D und wurde am selben Punkt geteilt
    a = _piece(1, [(0.0, 0.0, 500.0), (10.0, 0.0, 500.0)])
    b = _piece(2, [(10.0, 0.0, 500.0), (20.0, 0.0, 500.0)])
    path_a = _piece(3, [(10.0, -10.0, 900.0), (10.0, 0.0, 900.0)], width=2.0)
    path_b = _piece(4, [(10.0, 0.0, 900.0), (10.0, 10.0, 900.0)], width=2.0)

    chains = chain_tunnel_pieces([a, path_a, b, path_b])

    assert sorted(len(c["coords"]) for c in chains) == [3, 3]


def test_chain_tunnel_pieces_does_not_join_at_a_three_way_joint_or_across_widths():
    a = _piece(1, [(0.0, 0.0, 500.0), (10.0, 0.0, 500.0)])
    b = _piece(2, [(10.0, 0.0, 500.0), (20.0, 0.0, 500.0)])
    c = _piece(3, [(10.0, 0.0, 500.0), (10.0, 10.0, 500.0)])
    d = _piece(4, [(20.0, 0.0, 500.0), (30.0, 0.0, 500.0)], width=3.0)

    assert len(chain_tunnel_pieces([a, b, c, d])) == 4


def _plans(tunnels):
    return plan_tunnels(tunnels, width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=1.0)


def test_build_tunnels_returns_one_tube_and_two_portal_blocks_per_chain():
    pieces = [_piece(42, _straight_coords(length=100.0, n=6)), _piece(43, [(100.0, 0.0, 500.0), (200.0, 0.0, 500.0)])]

    meshes = build_tunnels(_plans(pieces), WALL, FRAME)

    assert [m["id"] for m in meshes] == ["tunnel_42", "tunnel_42_portal_start", "tunnel_42_portal_end"]
    assert FLOOR in meshes[0]["faces"]
    assert FRAME in meshes[1]["faces"] and FRAME in meshes[2]["faces"]


def test_portal_block_opening_matches_the_first_tube_ring():
    plans = _plans([_piece(1, _straight_coords(length=100.0))])
    meshes = build_tunnels(plans, WALL, FRAME, arc_segments=12)
    tube, block = meshes[0], meshes[1]

    ring = {tuple(np.round(v, 5)) for v in tube["vertices"] if abs(v[0]) < 1e-9}
    block_front = {tuple(np.round(v, 5)) for v in block["vertices"] if abs(v[0]) < 1e-9}
    assert len(ring) == 13  # 12 Bogen-Streifen -> 13 Ringpunkte
    assert ring <= block_front  # jede Ringkante der Röhre ist auch Kante der Portalöffnung


def test_portal_block_spans_the_configured_size_and_leaves_the_opening_free():
    plans = _plans([_piece(1, _straight_coords(length=100.0))])
    portal = plans[0]["portals"][0]
    block = build_tunnels(plans, WALL, FRAME)[1]
    v = block["vertices"]

    assert v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(3.5)  # Portalebene bis Blockende
    assert v[:, 1].max() == pytest.approx(portal["radius"] + 2.0)
    assert v[:, 2].max() == pytest.approx(portal["top_z"]) and v[:, 2].min() == pytest.approx(portal["bottom_z"])
    # Kein Stirnflächen-Dreieck überdeckt die Öffnung: der Schwerpunkt jedes Stirn-Dreiecks liegt außerhalb des
    # Röhrenquerschnitts (oder unter dem Boden).
    radius = portal["radius"]
    for face in block["faces"][FRAME]:
        pts = v[face]
        if not np.allclose(pts[:, 0], 0.0):
            continue
        cy, cz = pts[:, 1].mean(), pts[:, 2].mean() - 500.0
        inside = cz > 0.0 and math.hypot(cy, cz - radius / 2.0) < radius - 1e-6
        assert not inside


def test_build_tunnels_skips_too_short_tunnels():
    tunnels = [_piece(1, [(0.0, 0.0, 500.0)])]

    assert build_tunnels(_plans(tunnels), WALL, FRAME) == []


def test_chain_tunnel_pieces_joins_ends_a_few_millimetres_apart():
    a = _piece(1, [(0.0, 0.0, 500.0), (10.0, 0.0, 500.0)])
    b = _piece(2, [(10.004, 0.003, 500.0), (20.0, 0.0, 500.0)])

    assert len(chain_tunnel_pieces([a, b])) == 1
