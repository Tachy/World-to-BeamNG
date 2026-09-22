"""Tests für world_to_beamng.tunnels.tunnel_mesh: kreisrunde Tunnelröhre (Standardprofil: 240° Bogen über der
Fahrbahn, Boden als Sehne, Radius/Höhe aus der Breite abgeleitet) + hangneigungs-angepasste Portal-Rahmen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math

import numpy as np
import pytest

from world_to_beamng.tunnels.tunnel_mesh import (
    build_tunnel,
    build_tunnel_mesh,
    build_tunnels,
    portal_frame_corners,
    resample_tunnel_coords,
    tunnel_crown_height,
    tunnel_radius,
)

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


def test_portal_frame_corners_are_a_flat_rectangle_on_flat_ground():
    corners = np.array(portal_frame_corners((0.0, 0.0), (1.0, 0.0), width=8.0, height=12.0, margin=0.6, floor_z=500.0, slope_along_axis=0.0))

    assert corners[:, 0].max() == pytest.approx(0.0)  # keine Achsverschiebung bei Neigung 0
    assert corners[:, 1].min() == pytest.approx(-4.6) and corners[:, 1].max() == pytest.approx(4.6)
    assert corners[:, 2].min() == pytest.approx(500.0) and corners[:, 2].max() == pytest.approx(512.6)


def test_portal_frame_corners_shift_the_top_edge_with_slope():
    corners = np.array(portal_frame_corners((0.0, 0.0), (1.0, 0.0), width=8.0, height=12.0, margin=0.0, floor_z=500.0, slope_along_axis=0.2))

    assert corners[0, 0] == pytest.approx(0.0) and corners[1, 0] == pytest.approx(0.0)  # untere Ecken unverschoben
    assert corners[2, 0] == pytest.approx(2.4) and corners[3, 0] == pytest.approx(2.4)  # obere Ecken: 0.2 * 12m = 2.4m verschoben


def test_build_tunnel_returns_the_tube_plus_two_portal_frames_with_derived_height():
    tunnel = {"id": 42, "coords": _straight_coords(length=200.0, z=500.0), "width": 7.0, "floor_material": FLOOR}
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 500.0)

    meshes = build_tunnel(tunnel, ground_at, WALL, FRAME, width_margin=1.5, arc_segments=12, segment_step=10.0, portal_slope_sample_dist=5.0, frame_margin=0.6)

    assert [m["id"] for m in meshes] == ["tunnel_42", "tunnel_42_portal_start", "tunnel_42_portal_end"]
    assert FLOOR in meshes[0]["faces"]
    assert FRAME in meshes[1]["faces"] and FRAME in meshes[2]["faces"]
    expected_crown = tunnel_crown_height(7.0 + 1.5)
    # Rahmen-Oberkante liegt bei floor_z + Kronenhöhe + frame_margin (siehe portal_frame_corners(): "top" z ist
    # floor_z + height + margin), nicht bei floor_z + Kronenhöhe allein.
    assert max(v[2] for v in meshes[1]["vertices"]) == pytest.approx(500.0 + expected_crown + 0.6)


def test_build_tunnels_skips_too_short_tunnels():
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 500.0)
    tunnels = [{"id": 1, "coords": [(0.0, 0.0, 500.0)], "width": 7.0, "floor_material": FLOOR}]

    assert build_tunnels(tunnels, ground_at, WALL, FRAME) == []
