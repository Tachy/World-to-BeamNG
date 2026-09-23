"""Tests für world_to_beamng.walls.mesh_parts.MeshBuilder.quad(): Umlaufsinn folgt der Normalen.

quad() nutzt für Kreuz-/Skalarprodukt bewusst reines Python statt numpy (siehe Kommentar in
mesh_parts.py) - diese Tests sichern, dass das Ergebnis dasselbe bleibt wie mit np.cross()/np.dot().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.walls.mesh_parts import MeshBuilder


def _triangle_normal(vertices, face):
    a, b, c = (np.array(vertices[i], dtype=float) for i in face)
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n)


CORNERS_CCW = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
UVS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_quad_winds_both_triangles_to_face_the_given_normal():
    builder = MeshBuilder()

    builder.quad(CORNERS_CCW, UVS, normal=(0.0, 0.0, 1.0))

    assert len(builder.faces) == 2
    for face in builder.faces:
        assert _triangle_normal(builder.vertices, face) == pytest.approx([0.0, 0.0, 1.0])


def test_quad_flips_winding_when_corners_are_given_in_the_opposite_order():
    builder = MeshBuilder()
    corners_cw = list(reversed(CORNERS_CCW))

    builder.quad(corners_cw, UVS, normal=(0.0, 0.0, 1.0))

    assert len(builder.faces) == 2
    for face in builder.faces:
        assert _triangle_normal(builder.vertices, face) == pytest.approx([0.0, 0.0, 1.0])


def test_quad_works_for_an_arbitrary_non_axis_aligned_normal():
    builder = MeshBuilder()
    # Vierseitige Fläche in der xz-Ebene, Normale zeigt entlang -y.
    corners = [(0.0, 5.0, 0.0), (1.0, 5.0, 0.0), (1.0, 5.0, 1.0), (0.0, 5.0, 1.0)]

    builder.quad(corners, UVS, normal=(0.0, -1.0, 0.0))

    for face in builder.faces:
        assert _triangle_normal(builder.vertices, face) == pytest.approx([0.0, -1.0, 0.0])


def test_quad_appends_vertices_uvs_and_normals_for_all_four_corners():
    builder = MeshBuilder()

    builder.quad(CORNERS_CCW, UVS, normal=(0.0, 0.0, 1.0))

    assert builder.vertices == [list(c) for c in CORNERS_CCW]
    assert builder.uvs == [list(u) for u in UVS]
    assert builder.normals == [[0.0, 0.0, 1.0]] * 4
