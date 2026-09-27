"""Tests for world_to_beamng.sidewalks.sidewalk_mesh: kerb face, kerb top, sidewalk top, outer skirt, end caps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.sidewalks.sidewalk_mesh import build_sidewalk_mesh

KERB, SURFACE = "bridge_concrete", "asphalt_road_standard_structure"


def _mesh(points=((0.0, 0.0, 10.0), (4.0, 0.0, 10.0))):
    return build_sidewalk_mesh(np.array(points), 0.15, 1.0, 0.12, 0.3, 1.0, 2.0, KERB, SURFACE)


def _face_vertices(mesh, material):
    return mesh["vertices"][np.array(mesh["faces"][material]).reshape(-1)]


def test_sidewalk_top_is_flat_at_kerb_height_behind_the_kerb():
    top = _face_vertices(_mesh(), SURFACE)
    assert np.allclose(top[:, 2], 10.12)
    assert top[:, 1].min() == pytest.approx(0.15) and top[:, 1].max() == pytest.approx(1.15)


def test_mesh_spans_from_the_carriageway_edge_to_the_outer_edge_and_below_the_road():
    mesh = _mesh()
    assert mesh["vertices"][:, 1].min() == pytest.approx(0.0) and mesh["vertices"][:, 1].max() == pytest.approx(1.15)
    assert mesh["vertices"][:, 2].min() == pytest.approx(9.7) and mesh["vertices"][:, 2].max() == pytest.approx(10.12)


def test_kerb_face_points_to_the_carriageway_and_all_normals_are_unit():
    mesh = _mesh()
    kerb_face = np.isclose(mesh["vertices"][:, 1], 0.0) & np.isclose(mesh["normals"][:, 2], 0.0) & np.isclose(np.abs(mesh["normals"][:, 1]), 1.0)
    assert kerb_face.any() and np.allclose(mesh["normals"][kerb_face], [0.0, -1.0, 0.0])
    assert np.allclose(np.linalg.norm(mesh["normals"], axis=1), 1.0)


def test_long_segments_are_split_so_the_band_follows_the_road_height():
    mesh = _mesh(((0.0, 0.0, 10.0), (4.0, 0.0, 11.0)))
    top = _face_vertices(mesh, SURFACE)
    assert len(mesh["faces"][SURFACE]) == 8  # 4 segments of 1 m, 2 triangles each
    assert sorted(set(np.round(top[:, 0], 6))) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert np.allclose(top[:, 2], 10.12 + top[:, 0] / 4.0)


def test_faces_wind_along_their_normals():
    mesh = _mesh()
    for faces in mesh["faces"].values():
        for a, b, c in faces:
            v = mesh["vertices"]
            cross = np.cross(v[b] - v[a], v[c] - v[a])
            assert np.dot(cross, mesh["normals"][a]) > 0
