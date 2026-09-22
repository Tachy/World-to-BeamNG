"""Tests für world_to_beamng.walls.mesh_parts.add_box_column: rechteckige Stütze (Brücken-Pfeiler, Galerie-Stützen)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.walls.mesh_parts import MeshBuilder, add_box_column


def test_column_spans_the_given_height_and_footprint():
    builder = MeshBuilder()

    add_box_column(builder, cx=10.0, cy=20.0, bottom_z=100.0, top_z=105.0, size=1.5, tile_m=1.0)

    v = np.array(builder.vertices)
    assert v[:, 2].min() == pytest.approx(100.0) and v[:, 2].max() == pytest.approx(105.0)
    assert v[:, 0].min() == pytest.approx(10.0 - 0.75) and v[:, 0].max() == pytest.approx(10.0 + 0.75)
    assert v[:, 1].min() == pytest.approx(20.0 - 0.75) and v[:, 1].max() == pytest.approx(20.0 + 0.75)


def test_column_has_four_outward_facing_side_quads():
    builder = MeshBuilder()

    add_box_column(builder, cx=0.0, cy=0.0, bottom_z=0.0, top_z=1.0, size=1.0, tile_m=1.0)

    assert len(builder.faces) == 4 * 2  # 4 Seiten, je 2 Dreiecke
    directions = {tuple(np.round(n, 2)) for n in builder.normals}
    assert directions == {(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)}
    tris = np.array([[builder.vertices[i] for i in face] for face in builder.faces])
    for face, tri in zip(builder.faces, tris):
        geometric = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        assert np.dot(geometric, builder.normals[face[0]]) > 0
