"""Tests for world_to_beamng.walls.mesh_parts.add_box_column: rectangular column (bridge pier, gallery columns)."""

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


def test_column_footprint_is_rotated_with_the_given_direction():
    axis_aligned, rotated = MeshBuilder(), MeshBuilder()

    add_box_column(axis_aligned, cx=0.0, cy=0.0, bottom_z=0.0, top_z=1.0, size=2.0, tile_m=1.0, direction=(1.0, 0.0))
    # 30° instead of axis-parallel - the profile edges must rotate along, not stay on world X/Y.
    add_box_column(rotated, cx=0.0, cy=0.0, bottom_z=0.0, top_z=1.0, size=2.0, tile_m=1.0, direction=(np.cos(np.radians(30)), np.sin(np.radians(30))))

    xy_axis = {tuple(np.round(p, 3)) for p in np.array(axis_aligned.vertices)[:, :2]}
    xy_rotated = {tuple(np.round(p, 3)) for p in np.array(rotated.vertices)[:, :2]}
    assert xy_axis != xy_rotated  # different corners than with axis-parallel orientation

    # Unchanged cross-section (half the diagonal as radius), only rotated.
    radii = np.linalg.norm(np.array(rotated.vertices)[:, :2], axis=1)
    assert radii == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_column_direction_does_not_need_to_be_normalized():
    a, b = MeshBuilder(), MeshBuilder()

    add_box_column(a, cx=5.0, cy=5.0, bottom_z=0.0, top_z=1.0, size=1.0, tile_m=1.0, direction=(1.0, 0.0))
    add_box_column(b, cx=5.0, cy=5.0, bottom_z=0.0, top_z=1.0, size=1.0, tile_m=1.0, direction=(3.0, 0.0))

    assert np.array(a.vertices) == pytest.approx(np.array(b.vertices))


def test_column_has_four_outward_facing_side_quads():
    builder = MeshBuilder()

    add_box_column(builder, cx=0.0, cy=0.0, bottom_z=0.0, top_z=1.0, size=1.0, tile_m=1.0)

    assert len(builder.faces) == 4 * 2  # 4 sides, 2 triangles each
    directions = {tuple(np.round(n, 2)) for n in builder.normals}
    assert directions == {(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)}
    tris = np.array([[builder.vertices[i] for i in face] for face in builder.faces])
    for face, tri in zip(builder.faces, tris):
        geometric = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        assert np.dot(geometric, builder.normals[face[0]]) > 0
