"""Fill mesh of the junction corners: fan triangles from the corner point, lifted, world UVs, one mesh per node."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.junctions.fill_mesh import build_junction_meshes


def _corner(node=(0.0, 0.0), surface="asphalt_road_standard"):
    angles = np.linspace(-np.pi / 2, -np.pi, 7)
    arc = np.column_stack([8.5 + 6 * np.cos(angles), 9.0 + 6 * np.sin(angles), np.full(7, 100.0)])
    return {"node": np.array([*node, 100.0]), "corner_point": np.array([2.5, 3.0, 100.0]), "arc": arc, "surface": surface}


def test_one_mesh_per_node_with_fan_triangles_lifted_and_facing_up():
    meshes = build_junction_meshes([_corner(), _corner(), _corner(node=(100.0, 0.0))], lift=0.02, tile_m=5.0)
    assert len(meshes) == 2
    first = meshes[0]
    faces = first["faces"]["asphalt_road_standard_junction"]
    assert len(faces) == 2 * 6  # two corners, 6 fan triangles each
    assert np.allclose(first["vertices"][:, 2], 100.02)
    assert np.all(first["normals"][:, 2] > 0.99)


def test_uvs_are_world_aligned():
    mesh = build_junction_meshes([_corner()], lift=0.02, tile_m=5.0)[0]
    assert np.allclose(mesh["uvs"], mesh["vertices"][:, :2] / 5.0)


def test_no_corners_no_meshes():
    assert build_junction_meshes([], lift=0.02, tile_m=5.0) == []
