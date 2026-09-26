"""Tests for world_to_beamng.geometry.marking_mesh: road marking lines as thin mesh strips on structure floors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.marking_mesh import build_marking_meshes

TEXTURE_LENGTHS = {"line_edge_white": 10.0, "line_divider_dashed": 24.0}


def _line(name, material, y=0.0, z=100.0, length=48.0, width=0.15):
    return {"name": name, "material": material, "nodes": [[0.0, y, z, width], [length / 2, y, z, width], [length, y, z, width]]}


def test_strip_lies_just_above_the_floor_with_the_line_width():
    meshes = build_marking_meshes([_line("marking_1_0_0", "line_edge_white", y=2.0)], TEXTURE_LENGTHS, lift=0.01)
    v = meshes[0]["vertices"]

    assert np.allclose(v[:, 2], 100.01)
    assert v[:, 1].min() == pytest.approx(2.0 - 0.075) and v[:, 1].max() == pytest.approx(2.0 + 0.075)
    assert v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(48.0)


def test_strip_faces_up_and_uses_the_marking_material():
    mesh = build_marking_meshes([_line("marking_1_1_0", "line_divider_dashed")], TEXTURE_LENGTHS, lift=0.01)[0]

    assert list(mesh["faces"]) == ["line_divider_dashed"]
    assert np.allclose(mesh["normals"], [0.0, 0.0, 1.0])


def test_uvs_follow_the_decal_road_layout():
    # Like a DecalRoad: u across the line 0..1, v along the line in texture repeats (length / textureLength), so the
    # dash pattern matches the marking decals on the approach
    mesh = build_marking_meshes([_line("marking_1_1_0", "line_divider_dashed", length=48.0)], TEXTURE_LENGTHS, lift=0.01)[0]
    uv = mesh["uvs"]

    assert set(np.round(uv[:, 0], 6)) == {0.0, 1.0}
    assert uv[:, 1].min() == pytest.approx(0.0) and uv[:, 1].max() == pytest.approx(48.0 / 24.0)


def test_one_mesh_per_line_named_after_the_line():
    lines = [_line("marking_1_0_0", "line_edge_white"), _line("marking_1_1_0", "line_divider_dashed", y=3.0)]

    assert [m["id"] for m in build_marking_meshes(lines, TEXTURE_LENGTHS, lift=0.01)] == ["marking_1_0_0", "marking_1_1_0"]
