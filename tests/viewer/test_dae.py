"""Tests for tools/level_viewer/dae.py: COLLADA parsing with separate position/UV indices and material groups."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from tools.level_viewer.dae import load_dae_tile, read_dae

DAE = """<?xml version="1.0" encoding="UTF-8"?>
<COLLADA version="1.4.1" xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <library_geometries>
    <geometry id="g_geometry" name="quad">
      <mesh>
        <source id="g_vertices">
          <float_array id="g_vertices_array" count="12">0 0 0 1 0 0 1 1 0 0 1 0</float_array>
          <technique_common><accessor source="#g_vertices_array" count="4" stride="3"/></technique_common>
        </source>
        <source id="g_uvs">
          <float_array id="g_uvs_array" count="6">0 0 1 0 0.5 1</float_array>
          <technique_common><accessor source="#g_uvs_array" count="3" stride="2"/></technique_common>
        </source>
        <vertices id="g_vertices_input"><input semantic="POSITION" source="#g_vertices"/></vertices>
        <triangles material="concrete" count="1">
          <input semantic="VERTEX" source="#g_vertices_input" offset="0"/>
          <input semantic="TEXCOORD" source="#g_uvs" offset="1" set="0"/>
          <p>0 0 1 1 2 2</p>
        </triangles>
        <triangles material="railing" count="1">
          <input semantic="VERTEX" source="#g_vertices_input" offset="0"/>
          <input semantic="TEXCOORD" source="#g_uvs" offset="1" set="0"/>
          <p>0 2 2 0 3 1</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
</COLLADA>
"""


def test_each_position_uv_pair_becomes_one_vertex(tmp_path):
    path = tmp_path / "quad.dae"
    path.write_text(DAE, encoding="utf-8")

    (mesh,) = read_dae(path)

    assert mesh.name == "quad"
    assert set(mesh.triangles) == {"concrete", "railing"}
    # corner (0, 0) and (0, 2) share position 0 but differ in UV -> 2 vertices; (2, 2) and (2, 0) likewise
    assert len(mesh.positions) == 6
    for material, tris in mesh.triangles.items():
        assert tris.shape == (1, 3)
    concrete = mesh.triangles["concrete"][0]
    assert mesh.positions[concrete].tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    assert mesh.uvs[concrete].tolist() == [[0, 0], [1, 0], [0.5, 1]]


def test_load_dae_tile_compat_view(tmp_path):
    path = tmp_path / "quad.dae"
    path.write_text(DAE, encoding="utf-8")

    data = load_dae_tile(path)

    assert data["vertices"].shape == (6, 3)
    assert len(data["faces"]) == 2 and data["materials"] == ["concrete", "railing"]
    assert data["materials_per_face"] == {"concrete": [0], "railing": [1]}
    assert np.asarray(data["faces"]).max() < 6
