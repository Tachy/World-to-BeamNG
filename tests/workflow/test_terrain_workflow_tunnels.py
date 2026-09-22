"""Tests für TerrainWorkflow._build_tunnels() und export_tunnels(): Tunnel (Röhre+Portale) und Galerien
(Dach+Stützen) als eine gemeinsame DAE mit einem TSStatic."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.textures import registry
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _Items:
    def __init__(self):
        self.objects = {}

    def add_item(self, name, item_class="TSStatic", position=(0, 0, 0), **fields):
        self.objects[name] = {"class": item_class, "position": list(position), **fields}


class _Materials:
    def __init__(self):
        self.added = {}

    def get_templates(self):
        return {"buildings": {"wall": {"material_hints": {"groundType": "concrete", "materialTag0": "beamng", "materialTag1": "Building"}}}}

    def add_building_material(self, name, **fields):
        self.added[name] = fields


class _Dae:
    def __init__(self):
        self.calls = []

    def export_multi_mesh(self, output_path, meshes, with_uv=False, material_textures=None):
        self.calls.append((Path(output_path), meshes, with_uv))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<COLLADA/>", encoding="utf-8")
        return output_path


def _stub():
    return SimpleNamespace(items=_Items(), materials=_Materials(), dae=_Dae())


CONCRETE = {
    "baseColorMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_b.color.dds",
    "normalMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_nm.normal.dds",
    "roughnessMap": "levels/world_to_beamng/art/shapes/textures/tunnel_concrete_r.data.dds",
}


@pytest.fixture
def shapes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    monkeypatch.setattr(registry, "prepared_textures", lambda: {config.CONCRETE_TEXTURE_NAME: CONCRETE})
    return tmp_path / "shapes"


def _mesh(name):
    return {
        "id": name,
        "vertices": np.zeros((8, 3)),
        "uvs": np.zeros((8, 2)),
        "normals": np.tile([0.0, 0.0, 1.0], (8, 1)),
        "faces": {"asphalt_road_standard": [[0, 1, 2]], config.TUNNEL_MATERIAL_NAME: [[4, 5, 6]]},
    }


def _road(road_id, structure_type, tags):
    return {
        "road_id": road_id,
        "trimmed_centerline": np.array([[0.0, 0.0, 500.0], [200.0, 0.0, 500.0]]),
        "osm_tags": tags,
        "structure_type": structure_type,
    }


def test_export_tunnels_writes_one_dae_one_item_and_registers_floor_and_concrete_materials(shapes_dir):
    stub = _stub()
    roads = [_road(1, "tunnel", {"highway": "trunk", "tunnel": "yes"}), _road(2, "gallery", {"highway": "primary", "tunnel": "avalanche_protector"})]

    count = TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1"), _mesh("tunnel_1_portal_start"), _mesh("tunnel_1_portal_end"), _mesh("gallery_2")], "structure_road_polygons": roads})

    assert count == 4
    dae_path, meshes, with_uv = stub.dae.calls[0]
    assert dae_path == shapes_dir / "tunnels" / "tunnels.dae" and with_uv is True and len(meshes) == 4
    item = stub.items.objects["tunnels"]
    assert item["class"] == "TSStatic" and item["shape_name"] == "levels/world_to_beamng/art/shapes/tunnels/tunnels.dae"
    assert item["collisionType"] == "Visible Mesh Final"
    assert config.TUNNEL_MATERIAL_NAME in stub.materials.added
    assert "asphalt_road_standard" in stub.materials.added
    assert stub.materials.added["asphalt_road_standard"]["groundType"] == "ASPHALT"


def test_export_tunnels_takes_the_concrete_texture_from_the_registry_check_and_never_falls_back(shapes_dir, monkeypatch):
    def aborted():
        raise registry.MissingTexturesError("Beton-Textur fehlt")

    monkeypatch.setattr(registry, "prepared_textures", aborted)
    stub = _stub()

    with pytest.raises(registry.MissingTexturesError):
        TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1")], "structure_road_polygons": [_road(1, "tunnel", {"tunnel": "yes"})]})
    assert not stub.materials.added and not stub.dae.calls


def test_nothing_is_exported_and_stale_files_are_removed_without_tunnels(shapes_dir, monkeypatch):
    stale = shapes_dir / "tunnels"
    stale.mkdir(parents=True)
    (stale / "tunnels.dae").write_text("alt", encoding="utf-8")
    (stale / "tunnels.cdae").write_text("alt", encoding="utf-8")
    stub = _stub()

    assert TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [], "structure_road_polygons": []}) == 0
    assert not (stale / "tunnels.dae").exists() and not (stale / "tunnels.cdae").exists()

    monkeypatch.setattr(config, "TUNNELS_ENABLED", False)
    assert TerrainWorkflow.export_tunnels(stub, {"tunnel_meshes": [_mesh("tunnel_1")], "structure_road_polygons": [_road(1, "tunnel", {"tunnel": "yes"})]}) == 0
    assert not stub.dae.calls


def test_build_tunnels_creates_tube_plus_portals_for_a_tunnel_and_one_mesh_per_gallery():
    heights = np.full((50, 50), 495.0)
    tunnel_road = _road(1, "tunnel", {"highway": "trunk", "tunnel": "yes"})
    gallery_road = _road(2, "gallery", {"highway": "primary", "tunnel": "avalanche_protector"})

    meshes = TerrainWorkflow._build_tunnels(SimpleNamespace(), [tunnel_road, gallery_road], heights, 0.0, 0.0)

    ids = [m["id"] for m in meshes]
    assert "tunnel_1" in ids and "tunnel_1_portal_start" in ids and "tunnel_1_portal_end" in ids and "gallery_2" in ids


def test_build_tunnels_skips_surface_and_bridge_roads():
    heights = np.full((10, 10), 495.0)
    road = _road(1, "bridge", {"bridge": "yes"})

    assert TerrainWorkflow._build_tunnels(SimpleNamespace(), [road], heights, 0.0, 0.0) == []
