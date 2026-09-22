"""Tests für TerrainWorkflow._build_bridges() und export_bridges(): Brücken (Deck + Pfeiler) als eigene DAE mit einem TSStatic."""

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
    monkeypatch.setattr(
        registry, "prepared_textures", lambda: {config.CONCRETE_TEXTURE_NAME: CONCRETE, config.RAILING_TEXTURE_NAME: CONCRETE}
    )
    return tmp_path / "shapes"


def _mesh(name="bridge_1"):
    return {
        "id": name,
        "vertices": np.zeros((8, 3)),
        "uvs": np.zeros((8, 2)),
        "normals": np.tile([0.0, 0.0, 1.0], (8, 1)),
        "faces": {"asphalt_road_standard": [[0, 1, 2]], config.BRIDGE_MATERIAL_NAME: [[4, 5, 6]]},
    }


def _road(road_id, highway="primary"):
    return {
        "road_id": road_id,
        "trimmed_centerline": np.array([[0.0, 0.0, 200.0], [50.0, 0.0, 200.0]]),
        "osm_tags": {"highway": highway, "bridge": "yes"},
        "structure_type": "bridge",
    }


def test_export_bridges_writes_one_dae_one_item_and_registers_deck_and_pier_materials(shapes_dir):
    stub = _stub()

    count = TerrainWorkflow.export_bridges(stub, {"bridge_meshes": [_mesh("bridge_1"), _mesh("bridge_2")], "structure_road_polygons": [_road(1), _road(2)]})

    assert count == 2
    dae_path, meshes, with_uv = stub.dae.calls[0]
    assert dae_path == shapes_dir / "bridges" / "bridges.dae" and with_uv is True and len(meshes) == 2
    item = stub.items.objects["bridges"]
    assert item["class"] == "TSStatic" and item["shape_name"] == "levels/world_to_beamng/art/shapes/bridges/bridges.dae"
    assert item["collisionType"] == "Visible Mesh Final"
    assert config.BRIDGE_MATERIAL_NAME in stub.materials.added
    assert config.BRIDGE_RAILING_MATERIAL_NAME in stub.materials.added
    assert "asphalt_road_standard_structure" in stub.materials.added  # Fahrbahn-Deckmaterial (highway=primary)
    assert stub.materials.added["asphalt_road_standard_structure"]["groundType"] == "ASPHALT"


def test_export_bridges_takes_the_concrete_texture_from_the_registry_check_and_never_falls_back(shapes_dir, monkeypatch):
    def aborted():
        raise registry.MissingTexturesError("Beton-Textur fehlt")

    monkeypatch.setattr(registry, "prepared_textures", aborted)
    stub = _stub()

    with pytest.raises(registry.MissingTexturesError):
        TerrainWorkflow.export_bridges(stub, {"bridge_meshes": [_mesh()], "structure_road_polygons": [_road(1)]})
    assert not stub.materials.added and not stub.dae.calls


def test_nothing_is_exported_and_stale_files_are_removed_without_bridges(shapes_dir, monkeypatch):
    stale = shapes_dir / "bridges"
    stale.mkdir(parents=True)
    (stale / "bridges.dae").write_text("alt", encoding="utf-8")
    (stale / "bridges.cdae").write_text("alt", encoding="utf-8")
    stub = _stub()

    assert TerrainWorkflow.export_bridges(stub, {"bridge_meshes": [], "structure_road_polygons": []}) == 0
    assert not (stale / "bridges.dae").exists() and not (stale / "bridges.cdae").exists()

    monkeypatch.setattr(config, "BRIDGES_ENABLED", False)
    assert TerrainWorkflow.export_bridges(stub, {"bridge_meshes": [_mesh()], "structure_road_polygons": [_road(1)]}) == 0
    assert not stub.dae.calls


def test_build_bridges_creates_a_mesh_per_bridge_with_a_pier_over_a_deep_span():
    heights = np.full((50, 50), 150.0)  # flaches Tal, 50 m unter der Brücke
    coords = np.array([[x, 5.0, 200.0] for x in np.linspace(0.0, 60.0, 7)])  # Brücke auf 200 m Höhe
    road = {"road_id": 1, "trimmed_centerline": coords, "osm_tags": {"highway": "primary", "bridge": "yes"}, "structure_type": "bridge"}

    meshes = TerrainWorkflow._build_bridges(SimpleNamespace(), [road], heights, 0.0, 0.0)

    assert len(meshes) == 1 and meshes[0]["id"] == "bridge_1"
    assert "asphalt_road_standard_structure" in meshes[0]["faces"] and config.BRIDGE_MATERIAL_NAME in meshes[0]["faces"]
    assert len(meshes[0]["faces"][config.BRIDGE_MATERIAL_NAME]) > 0  # mindestens ein Pfeiler bei 60 m Spannweite
    assert len(meshes[0]["faces"][config.BRIDGE_RAILING_MATERIAL_NAME]) > 0  # Geländer über die volle Länge


def test_build_bridges_skips_non_bridge_roads():
    heights = np.full((10, 10), 150.0)
    road = {"road_id": 1, "trimmed_centerline": np.array([[0.0, 0.0, 200.0], [5.0, 0.0, 200.0]]), "osm_tags": {}, "structure_type": "tunnel"}

    assert TerrainWorkflow._build_bridges(SimpleNamespace(), [road], heights, 0.0, 0.0) == []
