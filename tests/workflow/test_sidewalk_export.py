"""Sidewalks in the export: built in export_decal_roads() from the DecalRoad nodes, written by export_sidewalks()."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.textures import registry
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _RecordingItems:
    def __init__(self):
        self.roads, self.objects = {}, {}

    def add_decal_road(self, name, nodes, material, **extra):
        self.roads[name] = {"nodes": nodes, "material": material, **extra}

    def add_item(self, name, item_class="TSStatic", position=(0, 0, 0), **fields):
        self.objects[name] = {"class": item_class, "position": list(position), **fields}


class _Materials:
    def __init__(self):
        self.materials, self.added = {}, {}

    def get_templates(self):
        return {"buildings": {"wall": {"material_hints": {}}}}

    def add_building_material(self, name, **fields):
        self.added[name] = fields


class _Dae:
    def __init__(self):
        self.calls = []

    def export_multi_mesh(self, output_path, meshes, with_uv=False, material_textures=None):
        self.calls.append((Path(output_path), meshes))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<COLLADA/>", encoding="utf-8")
        return output_path


def _stub():
    return SimpleNamespace(items=_RecordingItems(), materials=_Materials(), dae=_Dae(), _export_structure_road_assets=lambda lines: None)


def _mesh_data(sides, drop=False):
    heights = np.full((200, 200), 100.0)
    if drop:
        heights[:95, :] = 90.0  # 10 m drop south of the road: a guard rail case without sidewalk
    centerline = np.array([[x, 100.0, 100.0] for x in np.arange(20.0, 180.5, 1.0)])
    poly = {"road_id": 1, "trimmed_centerline": centerline, "osm_tags": {"highway": "residential"}}
    if sides:
        poly["sidewalk_sides"] = sides
    return {"road_slope_polygons_2d": [poly], "heightmap": heights, "terrain_origin_x": 0.0, "terrain_origin_y": 0.0}


def test_export_decal_roads_builds_one_mesh_per_sidewalk_side():
    mesh_data = _mesh_data({"left": "asphalt_road_standard", "right": "cobblestone_road"})
    TerrainWorkflow.export_decal_roads(_stub(), mesh_data)
    meshes = mesh_data["sidewalk_meshes"]
    assert len(meshes) == 2
    assert {m for mesh in meshes for m in mesh["faces"]} == {
        config.BRIDGE_MATERIAL_NAME, "asphalt_road_standard_structure", "cobblestone_road_structure"
    }
    top = max(float(mesh["vertices"][:, 2].max()) for mesh in meshes)
    assert top == pytest.approx(100.0 + config.SIDEWALK_KERB_HEIGHT)


def test_roads_without_sidewalk_get_no_mesh():
    mesh_data = _mesh_data({})
    TerrainWorkflow.export_decal_roads(_stub(), mesh_data)
    assert mesh_data["sidewalk_meshes"] == []


def test_road_with_sidewalk_gets_no_guard_rail(monkeypatch):
    monkeypatch.setattr(config, "GUARDRAILS_ENABLED", True)
    with_rail, without_rail = _mesh_data({}, drop=True), _mesh_data({"right": "asphalt_road_standard"}, drop=True)
    TerrainWorkflow.export_decal_roads(_stub(), with_rail)
    TerrainWorkflow.export_decal_roads(_stub(), without_rail)
    assert with_rail["guardrail_instances"] and not without_rail["guardrail_instances"]


CONCRETE = {"baseColorMap": "levels/world_to_beamng/art/shapes/textures/concrete_b.color.dds"}


def test_export_sidewalks_writes_one_dae_one_tsstatic_and_the_materials(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    monkeypatch.setattr(registry, "prepared_textures", lambda: {config.CONCRETE_TEXTURE_NAME: CONCRETE})
    stub = _stub()
    mesh = {"id": "sidewalk_0", "vertices": np.zeros((3, 3)), "uvs": np.zeros((3, 2)), "normals": np.tile([0.0, 0.0, 1.0], (3, 1)),
            "faces": {config.BRIDGE_MATERIAL_NAME: [[0, 1, 2]], "asphalt_road_standard_structure": [[0, 1, 2]]}}
    assert TerrainWorkflow.export_sidewalks(stub, {"sidewalk_meshes": [mesh]}) == 1
    assert (tmp_path / "shapes" / "sidewalks" / "sidewalks.dae").is_file()
    item = stub.items.objects["sidewalks"]
    assert item["collisionType"] == "Visible Mesh Final" and "rotation" not in item
    assert set(stub.materials.added) == {config.BRIDGE_MATERIAL_NAME, "asphalt_road_standard_structure"}


def test_export_sidewalks_removes_leftovers_without_meshes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    old = tmp_path / "shapes" / "sidewalks" / "sidewalks.dae"
    old.parent.mkdir(parents=True)
    old.write_text("old", encoding="utf-8")
    assert TerrainWorkflow.export_sidewalks(_stub(), {"sidewalk_meshes": []}) == 0
    assert not old.exists()
