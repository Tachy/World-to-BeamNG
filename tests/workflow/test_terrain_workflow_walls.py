"""Tests für TerrainWorkflow._build_wall_meshes() und export_walls(): Bruchsteinmauern als eigene DAE mit einem TSStatic."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
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


def _mesh(name="wall_1"):
    return {
        "id": name,
        "vertices": np.zeros((4, 3)),
        "uvs": np.zeros((4, 2)),
        "normals": np.tile([0.0, 0.0, 1.0], (4, 1)),
        "faces": {config.WALL_MATERIAL_NAME: [[0, 1, 2], [0, 2, 3]]},
    }


@pytest.fixture
def shapes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEAMNG_DIR_SHAPES", tmp_path / "shapes")
    return tmp_path / "shapes"


def test_export_walls_writes_one_dae_one_item_and_a_stone_material(shapes_dir):
    stub = _stub()

    count = TerrainWorkflow.export_walls(stub, {"wall_meshes": [_mesh("wall_1"), _mesh("wall_2")]})

    assert count == 2
    dae_path, meshes, with_uv = stub.dae.calls[0]
    assert dae_path == shapes_dir / "walls" / "walls.dae" and with_uv is True and len(meshes) == 2
    item = stub.items.objects["walls"]
    assert item["class"] == "TSStatic" and item["position"] == [0, 0, 0]
    assert item["shape_name"] == "levels/world_to_beamng/art/shapes/walls/walls.dae"  # Schrägstriche wie BeamNG sie erwartet
    assert item["collisionType"] == "Visible Mesh Final"  # Fahrzeuge stoßen an die Mauer
    textures = stub.materials.added[config.WALL_MATERIAL_NAME]["textures"]
    for key in ("baseColorMap", "normalMap", "roughnessMap"):
        assert textures[key].startswith("levels/world_to_beamng/art/shapes/assets/materials/tileable/stone/"), key


def test_nothing_is_exported_and_stale_files_are_removed_without_walls(shapes_dir, monkeypatch):
    stale = shapes_dir / "walls"
    stale.mkdir(parents=True)
    (stale / "walls.dae").write_text("alt", encoding="utf-8")
    (stale / "walls.cdae").write_text("alt", encoding="utf-8")
    stub = _stub()

    assert TerrainWorkflow.export_walls(stub, {"wall_meshes": []}) == 0
    assert not (stale / "walls.dae").exists() and not (stale / "walls.cdae").exists()
    assert "walls" not in stub.items.objects and not stub.dae.calls

    monkeypatch.setattr(config, "WALLS_ENABLED", False)
    assert TerrainWorkflow.export_walls(stub, {"wall_meshes": [_mesh()]}) == 0
    assert not stub.dae.calls


def test_build_wall_meshes_uses_only_walls_with_a_height_and_the_terrain_heights():
    from world_to_beamng.osm.landuse_polygons import make_local_transform

    offset = (412000.0, 5297000.0)

    def way(way_id, points, **tags):
        return {"type": "way", "id": way_id, "tags": {"barrier": "wall", **tags}, "geometry": [{"lat": lat, "lon": lon} for lon, lat in points]}

    osm = [
        way(1, [(7.6800, 47.8300), (7.6802, 47.8300)], height="2", material="stone"),
        way(2, [(7.6800, 47.8302), (7.6802, 47.8302)]),  # ohne Höhe
    ]
    ground = lambda x, y: np.full_like(np.asarray(x, float), 300.0)

    meshes, stats = TerrainWorkflow._build_wall_meshes(SimpleNamespace(), osm, offset, ground)

    assert [m["id"] for m in meshes] == ["wall_1"]
    assert stats["built"] == 1 and stats["without_height"] == 1
    z = meshes[0]["vertices"][:, 2]
    assert z.max() == pytest.approx(302.0) and z.min() == pytest.approx(300.0 - config.WALL_SINK)
    # Dicke quer zur Mauerrichtung (die Karte ist gegen Nord gedreht: UTM-Meridiankonvergenz)
    xy = meshes[0]["vertices"][:, :2]
    axis = np.linalg.svd(xy - xy.mean(axis=0))[2][0]  # Hauptrichtung der Mauer
    across = xy @ np.array([-axis[1], axis[0]])
    assert (across.max() - across.min()) == pytest.approx(0.5, abs=0.05)
    assert config.WALL_THICKNESS == pytest.approx(0.5)
