"""Tests für TerrainWorkflow.export_ground_cover() mit den echten Config-Daten."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng import config
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _RecordingItems:
    def __init__(self):
        self.objects = {}

    def add_ground_cover(self, name, material, types, **fields):
        self.objects[name] = {"material": material, "Types": types, **fields}


def _export(layer_map, names):
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}))
    count = TerrainWorkflow.export_ground_cover(stub, layer_map, names)
    return count, stub


def test_ground_cover_is_exported_only_for_layers_present_in_the_layer_map():
    layer_map = np.zeros((10, 10), dtype=np.uint8)
    layer_map[5:, :] = 1  # nur mat_grass kommt vor, mat_forest ist nicht gemalt
    names = ["aerial_photo", "mat_grass", "mat_forest"]

    count, stub = _export(layer_map, names)

    assert count == len(stub.items.objects) > 0
    assert all(name.startswith("gc_mat_grass_") for name in stub.items.objects)
    for obj in stub.items.objects.values():
        assert {t["layer"] for t in obj["Types"]} == {"mat_grass"}


def test_meadow_gets_grass_and_flowers_billboards():
    layer_map = np.ones((4, 4), dtype=np.uint8)

    _, stub = _export(layer_map, ["aerial_photo", "mat_grass"])

    materials = {o["material"] for o in stub.items.objects.values()}
    assert {"m_grass_green_short_01", "m_grass_green_long_01", "m_flowers_01"} <= materials
    # Billboard-Materialien sind registriert und nutzen gemeinsame /assets/-Texturen
    for name in materials:
        stage = stub.materials.materials[name]["Stages"][0]
        assert stage["baseColorMap"].startswith("/assets/")


def test_limits_come_from_config():
    layer_map = np.ones((4, 4), dtype=np.uint8)

    _, stub = _export(layer_map, ["aerial_photo", "mat_grass"])

    for obj in stub.items.objects.values():
        assert obj["maxElements"] == config.GROUND_COVER_MAX_ELEMENTS
        assert obj["radius"] <= config.GROUND_COVER_MAX_RADIUS


def test_only_photo_layer_means_no_ground_cover():
    count, stub = _export(np.zeros((4, 4), dtype=np.uint8), ["aerial_photo"])

    assert count == 0
    assert stub.items.objects == {}


def test_every_material_category_produces_ground_cover_when_painted():
    # Ziel: jede Landnutzung bekommt Bodenbewuchs
    mappings = config.OSM_MAPPER.config["landuse_mappings"]
    layers = ["aerial_photo"] + [d["internal_name"] for d in mappings.values() if d.get("internal_name") and not d.get("keep_photo")]
    layer_map = np.arange(len(layers), dtype=np.uint8).reshape(1, -1)

    _, stub = _export(layer_map, layers)

    for layer in layers[1:]:
        assert any(name.startswith(f"gc_{layer}_") for name in stub.items.objects), f"{layer} ohne Bodenbewuchs"


def test_disabled_ground_cover_exports_nothing(monkeypatch):
    monkeypatch.setattr(config, "GROUND_COVER_ENABLED", False)

    count, stub = _export(np.ones((4, 4), dtype=np.uint8), ["aerial_photo", "mat_grass"])

    assert count == 0
    assert stub.items.objects == {}
