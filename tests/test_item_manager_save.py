"""ItemManager.save(): Ausgabeformat (JSONL) und Spawn-Höhe aus dem nächsten Höhenpunkt."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.geometry.coordinates import transformer_to_utm
from world_to_beamng.managers.item_manager import ItemManager


@pytest.fixture
def manager(tmp_path):
    ItemManager._instance = None
    instance = ItemManager.get_instance(tmp_path) if hasattr(ItemManager, "get_instance") else ItemManager(tmp_path)
    yield instance
    ItemManager._instance = None


def test_spawn_height_is_the_nearest_height_point_plus_ten_metres(manager, monkeypatch):
    lat, lon = config.SPAWN_POINT
    ox, oy = 500000.0, 5300000.0
    x_utm, y_utm = transformer_to_utm.transform(lon, lat)
    x, y = x_utm - ox, y_utm - oy
    # Höhenraster im Meterabstand um den Spawn-Punkt; z = 100 + Zellindex, damit der nächste Punkt eindeutig ist
    gx, gy = np.meshgrid(np.arange(-20, 21) + np.floor(x), np.arange(-20, 21) + np.floor(y))
    points = np.column_stack([gx.ravel(), gy.ravel()])
    elevations = 100.0 + np.arange(len(points), dtype=float)

    position = manager._get_spawn_position_with_height(points, elevations, (ox, oy))

    nearest = int(np.argmin((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2))
    assert position[:2] == pytest.approx([x, y])
    assert position[2] == pytest.approx(elevations[nearest] + 10.0)


def test_spawn_height_falls_back_without_height_data(manager):
    position = manager._get_spawn_position_with_height(np.empty((0, 2)), np.empty(0), (0.0, 0.0))

    assert position[2] == pytest.approx(410.0)


def test_save_writes_one_json_object_per_line_with_the_spawn_position(manager, tmp_path):
    manager.items["road_1"] = {"class": "DecalRoad", "name": "straße_ü", "nodes": [[1.5, 2.0, 3.0, 4.0]]}

    manager.save(height_points=np.array([[0.0, 0.0]]), height_elevations=np.array([50.0]), global_offset=(0.0, 0.0))

    lines = (tmp_path / "main" / "MissionGroup" / "items.level.json").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == len(manager.base_lines) + 1
    assert parsed[-1]["name"] == "straße_ü"  # kein ASCII-Escaping
    assert json.loads((tmp_path / "main" / "items.level.json").read_text(encoding="utf-8"))["class"] == "SimGroup"
    drop = json.loads((tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8").splitlines()[0])
    assert drop["class"] in {"SpawnSphere", "BeamNGSpawnSphere"} or "position" in drop
