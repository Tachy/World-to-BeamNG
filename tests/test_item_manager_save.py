"""ItemManager.save(): Ausgabeformat (JSONL) und automatische Fahrzeug-Spawn-Position auf der
zur Gebietsmitte (lokal (0, 0)) nächstgelegenen Straße."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from world_to_beamng.managers.item_manager import ItemManager

IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]


@pytest.fixture
def manager(tmp_path):
    ItemManager._instance = None
    instance = ItemManager.get_instance(tmp_path) if hasattr(ItemManager, "get_instance") else ItemManager(tmp_path)
    yield instance
    ItemManager._instance = None


def _road(centerline):
    """centerline: Liste von (x, y, z) - wie result["road_slope_polygons_2d"][i]["trimmed_centerline"]."""
    return {"trimmed_centerline": np.array(centerline, dtype=float)}


def test_spawn_uses_the_point_on_the_nearest_road_to_the_area_centre(manager):
    near_road = _road([(10.0, 0.0, 100.0), (10.0, 50.0, 105.0)])  # naechster Punkt: (10, 0, 100)
    far_road = _road([(500.0, 500.0, 200.0), (500.0, 550.0, 205.0)])

    position, _rotation = manager._compute_vehicle_spawn([far_road, near_road])

    assert position[0] == pytest.approx(10.0)
    assert position[1] == pytest.approx(0.0)
    assert position[2] == pytest.approx(100.3)  # +0.3 m Sicherheitsabstand über der Strassenhoehe


def test_spawn_rotation_points_along_the_road_tangent(manager):
    # Strasse verlaeuft entlang +Y (Welt-Norden) - die Fahrtrichtung sollte lokal +Y entsprechen,
    # die rotationMatrix also die Identitaet sein (siehe _compute_vehicle_spawn()-Docstring).
    road = _road([(0.0, 0.0, 100.0), (0.0, 10.0, 100.0)])

    _position, rotation = manager._compute_vehicle_spawn([road])

    assert rotation == pytest.approx(IDENTITY)


def test_spawn_rotation_is_a_proper_rotation_matrix_for_a_diagonal_road(manager):
    road = _road([(0.0, 0.0, 100.0), (10.0, 10.0, 100.0)])  # 45 Grad zur Welt-X-Achse

    _position, rotation = manager._compute_vehicle_spawn([road])

    matrix = np.array(rotation).reshape(3, 3)
    # Orthonormal (reine Rotation, keine Skalierung/Scherung): M @ M^T = Identität
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_spawn_falls_back_without_road_data(manager):
    assert manager._compute_vehicle_spawn(None) == ([0, 0, 400], IDENTITY)
    assert manager._compute_vehicle_spawn([]) == ([0, 0, 400], IDENTITY)


def test_spawn_falls_back_for_a_road_with_a_degenerate_centerline(manager):
    # Zwei identische Punkte - keine brauchbare Tangente, aber auch kein Absturz
    road = _road([(0.0, 0.0, 100.0), (0.0, 0.0, 100.0)])

    position, rotation = manager._compute_vehicle_spawn([road])

    assert position == [0, 0, 400]
    assert rotation == IDENTITY


def test_save_writes_one_json_object_per_line_with_the_spawn_position(manager, tmp_path):
    manager.items["road_1"] = {"class": "DecalRoad", "name": "straße_ü", "nodes": [[1.5, 2.0, 3.0, 4.0]]}
    road = _road([(0.0, 0.0, 50.0), (0.0, 10.0, 50.0)])

    manager.save(road_polygons=[road])

    lines = (tmp_path / "main" / "MissionGroup" / "items.level.json").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == len(manager.base_lines) + 1
    assert parsed[-1]["name"] == "straße_ü"  # kein ASCII-Escaping
    assert json.loads((tmp_path / "main" / "items.level.json").read_text(encoding="utf-8"))["class"] == "SimGroup"
    drop = json.loads((tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8").splitlines()[0])
    assert drop["position"] == pytest.approx([0.0, 0.0, 50.3])
    assert drop["rotationMatrix"] == pytest.approx(IDENTITY)


def test_save_without_road_data_falls_back_to_the_origin(manager, tmp_path):
    manager.save(road_polygons=None)

    drop = json.loads((tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8").splitlines()[0])
    assert drop["position"] == [0, 0, 400]
    assert drop["rotationMatrix"] == IDENTITY
