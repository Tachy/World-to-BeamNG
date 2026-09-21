"""ItemManager: kein Feld "rotation" in den Level-Objekten.

Hintergrund: BeamNG liest die Ausrichtung eines Objekts nur aus "rotationMatrix" (in den offiziellen Leveln steht bei
77.678 TSStatic-Einträgen nie "rotation"). Unser früheres "rotation": [0, 0, 1, 0] hat jedes DAE-Objekt um ca. -0,04 Grad
um die x-Achse durch den Ursprung gekippt: bei y = 1660 saß eine Mauer 1,15 m zu tief, im Süden (y = -2000) 1,4 m zu hoch.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.managers.item_manager import ItemManager

IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]


@pytest.fixture
def manager(tmp_path):
    ItemManager._instance = None
    instance = ItemManager.get_instance(tmp_path) if hasattr(ItemManager, "get_instance") else ItemManager(tmp_path)
    yield instance
    ItemManager._instance = None


def test_a_plain_item_has_no_rotation_field(manager):
    manager.add_item("walls", item_class="TSStatic", shape_name="levels/x/walls.dae", position=(0, 0, 0))

    item = manager.items["walls"]
    assert "rotation" not in item and "rotationMatrix" not in item  # Ausrichtung = BeamNG-Standard (keine Drehung)
    assert item["position"] == [0, 0, 0] and item["scale"] == [1, 1, 1]


def test_an_explicit_orientation_is_written_as_a_rotation_matrix(manager):
    matrix = [0, 1, 0, -1, 0, 0, 0, 0, 1]

    manager.add_item("turned", item_class="TSStatic", shape_name="levels/x/a.dae", rotation_matrix=matrix)

    item = manager.items["turned"]
    assert item["rotationMatrix"] == matrix and "rotation" not in item


def test_the_old_quaternion_style_argument_is_gone(manager):
    with pytest.raises(TypeError):
        manager.add_item("x", item_class="TSStatic", rotation=(0, 0, 1, 0))


def test_convenience_items_have_no_rotation_field(manager):
    manager.add_building("b", "buildings_a.dae", position=(0, 0, 0))
    manager.add_horizon()
    manager.add_terrain_block("theTerrain", "world_to_beamng.ter", "world_to_beamngTerrainMaterialTextureSet", 500.0, 236.0, -2000.0, -2000.0, 1.0)

    for name, item in manager.items.items():
        assert "rotation" not in item, name


def test_the_saved_level_has_no_rotation_field_in_any_object(manager, tmp_path):
    manager.add_item("walls", item_class="TSStatic", shape_name="levels/x/walls.dae")
    manager.add_item("pond_0_0", item_class="WaterBlock", position=(1, 2, 3), scale=(4, 2, 3), rotation_matrix=IDENTITY)
    manager.add_item("river", item_class="River")

    manager.save(height_points=np.array([[0.0, 0.0]]), height_elevations=np.array([50.0]), global_offset=(0.0, 0.0))

    saved = list((tmp_path / "main").rglob("items.level.json"))
    assert len(saved) >= 3  # Wurzel, MissionGroup, PlayerDropPoints
    objects = [json.loads(line) for path in saved for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(objects) >= 5
    assert [o.get("name") for o in objects if "rotation" in o] == []
    pond = next(o for o in objects if o.get("name") == "pond_0_0")
    assert pond["rotationMatrix"] == IDENTITY and pond["scale"] == [4, 2, 3]  # Wasserblöcke behalten Ausrichtung und Größe


def test_the_spawn_sphere_has_no_rotation_field():
    for line in ItemManager.PLAYER_DROPPOINTS_LINE:
        assert "rotation" not in line, line.get("name")

