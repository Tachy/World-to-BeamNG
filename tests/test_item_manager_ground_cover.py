"""Tests for ItemManager.add_ground_cover()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.managers.item_manager import ItemManager


def test_add_ground_cover_creates_ground_cover_item(tmp_path):
    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)
    types = [{"billboardUVs": [0, 0, 1, 1], "layer": "mat_grass", "sizeMin": 0.1, "sizeMax": 0.2}]

    items.add_ground_cover(
        name="gc_mat_grass_grass_short",
        material="m_grass_green_short_01",
        types=types,
        radius=100.0,
        maxElements=50000,
        gridSize=6,
    )

    item = items.items["gc_mat_grass_grass_short"]
    assert item["class"] == "GroundCover"
    assert item["material"] == "m_grass_green_short_01"
    assert item["Types"] == types
    assert item["radius"] == 100.0
    assert item["maxElements"] == 50000
    assert item["gridSize"] == 6
    assert item["parentId"] == "MissionGroup"
    assert item["persistentId"]
    ItemManager.reset_instance()


def test_add_ground_cover_overwrites_existing_item_on_reexport(tmp_path):
    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)

    items.add_ground_cover("gc_a", "m1", [{"layer": "x"}], radius=10.0)
    items.add_ground_cover("gc_a", "m2", [{"layer": "x"}], radius=20.0)

    assert items.items["gc_a"]["material"] == "m2"
    ItemManager.reset_instance()
