"""Tests für ItemManager.add_terrain_block()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.managers.item_manager import ItemManager


def test_add_terrain_block_creates_correct_item(tmp_path):
    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)

    items.add_terrain_block(
        name="theTerrain",
        terrain_filename="world_to_beamng.ter",
        material_texture_set="world_to_beamngTerrainMaterialTextureSet",
        max_height=574.0,
        z_min=263.0,
        origin_x=-1024.0,
        origin_y=-1024.0,
    )

    item = items.items["theTerrain"]
    assert item["class"] == "TerrainBlock"
    assert item["position"] == [-1024.0, -1024.0, 263.0]
    assert item["maxHeight"] == 574.0
    assert item["materialTextureSet"] == "world_to_beamngTerrainMaterialTextureSet"
    assert item["terrainFile"] == "/levels/world_to_beamng/world_to_beamng.ter"
    ItemManager.reset_instance()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_add_terrain_block_creates_correct_item(Path(tmp))
        print("[OK] test_add_terrain_block_creates_correct_item")
        print("Alle Tests bestanden.")
