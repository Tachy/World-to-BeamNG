"""Tests for MaterialManager.add_terrain_materials()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.managers.material_manager import MaterialManager


def test_add_terrain_materials_registers_entries(tmp_path):
    MaterialManager.reset_instance()
    materials = MaterialManager.get_instance(tmp_path)

    entries = {
        "mat_forest": {
            "internalName": "mat_forest",
            "class": "TerrainMaterial",
            "persistentId": "abc-123",
            "baseColorBaseTex": "a/forest_b.png",
            "baseColorBaseTexSize": 4.0,
        }
    }

    materials.add_terrain_materials(entries)

    assert "mat_forest" in materials.materials
    assert materials.materials["mat_forest"]["class"] == "TerrainMaterial"
    MaterialManager.reset_instance()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_add_terrain_materials_registers_entries(Path(tmp))
        print("[OK] test_add_terrain_materials_registers_entries")
        print("Alle Tests bestanden.")
