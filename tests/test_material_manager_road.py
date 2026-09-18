"""Tests für MaterialManager.add_road_material() - groundType/materialTag-Mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.managers.material_manager import MaterialManager


def test_add_road_material_maps_ground_model_name_to_uppercase_ground_type(tmp_path):
    MaterialManager.reset_instance()
    materials = MaterialManager.get_instance(tmp_path)

    materials.add_road_material(
        "dirt_road",
        {"internal_name": "dirt_road", "groundModelName": "dirt", "textures": {"baseColorMap": "x.dds"}},
    )

    mat = materials.materials["dirt_road"]
    # BeamNGs art/groundmodels.json kennt nur GROSSGESCHRIEBENE Bezeichner (z.B.
    # "DIRT") - ohne Uppercase/richtigen Key würde stillschweigend der
    # ASPHALT-Fallback für Reifenphysik/-sound greifen.
    assert mat["groundType"] == "DIRT"
    assert mat["materialTag0"] == "RoadAndPath"
    assert mat["annotation"] == "NATURE"
    MaterialManager.reset_instance()


def test_add_road_material_asphalt_annotation(tmp_path):
    MaterialManager.reset_instance()
    materials = MaterialManager.get_instance(tmp_path)

    materials.add_road_material(
        "asphalt_road_standard",
        {"internal_name": "asphalt_road_standard", "groundModelName": "asphalt", "textures": {}},
    )

    mat = materials.materials["asphalt_road_standard"]
    assert mat["groundType"] == "ASPHALT"
    assert mat["annotation"] == "ASPHALT"
    MaterialManager.reset_instance()


def test_add_road_material_defaults_to_asphalt_without_ground_model_name(tmp_path):
    MaterialManager.reset_instance()
    materials = MaterialManager.get_instance(tmp_path)

    materials.add_road_material("unknown_road", {"internal_name": "unknown_road"})

    assert materials.materials["unknown_road"]["groundType"] == "ASPHALT"
    MaterialManager.reset_instance()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_add_road_material_maps_ground_model_name_to_uppercase_ground_type(Path(tmp))
        print("[OK] test_add_road_material_maps_ground_model_name_to_uppercase_ground_type")
        test_add_road_material_asphalt_annotation(Path(tmp))
        print("[OK] test_add_road_material_asphalt_annotation")
        test_add_road_material_defaults_to_asphalt_without_ground_model_name(Path(tmp))
        print("[OK] test_add_road_material_defaults_to_asphalt_without_ground_model_name")
        print("Alle Tests bestanden.")
