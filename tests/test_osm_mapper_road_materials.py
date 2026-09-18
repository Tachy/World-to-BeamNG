"""Tests für OSMMapper.generate_materials_json_entry() - groundType/materialTag-Mapping.

Dies ist der tatsächlich aktive Code-Pfad für Straßen-Materialien im
Haupt-Export (workflow/terrain_workflow.py::export_decal_roads() ruft ihn
direkt auf und schreibt das Ergebnis in MaterialManager.materials, OHNE über
MaterialManager.add_road_material() zu gehen).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.osm.osm_mapper import OSMMapper


def _mapper():
    # Nicht existierender Pfad -> OSMMapper fällt auf leere Defaults zurück;
    # generate_materials_json_entry() braucht kein geladenes self.config.
    return OSMMapper(config_path="__does_not_exist__.json")


def test_generate_materials_json_entry_maps_ground_type_uppercase():
    entry = _mapper().generate_materials_json_entry("dirt_road", {"groundModelName": "dirt", "textures": {}})

    # BeamNGs art/groundmodels.json kennt nur GROSSGESCHRIEBENE Bezeichner
    # (z.B. "DIRT") unter dem Key "groundType" - ohne diese Umsetzung würde
    # stillschweigend der ASPHALT-Fallback für Reifenphysik/-sound greifen.
    assert entry["groundType"] == "DIRT"
    assert "groundModelName" not in entry
    assert entry["materialTag0"] == "RoadAndPath"
    assert entry["materialTag1"] == "beamng"
    assert entry["annotation"] == "NATURE"


def test_generate_materials_json_entry_asphalt_annotation():
    entry = _mapper().generate_materials_json_entry("asphalt_road_standard", {"groundModelName": "asphalt"})

    assert entry["groundType"] == "ASPHALT"
    assert entry["annotation"] == "ASPHALT"


def test_generate_materials_json_entry_defaults_to_asphalt():
    entry = _mapper().generate_materials_json_entry("unknown", {})

    assert entry["groundType"] == "ASPHALT"


def test_generate_materials_json_entry_is_translucent_decal_material():
    # DecalRoad-Materialien MÜSSEN translucent sein, sonst kann BeamNG das
    # Decal nicht auf die Terrain-Oberfläche darunter verblenden (verifiziert
    # gegen west_coast_usa/art/road/main.materials.json -> "road_asphalt_2lane").
    entry = _mapper().generate_materials_json_entry("asphalt_road_standard", {"groundModelName": "asphalt"})

    assert entry["translucent"] is True
    assert entry["translucentZWrite"] is True


def test_generate_materials_json_entry_includes_full_pbr_stage():
    textures = {
        "baseColorMap": "a.dds",
        "normalMap": "b.dds",
        "roughnessMap": "c.dds",
        "ambientOcclusionMap": "d.dds",
        "opacityMap": "e.dds",
    }
    entry = _mapper().generate_materials_json_entry("asphalt_road_standard", {"textures": textures})

    stage = entry["Stages"][0]
    assert stage["baseColorMap"] == "a.dds"
    assert stage["normalMap"] == "b.dds"
    assert stage["roughnessMap"] == "c.dds"
    assert stage["ambientOcclusionMap"] == "d.dds"
    assert stage["opacityMap"] == "e.dds"


if __name__ == "__main__":
    test_generate_materials_json_entry_maps_ground_type_uppercase()
    print("[OK] test_generate_materials_json_entry_maps_ground_type_uppercase")
    test_generate_materials_json_entry_asphalt_annotation()
    print("[OK] test_generate_materials_json_entry_asphalt_annotation")
    test_generate_materials_json_entry_defaults_to_asphalt()
    print("[OK] test_generate_materials_json_entry_defaults_to_asphalt")
    test_generate_materials_json_entry_is_translucent_decal_material()
    print("[OK] test_generate_materials_json_entry_is_translucent_decal_material")
    test_generate_materials_json_entry_includes_full_pbr_stage()
    print("[OK] test_generate_materials_json_entry_includes_full_pbr_stage")
    print("Alle Tests bestanden.")
