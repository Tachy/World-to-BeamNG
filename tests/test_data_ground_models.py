"""Prüft, dass alle groundModelName-Werte in data/osm_to_beamng.json echte
BeamNG-Groundmodel-Bezeichner sind (aus art/groundmodels.json, Stand BeamNG
0.36/0.37 - siehe https://documentation.beamng.com/modding/levels/physics_materials/).

Ein hier nicht gelisteter Wert würde in generate_materials_json_entry() zwar
großgeschrieben, aber BeamNG kennt ihn trotzdem nicht - stillschweigender
ASPHALT-Fallback für Reifenphysik/-sound (siehe osm_mapper.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# BeamNGs offizielle Top-Level-Groundmodel-Namen (Aliase wie "concrete" zählen
# NICHT - die müssen in der Config bereits auf ihr kanonisches Ziel zeigen,
# z.B. "concrete" -> "asphalt", da BeamNGs Loader nur die hier gelisteten
# GROSSGESCHRIEBENEN Top-Level-Keys direkt auflöst).
OFFICIAL_GROUND_MODELS = {
    "ASPHALT",
    "ASPHALT_OLD",
    "ASPHALT_PREPPED",
    "ASPHALT_WET",
    "BRANCHES_STRONG",
    "COBBLESTONE",
    "DIRT",
    "DIRT_DUSTY",
    "DIRT_DUSTY_LOOSE",
    "FRICTIONLESS",
    "GRASS",
    "GRAVEL",
    "GRAVEL_WET",
    "ICE",
    "KICKPLATE",
    "LEAVES_STRONG",
    "LEAVES_THIN",
    "METAL",
    "METAL_TREAD",
    "MUD",
    "PLASTIC",
    "ROCK",
    "RUMBLE_STRIP",
    "SAND",
    "SHOCK_ABSORBER",
    "SLIPPERY",
    "SNOW",
    "SNOWBANK",
    "SOFT_COLLISION_GENERAL",
    "SPIKE_STRIP",
    "VOID",
    "WOOD",
}


def test_surface_types_ground_model_names_are_official():
    config_path = Path(__file__).parent.parent / "data" / "osm_to_beamng.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for name, surface_type in config.get("surface_types", {}).items():
        ground_model = surface_type.get("groundModelName", "asphalt").upper()
        assert ground_model in OFFICIAL_GROUND_MODELS, (
            f"surface_types.{name}.groundModelName={surface_type.get('groundModelName')!r} "
            f"-> {ground_model!r} ist kein offizieller BeamNG-Groundmodel-Name"
        )


def test_landuse_mappings_ground_model_names_are_official():
    config_path = Path(__file__).parent.parent / "data" / "osm_to_beamng.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for name, category in config.get("landuse_mappings", {}).items():
        if "groundModelName" not in category:
            continue  # Foto-Kategorien (Wohngebiet, Wasser) haben kein eigenes Material
        ground_model = category["groundModelName"].upper()
        assert ground_model in OFFICIAL_GROUND_MODELS, (
            f"landuse_mappings.{name}.groundModelName={category['groundModelName']!r} "
            f"-> {ground_model!r} ist kein offizieller BeamNG-Groundmodel-Name"
        )


if __name__ == "__main__":
    test_surface_types_ground_model_names_are_official()
    print("[OK] test_surface_types_ground_model_names_are_official")
    test_landuse_mappings_ground_model_names_are_official()
    print("[OK] test_landuse_mappings_ground_model_names_are_official")
