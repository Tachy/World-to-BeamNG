"""Checks that all groundModelName values in data/osm_to_beamng.json are real
BeamNG groundmodel identifiers (from art/groundmodels.json, as of BeamNG
0.36/0.37 - see https://documentation.beamng.com/modding/levels/physics_materials/).

A value not listed here would be uppercased in generate_materials_json_entry(),
but BeamNG still does not know it - silent ASPHALT fallback for tire
physics/sound (see osm_mapper.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# BeamNG's official top-level groundmodel names (aliases like "concrete" do NOT
# count - they must already point to their canonical target in the config,
# e.g. "concrete" -> "asphalt", since BeamNG's loader only resolves the
# UPPERCASE top-level keys listed here directly).
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
            f"-> {ground_model!r} is not an official BeamNG ground model name"
        )


def test_landuse_mappings_ground_model_names_are_official():
    config_path = Path(__file__).parent.parent / "data" / "osm_to_beamng.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    for name, category in config.get("landuse_mappings", {}).items():
        if "groundModelName" not in category:
            continue  # photo categories (residential area, water) have no material of their own
        ground_model = category["groundModelName"].upper()
        assert ground_model in OFFICIAL_GROUND_MODELS, (
            f"landuse_mappings.{name}.groundModelName={category['groundModelName']!r} "
            f"-> {ground_model!r} is not an official BeamNG ground model name"
        )


if __name__ == "__main__":
    test_surface_types_ground_model_names_are_official()
    print("[OK] test_surface_types_ground_model_names_are_official")
    test_landuse_mappings_ground_model_names_are_official()
    print("[OK] test_landuse_mappings_ground_model_names_are_official")
