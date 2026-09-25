"""
Extract Ground Cover Templates: takes over proven ground vegetation definitions (grass,
flowers, fern, weeds) from BeamNG's own levels into data/ground_cover_templates.json.

Background: blades of grass etc. are not part of the terrain texture in BeamNG but
separate `GroundCover` objects. An object has ONE billboard material (texture atlas,
see /assets/materials/foliage/...) and several `Types`, each with `billboardUVs`
(region in the atlas), size, clumping and the terrain layer they grow on.
The UV rectangles cannot sensibly be invented by hand - so they are copied from
original objects. The layer is deliberately omitted here and only set to our terrain
material during export.

Usage: python tools/extract_ground_cover_templates.py
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from vendor_shared_textures import get_beamng_install_dir

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "ground_cover_templates.json"

# Template name -> (level, object name in the level). Chosen after reviewing the original
# levels: per billboard material the object with the most types, and among several
# variants the one with the longer range ("distant").
SOURCES = {
    "grass_short": ("Industrial", "small_grass_green_distant"),
    "grass_long": ("Industrial", "long_grass_green_distant"),
    "grass_close": ("automation_test_track", "grass_green_1"),
    "flowers": ("Industrial", "flowers_03"),
    "dry_grass_short": ("Industrial", "small_grass_dry_close"),
    "dry_grass": ("Cliff", "grass_dry_7"),
    "weed": ("Cliff", "weed1"),
    "fern": ("driver_training", "fern1"),
    "wet_plant": ("Industrial", "wet_weed_02"),
    # Dense "close" presets (grid 4, radius 50, ~1.3 elements/m², blades up to 1.2 m): this is the
    # dense tall grass of the original levels. The "distant" templates above are only the thin far layer
    # (grid 6-8, ~0.3-0.6/m²) - on their own they yield isolated small blades.
    "grass_medium_close": ("east_coast_usa", "medium_grass_close"),
    "dry_grass_medium_close": ("Industrial", "medium_grass_dry_close"),
}

# Object fields that are taken over 1:1 (everything else - persistentId, position,
# __parent - belongs to the original level).
OBJECT_FIELDS = (
    "radius",
    "gridSize",
    "dissolveRadius",
    "shapeCullRadius",
    "maxBillboardTiltAngle",
    "windGustFrequency",
    "windGustLength",
    "windGustStrength",
    "windTurbulenceFrequency",
    "windTurbulenceStrength",
    "zOffset",
    "reflectScale",
)


def _iter_level_lines(zip_file: zipfile.ZipFile, suffix: str):
    for name in zip_file.namelist():
        if name.endswith(suffix):
            yield from zip_file.read(name).decode("utf-8", "ignore").splitlines()


def _find_ground_cover(zip_file: zipfile.ZipFile, object_name: str) -> dict:
    for line in _iter_level_lines(zip_file, "items.level.json"):
        if '"GroundCover"' not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("class") == "GroundCover" and obj.get("name") == object_name:
            return obj
    raise LookupError(f"GroundCover '{object_name}' not found")


def _find_material(levels_dir: Path, material_name: str) -> dict:
    for zip_path in sorted(levels_dir.glob("*.zip")):
        zip_file = zipfile.ZipFile(zip_path)
        for name in zip_file.namelist():
            if not name.endswith("materials.json"):
                continue
            try:
                data = json.loads(zip_file.read(name).decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                continue
            material = data.get(material_name)
            if not isinstance(material, dict) or material.get("class") != "Material":
                continue
            stage_paths = [
                v for stage in material.get("Stages", []) for k, v in stage.items() if k.endswith("Map") and isinstance(v, str)
            ]
            if stage_paths and all(p.startswith("/assets/") for p in stage_paths):
                return material
    raise LookupError(f"Material '{material_name}' with shared /assets/ textures not found")


def _clean_material(material: dict) -> dict:
    cleaned = {k: v for k, v in material.items() if k != "persistentId"}
    cleaned["Stages"] = [
        {k: v for k, v in stage.items() if v is not None} for stage in material.get("Stages", []) if any(stage.values())
    ]
    return cleaned


def _clean_type(ground_cover_type: dict) -> dict:
    return {k: v for k, v in ground_cover_type.items() if k != "layer" and v is not None}


def main():
    levels_dir = get_beamng_install_dir() / "content" / "levels"
    templates = {}
    materials = {}

    for template_name, (level, object_name) in SOURCES.items():
        with zipfile.ZipFile(levels_dir / f"{level}.zip") as zip_file:
            source = _find_ground_cover(zip_file, object_name)

        billboard_types = [_clean_type(t) for t in source["Types"] if t.get("billboardUVs")]
        material_name = source["material"]
        if material_name not in materials:
            materials[material_name] = _clean_material(_find_material(levels_dir, material_name))

        template = {"material": material_name, "source": f"{level}/{object_name}"}
        template.update({field: source[field] for field in OBJECT_FIELDS if field in source})
        template["types"] = billboard_types
        templates[template_name] = template
        print(f"[OK] {template_name}: {level}/{object_name} -> {material_name}, {len(billboard_types)} types")

    OUTPUT_PATH.write_text(
        json.dumps({"billboard_materials": materials, "templates": templates}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
