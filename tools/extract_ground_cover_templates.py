"""
Extract Ground Cover Templates: Übernimmt erprobte Bodenbewuchs-Definitionen (Gras,
Blumen, Farn, Unkraut) aus BeamNGs eigenen Levels nach data/ground_cover_templates.json.

Hintergrund: Grashalme etc. sind in BeamNG kein Teil der Terrain-Textur, sondern
separate `GroundCover`-Objekte. Ein Objekt hat EIN Billboard-Material (Textur-Atlas,
siehe /assets/materials/foliage/...) und mehrere `Types`, jeweils mit `billboardUVs`
(Ausschnitt im Atlas), Größe, Klumpung und dem Terrain-Layer, auf dem sie wachsen.
Die UV-Rechtecke sind von Hand nicht sinnvoll zu erfinden - deshalb werden sie aus
Original-Objekten kopiert. Der Layer wird hier bewusst weggelassen und erst beim
Export auf unser Terrain-Material gesetzt.

Aufruf: python tools/extract_ground_cover_templates.py
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from vendor_shared_textures import get_beamng_install_dir

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "ground_cover_templates.json"

# Vorlagenname -> (Level, Objektname im Level). Auswahl nach Sichtung der Original-
# Levels: je Billboard-Material das Objekt mit den meisten Typen, bei mehreren
# Varianten die weiter reichende ("distant").
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
    # Dichte "close"-Presets (Raster 4, Radius 50, ~1,3 Elemente/m², Halme bis 1,2 m): das ist das
    # dichte hohe Gras der Original-Levels. Die "distant"-Vorlagen oben sind nur die dünne Fernschicht
    # (Raster 6-8, ~0,3-0,6/m²) - allein ergeben sie einzelne kleine Halme.
    "grass_medium_close": ("east_coast_usa", "medium_grass_close"),
    "dry_grass_medium_close": ("Industrial", "medium_grass_dry_close"),
}

# Objekt-Felder, die 1:1 übernommen werden (alles andere - persistentId, Position,
# __parent - gehört zum Original-Level).
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
    raise LookupError(f"GroundCover '{object_name}' nicht gefunden")


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
    raise LookupError(f"Material '{material_name}' mit gemeinsamen /assets/-Texturen nicht gefunden")


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
        print(f"[OK] {template_name}: {level}/{object_name} -> {material_name}, {len(billboard_types)} Typen")

    OUTPUT_PATH.write_text(
        json.dumps({"billboard_materials": materials, "templates": templates}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
