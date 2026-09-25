"""
Extract Water Templates: takes over proven water objects (`River` for streams, `WaterBlock` for
ponds/lakes) from BeamNG's own levels into data/water_templates.json.

Background: real water (waves, Fresnel, underwater fog, buoyancy for vehicles) consists of dedicated
objects in BeamNG - the aerial photo alone is not enough. The many render parameters (waves,
ripples, foam, color, fog) cannot sensibly be invented by hand, so they are copied from original
objects that use only `core/` textures (always present, nothing to vendor). Location-dependent data
(name, position, nodes, scale, rotation, persistentId) is dropped.

Usage: python tools/extract_water_templates.py
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from vendor_shared_textures import get_beamng_install_dir

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "water_templates.json"

SOURCE_LEVEL = "east_coast_usa"
TEXTURE_KEYS = ("rippleTex", "foamTex", "depthGradientTex")

# Location-dependent fields belong to the original level
LOCATION_FIELDS = ("name", "persistentId", "__parent", "position", "nodes", "rotationMatrix", "rotation", "scale")


def _core_textures_only(obj: dict) -> bool:
    return all(str(obj.get(key, "core/")).startswith("core/") for key in TEXTURE_KEYS)


def _iter_objects(zip_file: zipfile.ZipFile, class_name: str):
    for name in zip_file.namelist():
        if not name.endswith("items.level.json"):
            continue
        for line in zip_file.read(name).decode("utf-8", "ignore").splitlines():
            if f'"{class_name}"' not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("class") == class_name:
                yield obj


def _pick_stream(zip_file: zipfile.ZipFile) -> dict:
    """Small stream: `River` with many nodes, 2-3 m wide, 1 m deep, core textures only."""
    for obj in _iter_objects(zip_file, "River"):
        nodes = obj.get("nodes") or []
        if len(nodes) < 10 or not _core_textures_only(obj):
            continue
        widths = [n[3] for n in nodes]
        depths = [n[4] for n in nodes]
        if max(widths) <= 3.0 and max(depths) <= 1.0:
            return obj
    raise LookupError("No suitable stream River found")


def _pick_pond(zip_file: zipfile.ZipFile) -> dict:
    """Still water: the `WaterBlock` 'creek2' (core textures, calm, greenish-brown)."""
    for obj in _iter_objects(zip_file, "WaterBlock"):
        if obj.get("name") == "creek2" and _core_textures_only(obj):
            return obj
    raise LookupError("WaterBlock 'creek2' not found")


def _template(obj: dict, source: str) -> dict:
    fields = {k: v for k, v in obj.items() if k not in LOCATION_FIELDS and v is not None}
    return {"source": source, "class": obj["class"], "fields": fields}


def main():
    levels_dir = get_beamng_install_dir() / "content" / "levels"
    with zipfile.ZipFile(levels_dir / f"{SOURCE_LEVEL}.zip") as zip_file:
        stream = _pick_stream(zip_file)
        pond = _pick_pond(zip_file)

    data = {
        "stream": _template(stream, f"{SOURCE_LEVEL}/River (stream, {len(stream['nodes'])} nodes)"),
        "pond": _template(pond, f"{SOURCE_LEVEL}/WaterBlock creek2"),
    }
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, template in data.items():
        print(f"[OK] {key}: {template['source']} -> {template['class']}, {len(template['fields'])} fields")
    print(f"[DONE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
