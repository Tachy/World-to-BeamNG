"""
Extract Water Templates: Übernimmt erprobte Wasser-Objekte (`River` für Bäche, `WaterBlock` für
Teiche/Seen) aus BeamNGs eigenen Levels nach data/water_templates.json.

Hintergrund: Echtes Wasser (Wellen, Fresnel, Unterwasser-Nebel, Auftrieb für Fahrzeuge) sind in
BeamNG eigene Objekte - das Luftbild allein reicht nicht. Die vielen Render-Parameter (Wellen,
Ripples, Schaum, Farbe, Nebel) sind von Hand nicht sinnvoll zu erfinden, deshalb werden sie aus
Original-Objekten kopiert, die ausschließlich `core/`-Texturen nutzen (immer vorhanden, nichts zu
vendoren). Ortsabhängiges (Name, Position, Knoten, Skalierung, Rotation, persistentId) entfällt.

Aufruf: python tools/extract_water_templates.py
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

# Ortsabhängige Felder gehören zum Original-Level
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
    """Kleiner Bach: `River` mit vielen Knoten, 2-3 m breit, 1 m tief, nur core-Texturen."""
    for obj in _iter_objects(zip_file, "River"):
        nodes = obj.get("nodes") or []
        if len(nodes) < 10 or not _core_textures_only(obj):
            continue
        widths = [n[3] for n in nodes]
        depths = [n[4] for n in nodes]
        if max(widths) <= 3.0 and max(depths) <= 1.0:
            return obj
    raise LookupError("Kein passender Bach-River gefunden")


def _pick_pond(zip_file: zipfile.ZipFile) -> dict:
    """Stillgewässer: das `WaterBlock` 'creek2' (core-Texturen, ruhig, grünlich-braun)."""
    for obj in _iter_objects(zip_file, "WaterBlock"):
        if obj.get("name") == "creek2" and _core_textures_only(obj):
            return obj
    raise LookupError("WaterBlock 'creek2' nicht gefunden")


def _template(obj: dict, source: str) -> dict:
    fields = {k: v for k, v in obj.items() if k not in LOCATION_FIELDS and v is not None}
    return {"source": source, "class": obj["class"], "fields": fields}


def main():
    levels_dir = get_beamng_install_dir() / "content" / "levels"
    with zipfile.ZipFile(levels_dir / f"{SOURCE_LEVEL}.zip") as zip_file:
        stream = _pick_stream(zip_file)
        pond = _pick_pond(zip_file)

    data = {
        "stream": _template(stream, f"{SOURCE_LEVEL}/River (Bach, {len(stream['nodes'])} Knoten)"),
        "pond": _template(pond, f"{SOURCE_LEVEL}/WaterBlock creek2"),
    }
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, template in data.items():
        print(f"[OK] {key}: {template['source']} -> {template['class']}, {len(template['fields'])} Felder")
    print(f"[DONE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
