"""
Extract Environment Defaults: Übernimmt Licht, Himmel, Nebel, Wolken und Regen aus BeamNGs eigenen Vorgaben
nach data/environment_defaults.json, statt Werte zu raten.

Quellen (alles aus der BeamNG-Installation):
- Objekt-Felder von LevelInfo, ScatterSky, TimeOfDay, CloudLayer, Precipitation aus dem italy-Level: EIN konsistenter,
  fertig abgestimmter Satz (statt Werte aus verschiedenen Leveln zu mischen).
- Wetter-Vorgabe `sunny_noon` aus gameengine.zip: art/weather/defaults.json (die Farb-/Lichtwerte des Spiels für
  "sonnig"). Deren `time` und `fogDensity` werden bewusst NICHT übernommen: die Uhrzeit wird aus einer Uhrzeit berechnet
  (siehe managers/environment.py), die Nebeldichte ist levelspezifisch wie in den Original-Leveln.

Aufruf: python tools/extract_environment_defaults.py
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from vendor_shared_textures import get_beamng_install_dir

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "environment_defaults.json"
SOURCE_LEVEL = "italy"
WEATHER_PRESET = "sunny_noon"
CLASSES = ("LevelInfo", "ScatterSky", "TimeOfDay", "CloudLayer", "Precipitation")

# Gehört zum Original-Level bzw. wird von uns gesetzt
LOCATION_FIELDS = ("name", "class", "persistentId", "__parent", "position", "rotation", "rotationMatrix", "scale")
# Nicht sinnvoll übertragbar: level-spezifische Umgebungs-Map (wir nutzen eine globale) und Sternbild-Bezeichnungen
DROP_FIELDS = {
    "LevelInfo": ("globalEnviromentMap", "gravity"),
    "ScatterSky": ("constellationNames",),
}
# Aus der Wetter-Vorgabe nur diese Klassen/Felder als Startzustand (keine Uhrzeit, keine Nebeldichte)
WEATHER_CLASSES = ("ScatterSky", "CloudLayer", "Precipitation")


def clean_object(obj: dict, class_name: str) -> dict:
    """Felder eines Original-Objekts ohne Ortsangaben, IDs, None-Werte und level-spezifische Felder."""
    drop = set(LOCATION_FIELDS) | set(DROP_FIELDS.get(class_name, ()))
    return {k: v for k, v in obj.items() if k not in drop and v is not None}


def weather_start_values(preset: dict) -> dict:
    """Startzustand aus einer Wetter-Vorgabe: nur Licht-/Farbwerte, Wolken, Regen (ohne TimeOfDay/LevelInfo/Wind)."""
    return {cls: dict(preset[cls]) for cls in WEATHER_CLASSES if cls in preset}


def gradient_paths(scatter_sky: dict) -> list:
    return [v for k, v in scatter_sky.items() if k.endswith("GradientFile") and isinstance(v, str)]


def _find_objects(zip_file: zipfile.ZipFile) -> dict:
    found = {}
    for name in zip_file.namelist():
        if not name.endswith("items.level.json"):
            continue
        for line in zip_file.read(name).decode("utf-8", "ignore").splitlines():
            if not any(f'"{c}"' in line for c in CLASSES):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("class") in CLASSES and obj["class"] not in found:
                found[obj["class"]] = obj
    return found


def main():
    install = get_beamng_install_dir()
    with zipfile.ZipFile(install / "content" / "levels" / f"{SOURCE_LEVEL}.zip") as z:
        objects = _find_objects(z)
    missing = [c for c in CLASSES if c not in objects]
    if missing:
        raise LookupError(f"{SOURCE_LEVEL}: keine Objekte der Klassen {missing}")

    with zipfile.ZipFile(install / "gameengine.zip") as z:
        presets = json.loads(z.read("art/weather/defaults.json").decode("utf-8", "ignore"))
        available = set(z.namelist())

    fields = {cls: clean_object(objects[cls], cls) for cls in CLASSES}

    # Die Gradienten müssen GLOBAL im Spiel liegen (nichts zu vendoren, kein Bezug auf ein anderes Level)
    for path in gradient_paths(fields["ScatterSky"]):
        if path.startswith("/") or path.startswith("levels/") or path not in available:
            raise ValueError(f"Himmels-Gradient nicht global im Spiel vorhanden: {path}")

    data = {
        "source": f"{SOURCE_LEVEL}/items.level.json + gameengine.zip:art/weather/defaults.json ({WEATHER_PRESET})",
        **fields,
        "weather": {"preset": WEATHER_PRESET, **weather_start_values(presets[WEATHER_PRESET])},
    }
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cls in CLASSES:
        print(f"[OK] {cls}: {len(fields[cls])} Felder")
    print(f"[OK] Wetter-Vorgabe {WEATHER_PRESET}: {sorted(data['weather'])}")
    print(f"[DONE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
