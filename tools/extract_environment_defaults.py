"""
Extract Environment Defaults: takes over light, sky, fog, clouds and rain from BeamNG's own defaults
into data/environment_defaults.json instead of guessing values.

Sources (all from the BeamNG installation):
- Object fields of LevelInfo, ScatterSky, TimeOfDay, CloudLayer, Precipitation from the italy level: ONE consistent,
  already tuned set (instead of mixing values from different levels).
- Weather preset `sunny_noon` from gameengine.zip: art/weather/defaults.json (the game's color/light values for
  "sunny"). Its `time` and `fogDensity` are deliberately NOT taken over: the time of day is computed from a clock time
  (see managers/environment.py), the fog density is level-specific as in the original levels.

Usage: python tools/extract_environment_defaults.py
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.io.beamng_install import get_beamng_install_dir

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "environment_defaults.json"
SOURCE_LEVEL = "italy"
WEATHER_PRESET = "sunny_noon"
CLASSES = ("LevelInfo", "ScatterSky", "TimeOfDay", "CloudLayer", "Precipitation")

# Belongs to the original level or is set by us
LOCATION_FIELDS = ("name", "class", "persistentId", "__parent", "position", "rotation", "rotationMatrix", "scale")
# Cannot sensibly be transferred: level-specific environment map (we use a global one) and constellation names
DROP_FIELDS = {
    "LevelInfo": ("globalEnviromentMap", "gravity"),
    "ScatterSky": ("constellationNames",),
}
# From the weather preset only these classes/fields as the initial state (no time of day, no fog density)
WEATHER_CLASSES = ("ScatterSky", "CloudLayer", "Precipitation")


def clean_object(obj: dict, class_name: str) -> dict:
    """Fields of an original object without location data, IDs, None values and level-specific fields."""
    drop = set(LOCATION_FIELDS) | set(DROP_FIELDS.get(class_name, ()))
    return {k: v for k, v in obj.items() if k not in drop and v is not None}


def weather_start_values(preset: dict) -> dict:
    """Initial state from a weather preset: only light/color values, clouds, rain (without TimeOfDay/LevelInfo/wind)."""
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
        raise LookupError(f"{SOURCE_LEVEL}: no objects of the classes {missing}")

    with zipfile.ZipFile(install / "gameengine.zip") as z:
        presets = json.loads(z.read("art/weather/defaults.json").decode("utf-8", "ignore"))
        available = set(z.namelist())

    fields = {cls: clean_object(objects[cls], cls) for cls in CLASSES}

    # The gradients must live GLOBALLY in the game (nothing to vendor, no reference to another level)
    for path in gradient_paths(fields["ScatterSky"]):
        if path.startswith("/") or path.startswith("levels/") or path not in available:
            raise ValueError(f"Sky gradient not available globally in the game: {path}")

    data = {
        "source": f"{SOURCE_LEVEL}/items.level.json + gameengine.zip:art/weather/defaults.json ({WEATHER_PRESET})",
        **fields,
        "weather": {"preset": WEATHER_PRESET, **weather_start_values(presets[WEATHER_PRESET])},
    }
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for cls in CLASSES:
        print(f"[OK] {cls}: {len(fields[cls])} fields")
    print(f"[OK] Weather preset {WEATHER_PRESET}: {sorted(data['weather'])}")
    print(f"[DONE] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
