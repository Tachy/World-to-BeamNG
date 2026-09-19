"""
Licht, Himmel, Nebel, Wolken und Regen als BeamNG-Umgebungsobjekte.

Statt geratener Werte kommen alle Felder aus BeamNGs eigenen Vorgaben (data/environment_defaults.json, erzeugt von
tools/extract_environment_defaults.py): ein abgestimmter Satz des italy-Levels plus die Farb-/Lichtwerte der Wetter-
Vorgabe "sunny_noon". Den Sonnenstand berechnet BeamNG selbst aus TimeOfDay (Datum, Breite/Länge, Uhrzeit); ein
separates Sun-Objekt gibt es wie in den Original-Leveln nicht (der ScatterSky liefert die Sonne).

Damit funktionieren im Spiel Uhrzeit, Wolken und die Wetter-Vorgaben (art/weather/*.json), die über genau diese
Objektklassen wirken (lua/ge/extensions/core/environment.lua, weather.lua).
"""

import copy
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULTS_PATH = Path(__file__).parent.parent.parent / "data" / "environment_defaults.json"
CLOUD_TEXTURE = "art/skies/clouds/clouds_normal_displacement.png"  # globales Asset in gameengine.zip

# Feste IDs der beiden schon früher vorhandenen Objekte (stabil über Exporte hinweg)
_FIXED_IDS = {
    "LevelInfo": "64e00688-24f4-417d-a0c8-25e1e7d59cce",
    "ScatterSky": "f0c7b6f6-7e4a-4b2a-8c4f-5c6f0c2a9c55",
}
_NAMES = {
    "LevelInfo": "the_level_info",
    "ScatterSky": "the_sky",
    "TimeOfDay": "tod",
    "CloudLayer": "clouds1",
    "Precipitation": "rain_coverage",
}


def load_environment_defaults(path: Optional[Path] = None) -> Dict:
    """Lädt data/environment_defaults.json (Referenzwerte aus dem Spiel)."""
    return json.loads(Path(path or DEFAULTS_PATH).read_text(encoding="utf-8"))


def clock_to_time_of_day(clock: str) -> float:
    """
    Uhrzeit "HH:MM" -> Wert für TimeOfDay.time, nach BeamNGs eigener Formel (core/solarTimeOfDay.lua,
    timeFromMinutes): (Minuten/1440 - 0.5) mod 1. Also 12:00 = 0.0, 18:00 = 0.25, Mitternacht = 0.5, 06:00 = 0.75
    (das italy-Level startet mit 0.92 = 10:05 Uhr).
    """
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", clock or "")
    if not match:
        raise ValueError(f"Uhrzeit muss HH:MM sein, war: {clock!r}")
    hours, minutes = int(match.group(1)), int(match.group(2))
    if hours > 24 or minutes > 59 or (hours == 24 and minutes):
        raise ValueError(f"Ungültige Uhrzeit: {clock!r}")
    return ((hours * 60 + minutes) / 1440.0 - 0.5) % 1.0


def _line(class_name: str, fields: Dict) -> Dict:
    persistent_id = _FIXED_IDS.get(class_name) or str(uuid.uuid5(uuid.NAMESPACE_URL, f"world_to_beamng/environment/{class_name}"))
    line = {"name": _NAMES[class_name], "class": class_name, "persistentId": persistent_id, "parentId": "MissionGroup"}
    line.update(copy.deepcopy(fields))
    return line


def build_environment_lines(
    defaults: Dict,
    latitude: float,
    longitude: float,
    date: Tuple[int, int, int],
    clock: str,
    fog_color: Sequence[float],
    fog_density: float,
    visible_distance: float,
    environment_map: str,
) -> List[Dict]:
    """
    Baut LevelInfo, ScatterSky, TimeOfDay, CloudLayer und Precipitation für main/MissionGroup/items.level.json.

    Args:
        defaults: Ergebnis von load_environment_defaults()
        latitude, longitude: Standort für den Sonnenstand
        date: (Jahr, Monat, Tag) für den Sonnenstand
        clock: Startuhrzeit "HH:MM"
        fog_color, fog_density: Dunst (wie die Original-Level levelspezifisch gesetzt)
        visible_distance: Sichtweite in Metern
        environment_map: globale Umgebungs-Map für Fahrzeug-Reflexionen
    """
    weather = defaults.get("weather", {})
    time_value = clock_to_time_of_day(clock)
    year, month, day = date

    level_info = {
        **defaults["LevelInfo"],
        "gravity": -9.81,
        "levelName": "world_to_beamng",
        "decalsEnabled": True,
        "canSave": True,
        "globalEnviromentMap": environment_map,  # so schreibt es die Engine (mit dem Tippfehler)
        "visibleDistance": visible_distance,
        "fogDensity": fog_density,
        "fogColor": list(fog_color),
    }
    scatter_sky = {**defaults["ScatterSky"], **weather.get("ScatterSky", {})}
    time_of_day = {
        **defaults["TimeOfDay"],
        "latitude": latitude,
        "longitude": longitude,
        "year": year,
        "month": month,
        "day": day,
        "time": time_value,
        "startTime": time_value,
        "play": False,
    }
    clouds = {**defaults["CloudLayer"], **weather.get("CloudLayer", {}), "texture": CLOUD_TEXTURE, "position": [0, 0, 0]}
    rain = {**defaults["Precipitation"], **weather.get("Precipitation", {}), "position": [0, 0, 0]}

    return [
        _line("LevelInfo", level_info),
        _line("ScatterSky", scatter_sky),
        _line("TimeOfDay", time_of_day),
        _line("CloudLayer", clouds),
        _line("Precipitation", rain),
    ]
