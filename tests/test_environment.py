"""
Tests für managers/environment.py: Licht, Himmel, Nebel, Wolken, Regen aus BeamNGs eigenen Vorgaben.

Die Werte stammen aus data/environment_defaults.json (siehe tools/extract_environment_defaults.py): ein abgestimmter
Satz des italy-Levels plus die Wetter-Vorgabe `sunny_noon` des Spiels - nichts davon ist geraten. Sonnenstand
berechnet BeamNG selbst aus Datum, Breite/Länge und Uhrzeit (TimeOfDay).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import pytest

from world_to_beamng import config
from world_to_beamng.managers.environment import build_environment_lines, clock_to_time_of_day, load_environment_defaults

DEFAULTS = load_environment_defaults()


def _lines(**overrides):
    args = dict(
        latitude=47.84,
        longitude=7.68,
        date=(2026, 6, 21),
        clock="11:00",
        fog_color=[0.7, 0.8, 0.9, 1.0],
        fog_density=0.0002,
        visible_distance=25000,
        environment_map="BNG_Sky_02_cubemap",
    )
    args.update(overrides)
    return build_environment_lines(DEFAULTS, **args)


def _by_class(lines):
    return {line["class"]: line for line in lines}


# --- Uhrzeit: BeamNGs eigene Formel (core/solarTimeOfDay.lua: timeFromMinutes) -----------------------


@pytest.mark.parametrize(
    "clock, expected",
    [("12:00", 0.0), ("00:00", 0.5), ("18:00", 0.25), ("06:00", 0.75), ("11:00", 0.9583333), ("10:05", 0.9201389)],
)
def test_clock_time_uses_the_games_own_formula(clock, expected):
    # 0.0 = 12 Uhr, 0.5 = Mitternacht; italy startet mit 0.92 = 10:05 Uhr
    assert clock_to_time_of_day(clock) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("bad", ["", "25:00", "12:60", "abc", "12"])
def test_invalid_clock_times_are_rejected(bad):
    with pytest.raises(ValueError):
        clock_to_time_of_day(bad)


# --- Objekte ---------------------------------------------------------------------------------------


def test_all_environment_objects_exist_and_there_is_no_separate_sun():
    by_class = _by_class(_lines())

    assert set(by_class) == {"LevelInfo", "ScatterSky", "TimeOfDay", "CloudLayer", "Precipitation"}
    # ScatterSky liefert die Sonne selbst; nur die Innenraum-Level der Originale haben ein Sun-Objekt
    assert "Sun" not in by_class


def test_names_and_persistent_ids_are_unique_stable_and_parented():
    first, second = _lines(), _lines()

    assert first == second  # deterministisch
    assert len({l["name"] for l in first}) == len(first)
    assert len({l["persistentId"] for l in first}) == len(first)
    assert all(l["parentId"] == "MissionGroup" for l in first)


def test_level_info_carries_fog_color_distance_and_the_correctly_spelled_environment_map():
    info = _by_class(_lines())["LevelInfo"]

    assert info["fogColor"] == [0.7, 0.8, 0.9, 1.0]
    assert info["fogDensity"] == 0.0002 and info["visibleDistance"] == 25000
    assert info["globalEnviromentMap"] == "BNG_Sky_02_cubemap"  # Schreibweise der Engine
    assert "globalEnvironmentMap" not in info
    assert info["levelName"] == "world_to_beamng" and info["decalsEnabled"] is True


def test_scatter_sky_is_the_games_tuned_set_with_global_gradients_and_sunny_values():
    sky = _by_class(_lines())["ScatterSky"]

    gradients = [v for k, v in sky.items() if k.endswith("GradientFile")]
    assert gradients and all(g.startswith("art/sky_gradients/") for g in gradients)  # global, nicht /levels/italy/...
    sunny = DEFAULTS["weather"]["ScatterSky"]
    for key, value in sunny.items():
        assert sky[key] == value  # Farb-/Lichtwerte der Vorgabe "sunny_noon"
    assert "constellationNames" not in sky
    # keine der früher geratenen Handwerte
    assert "fadeStartDistance" not in sky and "texSize" not in sky


def test_time_of_day_follows_location_date_and_clock():
    tod = _by_class(_lines(clock="11:00", date=(2026, 6, 21)))["TimeOfDay"]

    assert (tod["latitude"], tod["longitude"]) == (47.84, 7.68)
    assert (tod["year"], tod["month"], tod["day"]) == (2026, 6, 21)
    assert tod["time"] == tod["startTime"] == pytest.approx(clock_to_time_of_day("11:00"))
    assert tod["play"] is False  # Uhrzeit steht, im Spiel regelbar


def test_clouds_use_a_global_texture_and_the_sunny_coverage():
    clouds = _by_class(_lines())["CloudLayer"]

    assert clouds["texture"].startswith("art/skies/clouds/")  # globales Asset, nichts zu vendoren
    assert clouds["coverage"] == DEFAULTS["weather"]["CloudLayer"]["coverage"]


def test_rain_object_exists_but_is_off():
    rain = _by_class(_lines())["Precipitation"]

    assert rain["dataBlock"] == "rain_medium"  # im Spiel global definiert
    assert rain["numDrops"] == 0


def test_the_defaults_are_the_extracted_game_data():
    assert DEFAULTS["weather"]["preset"] == "sunny_noon"
    assert "time" not in DEFAULTS["weather"].get("ScatterSky", {})
    assert "TimeOfDay" not in DEFAULTS["weather"]  # Uhrzeit wird aus einer Uhrzeit berechnet, nicht aus der Vorgabe


# --- Config ----------------------------------------------------------------------------------------


def test_environment_config_has_sane_values():
    assert len(config.ENV_FOG_COLOR) == 4 and all(0.0 <= c <= 1.0 for c in config.ENV_FOG_COLOR)
    assert config.ENV_FOG_COLOR[2] > config.ENV_FOG_COLOR[0]  # bläulicher Dunst, nicht grau
    assert config.ENV_FOG_HEIGHT_MARGIN >= 0
    assert clock_to_time_of_day(config.ENV_CLOCK_TIME) >= 0.0
    year, month, day = config.ENV_DATE
    assert 1 <= month <= 12 and 1 <= day <= 31


# --- Extraktions-Tool (reine Funktionen) -----------------------------------------------------------


def test_extraction_removes_location_ids_and_level_specific_fields():
    import extract_environment_defaults as tool

    cleaned = tool.clean_object(
        {"name": "x", "class": "LevelInfo", "persistentId": "p", "position": [1, 2, 3], "fogDensity": 0.1,
         "globalEnviromentMap": "cubemap_italy_reflection", "canvasClearColor": None, "visibleDistance": 25000},
        "LevelInfo",
    )

    assert cleaned == {"fogDensity": 0.1, "visibleDistance": 25000}


def test_weather_start_values_skip_time_fog_and_wind():
    import extract_environment_defaults as tool

    preset = {"TimeOfDay": {"time": 0.25}, "LevelInfo": {"fogDensity": 0.003}, "ForestWindEmitter": {"strength": 0.5},
              "ScatterSky": {"colorize": [1, 1, 1, 1]}, "CloudLayer": {"coverage": 0}, "Precipitation": {"numDrops": 0}}

    assert tool.weather_start_values(preset) == {
        "ScatterSky": {"colorize": [1, 1, 1, 1]}, "CloudLayer": {"coverage": 0}, "Precipitation": {"numDrops": 0}}
