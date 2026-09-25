"""
Tests for managers/environment.py: light, sky, fog, clouds, rain from BeamNG's own defaults.

The values come from data/environment_defaults.json (see tools/extract_environment_defaults.py): a tuned
set from the italy level plus the game's `sunny_noon` weather preset - none of it is guessed. BeamNG computes the
sun position itself from date, latitude/longitude and time of day (TimeOfDay).
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


# --- Time of day: BeamNG's own formula (core/solarTimeOfDay.lua: timeFromMinutes) --------------------


@pytest.mark.parametrize(
    "clock, expected",
    [("12:00", 0.0), ("00:00", 0.5), ("18:00", 0.25), ("06:00", 0.75), ("11:00", 0.9583333), ("10:05", 0.9201389)],
)
def test_clock_time_uses_the_games_own_formula(clock, expected):
    # 0.0 = 12:00 noon, 0.5 = midnight; italy starts at 0.92 = 10:05 AM
    assert clock_to_time_of_day(clock) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("bad", ["", "25:00", "12:60", "abc", "12"])
def test_invalid_clock_times_are_rejected(bad):
    with pytest.raises(ValueError):
        clock_to_time_of_day(bad)


# --- Objects ---------------------------------------------------------------------------------------


def test_all_environment_objects_exist_and_there_is_no_separate_sun():
    by_class = _by_class(_lines())

    assert set(by_class) == {"LevelInfo", "ScatterSky", "TimeOfDay", "CloudLayer", "Precipitation"}
    # ScatterSky provides the sun itself; only the originals' indoor levels have a Sun object
    assert "Sun" not in by_class


def test_names_and_persistent_ids_are_unique_stable_and_parented():
    first, second = _lines(), _lines()

    assert first == second  # deterministic
    assert len({l["name"] for l in first}) == len(first)
    assert len({l["persistentId"] for l in first}) == len(first)
    assert all(l["parentId"] == "MissionGroup" for l in first)


def test_level_info_carries_fog_color_distance_and_the_correctly_spelled_environment_map():
    info = _by_class(_lines())["LevelInfo"]

    assert info["fogColor"] == [0.7, 0.8, 0.9, 1.0]
    assert info["fogDensity"] == 0.0002 and info["visibleDistance"] == 25000
    assert info["globalEnviromentMap"] == "BNG_Sky_02_cubemap"  # spelling as used by the engine
    assert "globalEnvironmentMap" not in info
    assert info["levelName"] == "world_to_beamng" and info["decalsEnabled"] is True


def test_scatter_sky_is_the_games_tuned_set_with_global_gradients_and_sunny_values():
    sky = _by_class(_lines())["ScatterSky"]

    gradients = [v for k, v in sky.items() if k.endswith("GradientFile")]
    assert gradients and all(g.startswith("art/sky_gradients/") for g in gradients)  # global, not /levels/italy/...
    sunny = DEFAULTS["weather"]["ScatterSky"]
    for key, value in sunny.items():
        assert sky[key] == value  # color/light values of the "sunny_noon" preset
    assert "constellationNames" not in sky
    # none of the previously guessed hand-tuned values
    assert "fadeStartDistance" not in sky and "texSize" not in sky


def test_time_of_day_follows_location_date_and_clock():
    tod = _by_class(_lines(clock="11:00", date=(2026, 6, 21)))["TimeOfDay"]

    assert (tod["latitude"], tod["longitude"]) == (47.84, 7.68)
    assert (tod["year"], tod["month"], tod["day"]) == (2026, 6, 21)
    assert tod["time"] == tod["startTime"] == pytest.approx(clock_to_time_of_day("11:00"))
    assert tod["play"] is False  # time of day is fixed, adjustable in game


def test_clouds_use_a_global_texture_and_the_sunny_coverage():
    clouds = _by_class(_lines())["CloudLayer"]

    assert clouds["texture"].startswith("art/skies/clouds/")  # global asset, nothing to vendor
    assert clouds["coverage"] == DEFAULTS["weather"]["CloudLayer"]["coverage"]


def test_rain_object_exists_but_is_off():
    rain = _by_class(_lines())["Precipitation"]

    assert rain["dataBlock"] == "rain_medium"  # defined globally in the game
    assert rain["numDrops"] == 0


def test_the_defaults_are_the_extracted_game_data():
    assert DEFAULTS["weather"]["preset"] == "sunny_noon"
    assert "time" not in DEFAULTS["weather"].get("ScatterSky", {})
    assert "TimeOfDay" not in DEFAULTS["weather"]  # time of day is computed from a clock time, not taken from the preset


# --- Config ----------------------------------------------------------------------------------------


def test_environment_config_has_sane_values():
    assert len(config.ENV_FOG_COLOR) == 4 and all(0.0 <= c <= 1.0 for c in config.ENV_FOG_COLOR)
    assert config.ENV_FOG_COLOR[2] > config.ENV_FOG_COLOR[0]  # bluish haze, not gray
    assert config.ENV_FOG_HEIGHT_MARGIN >= 0
    assert clock_to_time_of_day(config.ENV_CLOCK_TIME) >= 0.0
    year, month, day = config.ENV_DATE
    assert 1 <= month <= 12 and 1 <= day <= 31


# --- Extraction tool (pure functions) --------------------------------------------------------------


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
