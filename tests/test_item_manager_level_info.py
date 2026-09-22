"""Tests für die LevelInfo-Basiszeile (Sichtweite/Nebel) des ItemManagers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng import config
from world_to_beamng.managers.item_manager import ItemManager


def _level_info():
    return next(line for line in ItemManager.OTHER_BASE_LINES if line["class"] == "LevelInfo")


def test_level_info_sets_visible_distance_from_config():
    # Ohne visibleDistance greift der BeamNG-Standard (~1 km): der Rest wird geclippt.
    info = _level_info()

    assert info["visibleDistance"] == config.LEVEL_VISIBLE_DISTANCE
    assert info["visibleDistance"] > 5000  # mindestens so weit wie die kleinsten Original-Level


def test_level_info_sets_fog_density_from_config():
    assert _level_info()["fogDensity"] == config.LEVEL_FOG_DENSITY


# --- Umgebungsobjekte (Licht/Wetter aus BeamNGs Vorgaben) ---------------------------------------------


def _classes(lines):
    return [line["class"] for line in lines]


def test_base_lines_have_the_environment_objects_and_no_guessed_sun():
    classes = _classes(ItemManager.OTHER_BASE_LINES)

    assert {"LevelInfo", "ScatterSky", "TimeOfDay", "CloudLayer", "Precipitation", "SimGroup"} <= set(classes)
    assert "Sun" not in classes  # der ScatterSky liefert die Sonne


def test_level_info_has_the_fog_color_and_the_engines_spelling_of_the_environment_map():
    info = _level_info()

    assert info["fogColor"] == config.ENV_FOG_COLOR
    assert info["globalEnviromentMap"] and "globalEnvironmentMap" not in info


def test_set_base_line_fields_changes_only_this_instance(tmp_path):
    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)
    original = _level_info().get("fogAtmosphereHeight")

    items.set_base_line_fields("theLevelInfo", fogAtmosphereHeight=812.5)

    changed = next(l for l in items.base_lines if l["name"] == "theLevelInfo")
    assert changed["fogAtmosphereHeight"] == 812.5
    assert _level_info().get("fogAtmosphereHeight") == original  # Klassen-Liste bleibt unverändert
    ItemManager.reset_instance()
    assert next(l for l in ItemManager.get_instance(tmp_path).base_lines if l["name"] == "theLevelInfo").get("fogAtmosphereHeight") == original
    ItemManager.reset_instance()


def test_set_base_line_fields_rejects_unknown_objects(tmp_path):
    import pytest

    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)

    with pytest.raises(KeyError):
        items.set_base_line_fields("gibt_es_nicht", foo=1)
    ItemManager.reset_instance()


def test_save_writes_the_exported_fog_height_and_all_environment_objects(tmp_path):
    import json

    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)
    items.set_base_line_fields("theLevelInfo", fogAtmosphereHeight=777.0)

    items.save()

    lines = [json.loads(l) for l in (tmp_path / "main" / "MissionGroup" / "items.level.json").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_class = {l["class"]: l for l in lines}
    assert by_class["LevelInfo"]["fogAtmosphereHeight"] == 777.0
    assert {"ScatterSky", "TimeOfDay", "CloudLayer", "Precipitation"} <= set(by_class)
    assert "Sun" not in by_class
    ItemManager.reset_instance()


def test_info_json_declares_the_default_spawn_point_name(tmp_path):
    import json

    ItemManager.reset_instance()
    ItemManager.get_instance(tmp_path).save_info_json()

    info = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    # Name des SpawnSphere-Objekts ("spawn"), NICHT der PlayerDropPoints-SimGroup - siehe setSpawnpoint.lua
    assert info["defaultSpawnPointName"] == "spawn"
    assert "spawnPointName" not in info  # falscher Schlüssel, BeamNG liest nur defaultSpawnPointName
    ItemManager.reset_instance()


def test_info_json_declares_time_of_day_support(tmp_path):
    import json

    ItemManager.reset_instance()
    ItemManager.get_instance(tmp_path).save_info_json()

    assert json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))["supportsTimeOfDay"] is True
    ItemManager.reset_instance()


# --- info_json / set_info_json_fields (Minimap/Terrain-Größe zur Exportzeit) --------------------------


def test_info_json_defaults_to_a_copy_of_level_info(tmp_path):
    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)

    items.info_json["size"] = [1234, 1234]

    assert ItemManager.LEVEL_INFO["size"] == [2000, 2000]  # Klassen-Konstante bleibt unverändert
    ItemManager.reset_instance()


def test_set_info_json_fields_changes_only_this_instance(tmp_path):
    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)

    items.set_info_json_fields(size=[4096, 4096], minimap=[{"file": "minimap/terrain.png", "size": [4096, 4096], "offset": [-2048, 2048]}])

    assert items.info_json["size"] == [4096, 4096]
    assert ItemManager.LEVEL_INFO["size"] == [2000, 2000]
    ItemManager.reset_instance()
    assert ItemManager.get_instance(tmp_path).info_json["size"] == [2000, 2000]  # neue Instanz: wieder Default
    ItemManager.reset_instance()


def test_save_info_json_writes_the_minimap_only_when_it_was_set(tmp_path):
    import json

    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)
    items.save_info_json()
    without_minimap = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    ItemManager.reset_instance()

    items = ItemManager.get_instance(tmp_path)
    items.set_info_json_fields(size=[4096, 4096], minimap=[{"file": "minimap/terrain.png", "size": [4096, 4096], "offset": [0, 0]}])
    items.save_info_json()
    with_minimap = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    ItemManager.reset_instance()

    assert "minimap" not in without_minimap  # wie ein echtes Level ohne Minimap (z.B. italy)
    assert with_minimap["minimap"][0]["file"] == "minimap/terrain.png"
    assert with_minimap["size"] == [4096, 4096]
