"""ItemManager: zusätzliche, in der BeamNG-Fahrzeugauswahl wählbare Spawn-Punkte - ein SpawnSphere je
POI (Ort oder großer Parkplatz, siehe osm/poi_points.py), ergänzt den automatischen Standard-Spawn."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from world_to_beamng.managers.item_manager import ItemManager

IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.fixture
def manager(tmp_path):
    ItemManager._instance = None
    instance = ItemManager.get_instance(tmp_path)
    yield instance
    ItemManager._instance = None


def _poi(name, position=(0.0, 0.0, 100.0), kind="place", rank=1.0):
    return {"name": name, "position": list(position), "kind": kind, "rank": rank}


def test_one_spawn_point_per_poi(manager):
    pois = [_poi("Hospental", kind="place", rank=2), _poi("Andermatt", kind="place", rank=3)]

    spawns = manager._compute_poi_spawn_points(pois)

    assert {s["display_name"] for s in spawns} == {"Hospental", "Andermatt"}
    assert {s["object_name"] for s in spawns} == {"spawn_hospental", "spawn_andermatt"}


def test_places_are_ranked_before_parking_lots_regardless_of_rank_value(manager):
    pois = [_poi("Riesenparkplatz", kind="parking", rank=1_000_000.0), _poi("Kleinweiler", kind="place", rank=1)]

    spawns = manager._compute_poi_spawn_points(pois)

    assert spawns[0]["display_name"] == "Kleinweiler"
    assert spawns[1]["display_name"] == "Riesenparkplatz"


def test_within_the_same_kind_higher_rank_wins(manager):
    pois = [_poi("Dorf", kind="place", rank=2), _poi("Stadt", kind="place", rank=5), _poi("Weiler", kind="place", rank=1)]

    spawns = manager._compute_poi_spawn_points(pois)

    assert [s["display_name"] for s in spawns] == ["Stadt", "Dorf", "Weiler"]


def test_object_names_are_deduplicated_when_slugs_collide(manager):
    pois = [_poi("Weg!"), _poi("Weg?")]

    spawns = manager._compute_poi_spawn_points(pois)

    object_names = {s["object_name"] for s in spawns}
    assert len(object_names) == 2  # beide slugifizieren zu "spawn_weg" - müssen trotzdem eindeutig bleiben


def test_duplicate_display_names_are_numbered(manager):
    pois = [_poi("Parkplatz", kind="parking", rank=300), _poi("Parkplatz", kind="parking", rank=200)]

    spawns = manager._compute_poi_spawn_points(pois)

    display_names = sorted(s["display_name"] for s in spawns)
    assert display_names == ["Parkplatz", "Parkplatz 2"]


def test_result_is_capped_at_max_points_keeping_the_highest_ranked(manager):
    pois = [_poi(f"Weiler{i}", kind="place", rank=i) for i in range(5)]

    spawns = manager._compute_poi_spawn_points(pois, max_points=1)

    assert len(spawns) == 1 and spawns[0]["display_name"] == "Weiler4"  # höchster rank gewinnt


def test_position_gets_a_small_clearance_offset_and_identity_rotation(manager):
    pois = [_poi("Ort", position=(10.0, 20.0, 100.0))]

    spawns = manager._compute_poi_spawn_points(pois)

    assert spawns[0]["position"] == pytest.approx([10.0, 20.0, 100.3])
    assert spawns[0]["rotationMatrix"] == pytest.approx(IDENTITY)


def test_no_poi_points_returns_an_empty_list(manager):
    assert manager._compute_poi_spawn_points(None) == []
    assert manager._compute_poi_spawn_points([]) == []


def test_preview_builder_is_called_with_object_name_and_xy(manager):
    calls = []

    def preview_builder(object_name, xy):
        calls.append((object_name, xy))
        return f"spawn_previews/{object_name}.jpg"

    spawns = manager._compute_poi_spawn_points([_poi("Andermatt", position=(5.0, 6.0, 100.0))], preview_builder=preview_builder)

    assert calls == [("spawn_andermatt", (5.0, 6.0))]
    assert spawns[0]["preview"] == "spawn_previews/spawn_andermatt.jpg"


def test_without_a_preview_builder_preview_is_none(manager):
    spawns = manager._compute_poi_spawn_points([_poi("Andermatt")])

    assert spawns[0]["preview"] is None


# --- save() Integration --------------------------------------------------------------------------------


def test_save_writes_poi_spawn_spheres_after_the_default(manager, tmp_path):
    manager.save(poi_points=[_poi("Andermatt", position=(100.0, 100.0, 200.0))])

    lines = (tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8").splitlines()
    objects = [json.loads(line) for line in lines]

    assert objects[0]["name"] == "spawn"  # Standard-Spawn bleibt zuerst
    assert objects[1]["name"] == "spawn_andermatt"
    assert objects[1]["parentId"] == "PlayerDropPoints"
    assert objects[1]["class"] == "SpawnSphere"


def test_save_declares_spawn_points_in_info_json_when_pois_exist(manager, tmp_path):
    manager.save(poi_points=[_poi("Andermatt", position=(0.0, 0.0, 100.0))])
    manager.save_info_json()

    info = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    assert info["spawnPoints"][0] == {"objectname": "spawn"}
    assert {"objectname": "spawn_andermatt", "name": "Andermatt"} in info["spawnPoints"]


def test_save_includes_the_preview_path_when_a_builder_is_given(manager, tmp_path):
    manager.save(
        poi_points=[_poi("Andermatt", position=(0.0, 0.0, 100.0))],
        preview_builder=lambda object_name, xy: f"spawn_previews/{object_name}.jpg",
    )
    manager.save_info_json()

    info = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    assert {
        "objectname": "spawn_andermatt", "name": "Andermatt", "preview": "spawn_previews/spawn_andermatt.jpg",
    } in info["spawnPoints"]


def test_save_without_pois_does_not_add_a_spawn_points_field(manager, tmp_path):
    manager.save(poi_points=None)
    manager.save_info_json()

    info = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    assert "spawnPoints" not in info
