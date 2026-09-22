"""ItemManager: zusätzliche, in der BeamNG-Fahrzeugauswahl wählbare Spawn-Punkte - ein SpawnSphere je
eindeutig benannter OSM-Straße (osm_tags["name"]), ergänzt den automatischen Standard-Spawn."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from world_to_beamng.managers.item_manager import ItemManager

IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]


@pytest.fixture
def manager(tmp_path):
    ItemManager._instance = None
    instance = ItemManager.get_instance(tmp_path)
    yield instance
    ItemManager._instance = None


def _road(centerline, name=None, structure_type=None):
    road = {"trimmed_centerline": np.array(centerline, dtype=float)}
    if name is not None:
        road["osm_tags"] = {"name": name}
    if structure_type is not None:
        road["structure_type"] = structure_type
    return road


def test_one_spawn_point_per_uniquely_named_road(manager):
    roads = [
        _road([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (20.0, 0.0, 100.0)], name="Gotthardstrasse"),
        _road([(0.0, 100.0, 200.0), (0.0, 110.0, 200.0), (0.0, 120.0, 200.0)], name="Tremolastrasse"),
    ]

    spawns = manager._compute_named_spawn_points(roads)

    assert {s["display_name"] for s in spawns} == {"Gotthardstrasse", "Tremolastrasse"}
    assert {s["object_name"] for s in spawns} == {"spawn_gotthardstrasse", "spawn_tremolastrasse"}


def test_unnamed_roads_are_skipped(manager):
    roads = [_road([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)])]

    assert manager._compute_named_spawn_points(roads) == []


def test_tunnels_and_galleries_are_excluded_but_bridges_are_kept(manager):
    roads = [
        _road([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)], name="Gotthardtunnel", structure_type="tunnel"),
        _road([(0.0, 50.0, 100.0), (10.0, 50.0, 100.0)], name="Lawinengalerie", structure_type="gallery"),
        _road([(0.0, 100.0, 100.0), (10.0, 100.0, 100.0)], name="Talbruecke", structure_type="bridge"),
    ]

    spawns = manager._compute_named_spawn_points(roads)

    assert {s["display_name"] for s in spawns} == {"Talbruecke"}


def test_multiple_ways_with_the_same_name_use_the_one_with_more_points(manager):
    short_way = _road([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)], name="Passstrasse")
    long_way = _road([(100.0, 0.0, 200.0), (110.0, 0.0, 200.0), (120.0, 0.0, 200.0), (130.0, 0.0, 200.0)], name="Passstrasse")

    spawns = manager._compute_named_spawn_points([short_way, long_way])

    assert len(spawns) == 1
    assert spawns[0]["position"][0] == pytest.approx(120.0)  # Mittelpunkt des längeren Ways (Index 2 von 4)


def test_object_names_are_deduplicated_when_slugs_collide(manager):
    roads = [
        _road([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)], name="Weg!"),
        _road([(0.0, 50.0, 100.0), (10.0, 50.0, 100.0)], name="Weg?"),
    ]

    spawns = manager._compute_named_spawn_points(roads)

    object_names = {s["object_name"] for s in spawns}
    assert len(object_names) == 2  # beide slugifizieren zu "spawn_weg" - müssen trotzdem eindeutig bleiben


def test_result_is_capped_at_max_points_keeping_the_longest_roads(manager):
    roads = [_road([(float(i), 0.0, 100.0), (float(i), 10.0, 100.0)], name=f"Weg{i}") for i in range(5)]
    longer = _road([(0.0, 100.0, 100.0), (0.0, 110.0, 100.0), (0.0, 120.0, 100.0)], name="Hauptstrasse")
    roads.append(longer)

    spawns = manager._compute_named_spawn_points(roads, max_points=1)

    assert len(spawns) == 1 and spawns[0]["display_name"] == "Hauptstrasse"


def test_a_road_with_a_degenerate_midpoint_segment_is_skipped(manager):
    road = _road([(0.0, 0.0, 100.0), (0.0, 0.0, 100.0)], name="Nullstrasse")  # zwei identische Punkte

    assert manager._compute_named_spawn_points([road]) == []


def test_rotation_matrix_points_along_the_road_tangent(manager):
    road = _road([(0.0, 0.0, 100.0), (0.0, 10.0, 100.0), (0.0, 20.0, 100.0)], name="Nordstrasse")

    spawns = manager._compute_named_spawn_points([road])

    assert spawns[0]["rotationMatrix"] == pytest.approx(IDENTITY)


def test_no_named_roads_returns_an_empty_list(manager):
    assert manager._compute_named_spawn_points(None) == []
    assert manager._compute_named_spawn_points([]) == []


# --- save() Integration --------------------------------------------------------------------------------


def test_save_writes_named_spawn_spheres_after_the_default(manager, tmp_path):
    default_road = _road([(0.0, 0.0, 50.0), (0.0, 10.0, 50.0)])
    named_road = _road([(100.0, 100.0, 200.0), (110.0, 100.0, 200.0), (120.0, 100.0, 200.0)], name="Passstrasse")

    manager.save(road_polygons=[default_road, named_road])

    lines = (tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8").splitlines()
    objects = [json.loads(line) for line in lines]

    assert objects[0]["name"] == "spawn"  # Standard-Spawn bleibt zuerst
    assert objects[1]["name"] == "spawn_passstrasse"
    assert objects[1]["parentId"] == "PlayerDropPoints"
    assert objects[1]["class"] == "SpawnSphere"


def test_save_declares_spawn_points_in_info_json_when_named_roads_exist(manager, tmp_path):
    named_road = _road([(0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (20.0, 0.0, 100.0)], name="Passstrasse")

    manager.save(road_polygons=[named_road])
    manager.save_info_json()

    info = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    assert info["spawnPoints"][0] == {"objectname": "spawn"}
    assert {"objectname": "spawn_passstrasse", "name": "Passstrasse"} in info["spawnPoints"]


def test_save_without_named_roads_does_not_add_a_spawn_points_field(manager, tmp_path):
    unnamed_road = _road([(0.0, 0.0, 50.0), (0.0, 10.0, 50.0)])

    manager.save(road_polygons=[unnamed_road])
    manager.save_info_json()

    info = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    assert "spawnPoints" not in info
