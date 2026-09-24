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


def _roads_along_x():
    """Eine Straße entlang der x-Achse (-500..500, Höhe 100) - nah genug an allen POIs der Tests unten."""
    return [_road([(-500.0, 0.0, 100.0), (500.0, 0.0, 100.0)])]


def _spawns(manager, pois, **kwargs):
    kwargs.setdefault("road_polygons", _roads_along_x())
    return manager._compute_poi_spawn_points(pois, **kwargs)


def test_one_spawn_point_per_poi(manager):
    pois = [_poi("Hospental", kind="place", rank=2), _poi("Andermatt", kind="place", rank=3)]

    spawns = _spawns(manager, pois)

    assert {s["display_name"] for s in spawns} == {"Hospental", "Andermatt"}
    assert {s["object_name"] for s in spawns} == {"spawn_hospental", "spawn_andermatt"}


def test_places_are_ranked_before_parking_lots_regardless_of_rank_value(manager):
    pois = [_poi("Riesenparkplatz", kind="parking", rank=1_000_000.0), _poi("Kleinweiler", kind="place", rank=1)]

    spawns = _spawns(manager, pois)

    assert spawns[0]["display_name"] == "Kleinweiler"
    assert spawns[1]["display_name"] == "Riesenparkplatz"


def test_within_the_same_kind_higher_rank_wins(manager):
    pois = [_poi("Dorf", kind="place", rank=2), _poi("Stadt", kind="place", rank=5), _poi("Weiler", kind="place", rank=1)]

    spawns = _spawns(manager, pois)

    assert [s["display_name"] for s in spawns] == ["Stadt", "Dorf", "Weiler"]


def test_object_names_are_deduplicated_when_slugs_collide(manager):
    pois = [_poi("Weg!"), _poi("Weg?")]

    spawns = _spawns(manager, pois)

    object_names = {s["object_name"] for s in spawns}
    assert len(object_names) == 2  # beide slugifizieren zu "spawn_weg" - müssen trotzdem eindeutig bleiben


def test_duplicate_display_names_are_numbered(manager):
    pois = [_poi("Parkplatz", kind="parking", rank=300), _poi("Parkplatz", kind="parking", rank=200)]

    spawns = _spawns(manager, pois)

    display_names = sorted(s["display_name"] for s in spawns)
    assert display_names == ["Parkplatz", "Parkplatz 2"]


def test_result_is_capped_at_max_points_keeping_the_highest_ranked(manager):
    pois = [_poi(f"Weiler{i}", kind="place", rank=i) for i in range(5)]

    spawns = _spawns(manager, pois, max_points=1)

    assert len(spawns) == 1 and spawns[0]["display_name"] == "Weiler4"  # höchster rank gewinnt


@pytest.mark.parametrize("roads", [None, []])
def test_without_road_data_there_are_no_poi_spawns(manager, roads):
    pois = [_poi("Ort", position=(10.0, 20.0, 100.0))]

    assert _spawns(manager, pois, road_polygons=roads) == []


def test_save_without_road_data_writes_only_the_default_spawn(manager, tmp_path):
    manager.save(road_polygons=None, poi_points=[_poi("Andermatt")])

    lines = (tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["name"] for line in lines] == ["spawn"]


def test_no_poi_points_returns_an_empty_list(manager):
    assert _spawns(manager, None) == []
    assert _spawns(manager, []) == []


def test_preview_builder_is_called_with_object_name_and_xy(manager):
    calls = []

    def preview_builder(object_name, xy):
        calls.append((object_name, xy))
        return f"spawn_previews/{object_name}.jpg"

    spawns = _spawns(manager, [_poi("Andermatt", position=(5.0, 6.0, 100.0))], preview_builder=preview_builder)

    assert calls == [("spawn_andermatt", pytest.approx((5.0, 0.0)))]  # auf die Straße gesetzt
    assert spawns[0]["preview"] == "spawn_previews/spawn_andermatt.jpg"


def test_without_a_preview_builder_preview_is_none(manager):
    spawns = _spawns(manager, [_poi("Andermatt")])

    assert spawns[0]["preview"] is None


# --- save() Integration --------------------------------------------------------------------------------


def test_save_writes_poi_spawn_spheres_after_the_default(manager, tmp_path):
    manager.save(road_polygons=_roads_along_x(), poi_points=[_poi("Andermatt", position=(100.0, 100.0, 200.0))])

    lines = (tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8").splitlines()
    objects = [json.loads(line) for line in lines]

    assert objects[0]["name"] == "spawn"  # Standard-Spawn bleibt zuerst
    assert objects[1]["name"] == "spawn_andermatt"
    assert objects[1]["parentId"] == "PlayerDropPoints"
    assert objects[1]["class"] == "SpawnSphere"


def test_save_declares_spawn_points_in_info_json_when_pois_exist(manager, tmp_path):
    manager.save(road_polygons=_roads_along_x(), poi_points=[_poi("Andermatt", position=(0.0, 0.0, 100.0))])
    manager.save_info_json()

    info = json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))
    assert info["spawnPoints"][0] == {"objectname": "spawn"}
    assert {"objectname": "spawn_andermatt", "name": "Andermatt"} in info["spawnPoints"]


def test_save_includes_the_preview_path_when_a_builder_is_given(manager, tmp_path):
    manager.save(
        road_polygons=_roads_along_x(),
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


# --- Spawn auf die nächste Straße setzen, Heading parallel zur Centerline ---

import numpy as np

from world_to_beamng import config


def _road(points, highway="secondary", structure="surface"):
    return {
        "trimmed_centerline": np.array(points, dtype=float),
        "osm_tags": {"highway": highway},
        "structure_type": structure,
    }


def _heading_matrix(dx, dy):
    # wie _compute_vehicle_spawn(): BeamNG speichert die Bilder der lokalen Achsen in den ZEILEN (aus Vanilla-
    # Spawnpunkten auf diagonalen Straßen abgeleitet: 20 von 24). Die Fahrzeugfront liegt auf lokal -Y (jbeam-
    # Konvention, im Spiel bestätigt: mit Zeile 1 = Fahrtrichtung schaute das Auto rückwärts) -> Zeile 1 = -(dx, dy)
    return [-dy, dx, 0.0, -dx, -dy, 0.0, 0.0, 0.0, 1.0]


def test_poi_spawn_is_moved_onto_the_nearest_road_with_heading_along_it(manager):
    roads = [_road([(0.0, 0.0, 100.0), (100.0, 0.0, 110.0)])]
    pois = [_poi("Ort", position=(30.0, 20.0, 50.0))]

    spawns = manager._compute_poi_spawn_points(pois, road_polygons=roads)

    assert spawns[0]["position"] == pytest.approx([30.0, 0.0, 103.3])  # Lotfußpunkt, Höhe entlang der Straße
    assert spawns[0]["rotationMatrix"] == pytest.approx(_heading_matrix(1.0, 0.0))


def test_heading_follows_the_segment_the_spawn_lands_on(manager):
    roads = [_road([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)])]
    pois = [_poi("Ort", position=(14.0, 6.0, 0.0))]

    spawns = manager._compute_poi_spawn_points(pois, road_polygons=roads)

    assert spawns[0]["position"][:2] == pytest.approx([10.0, 6.0])
    assert spawns[0]["rotationMatrix"] == pytest.approx(_heading_matrix(0.0, 1.0))


def test_nearest_of_several_roads_wins(manager):
    roads = [_road([(0.0, 50.0, 0.0), (100.0, 50.0, 0.0)]), _road([(0.0, 10.0, 5.0), (100.0, 10.0, 5.0)])]
    pois = [_poi("Ort", position=(40.0, 0.0, 0.0))]

    spawns = manager._compute_poi_spawn_points(pois, road_polygons=roads)

    assert spawns[0]["position"] == pytest.approx([40.0, 10.0, 5.3])


@pytest.mark.parametrize(
    "excluded", [_road([(0.0, 2.0, 0.0), (100.0, 2.0, 0.0)], structure="tunnel"),
                 _road([(0.0, 2.0, 0.0), (100.0, 2.0, 0.0)], highway="footway"),
                 _road([(0.0, 2.0, 0.0), (100.0, 2.0, 0.0)], highway="path")]
)
def test_tunnels_and_non_car_ways_are_never_spawn_roads(manager, excluded):
    roads = [excluded, _road([(0.0, 30.0, 7.0), (100.0, 30.0, 7.0)])]
    pois = [_poi("Ort", position=(50.0, 0.0, 0.0))]

    spawns = manager._compute_poi_spawn_points(pois, road_polygons=roads)

    assert spawns[0]["position"] == pytest.approx([50.0, 30.0, 7.3])


def test_poi_without_a_road_nearby_is_left_out(manager):
    far = config.POI_SPAWN_MAX_ROAD_DISTANCE + 50.0
    roads = [_road([(0.0, far, 0.0), (100.0, far, 0.0)])]
    pois = [_poi("Abseits", position=(50.0, 0.0, 100.0)), _poi("Dorf", position=(50.0, far - 10.0, 0.0))]

    spawns = manager._compute_poi_spawn_points(pois, road_polygons=roads)

    assert [s["display_name"] for s in spawns] == ["Dorf"]


def test_poi_with_only_footpaths_nearby_is_left_out(manager):
    roads = [_road([(0.0, 5.0, 0.0), (100.0, 5.0, 0.0)], highway="path")]

    assert manager._compute_poi_spawn_points([_poi("Alp", position=(50.0, 0.0, 0.0))], road_polygons=roads) == []


def test_left_out_pois_do_not_count_against_the_cap(manager):
    roads = [_road([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)])]
    far_y = config.POI_SPAWN_MAX_ROAD_DISTANCE + 50.0
    pois = [_poi("Abseits", position=(50.0, far_y, 0.0), rank=9), _poi("Dorf", position=(50.0, 5.0, 0.0), rank=1)]

    spawns = manager._compute_poi_spawn_points(pois, max_points=1, road_polygons=roads)

    assert [s["display_name"] for s in spawns] == ["Dorf"]


def test_preview_is_centred_on_the_snapped_spawn(manager):
    roads = [_road([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)])]
    calls = []

    manager._compute_poi_spawn_points(
        [_poi("Ort", position=(30.0, 20.0, 0.0))],
        road_polygons=roads,
        preview_builder=lambda name, xy: calls.append(xy),
    )

    assert calls == [pytest.approx((30.0, 0.0))]


def test_save_snaps_poi_spawn_spheres_onto_roads(manager, tmp_path):
    roads = [_road([(0.0, 0.0, 100.0), (100.0, 0.0, 100.0)])]
    manager.save(road_polygons=roads, poi_points=[_poi("Airolo", position=(30.0, 20.0, 50.0))])

    lines = (tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8")
    sphere = next(json.loads(l) for l in lines.splitlines() if '"spawn_airolo"' in l)
    assert sphere["position"] == pytest.approx([30.0, 0.0, 100.3])
    assert sphere["rotationMatrix"] == pytest.approx(_heading_matrix(1.0, 0.0))
