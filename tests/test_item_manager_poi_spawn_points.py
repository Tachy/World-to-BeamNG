"""ItemManager: additional spawn points selectable in BeamNG's vehicle selection - one SpawnSphere per
POI (village/town or large parking lot, see osm/poi_points.py), supplementing the automatic default spawn."""

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
    """A road along the x axis (-500..500, height 100) - close enough to all POIs of the tests below."""
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
    assert len(object_names) == 2  # both slugify to "spawn_weg" - must still stay unique


def test_duplicate_display_names_are_numbered(manager):
    pois = [_poi("Parkplatz", kind="parking", rank=300), _poi("Parkplatz", kind="parking", rank=200)]

    spawns = _spawns(manager, pois)

    display_names = sorted(s["display_name"] for s in spawns)
    assert display_names == ["Parkplatz", "Parkplatz 2"]


def test_result_is_capped_at_max_points_keeping_the_highest_ranked(manager):
    pois = [_poi(f"Weiler{i}", kind="place", rank=i) for i in range(5)]

    spawns = _spawns(manager, pois, max_points=1)

    assert len(spawns) == 1 and spawns[0]["display_name"] == "Weiler4"  # highest rank wins


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

    assert calls == [("spawn_andermatt", pytest.approx((5.0, 0.0)))]  # placed on the road
    assert spawns[0]["preview"] == "spawn_previews/spawn_andermatt.jpg"


def test_without_a_preview_builder_preview_is_none(manager):
    spawns = _spawns(manager, [_poi("Andermatt")])

    assert spawns[0]["preview"] is None


# --- save() integration --------------------------------------------------------------------------------


def test_save_writes_poi_spawn_spheres_after_the_default(manager, tmp_path):
    manager.save(road_polygons=_roads_along_x(), poi_points=[_poi("Andermatt", position=(100.0, 100.0, 200.0))])

    lines = (tmp_path / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json").read_text(encoding="utf-8").splitlines()
    objects = [json.loads(line) for line in lines]

    assert objects[0]["name"] == "spawn"  # default spawn stays first
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


# --- Place spawn on the nearest road, heading parallel to the centerline ---

import numpy as np

from world_to_beamng import config


def _road(points, highway="secondary", structure="surface"):
    return {
        "trimmed_centerline": np.array(points, dtype=float),
        "osm_tags": {"highway": highway},
        "structure_type": structure,
    }


def _heading_matrix(dx, dy):
    # like _compute_vehicle_spawn(): BeamNG stores the images of the local axes in the ROWS (derived from vanilla
    # spawn points on diagonal roads: 20 of 24). The vehicle front lies on local -Y (jbeam
    # convention, confirmed in game: with row 1 = driving direction the car faced backwards) -> row 1 = -(dx, dy)
    return [-dy, dx, 0.0, -dx, -dy, 0.0, 0.0, 0.0, 1.0]


def test_poi_spawn_is_moved_onto_the_nearest_road_with_heading_along_it(manager):
    roads = [_road([(0.0, 0.0, 100.0), (100.0, 0.0, 110.0)])]
    pois = [_poi("Ort", position=(30.0, 20.0, 50.0))]

    spawns = manager._compute_poi_spawn_points(pois, road_polygons=roads)

    assert spawns[0]["position"] == pytest.approx([30.0, 0.0, 103.3])  # foot of the perpendicular, height along the road
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


def _nearest_road_pose_reference(road_polygons, target_xy, max_distance):
    """Earlier version (loop per road) - reference for the vectorized _nearest_road_pose()."""
    import numpy as np
    from world_to_beamng import config

    target = np.asarray(target_xy, dtype=float)[:2]
    best, best_dist = None, float(max_distance)
    for road in road_polygons or []:
        if road.get("structure_type", "surface") == "tunnel":
            continue
        if (road.get("osm_tags") or {}).get("highway") in config.POI_SPAWN_EXCLUDED_HIGHWAYS:
            continue
        centerline = road.get("trimmed_centerline")
        if centerline is None or len(centerline) < 2:
            continue
        coords = np.asarray(centerline, dtype=float)
        starts, ends = coords[:-1], coords[1:]
        seg = ends[:, :2] - starts[:, :2]
        seg_len_sq = np.einsum("ij,ij->i", seg, seg)
        valid = seg_len_sq > 1e-12
        if not valid.any():
            continue
        t = np.zeros(len(seg))
        t[valid] = np.clip(np.einsum("ij,ij->i", target - starts[valid, :2], seg[valid]) / seg_len_sq[valid], 0.0, 1.0)
        foot = starts + t[:, None] * (ends - starts)
        dist = np.hypot(foot[:, 0] - target[0], foot[:, 1] - target[1])
        dist[~valid] = np.inf
        i = int(np.argmin(dist))
        if dist[i] < best_dist:
            best_dist = float(dist[i])
            best = (tuple(float(v) for v in foot[i]), tuple(float(v) for v in seg[i] / np.sqrt(seg_len_sq[i])))
    return best


def test_nearest_road_pose_matches_the_per_road_reference_loop():
    import numpy as np

    rng = np.random.default_rng(7)
    highways = ["primary", "residential", "footway", "track", "service"]
    roads = []
    for k in range(200):
        n = int(rng.integers(1, 12))
        points = np.cumsum(rng.normal(0.0, 30.0, size=(n, 3)), axis=0) + rng.uniform(-1500, 1500, size=3)
        if k % 17 == 0 and n > 2:
            points[1] = points[0]  # zero-length segment
        roads.append({
            "trimmed_centerline": points,
            "osm_tags": {"highway": highways[k % len(highways)]},
            "structure_type": "tunnel" if k % 13 == 0 else "surface",
        })
    segments = ItemManager._spawn_road_segments(roads)

    for target in rng.uniform(-1800, 1800, size=(300, 2)):
        expected = _nearest_road_pose_reference(roads, target, 150.0)
        assert ItemManager._nearest_road_pose(roads, target, 150.0, segments) == expected
        assert ItemManager._nearest_road_pose(roads, target, 150.0) == expected


def test_nearest_road_pose_without_usable_roads_is_none():
    assert ItemManager._nearest_road_pose([], (0.0, 0.0), 100.0) is None
    tunnel = {"trimmed_centerline": [[0, 0, 0], [10, 0, 0]], "structure_type": "tunnel"}
    assert ItemManager._nearest_road_pose([tunnel], (0.0, 0.0), 100.0) is None
