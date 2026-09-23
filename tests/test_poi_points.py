"""Tests für world_to_beamng.osm.poi_points: Orts- und Parkplatz-POI-Kandidaten für Spawn-Punkte."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.osm.poi_points import extract_parking_points, extract_place_points


def _pt(x, y):
    # Identität statt WGS84: to_local() in den Tests ist (lon, lat) -> (x, y)
    return {"lon": float(x), "lat": float(y)}


def _to_local(points):
    return [(p["lon"], p["lat"]) for p in points]


def _node(tags, lon, lat, node_id=1):
    return {"type": "node", "id": node_id, "lat": lat, "lon": lon, "tags": tags}


def _ring(*xy):
    return [_pt(x, y) for x, y in xy]


def _way(tags, coords, way_id=1):
    return {"type": "way", "id": way_id, "tags": tags, "geometry": _ring(*coords)}


SQUARE_10 = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]  # 100 m²
SQUARE_100 = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]  # 10000 m²


# --- extract_place_points -------------------------------------------------------------------------------


def test_named_village_becomes_a_place_point():
    osm_data = [_node({"place": "village", "name": "Hospental"}, lon=8.56, lat=46.62)]

    result = extract_place_points(osm_data, _to_local)

    assert len(result) == 1
    assert result[0]["name"] == "Hospental"
    assert result[0]["position_xy"] == (8.56, 46.62)
    assert result[0]["kind"] == "place"


def test_unnamed_place_is_skipped():
    osm_data = [_node({"place": "hamlet"}, lon=8.56, lat=46.62)]

    assert extract_place_points(osm_data, _to_local) == []


def test_place_value_without_a_known_rank_is_skipped():
    osm_data = [_node({"place": "island", "name": "Irgendwo"}, lon=8.56, lat=46.62)]

    assert extract_place_points(osm_data, _to_local) == []


def test_city_ranks_higher_than_hamlet():
    osm_data = [
        _node({"place": "hamlet", "name": "Kleinweiler"}, lon=0, lat=0, node_id=1),
        _node({"place": "city", "name": "Grossstadt"}, lon=1, lat=1, node_id=2),
    ]

    result = extract_place_points(osm_data, _to_local)
    by_name = {p["name"]: p["rank"] for p in result}

    assert by_name["Grossstadt"] > by_name["Kleinweiler"]


def test_non_place_nodes_and_ways_are_ignored():
    osm_data = [_node({"amenity": "bench"}, lon=0, lat=0), _way({"place": "village", "name": "Fake"}, SQUARE_10)]

    assert extract_place_points(osm_data, _to_local) == []


# --- extract_parking_points -----------------------------------------------------------------------------


def test_large_named_parking_lot_is_found():
    osm_data = [_way({"amenity": "parking", "name": "Talstation"}, SQUARE_100)]

    result = extract_parking_points(osm_data, _to_local, min_area_m2=500.0)

    assert len(result) == 1
    assert result[0]["name"] == "Talstation"
    assert result[0]["kind"] == "parking"
    assert result[0]["position_xy"] == (50.0, 50.0)  # Zentroid des Quadrats
    assert result[0]["rank"] == 10000.0  # Fläche in m²


def test_small_parking_lot_is_filtered_out():
    osm_data = [_way({"amenity": "parking"}, SQUARE_10)]  # 100 m²

    assert extract_parking_points(osm_data, _to_local, min_area_m2=500.0) == []


def test_unnamed_parking_lot_falls_back_to_a_generic_name():
    osm_data = [_way({"amenity": "parking"}, SQUARE_100)]

    result = extract_parking_points(osm_data, _to_local, min_area_m2=500.0)

    assert result[0]["name"] == "Parkplatz"


def test_non_parking_amenity_is_ignored():
    osm_data = [_way({"amenity": "school", "name": "Schule"}, SQUARE_100)]

    assert extract_parking_points(osm_data, _to_local, min_area_m2=500.0) == []
