"""Erkennung von Junctions in Centerlines: gemeinsame Endpunkte, T-Einmündungen und Kreuzungen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.geometry.junctions import _JunctionIndex
from world_to_beamng.geometry.junctions import detect_junctions_in_centerlines


def _road(points, z=100.0):
    return {"coords": [(float(x), float(y), z) for x, y in points], "id": 0, "name": "r"}


def _with_seed(roads):
    """
    Die T-/Kreuzungs-Erkennung läuft nur, wenn es überhaupt gemeinsame Endpunkte gibt (die Funktion kehrt sonst
    früh zurück) - ein weit entferntes Straßenpaar mit gemeinsamem Endpunkt stellt das sicher.
    """
    seed = [_road(_line(5000, 0, 5050, 0)), _road(_line(5050, 0, 5050, 40))]
    return list(roads) + seed


def _near_origin(junctions):
    return [j for j in junctions if abs(j["position"][0]) < 1000]


def _line(x0, y0, x1, y1, step=5.0):
    """Gerade mit Stützpunkten im Abstand `step`."""
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    count = max(1, int(round(length / step)))
    return [(x0 + (x1 - x0) * i / count, y0 + (y1 - y0) * i / count) for i in range(count + 1)]


def test_no_roads_gives_no_junctions():
    assert detect_junctions_in_centerlines([]) == []


def test_two_roads_sharing_an_endpoint_form_one_junction():
    roads = [_road(_line(0, 0, 50, 0)), _road(_line(50, 0, 50, 40))]

    junctions = detect_junctions_in_centerlines(roads)

    assert len(junctions) == 1
    junction = junctions[0]
    assert junction["road_indices"] == [0, 1]
    assert junction["connection_types"][0][0] == "end" and junction["connection_types"][1][0] == "start"
    assert junction["position"][:2] == pytest.approx((50.0, 0.0))


def test_endpoint_on_a_through_road_is_a_t_junction():
    through = _road(_line(0, 0, 100, 0))
    side = _road(_line(47, 30, 47, 0))  # endet zwischen zwei Stützpunkten der durchgehenden Straße

    junctions = _near_origin(detect_junctions_in_centerlines(_with_seed([through, side])))

    assert len(junctions) == 1
    junction = junctions[0]
    assert junction["road_indices"] == [0, 1]
    assert junction["connection_types"][0] == ["mid"]
    assert junction["connection_types"][1] == ["end"]
    assert junction["position"][:2] == pytest.approx((47.0, 0.0))


def test_several_side_roads_at_one_spot_merge_into_one_junction():
    through = _road(_line(0, 0, 100, 0))
    north = _road(_line(60, 30, 60, 0))
    south = _road(_line(60, -30, 60, 0))

    junctions = _near_origin(detect_junctions_in_centerlines(_with_seed([through, north, south])))

    assert len(junctions) == 1
    assert sorted(junctions[0]["road_indices"]) == [0, 1, 2]
    assert junctions[0]["connection_types"][0] == ["mid"]


def test_crossing_roads_without_shared_vertices_form_an_x_junction():
    roads = [_road(_line(0, 0, 100, 0, step=20)), _road(_line(52, -40, 52, 40, step=20))]

    junctions = _near_origin(detect_junctions_in_centerlines(_with_seed(roads)))

    assert len(junctions) == 1
    assert junctions[0]["connection_types"] == {0: ["mid"], 1: ["mid"]}
    assert junctions[0]["position"][:2] == pytest.approx((52.0, 0.0))


def test_roads_that_do_not_touch_form_no_junction():
    roads = [_road(_line(0, 0, 100, 0)), _road(_line(0, 4, 100, 4))]

    assert _near_origin(detect_junctions_in_centerlines(_with_seed(roads))) == []


def test_near_parallel_roads_within_the_tolerance_meet_between_their_closest_vertices():
    roads = [_road(_line(0, 0, 100, 0)), _road(_line(40, 0.6, 60, 0.6))]

    junctions = _near_origin(detect_junctions_in_centerlines(_with_seed(roads)))

    assert len(junctions) >= 1
    assert all(0.0 <= j["position"][1] <= 0.6 for j in junctions)
    assert all(j["road_indices"] == [0, 1] for j in junctions)


# --- räumlicher Index -----------------------------------------------------------------------------------


def _junction(x, y):
    return {"position": (x, y, 0.0)}


def test_junction_index_finds_the_first_junction_within_the_tolerance():
    junctions = [_junction(10.0, 10.0), _junction(10.4, 10.0), _junction(50.0, 50.0)]
    index = _JunctionIndex(junctions, tolerance=1.0)

    assert index.find(10.2, 10.0) is junctions[0]  # beide im Radius: die erste der Liste
    assert index.find(11.3, 10.0) is junctions[1]  # 1,3 m von #0 (außerhalb), 0,9 m von #1
    assert index.find(30.0, 30.0) is None


def test_junction_index_boundary_is_inclusive_and_sees_junctions_added_later():
    junctions = [_junction(0.0, 0.0)]
    index = _JunctionIndex(junctions, tolerance=1.0)
    assert index.find(1.0, 0.0) is junctions[0]  # genau Toleranz
    assert index.find(1.01, 0.0) is None

    junctions.append(_junction(-5.3, 7.7))  # Zellen mit negativen Koordinaten
    assert index.find(-5.0, 7.5) is junctions[1]
