"""Tests für world_to_beamng.geometry.polygon.smooth_structure_transitions: smooth_roads_xy_only glättet
jede Straße unabhängig und hält ihre Endpunkte fix, wodurch am Übergang zu einer Brücke/Tunnel/Galerie ein
sichtbarer Knick entstehen kann (unterschiedliche Richtung/Gefälle beidseits, u.a. weil die Struktur ein
lineares statt das natürliche DGM-Höhenprofil hat). Dieser Schritt glättet die paar Punkte beidseits eines
eindeutigen Struktur-Übergangs gemeinsam, sodass der gemeinsame Punkt in beiden Straßen identisch bleibt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.polygon import smooth_structure_transitions


def _road(road_id, coords, **tags):
    return {"id": road_id, "coords": coords, "name": "r", "osm_tags": tags}


def test_blends_the_kink_at_a_unique_bridge_transition_and_keeps_the_boundary_shared():
    # A (Oberflächenstraße) läuft flach in den gemeinsamen Punkt P=(10,0,0); B (Brücke) läuft ab P steil
    # weiter (10 Höhenmeter auf 10m) - deutlicher Knick im Höhenprofil genau bei P.
    road_a = _road(1, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], highway="residential")
    road_b = _road(2, [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)], highway="primary", bridge="yes")

    result = smooth_structure_transitions([road_a, road_b], window=1, iterations=1, weight_center=0.5)

    a = next(r for r in result if r["id"] == 1)
    b = next(r for r in result if r["id"] == 2)

    # Chaikin-Schritt von Hand: temp[1] = 0.5*(10,0,0) + 0.25*(0,0,0) + 0.25*(20,0,10) = (10,0,2.5)
    assert a["coords"][-1] == (10.0, 0.0, 2.5)
    assert b["coords"][0] == (10.0, 0.0, 2.5)
    assert a["coords"][-1] == b["coords"][0]  # kein Spalt am Übergang

    # ferne Enden (Anker) bleiben unverändert
    assert a["coords"][0] == (0.0, 0.0, 0.0)
    assert b["coords"][-1] == (20.0, 0.0, 10.0)


def test_blends_a_structure_to_structure_chain_too():
    # Tunnel direkt gefolgt von einer Brücke (kein Oberflächen-Nachbar beteiligt) - auch hier soll geglättet werden.
    road_a = _road(1, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], highway="trunk", tunnel="yes")
    road_b = _road(2, [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)], highway="primary", bridge="yes")

    result = smooth_structure_transitions([road_a, road_b], window=1, iterations=1, weight_center=0.5)

    a = next(r for r in result if r["id"] == 1)
    b = next(r for r in result if r["id"] == 2)

    assert a["coords"][-1] == b["coords"][0]
    assert a["coords"][-1] == (10.0, 0.0, 2.5)


def test_real_multiway_junction_at_a_bridge_end_is_left_untouched():
    # Zwei Oberflächenstraßen A und C berühren beide denselben Brücken-Endpunkt P -> mehrdeutig, keine Glättung.
    road_a = _road(1, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], highway="residential")
    road_c = _road(3, [(10.0, 0.0, 0.0), (5.0, 5.0, 0.0)], highway="residential")
    road_b = _road(2, [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)], highway="primary", bridge="yes")

    result = smooth_structure_transitions([road_a, road_b, road_c], window=1, iterations=1, weight_center=0.5)

    a = next(r for r in result if r["id"] == 1)
    b = next(r for r in result if r["id"] == 2)
    c = next(r for r in result if r["id"] == 3)

    assert a["coords"] == [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    assert b["coords"] == [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)]
    assert c["coords"] == [(10.0, 0.0, 0.0), (5.0, 5.0, 0.0)]


def test_plain_surface_to_surface_junction_is_out_of_scope_and_untouched():
    road_a = _road(1, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], highway="residential")
    road_b = _road(2, [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)], highway="residential")

    result = smooth_structure_transitions([road_a, road_b], window=1, iterations=1, weight_center=0.5)

    a = next(r for r in result if r["id"] == 1)
    b = next(r for r in result if r["id"] == 2)

    assert a["coords"] == [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    assert b["coords"] == [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)]
