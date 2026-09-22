"""Tests für world_to_beamng.geometry.polygon.extend_short_bridges_to_natural_grade: zu kurz getaggte
OSM-Brücken (bridge=yes-Way beginnt erst mitten am Hang statt am echten Straßenniveau) werden in die
angrenzende Oberflächenstraße hinein verlängert, bis dort wieder normales Gefälle herrscht."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.geometry.polygon import extend_short_bridges_to_natural_grade

SLOPE = 0.10  # 10 % Schwelle
MAX_EXT = 40.0


def _road(road_id, coords, **tags):
    return {"id": road_id, "coords": coords, "name": "r", "osm_tags": tags}


def test_bridge_is_extended_through_the_steep_approach_until_the_grade_flattens():
    # Nachbarstraße: nahe der Brücke (x=10..20) steil (20 %, die Fortsetzung der Hangflanke, die die zu
    # kurze Brücke nicht abdeckt), weiter weg (x=0..10) flach (2 %, echtes Straßenniveau) - die Brücke
    # soll bis x=10 (dem Punkt, an dem wieder normales Gefälle herrscht) hineinwachsen, nicht weiter.
    neighbor_coords = [
        (0.0, 0.0, 110.0),
        (5.0, 0.0, 109.9),
        (10.0, 0.0, 109.8),
        (15.0, 0.0, 108.8),
        (20.0, 0.0, 107.8),
    ]
    bridge_coords = [(20.0, 0.0, 107.8), (30.0, 0.0, 107.8), (40.0, 0.0, 107.8)]
    roads = [_road(1, neighbor_coords, highway="residential"), _road(2, bridge_coords, highway="primary", bridge="yes")]

    result = extend_short_bridges_to_natural_grade(roads, slope_threshold=SLOPE, max_extension=MAX_EXT)

    bridge = next(r for r in result if r["id"] == 2)
    neighbor = next(r for r in result if r["id"] == 1)

    assert bridge["coords"][0] == (10.0, 0.0, 109.8)  # bis zum Punkt mit Gefälle < 10 % verlängert
    assert bridge["coords"][1:] == [(15.0, 0.0, 108.8), (20.0, 0.0, 107.8)] + bridge_coords[1:]
    assert neighbor["coords"] == [(0.0, 0.0, 110.0), (5.0, 0.0, 109.9), (10.0, 0.0, 109.8)]  # entsprechend gekürzt


def test_bridge_extension_stops_at_the_max_extension_cap_if_the_slope_never_flattens():
    # Durchgehend 20 % Gefälle über 50 m - die Verlängerung darf höchstens max_extension (hier 20 m) weit gehen.
    neighbor_coords = [(float(x), 0.0, 100.0 + 0.2 * (50 - x)) for x in range(0, 51, 5)]
    bridge_coords = [(50.0, 0.0, 100.0), (60.0, 0.0, 100.0)]
    roads = [_road(1, neighbor_coords, highway="residential"), _road(2, bridge_coords, highway="primary", bridge="yes")]

    result = extend_short_bridges_to_natural_grade(roads, slope_threshold=SLOPE, max_extension=20.0)

    bridge = next(r for r in result if r["id"] == 2)
    neighbor = next(r for r in result if r["id"] == 1)

    assert bridge["coords"][0] == pytest.approx((30.0, 0.0, 104.0))  # 20 m von x=50 rückwärts, bei x=30
    assert neighbor["coords"][-1] == pytest.approx((30.0, 0.0, 104.0))


def test_bridge_extends_from_the_far_end_of_a_reversed_neighbor():
    # Nachbarstraße läuft in umgekehrter Reihenfolge (ihr ERSTER Punkt berührt die Brücke) - nahe der
    # Brücke steil, weiter weg flach.
    neighbor_coords = [
        (40.0, 0.0, 99.8),
        (35.0, 0.0, 100.8),
        (30.0, 0.0, 101.8),
        (25.0, 0.0, 101.85),
        (20.0, 0.0, 101.9),
    ]
    bridge_coords = [(50.0, 0.0, 99.8), (45.0, 0.0, 99.8), (40.0, 0.0, 99.8)]
    roads = [_road(1, neighbor_coords, highway="residential"), _road(2, bridge_coords, highway="primary", bridge="yes")]

    result = extend_short_bridges_to_natural_grade(roads, slope_threshold=SLOPE, max_extension=MAX_EXT)

    bridge = next(r for r in result if r["id"] == 2)
    neighbor = next(r for r in result if r["id"] == 1)

    assert bridge["coords"][-1] == (30.0, 0.0, 101.8)
    assert neighbor["coords"] == [(30.0, 0.0, 101.8), (25.0, 0.0, 101.85), (20.0, 0.0, 101.9)]


def test_no_extension_when_the_approach_is_already_flat():
    neighbor_coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 99.95), (20.0, 0.0, 99.9)]
    bridge_coords = [(20.0, 0.0, 99.9), (30.0, 0.0, 99.9)]
    roads = [_road(1, neighbor_coords, highway="residential"), _road(2, bridge_coords, highway="primary", bridge="yes")]

    result = extend_short_bridges_to_natural_grade(roads, slope_threshold=SLOPE, max_extension=MAX_EXT)

    bridge = next(r for r in result if r["id"] == 2)
    neighbor = next(r for r in result if r["id"] == 1)

    assert bridge["coords"] == bridge_coords
    assert neighbor["coords"] == neighbor_coords


def test_no_neighbor_found_leaves_the_bridge_unchanged():
    bridge_coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 100.0)]
    roads = [_road(2, bridge_coords, highway="primary", bridge="yes")]

    result = extend_short_bridges_to_natural_grade(roads, slope_threshold=SLOPE, max_extension=MAX_EXT)

    assert result[0]["coords"] == bridge_coords


def test_does_not_extend_into_another_structure_road():
    # Der "Nachbar" ist selbst ein Tunnel - dessen Höhenprofil ist kein echtes Gelände, also nicht hineinlaufen.
    tunnel_coords = [(0.0, 0.0, 90.0), (10.0, 0.0, 95.0), (20.0, 0.0, 100.0)]
    bridge_coords = [(20.0, 0.0, 100.0), (30.0, 0.0, 100.0)]
    roads = [_road(1, tunnel_coords, highway="trunk", tunnel="yes"), _road(2, bridge_coords, highway="primary", bridge="yes")]

    result = extend_short_bridges_to_natural_grade(roads, slope_threshold=SLOPE, max_extension=MAX_EXT)

    bridge = next(r for r in result if r["id"] == 2)
    tunnel = next(r for r in result if r["id"] == 1)

    assert bridge["coords"] == bridge_coords
    assert tunnel["coords"] == tunnel_coords


def test_both_ends_can_extend_independently():
    left_neighbor = [(-20.0, 0.0, 102.05), (-10.0, 0.0, 102.0), (0.0, 0.0, 100.0)]
    right_neighbor = [(20.0, 0.0, 100.0), (30.0, 0.0, 103.0), (40.0, 0.0, 103.05)]
    bridge_coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (20.0, 0.0, 100.0)]
    roads = [
        _road(1, left_neighbor, highway="residential"),
        _road(2, bridge_coords, highway="primary", bridge="yes"),
        _road(3, right_neighbor, highway="residential"),
    ]

    result = extend_short_bridges_to_natural_grade(roads, slope_threshold=SLOPE, max_extension=MAX_EXT)

    bridge = next(r for r in result if r["id"] == 2)
    assert bridge["coords"][0] == (-10.0, 0.0, 102.0)
    assert bridge["coords"][-1] == (30.0, 0.0, 103.0)


def test_extension_stops_at_the_neighbors_own_far_end_without_reaching_into_a_third_road():
    # Nachbar ist selbst nur 15 m lang und komplett steil (20 %) - Verlängerung darf trotz max_extension=40
    # nicht über den Nachbarn hinaus in eine dritte Straße reichen.
    neighbor_coords = [(5.0, 0.0, 103.0), (10.0, 0.0, 101.0), (15.0, 0.0, 100.0), (20.0, 0.0, 99.0)]
    third_road_coords = [(0.0, 0.0, 103.05), (5.0, 0.0, 103.0)]
    bridge_coords = [(20.0, 0.0, 99.0), (30.0, 0.0, 99.0)]
    roads = [
        _road(0, third_road_coords, highway="residential"),
        _road(1, neighbor_coords, highway="residential"),
        _road(2, bridge_coords, highway="primary", bridge="yes"),
    ]

    result = extend_short_bridges_to_natural_grade(roads, slope_threshold=SLOPE, max_extension=MAX_EXT)

    bridge = next(r for r in result if r["id"] == 2)
    third_road = next(r for r in result if r["id"] == 0)

    assert bridge["coords"][0] == (5.0, 0.0, 103.0)  # der ganze (kurze) Nachbar wurde übernommen
    assert bridge["coords"][1:] == neighbor_coords[1:] + bridge_coords[1:]
    assert third_road["coords"] == third_road_coords  # unangetastet - keine dritte Straße betroffen
