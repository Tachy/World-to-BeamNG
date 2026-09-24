"""Tests für world_to_beamng.geometry.polygon.settle_tunnel_portals_to_approach_grade: der OSM-Tunnelanfang
liegt oft schon im Hang (DGM zeigt dort die Portal-Böschung statt Straßenniveau) - Portalhöhe und die letzten
Meter der Zufahrt werden auf die stabile Steigung der Zufahrt weiter draußen gebracht."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.geometry.polygon import settle_tunnel_portals_to_approach_grade

SLOPE = 0.10
STABLE = 6.0
MAX_DIST = 40.0


def _road(road_id, coords, **tags):
    return {"id": road_id, "coords": coords, "name": "r", "osm_tags": tags}


def _approach_with_hump():
    # Zufahrt x=0..30 mit stabilen 5 % Steigung Richtung Tunnel (z = 100 + 0.05 x); die letzten 8 m vor dem
    # Portal (x=22..30) steigen dagegen steil an (Hangflanke über dem Portal, bis +4 m über der Straßen-Linie).
    coords = []
    for x in range(0, 31, 2):
        z = 100.0 + 0.05 * x
        if x > 22:
            z += (x - 22) * 0.5
        coords.append((float(x), 0.0, z))
    return coords


def test_portal_height_follows_the_stable_approach_grade():
    approach = _approach_with_hump()
    portal = approach[-1]
    tunnel = [portal, (60.0, 0.0, 120.0), (100.0, 0.0, 125.0)]
    roads = [_road(1, approach, highway="primary"), _road(2, tunnel, highway="primary", tunnel="yes")]

    settle_tunnel_portals_to_approach_grade(roads, slope_threshold=SLOPE, stable_length=STABLE, max_distance=MAX_DIST)

    # stabile Linie z = 100 + 0.05 x bis ans Portal (x=30) verlängert -> 101.5 statt 105.5
    assert roads[1]["coords"][0] == pytest.approx((30.0, 0.0, 101.5))
    assert roads[1]["coords"][1:] == tunnel[1:]  # Rest des Tunnels unverändert (lineares Profil folgt später)
    for x, _, z in roads[0]["coords"]:
        assert z == pytest.approx(100.0 + 0.05 * x)  # Buckel der Zufahrt auf die stabile Steigung gebracht
    assert roads[0]["coords"][-1] == roads[1]["coords"][0]  # gemeinsamer Grenzpunkt bleibt identisch


def test_portal_on_an_already_stable_approach_is_left_alone():
    approach = [(float(x), 0.0, 100.0 + 0.05 * x) for x in range(0, 31, 2)]
    tunnel = [approach[-1], (60.0, 0.0, 120.0)]
    roads = [_road(1, list(approach), highway="primary"), _road(2, list(tunnel), highway="primary", tunnel="yes")]

    settle_tunnel_portals_to_approach_grade(roads, slope_threshold=SLOPE, stable_length=STABLE, max_distance=MAX_DIST)

    assert roads[0]["coords"] == approach
    assert roads[1]["coords"] == tunnel


def test_portal_is_left_alone_when_the_approach_never_becomes_stable_within_the_limit():
    approach = [(float(x), 0.0, 100.0 + 0.3 * x) for x in range(0, 61, 2)]  # durchgehend 30 %
    tunnel = [approach[-1], (100.0, 0.0, 120.0)]
    roads = [_road(1, list(approach), highway="primary"), _road(2, list(tunnel), highway="primary", tunnel="yes")]

    settle_tunnel_portals_to_approach_grade(roads, slope_threshold=SLOPE, stable_length=STABLE, max_distance=MAX_DIST)

    assert roads[0]["coords"] == approach
    assert roads[1]["coords"] == tunnel


def test_approach_given_in_reverse_direction_and_tunnel_end_are_handled():
    # Zufahrt beginnt am Portal (umgekehrte Laufrichtung), Tunnel endet dort.
    approach = list(reversed(_approach_with_hump()))
    portal = approach[0]
    tunnel = [(100.0, 0.0, 125.0), (60.0, 0.0, 120.0), portal]
    roads = [_road(1, approach, highway="primary"), _road(2, tunnel, highway="primary", tunnel="yes")]

    settle_tunnel_portals_to_approach_grade(roads, slope_threshold=SLOPE, stable_length=STABLE, max_distance=MAX_DIST)

    assert roads[1]["coords"][-1] == pytest.approx((30.0, 0.0, 101.5))
    assert roads[0]["coords"][0] == roads[1]["coords"][-1]
    for x, _, z in roads[0]["coords"]:
        assert z == pytest.approx(100.0 + 0.05 * x)


def test_ambiguous_portal_junction_is_left_alone():
    approach = _approach_with_hump()
    portal = approach[-1]
    other = [portal, (30.0, 20.0, 110.0)]
    tunnel = [portal, (60.0, 0.0, 120.0)]
    roads = [
        _road(1, list(approach), highway="primary"),
        _road(3, list(other), highway="track"),
        _road(2, list(tunnel), highway="primary", tunnel="yes"),
    ]

    settle_tunnel_portals_to_approach_grade(roads, slope_threshold=SLOPE, stable_length=STABLE, max_distance=MAX_DIST)

    assert roads[2]["coords"] == tunnel
