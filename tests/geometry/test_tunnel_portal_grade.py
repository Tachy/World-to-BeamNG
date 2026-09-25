"""Tests for world_to_beamng.geometry.polygon.settle_tunnel_portals_to_approach_grade: the OSM tunnel start
is often already inside the hillside (DGM shows the portal embankment there instead of road level) - the portal height
and the last few meters of the approach are brought to the stable grade of the approach further out."""

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
    # Approach x=0..30 with a stable 5 % grade toward the tunnel (z = 100 + 0.05 x); the last 8 m before the
    # portal (x=22..30), however, rise steeply (hillside above the portal, up to +4 m above the road line).
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

    # stable line z = 100 + 0.05 x extended to the portal (x=30) -> 101.5 instead of 105.5
    assert roads[1]["coords"][0] == pytest.approx((30.0, 0.0, 101.5))
    assert roads[1]["coords"][1:] == tunnel[1:]  # rest of the tunnel unchanged (linear profile follows later)
    for x, _, z in roads[0]["coords"]:
        assert z == pytest.approx(100.0 + 0.05 * x)  # hump of the approach brought to the stable grade
    assert roads[0]["coords"][-1] == roads[1]["coords"][0]  # shared boundary point stays identical


def test_portal_on_an_already_stable_approach_is_left_alone():
    approach = [(float(x), 0.0, 100.0 + 0.05 * x) for x in range(0, 31, 2)]
    tunnel = [approach[-1], (60.0, 0.0, 120.0)]
    roads = [_road(1, list(approach), highway="primary"), _road(2, list(tunnel), highway="primary", tunnel="yes")]

    settle_tunnel_portals_to_approach_grade(roads, slope_threshold=SLOPE, stable_length=STABLE, max_distance=MAX_DIST)

    assert roads[0]["coords"] == approach
    assert roads[1]["coords"] == tunnel


def test_portal_is_left_alone_when_the_approach_never_becomes_stable_within_the_limit():
    approach = [(float(x), 0.0, 100.0 + 0.3 * x) for x in range(0, 61, 2)]  # 30 % throughout
    tunnel = [approach[-1], (100.0, 0.0, 120.0)]
    roads = [_road(1, list(approach), highway="primary"), _road(2, list(tunnel), highway="primary", tunnel="yes")]

    settle_tunnel_portals_to_approach_grade(roads, slope_threshold=SLOPE, stable_length=STABLE, max_distance=MAX_DIST)

    assert roads[0]["coords"] == approach
    assert roads[1]["coords"] == tunnel


def test_approach_given_in_reverse_direction_and_tunnel_end_are_handled():
    # Approach starts at the portal (reversed direction), tunnel ends there.
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
