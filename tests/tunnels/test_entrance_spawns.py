"""Selectable spawn points in front of the entrances of tunnel chains (tunnels/entrance_spawns.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.road_structures import classify_structure
from world_to_beamng.tunnels.entrance_spawns import plan_entrance_spawns

KW = dict(distance=20.0, excluded_highways={"path", "footway", "steps", "cycleway", "bridleway"})


def _road(road_id, xs, z=100.0, **tags):
    tags = {"highway": "primary", **tags}
    return {"road_id": road_id, "trimmed_centerline": np.array([(float(x), 0.0, z) for x in xs]), "osm_tags": tags,
            "structure_type": classify_structure(tags)}


def test_both_entrances_of_a_tunnel_get_a_spawn_facing_into_it():
    roads = [
        _road(1, range(-100, 1, 10)),
        _road(2, range(0, 201, 10), tunnel="yes", **{"tunnel:name": "Tunnel Fieud"}),
        _road(3, range(200, 301, 10)),
    ]

    spawns = sorted(plan_entrance_spawns(roads, **KW), key=lambda s: s["position"][0])

    assert [s["position"][0] for s in spawns] == pytest.approx([-20.0, 220.0])
    assert spawns[0]["heading"] == pytest.approx((1.0, 0.0)) and spawns[1]["heading"] == pytest.approx((-1.0, 0.0))
    assert all(s["position"][2] == pytest.approx(100.0) for s in spawns)
    assert all(s["name"] == "Tunnel Fieud (Einfahrt)" for s in spawns)


def test_entrance_at_a_gallery_of_the_chain_uses_the_gallery_name():
    roads = [
        _road(1, range(-100, 1, 10)),
        _road(2, range(0, 51, 10), covered="yes", layer="-1", **{"tunnel:name": "Galleria artificiale Banchi"}),
        _road(3, range(50, 101, 10), tunnel="yes", **{"tunnel:name": "Tunnel Banchi"}),
        _road(4, range(100, 201, 10)),
    ]

    names = sorted(s["name"] for s in plan_entrance_spawns(roads, **KW))

    assert names == ["Galleria artificiale Banchi (Einfahrt)", "Tunnel Banchi (Einfahrt)"]


def test_chain_without_a_tunnel_gets_no_spawn():
    roads = [_road(1, range(-100, 1, 10)), _road(2, range(0, 301, 10), covered="yes", layer="-1"), _road(3, range(300, 401, 10))]

    assert plan_entrance_spawns(roads, **KW) == []


def test_footpath_tunnel_gets_no_spawn():
    roads = [_road(1, range(-100, 1, 10), highway="path"), _road(2, range(0, 201, 10), highway="path", tunnel="yes")]

    assert plan_entrance_spawns(roads, **KW) == []


def test_short_approach_puts_the_spawn_at_its_far_end():
    roads = [_road(1, [-8.0, 0.0]), _road(2, range(0, 201, 10), tunnel="yes", name="Nuova strada")]

    spawns = plan_entrance_spawns(roads, **KW)

    assert len(spawns) == 1 and spawns[0]["position"][0] == pytest.approx(-8.0)
    assert spawns[0]["name"] == "Nuova strada (Einfahrt)"


def test_unnamed_underpass_gets_no_spawn():
    # Short track underpasses (highway=track, tunnel=yes without a name) are not a tunnel chain to drive up to
    roads = [_road(1, range(-100, 1, 10), highway="track"), _road(2, [0.0, 15.0], highway="track", tunnel="yes"),
             _road(3, range(15, 116, 10), highway="track")]

    assert plan_entrance_spawns(roads, **KW) == []
