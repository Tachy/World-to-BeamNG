"""Tests for world_to_beamng.geometry.polygon.apply_structure_elevation_profiles: bridges/tunnels/galleries
get a linear elevation profile between their endpoints instead of the raw DGM height at every point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.geometry.polygon import apply_structure_elevation_profiles


def _road(coords, **tags):
    return {"id": 1, "coords": coords, "name": "r", "osm_tags": tags}


def test_surface_roads_keep_their_raw_elevation():
    coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 150.0), (20.0, 0.0, 90.0)]
    roads = [_road(coords, highway="residential")]

    result = apply_structure_elevation_profiles(roads)

    assert result[0]["coords"] == coords


def test_bridge_gets_a_linear_profile_between_its_endpoints():
    # Raw DGM height in the middle would be the valley floor (50 m) - the bridge should instead interpolate
    # smoothly between the two connection points (100 m).
    coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 50.0), (20.0, 0.0, 100.0)]
    roads = [_road(coords, highway="primary", bridge="yes")]

    result = apply_structure_elevation_profiles(roads)

    assert result[0]["coords"][0] == (0.0, 0.0, 100.0)
    assert result[0]["coords"][-1] == (20.0, 0.0, 100.0)
    assert result[0]["coords"][1][2] == pytest.approx(100.0)  # midpoint: no longer the valley floor


def test_tunnel_profile_is_weighted_by_arc_length_not_point_count():
    # Unevenly spaced points: at 0m, 10m, 40m (total length 40m) end heights 100m/180m
    coords = [(0.0, 0.0, 100.0), (10.0, 0.0, 999.0), (40.0, 0.0, 180.0)]
    roads = [_road(coords, highway="trunk", tunnel="yes")]

    result = apply_structure_elevation_profiles(roads)

    # At 10m of 40m total length: 100 + (10/40)*(180-100) = 120
    assert result[0]["coords"][1][2] == pytest.approx(120.0)


def test_gallery_profile_also_gets_linearised():
    coords = [(0.0, 0.0, 200.0), (5.0, 0.0, 500.0), (10.0, 0.0, 210.0)]
    roads = [_road(coords, highway="primary", tunnel="avalanche_protector")]

    result = apply_structure_elevation_profiles(roads)

    assert result[0]["coords"][1][2] == pytest.approx(205.0)


def test_degenerate_zero_length_road_is_left_unchanged():
    coords = [(5.0, 5.0, 100.0), (5.0, 5.0, 100.0)]
    roads = [_road(coords, bridge="yes")]

    result = apply_structure_elevation_profiles(roads)

    assert result[0]["coords"] == coords


def _bridge(coords, road_id=1):
    return {"id": road_id, "coords": coords, "name": "b", "osm_tags": {"highway": "primary", "bridge": "yes"}}


def _surface(coords, road_id=100):
    return {"id": road_id, "coords": coords, "name": "s", "osm_tags": {"highway": "primary"}}


def test_a_node_shared_only_by_bridges_gets_the_deck_height_not_the_valley_floor():
    # Four bridges meet in the middle of a valley (raw DGM 50 m there); their outer ends are abutments at 100 m, each
    # 20 m away - the deck at the shared node lies at 100 m.
    center = (0.0, 0.0, 50.0)
    ends = [(20.0, 0.0, 100.0), (-20.0, 0.0, 100.0), (0.0, 20.0, 100.0), (0.0, -20.0, 100.0)]
    roads = [_bridge([center, (e[0] / 2, e[1] / 2, 50.0), e], road_id=i) for i, e in enumerate(ends)]
    roads += [_surface([e, (e[0] * 2, e[1] * 2, 100.0)], road_id=100 + i) for i, e in enumerate(ends)]

    apply_structure_elevation_profiles(roads)

    for bridge in roads[:4]:
        assert bridge["coords"][0][2] == pytest.approx(100.0)
        assert bridge["coords"][1][2] == pytest.approx(100.0)


def test_the_shared_node_is_weighted_by_the_length_to_each_abutment():
    # Two bridges in a row, joint at raw 0 m: 10 m to an abutment at 100 m, 30 m to one at 140 m -> linear over 40 m
    first = _bridge([(-10.0, 0.0, 100.0), (0.0, 0.0, 0.0)], road_id=1)
    second = _bridge([(0.0, 0.0, 0.0), (15.0, 0.0, 0.0), (30.0, 0.0, 140.0)], road_id=2)
    roads = [first, second, _surface([(-20.0, 0.0, 100.0), (-10.0, 0.0, 100.0)]), _surface([(30.0, 0.0, 140.0), (40.0, 0.0, 140.0)], 101)]

    apply_structure_elevation_profiles(roads)

    assert first["coords"][-1][2] == pytest.approx(110.0)
    assert second["coords"][1][2] == pytest.approx(125.0)
    assert second["coords"][-1][2] == pytest.approx(140.0)


def test_a_bridge_end_at_a_surface_road_keeps_its_terrain_height():
    bridge = _bridge([(0.0, 0.0, 80.0), (20.0, 0.0, 30.0), (40.0, 0.0, 90.0)])
    roads = [bridge, _surface([(40.0, 0.0, 90.0), (40.0, 30.0, 95.0)])]

    apply_structure_elevation_profiles(roads)

    assert bridge["coords"][0][2] == pytest.approx(80.0)  # free end
    assert bridge["coords"][-1][2] == pytest.approx(90.0)
    assert bridge["coords"][1][2] == pytest.approx(85.0)
