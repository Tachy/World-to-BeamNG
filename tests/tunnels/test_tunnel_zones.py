"""Zone objects that darken tunnel tubes (tunnels/tunnel_zones.py) - schema as in BeamNG's own levels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.tunnel_portal import plan_tunnels
from world_to_beamng.tunnels.tunnel_zones import plan_tunnel_zones

KW = dict(max_length=50.0, max_deviation=0.5, end_overlap=0.5, width_margin=2.0, height_margin=2.0, portal_inset=1.0, portal_depth=3.0)


def _zones(plans):
    return [o for o in plan_tunnel_zones(plans, **KW) if o["class"] == "Zone"]


def _portals(plans):
    return [o for o in plan_tunnel_zones(plans, **KW) if o["class"] == "Portal"]


def _plans(coords, width=6.5):
    tunnel = {"id": 3, "coords": coords, "width": width, "floor_material": "f"}
    return plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, collar_ratio=0.1, flat_depth=1.5, length=3.5)


def _local_axes(zone):
    m = np.array(zone["rotation_matrix"]).reshape(3, 3)
    return m[0], m[1], m[2]  # BeamNG: ROWS = images of the local x (longitudinal axis), y, z


def test_straight_tunnel_is_covered_by_equal_zones_starting_one_metre_inside():
    plans = _plans([(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)])
    crown, tube_width = plans[0]["crown"], plans[0]["tube_width"]

    zones = _zones(plans)

    assert len(zones) == 2  # 98 m between the insets -> 2 x 49 m
    centers = sorted(z["position"][0] for z in zones)
    assert centers == pytest.approx([25.5, 74.5])
    for zone in zones:
        assert zone["scale"] == pytest.approx([49.0 + 1.0, tube_width + 2.0, crown + 2.0])
        assert zone["position"][2] == pytest.approx(500.0 + crown / 2.0)
        assert zone["rotation_matrix"] == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0, 1])
    # Portal zone stays bright: zones start 1 m behind the portal plane (incl. 0.5 m overlap -> 0.5 m)
    assert min(z["position"][0] - z["scale"][0] / 2.0 for z in zones) == pytest.approx(0.5)
    assert max(z["position"][0] + z["scale"][0] / 2.0 for z in zones) == pytest.approx(99.5)


def test_zone_follows_the_tunnel_grade():
    zones = _zones(_plans([(0.0, 0.0, 500.0), (100.0, 0.0, 510.0)]))

    forward, _, up = _local_axes(zones[0])
    assert forward == pytest.approx(np.array([100.0, 0.0, 10.0]) / np.hypot(100.0, 10.0), abs=1e-6)
    assert up[2] > 0.99  # not tilted, only inclined


def test_zones_follow_a_curved_tunnel_within_the_deviation_limit():
    angles = np.linspace(0.0, np.pi / 2.0, 30)
    coords = [(100.0 * np.sin(a), 100.0 - 100.0 * np.cos(a), 500.0) for a in angles]  # radius 100 m, 157 m long
    plans = _plans(coords)
    centerline = np.array(plans[0]["coords"])[:, :2]

    zones = _zones(plans)

    assert len(zones) > 4  # more than the pure length split, because of the curvature
    for zone in zones:
        forward, side, _ = _local_axes(zone)
        rel = centerline - np.array(zone["position"][:2])
        along = rel @ forward[:2] / np.linalg.norm(forward[:2])
        inside = np.abs(along) <= zone["scale"][0] / 2.0 - 0.5
        lateral = np.abs(rel[inside] @ side[:2] / np.linalg.norm(side[:2]))
        assert lateral.max() <= 0.5 + 1e-6


def test_zone_fields_follow_the_vanilla_schema():
    zone = _zones(_plans([(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)]))[0]

    assert zone["name"].startswith("tunnel_zone_3_")
    assert zone["fields"] == {
        "useAmbientLightColor": True, "ambientLightColor": [0, 0, 0, 1], "skyLightFactor": 0.05, "zoneGroup": 1,
    }


def test_all_zones_of_a_tunnel_share_one_zone_group_and_tunnels_differ():
    # As in the vanilla tunnel (jungle_rock_island: zoneGroup 1): the zones of one tube form a contiguous space
    plans = _plans([(0.0, 0.0, 500.0), (300.0, 0.0, 500.0)])
    plans += plan_tunnels([{"id": 4, "coords": [(0.0, 50.0, 500.0), (100.0, 50.0, 500.0)], "width": 6.5, "floor_material": "f"}],
                          width_margin=1.5, segment_step=10.0, collar_ratio=0.1, flat_depth=1.5, length=3.5)

    zones = _zones(plans)

    groups = {z["name"].split("_")[2]: set() for z in zones}
    for z in zones:
        groups[z["name"].split("_")[2]].add(z["fields"]["zoneGroup"])
    assert all(len(g) == 1 for g in groups.values())
    assert groups["3"] != groups["4"]


def test_each_tunnel_end_gets_a_portal_at_the_zone_face():
    plans = _plans([(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)])
    crown, tube_width = plans[0]["crown"], plans[0]["tube_width"]

    portals = sorted(_portals(plans), key=lambda p: p["position"][0])

    assert len(portals) == 2
    assert [p["position"][0] for p in portals] == pytest.approx([1.0, 99.0])  # at the end face of the zone chain
    for p in portals:
        assert p["position"][2] == pytest.approx(500.0 + crown / 2.0)
        # Vanilla convention: local y axis along the tunnel, dimensions [width, depth, height]
        assert p["scale"] == pytest.approx([tube_width + 2.0, 3.0, crown + 2.0])
        m = np.array(p["rotation_matrix"]).reshape(3, 3)
        assert np.abs(m[1]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)  # row 1 = local y axis along the tunnel
        assert p["fields"] == {}


def test_zone_on_a_diagonal_tunnel_is_aligned_with_the_tube():
    # Row and column only differ on diagonal tubes - here the bug had stayed invisible
    zones = _zones(_plans([(0.0, 0.0, 500.0), (60.0, 60.0, 500.0)]))

    forward, _, _ = _local_axes(zones[0])
    assert forward == pytest.approx([np.sqrt(0.5), np.sqrt(0.5), 0.0], abs=1e-9)
