"""Tests for world_to_beamng.tunnels.tunnel_lights: SpotLight fixtures along tunnel tubes (vanilla italy.zip pattern:
one shared preset per light, mounted near the crown, aimed straight down)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.tunnel_lights import plan_tunnel_lights
from world_to_beamng.tunnels.tunnel_portal import plan_tunnels

FIELDS = {
    "brightness": 4, "color": [1, 0.62263, 0.35778, 1], "innerAngle": 100, "outerAngle": 160, "intensity": 20000,
    "range": 15, "castShadows": True, "useColorTemperature": "true",
}
KW = dict(spacing=11.0, start_inset=5.0, ceiling_margin=0.3, fields=FIELDS)


def _plans(coords, width=6.5):
    tunnel = {"id": 7, "coords": coords, "width": width, "floor_material": "f"}
    return plan_tunnels([tunnel], segment_step=10.0, collar_ratio=0.1, flat_depth=1.5, length=3.5)


def test_lights_are_spaced_along_the_tube_starting_the_inset_behind_each_open_portal():
    plans = _plans([(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)])

    lights = plan_tunnel_lights(plans, **KW)
    xs = sorted(l["position"][0] for l in lights)

    assert xs[0] == pytest.approx(5.0)
    assert xs[-1] <= 95.0 + 1e-6
    assert np.allclose(np.diff(xs), 11.0)


def test_light_sits_near_the_crown_ceiling_margin_below_it():
    plans = _plans([(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)])
    crown = plans[0]["crown"]

    light = plan_tunnel_lights(plans, **KW)[0]

    assert light["position"][2] == pytest.approx(500.0 + crown - 0.3)


def test_light_aims_straight_down_with_x_along_the_tunnel():
    plans = _plans([(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)])

    m = np.array(plan_tunnel_lights(plans, **KW)[0]["rotation_matrix"]).reshape(3, 3)

    assert m[0] == pytest.approx([1.0, 0.0, 0.0])  # row 0 (x) = tunnel direction
    assert m[1] == pytest.approx([0.0, 0.0, -1.0])  # row 1 (y) = beam direction, straight down
    assert m[2] == pytest.approx([0.0, 1.0, 0.0])  # row 2 (z) = cross(x, y), matches italy.zip's tunnelLight sample


def test_fields_are_copied_from_the_given_preset():
    plans = _plans([(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)])

    light = plan_tunnel_lights(plans, **KW)[0]

    assert light["class"] == "SpotLight"
    assert light["fields"] == FIELDS
    assert light["name"].startswith("tunnel_light_7_")


def test_closed_tube_end_gets_lights_all_the_way_to_its_far_end():
    # A closed end (buried in the mountain, no portal structure to clip into) needs no inset - only an open end does
    plans = _plans([(0.0, 0.0, 500.0), (100.0, 0.0, 500.0)])
    plans[0]["portals"][1]["open"] = False  # "end" portal, at x=100

    xs = sorted(l["position"][0] for l in plan_tunnel_lights(plans, **KW))

    assert xs[0] == pytest.approx(5.0)  # inset still applies at the open "start" portal
    assert xs[-1] > 100.0 - 11.0  # reaches within one spacing of the true end - no inset held back there


def test_chain_too_short_for_the_insets_gets_no_lights():
    plans = _plans([(0.0, 0.0, 500.0), (9.0, 0.0, 500.0)])  # 9 m < 2 * 5 m inset

    assert plan_tunnel_lights(plans, **KW) == []
