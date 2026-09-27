"""build_road_embankment_profiles(): on a sidewalk side the embankment starts behind the sidewalk."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.terrain.road_embedding import build_road_embankment_profiles

MAPPER = SimpleNamespace(get_road_properties=lambda tags: {"width": 6.0})


def _profiles(**extra):
    road = {"trimmed_centerline": np.array([[x, 50.0, 10.0] for x in np.arange(20.0, 80.5, 1.0)]), "osm_tags": {}, **extra}
    heights = np.full((128, 128), 10.0)
    return build_road_embankment_profiles([road], heights, 0.0, 0.0, 1.0, MAPPER, 30.0, 1.0, max_slope_width=10.0)[0]


def test_sidewalk_side_edge_moves_out_by_the_extra():
    plain, widened = _profiles(), _profiles(sidewalk_extra={"left": 1.15})
    # road runs along +x at y = 50: standard left is +y
    assert plain["right_edge_xyz"][:, 1].max() == pytest.approx(53.0)
    assert widened["right_edge_xyz"][:, 1].max() == pytest.approx(54.15)
    np.testing.assert_allclose(widened["left_edge_xyz"], plain["left_edge_xyz"])  # standard right side unchanged
