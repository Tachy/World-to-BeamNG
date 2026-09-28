"""Tests for world_to_beamng.junctions.corners: junction corners with a tangent-circle fillet."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.junctions.corners import corner_radius

TABLE = {"default_radius": 6.0, "radius_by_highway": {"service": 4.0, "residential": 6.0, "secondary": 10.0}}


def test_corner_radius_is_the_smaller_of_both_arms_with_default_for_unknown():
    assert corner_radius("secondary", "residential", TABLE) == 6.0
    assert corner_radius("secondary", "service", TABLE) == 4.0
    assert corner_radius("secondary", "secondary", TABLE) == 10.0
    assert corner_radius("motorway", "secondary", TABLE) == 6.0
