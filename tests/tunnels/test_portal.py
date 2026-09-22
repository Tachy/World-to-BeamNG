"""Tests für world_to_beamng.tunnels.portal: Portal-Stirnflächen von Tunneln folgen der natürlichen Hangneigung
statt rechtwinklig zur Achse abgeschnitten zu werden (siehe Design-Spec Abschnitt 5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.portal import portal_axial_shift, sample_slope_along_axis


def test_sample_slope_along_axis_is_zero_on_flat_ground():
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 100.0)

    slope = sample_slope_along_axis(ground_at, (0.0, 0.0), (1.0, 0.0), sample_dist=5.0)

    assert slope == pytest.approx(0.0)


def test_sample_slope_along_axis_matches_a_known_incline():
    # Gelände steigt 1 m je 10 m in +x-Richtung
    ground_at = lambda x, y: 100.0 + 0.1 * np.asarray(x, float)

    slope = sample_slope_along_axis(ground_at, (20.0, 0.0), (1.0, 0.0), sample_dist=5.0)

    assert slope == pytest.approx(0.1)


def test_sample_slope_along_axis_flips_sign_with_direction():
    ground_at = lambda x, y: 100.0 + 0.1 * np.asarray(x, float)

    forward = sample_slope_along_axis(ground_at, (20.0, 0.0), (1.0, 0.0), sample_dist=5.0)
    backward = sample_slope_along_axis(ground_at, (20.0, 0.0), (-1.0, 0.0), sample_dist=5.0)

    assert forward == pytest.approx(-backward)


def test_portal_axial_shift_is_zero_at_floor_level():
    assert portal_axial_shift(0.0, slope_along_axis=0.2) == pytest.approx(0.0)


def test_portal_axial_shift_grows_with_height_and_slope():
    assert portal_axial_shift(5.0, slope_along_axis=0.2) == pytest.approx(1.0)
    assert portal_axial_shift(5.0, slope_along_axis=0.0) == pytest.approx(0.0)
