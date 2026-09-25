"""
Tests for the metric roof UV mapping.

Core invariant: the mapping roof plane -> UV is length-preserving (isometry up to the factor ROOF_REPEAT_M).
Thus a beaver-tail tile is always ROOF_TILE_WIDTH_M wide, no matter how steep the roof is.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.facade.roof_uv import RoofUvMapper

SLOPES_DEG = [0.0, 30.0, 45.0, 60.0, 75.0]

RECTANGLE = [(0.0, 0.0), (12.0, 0.0), (12.0, 7.0), (0.0, 7.0)]
L_SHAPE = [(0.0, 0.0), (12.0, 0.0), (12.0, 4.0), (5.0, 4.0), (5.0, 9.0), (0.0, 9.0)]  # concave


def _roof(outline, slope_deg, azimuth_deg=37.0, origin=(1000.0, 2000.0, 300.0)):
    """Roof polygon in space: a runs along the eave, b up the slope (b = length IN the roof plane)."""
    slope = np.radians(slope_deg)
    azimuth = np.radians(azimuth_deg)
    eave = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    horizontal_up = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up_slope = np.cos(slope) * horizontal_up + np.array([0.0, 0.0, np.sin(slope)])
    return np.array([np.array(origin) + a * eave + b * up_slope for a, b in outline])


def _closed(verts):
    """As in the CityGML ring: last point = first point."""
    return np.vstack([verts, verts[:1]])


def _assert_isometric(verts, uvs, repeat_m):
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            world = np.linalg.norm(verts[i] - verts[j])
            texture = np.linalg.norm(uvs[i] - uvs[j]) * repeat_m
            assert texture == pytest.approx(world, abs=1e-6), (i, j)


@pytest.mark.parametrize("slope", SLOPES_DEG)
@pytest.mark.parametrize("outline", [RECTANGLE, L_SHAPE], ids=["rectangle", "concave"])
def test_uv_distance_equals_world_distance(slope, outline):
    verts = _roof(outline, slope)

    uvs = RoofUvMapper().map_polygon(_closed(verts))

    _assert_isometric(_closed(verts), uvs, config.ROOF_REPEAT_M)


def test_steep_roof_is_not_stretched_along_the_slope():
    # Regression: 45° roof, 10 m in the roof plane -> the old footprint projection yielded only 7.07 m
    verts = _roof([(0.0, 0.0), (6.0, 0.0), (6.0, 10.0), (0.0, 10.0)], 45.0)

    uvs = RoofUvMapper().map_polygon(verts)

    assert (uvs[:, 1].max() - uvs[:, 1].min()) * config.ROOF_REPEAT_M == pytest.approx(10.0)


def test_one_tile_is_always_the_configured_width():
    # 0.20 m along the eave = exactly one tile = 1/ROOF_TILES_PER_REPEAT of the texture
    verts = _roof([(0.0, 0.0), (config.ROOF_TILE_WIDTH_M, 0.0), (config.ROOF_TILE_WIDTH_M, 3.0), (0.0, 3.0)], 60.0)

    uvs = RoofUvMapper().map_polygon(verts)

    assert uvs[1, 0] - uvs[0, 0] == pytest.approx(1.0 / config.ROOF_TILES_PER_REPEAT)


@pytest.mark.parametrize("slope", [s for s in SLOPES_DEG if s > 0])  # flat roofs have no eave direction
def test_tile_rows_run_parallel_to_the_eave(slope):
    verts = _roof(RECTANGLE, slope)

    uvs = RoofUvMapper().map_polygon(verts)

    # Eave = edge 0->1 (a direction): same V, its length runs along U
    assert uvs[0, 1] == pytest.approx(uvs[1, 1])
    assert uvs[1, 0] - uvs[0, 0] == pytest.approx(12.0 / config.ROOF_REPEAT_M)


@pytest.mark.parametrize("slope", [0.0, 45.0])
def test_origin_is_at_the_eave_corner(slope):
    uvs = RoofUvMapper().map_polygon(_roof(RECTANGLE, slope))

    assert uvs[:, 0].min() == pytest.approx(0.0)
    assert uvs[:, 1].min() == pytest.approx(0.0)


@pytest.mark.parametrize("slope", [30.0, 60.0])
def test_result_does_not_depend_on_ring_direction_or_closing_point(slope):
    verts = _roof(L_SHAPE, slope)

    reference = RoofUvMapper().map_polygon(verts)
    closed = RoofUvMapper().map_polygon(_closed(verts))
    reversed_ring = RoofUvMapper().map_polygon(verts[::-1])

    assert np.allclose(closed[:-1], reference)
    assert np.allclose(reversed_ring[::-1], reference)


def test_flat_roof_uses_world_axes():
    verts = np.array([[0.0, 0.0, 5.0], [8.0, 0.0, 5.0], [8.0, 4.0, 5.0], [0.0, 4.0, 5.0]])

    uvs = RoofUvMapper().map_polygon(verts)

    assert uvs[:, 0].max() == pytest.approx(8.0 / config.ROOF_REPEAT_M)
    assert uvs[:, 1].max() == pytest.approx(4.0 / config.ROOF_REPEAT_M)


def test_degenerate_polygon_gets_zero_uvs():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    assert np.allclose(RoofUvMapper().map_polygon(line), 0.0)
