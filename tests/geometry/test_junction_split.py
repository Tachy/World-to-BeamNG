"""Tests for world_to_beamng.geometry.junctions.split_roads_at_mid_junctions: split roads at mid-junctions.
Regression: a mid-junction whose projection falls exactly (or numerically almost) on an already
existing centerline point must not produce a duplicate endpoint (zero-length segment -> NaN in
downstream direction normalization, e.g. in bridge/tunnel/gallery meshes that do not go through their own
duplicate cleanup like drop_close_nodes())."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.junctions import _same_xy, split_roads_at_mid_junctions


def _road(road_id, points, z=0.0, **tags):
    return {
        "id": road_id,
        "coords": np.array([[float(x), float(y), z] for x, y in points]),
        "name": "r",
        "osm_tags": tags,
    }


def _min_gap(coords):
    xy = np.asarray(coords)[:, :2]
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).min())


def test_mid_junction_landing_exactly_on_an_existing_point_does_not_duplicate_it():
    road = _road(1, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    junctions = [{"position": (2.0, 0.0, 0.0), "connection_types": {0: ["mid"]}}]

    new_roads, _ = split_roads_at_mid_junctions([road], junctions)

    assert len(new_roads) == 2
    for r in new_roads:
        assert _min_gap(r["coords"]) > 0.0, f"road {r['id']} has a zero-length (duplicated-point) segment: {r['coords']}"


def test_mid_junction_landing_very_close_to_an_existing_point_does_not_produce_a_near_zero_segment():
    # Projection falls numerically almost (but not exactly) on the point at x=2 - the same failure mode can
    # also occur without exact equality (rounding in the projection calculation).
    road = _road(1, [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
    junctions = [{"position": (2.0 + 1e-9, 0.0, 0.0), "connection_types": {0: ["mid"]}}]

    new_roads, _ = split_roads_at_mid_junctions([road], junctions)

    for r in new_roads:
        assert _min_gap(r["coords"]) > 1e-6, f"road {r['id']} has a near-zero segment: {r['coords']}"


def test_mid_junction_in_the_middle_of_a_segment_still_splits_normally():
    road = _road(1, [(0.0, 0.0), (10.0, 0.0)])
    junctions = [{"position": (5.0, 0.0, 0.0), "connection_types": {0: ["mid"]}}]

    new_roads, _ = split_roads_at_mid_junctions([road], junctions)

    assert len(new_roads) == 2
    coords_a = np.asarray(new_roads[0]["coords"])
    coords_b = np.asarray(new_roads[1]["coords"])
    assert coords_a[-1][0] == pytest.approx(5.0) and coords_a[-1][1] == pytest.approx(0.0)
    assert coords_b[0][0] == pytest.approx(5.0) and coords_b[0][1] == pytest.approx(0.0)
    assert _min_gap(coords_a) == pytest.approx(5.0)
    assert _min_gap(coords_b) == pytest.approx(5.0)


def test_road_without_mid_junctions_is_returned_unchanged():
    road = _road(1, [(0.0, 0.0), (10.0, 0.0)])

    new_roads, _ = split_roads_at_mid_junctions([road], [])

    assert len(new_roads) == 1
    assert np.array_equal(np.asarray(new_roads[0]["coords"]), np.asarray(road["coords"]))


def test_same_xy_matches_np_allclose_including_the_relative_tolerance():
    """_same_xy replaces np.allclose(a[:2], b[:2], atol=1e-6) - same decision even right at the tolerance limit
    (rtol=1e-5 makes it about 1 cm large for coordinates around 1000 m)."""
    rng = np.random.default_rng(3)
    base = rng.uniform(-3000.0, 3000.0, size=(4000, 3))
    offsets = rng.choice([0.0, 1e-7, 1e-6, 5e-3, 1e-2, 2e-2, 1.0], size=(4000, 2)) * rng.choice([-1.0, 1.0], size=(4000, 2))
    other = base.copy()
    other[:, :2] += offsets
    other[::7, :2] = base[::7, :2] * (1 + 1e-5)  # exactly on the relative limit
    for a, b in zip(base, other):
        assert _same_xy(a, b) == bool(np.allclose(a[:2], b[:2], atol=1e-6))
        assert _same_xy(b, a) == bool(np.allclose(b[:2], a[:2], atol=1e-6))
