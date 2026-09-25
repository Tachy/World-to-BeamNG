"""Tests for the cap slabs of the walls (walls/wall_cap.py): stone slabs on top of the wall with overhang."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.walls.wall_cap import corner_arcs, plate_spans
from world_to_beamng.walls.wall_mesh import build_wall_mesh

GROUND, HEIGHT = 100.0, 1.5
CAP_T, OVERHANG, THICKNESS = 0.05, 0.04, 0.5
CAP_TOP = GROUND + HEIGHT


def _flat(z=GROUND):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _mesh(coords, **kwargs):
    return build_wall_mesh(coords, HEIGHT, _flat(), thickness=THICKNESS, **kwargs)


def _top_face_vertices(mesh):
    """Corner points of the slab tops: at slab height and with an upward normal."""
    v, n = mesh["vertices"], mesh["normals"]
    return v[(np.abs(v[:, 2] - CAP_TOP) < 1e-9) & (n[:, 2] > 0.99)]


# --- Config ---------------------------------------------------------------------------------------------------------


def test_config_has_the_agreed_cap_dimensions():
    assert config.WALL_CAP_THICKNESS == pytest.approx(0.05)
    assert 0.02 <= config.WALL_CAP_OVERHANG <= 0.06  # "a few centimeters"
    assert config.WALL_CAP_PLATE_LENGTH > 0.3 and 0 < config.WALL_CAP_JOINT < 0.03


# --- Slab layout ----------------------------------------------------------------------------------------------


def test_plate_spans_fill_the_range_with_joints_and_varying_lengths():
    spans = plate_spans(0.0, 10.0, 0.8, 0.01, np.random.default_rng(1))

    assert spans[0][0] == pytest.approx(0.0) and spans[-1][1] == pytest.approx(10.0)
    for (_, end), (start, _) in zip(spans[:-1], spans[1:]):
        assert start - end == pytest.approx(0.01)
    lengths = [end - start for start, end in spans]
    assert max(lengths) - min(lengths) > 0.05  # not all the same length
    assert all(0.2 < length < 1.3 for length in lengths)


def test_plate_spans_never_end_with_a_sliver():
    for seed in range(20):
        spans = plate_spans(0.0, 3.3 + 0.07 * seed, 0.8, 0.01, np.random.default_rng(seed))
        assert min(end - start for start, end in spans) > 0.2, seed


def test_plate_spans_are_deterministic_and_handle_short_runs():
    assert plate_spans(0.0, 5.0, 0.8, 0.01, np.random.default_rng(7)) == plate_spans(0.0, 5.0, 0.8, 0.01, np.random.default_rng(7))
    assert plate_spans(2.0, 2.2, 0.8, 0.01, np.random.default_rng(0)) == [(2.0, 2.2)]  # shorter than one slab: one small slab
    assert plate_spans(2.0, 2.0, 0.8, 0.01, np.random.default_rng(0)) == []


# --- Corners ----------------------------------------------------------------------------------------------------------


def test_corners_are_found_where_the_wall_turns():
    points = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [10.0, 8.0]])  # bend at the third point, point 2 lies on the straight line
    arc = np.array([0.0, 5.0, 10.0, 18.0])

    assert corner_arcs(points, closed=False, arc=arc) == [pytest.approx(10.0)]


def test_a_gentle_bend_is_not_a_corner_but_a_ring_has_four():
    gentle = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.5]])  # approx. 6 degrees
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

    assert corner_arcs(gentle, closed=False, arc=np.array([0.0, 5.0, 10.0])) == []
    assert len(corner_arcs(square, closed=True, arc=np.array([0.0, 10.0, 20.0, 30.0, 40.0]))) == 4


# --- Geometry ------------------------------------------------------------------------------------------------------


def test_the_wall_keeps_its_height_with_the_cap_on_top_and_the_body_below_it():
    mesh = _mesh([(0, 0), (10, 0)])

    z = mesh["vertices"][:, 2]
    assert z.max() == pytest.approx(CAP_TOP)
    body_top_faces = (np.abs(z - (CAP_TOP - CAP_T)) < 1e-9) & (mesh["normals"][:, 2] > 0.99)
    assert body_top_faces.any()  # the wall body stays visible between the joints


def test_the_plates_overhang_the_wall_by_a_few_centimetres_on_all_open_sides():
    top = _top_face_vertices(_mesh([(0, 0), (10, 0)]))

    assert np.abs(top[:, 1]).max() == pytest.approx(THICKNESS / 2 + OVERHANG)
    assert top[:, 0].min() == pytest.approx(-OVERHANG) and top[:, 0].max() == pytest.approx(10.0 + OVERHANG)  # also at the end faces


def test_the_plates_are_separated_by_joints():
    top = _top_face_vertices(_mesh([(0, 0), (10, 0)]))

    xs = np.unique(np.round(top[:, 0], 6))
    joints = np.isclose(np.diff(xs), config.WALL_CAP_JOINT, atol=1e-6).sum()
    assert 8 <= joints <= 14  # 10 m with approx. 0.8 m long slabs


def test_an_underside_makes_the_overhang_solid():
    mesh = _mesh([(0, 0), (10, 0)])

    v, n = mesh["vertices"], mesh["normals"]
    under = v[(n[:, 2] < -0.99) & (np.abs(v[:, 2] - (CAP_TOP - CAP_T)) < 1e-9)]
    assert len(under) > 0 and np.abs(under[:, 1]).max() == pytest.approx(THICKNESS / 2 + OVERHANG)


def test_plates_are_mitred_at_a_corner_and_do_not_bend_around_it():
    top = _top_face_vertices(_mesh([(0, 0), (10, 0), (10, 8)]))
    half = THICKNESS / 2 + OVERHANG

    def has(x, y):
        return bool((np.hypot(top[:, 0] - x, top[:, 1] - y) < 1e-6).any())

    assert has(10 + half, -half) and has(10 - half, half)  # miter points outside and inside
    assert top[:, 0].max() == pytest.approx(10 + half) and top[:, 1].min() == pytest.approx(-half)  # nothing protrudes beyond that


def test_a_closed_wall_is_covered_all_around_without_overlap_at_the_start():
    ring = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    mesh = _mesh(ring)

    top = _top_face_vertices(mesh)
    half = THICKNESS / 2 + OVERHANG
    assert top[:, 0].min() == pytest.approx(-half) and top[:, 0].max() == pytest.approx(10 + half)  # overhang only, no extension
    assert top[:, 1].min() == pytest.approx(-half) and top[:, 1].max() == pytest.approx(10 + half)


def test_the_cap_follows_the_terrain_slope():
    slope = lambda x, y: GROUND + 0.1 * np.asarray(x, float)
    mesh = build_wall_mesh([(0, 0), (10, 0)], HEIGHT, slope, thickness=THICKNESS)

    v, n = mesh["vertices"], mesh["normals"]
    top = v[(n[:, 2] > 0.9) & (v[:, 2] > GROUND + 0.1 * v[:, 0] + HEIGHT - CAP_T / 2)]
    assert len(top) > 0
    np.testing.assert_allclose(top[:, 2], GROUND + 0.1 * np.clip(top[:, 0], 0, 10) + HEIGHT, atol=0.1 * OVERHANG + 1e-6)


def test_all_faces_are_valid_and_normals_are_unit_length():
    mesh = _mesh([(0, 0), (6, 0), (6, 4), (0, 4), (0, 0)])

    assert len(mesh["vertices"]) == len(mesh["uvs"]) == len(mesh["normals"])
    assert np.isfinite(mesh["vertices"]).all() and np.isfinite(mesh["uvs"]).all()
    np.testing.assert_allclose(np.linalg.norm(mesh["normals"], axis=1), 1.0, atol=1e-9)
    assert max(max(f) for f in mesh["faces"]) < len(mesh["vertices"])


def test_faces_wind_counter_clockwise_towards_their_normals():
    mesh = _mesh([(0, 0), (10, 0), (10, 8)])
    v, n = mesh["vertices"], mesh["normals"]

    for a, b, c in mesh["faces"]:
        geometric = np.cross(v[b] - v[a], v[c] - v[a])
        assert np.dot(geometric, n[a]) > -1e-9


def test_the_plates_are_the_same_on_every_run_and_differ_between_walls():
    coords = [(0, 0), (10, 0)]

    np.testing.assert_array_equal(_mesh(coords, seed=5)["vertices"], _mesh(coords, seed=5)["vertices"])
    assert not np.array_equal(_mesh(coords, seed=5)["vertices"], _mesh(coords, seed=6)["vertices"])


def test_without_a_cap_the_wall_is_a_plain_body_up_to_the_full_height():
    mesh = _mesh([(0, 0), (10, 0)], cap_thickness=0.0)

    assert mesh["vertices"][:, 2].max() == pytest.approx(CAP_TOP)
    assert np.abs(mesh["vertices"][:, 1]).max() == pytest.approx(THICKNESS / 2)
