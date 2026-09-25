"""
Tests for the roof overhang: 60 cm (horizontal) at the eave, 30 cm at the gable, 10 cm thick.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.facade.edge_topology import roof_rings, wall_lines
from world_to_beamng.facade.roof_overhang import RoofOverhangBuilder

TRIS = np.array([[0, 1, 2], [0, 2, 3]])


def _wall(a, b, z0, z1):
    return (np.array([[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1], [a[0], a[1], z0]], float), TRIS)


def _gable_wall(a, b, z_eave, z_ridge):
    """Gable wall: rectangle up to the eave plus a triangle up to the ridge in the middle."""
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    ring = np.array(
        [[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], z_eave], [mid[0], mid[1], z_ridge], [a[0], a[1], z_eave], [a[0], a[1], 0]], float
    )
    return ring, np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])


def _gabled_house(width=10.0, depth=8.0, eave=6.0, ridge=9.0):
    """Gable roof: ridge parallel to the x axis, gables at x = 0 and x = width."""
    half = depth / 2
    walls = [
        _wall((0, 0), (width, 0), 0, eave),  # eave side south
        _gable_wall((width, 0), (width, depth), eave, ridge),  # gable east
        _wall((width, depth), (0, depth), 0, eave),  # eave side north
        _gable_wall((0, depth), (0, 0), eave, ridge),  # gable west
    ]
    south = np.array([[0, 0, eave], [width, 0, eave], [width, half, ridge], [0, half, ridge]], float)
    north = np.array([[width, depth, eave], [0, depth, eave], [0, half, ridge], [width, half, ridge]], float)
    return {"id": "H", "walls": walls, "roofs": [(south, TRIS), (north, TRIS)], "bounds": (0, 0, 0, width, depth, ridge)}


def _build(building, index=0):
    roofs = roof_rings(building)
    return RoofOverhangBuilder().build(roofs[index], index, roofs, wall_lines(building))


def test_eave_projects_sixty_centimetres_horizontally():
    roof = _build(_gabled_house())

    assert roof.ring[:, 1].min() == pytest.approx(-config.ROOF_EAVE_OVERHANG_M)  # south wall is at y = 0


def test_verge_projects_thirty_centimetres_beyond_the_gable_wall():
    roof = _build(_gabled_house())

    assert roof.ring[:, 0].min() == pytest.approx(-config.ROOF_VERGE_OVERHANG_M)
    assert roof.ring[:, 0].max() == pytest.approx(10.0 + config.ROOF_VERGE_OVERHANG_M)


def test_ridge_and_the_shared_edge_are_not_extended():
    roof = _build(_gabled_house())

    ridge = roof.ring[np.isclose(roof.ring[:, 2], 9.0)]
    assert len(ridge) >= 2 and np.allclose(ridge[:, 1], 4.0)  # ridge stays at y = 4


def test_extended_roof_stays_in_its_plane():
    roof = _build(_gabled_house())

    slope = 3.0 / 4.0  # dz/dy of the south face
    assert np.allclose(roof.ring[:, 2], 6.0 + slope * roof.ring[:, 1])


SOUTH_NORMAL = np.array([0.0, -0.6, 0.8])  # south face of the gable roof (dz/dy = 0.75), points up and south


def test_overhang_is_a_slab_ten_centimetres_thick_perpendicular_to_the_roof():
    roof = _build(_gabled_house())

    top = np.array([0.0, 0.0, 6.0])  # a point of the roof plane
    distance = (roof.trim_vertices - top) @ SOUTH_NORMAL
    assert set(np.round(distance, 6)) == {0.0, -config.ROOF_OVERHANG_THICKNESS_M}  # top edge in the roof plane, bottom edge 10 cm below


def test_trim_faces_point_down_for_the_soffit_and_outward_for_the_fascia():
    roof = _build(_gabled_house())
    vertices = roof.trim_vertices
    soffit = fascia = 0
    centre = np.array([5.0, 2.0])
    for i, j, k in roof.trim_faces:
        normal = np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i])
        unit = normal / np.linalg.norm(normal)
        if np.allclose(unit, -SOUTH_NORMAL):  # soffit: parallel to the roof face, points down
            soffit += 1
        elif abs(unit @ SOUTH_NORMAL) < 1e-9:  # fascia board: perpendicular to the roof face, points outward
            fascia += float(unit[:2] @ (vertices[[i, j, k], :2].mean(axis=0) - centre)) > 0
    assert soffit > 0 and fascia > 0
    assert soffit + fascia == len(roof.trim_faces)  # no faces pointing up/inward


def test_result_does_not_depend_on_ring_direction():
    house = _gabled_house()
    reversed_house = {**house, "roofs": [(verts[::-1].copy(), faces) for verts, faces in house["roofs"]]}

    a, b = _build(house), _build(reversed_house)

    assert np.allclose(np.sort(a.ring, axis=0), np.sort(b.ring, axis=0))


def test_edges_without_a_wall_underneath_get_no_overhang():
    house = _gabled_house()
    house["walls"] = []  # nothing supports the roof

    roof = _build(house)

    assert np.allclose(np.sort(roof.ring, axis=0), np.sort(roof_rings(house)[0], axis=0))
    assert len(roof.trim_faces) == 0


def test_edge_with_a_higher_wall_gets_no_overhang():
    house = _gabled_house()
    house["walls"][0] = _wall((0, 0), (10, 0), 0, 9.0)  # south wall as a firewall extending above the eave

    roof = _build(house)

    assert roof.ring[:, 1].min() == pytest.approx(0.0)  # no eave extended


def test_hip_roof_planes_meet_on_the_hip_line():
    """Hip roof: two faces share the hip line; their overhangs must end on the same line."""
    width, depth, eave = 10.0, 6.0, 6.0
    ridge_z, x0, x1 = 8.0, 3.0, 7.0
    walls = [
        _wall((0, 0), (width, 0), 0, eave),
        _wall((width, 0), (width, depth), 0, eave),
        _wall((width, depth), (0, depth), 0, eave),
        _wall((0, depth), (0, 0), 0, eave),
    ]
    south = np.array([[0, 0, eave], [width, 0, eave], [x1, 3, ridge_z], [x0, 3, ridge_z]], float)
    east = np.array([[width, 0, eave], [width, depth, eave], [x1, 3, ridge_z]], float)
    house = {"id": "W", "walls": walls, "roofs": [(south, TRIS), (east, np.array([[0, 1, 2]]))], "bounds": (0, 0, 0, width, depth, ridge_z)}

    south_roof, east_roof = _build(house, 0), _build(house, 1)

    def on_hip_line(point):  # hip line extended from the corner point (10, 0, 6) to (7, 3, 8)
        direction = np.array([-3.0, 3.0, 2.0])
        offset = point - np.array([10.0, 0.0, 6.0])
        return np.linalg.norm(np.cross(offset, direction)) / np.linalg.norm(direction) < 1e-6

    south_corner = south_roof.ring[np.argmax(south_roof.ring[:, 0] + (south_roof.ring[:, 2] < 6.5) * 100)]
    east_corner = east_roof.ring[np.argmin(east_roof.ring[:, 1] + (east_roof.ring[:, 2] > 6.5) * 100)]
    assert on_hip_line(south_corner) and on_hip_line(east_corner)
    assert np.allclose(south_corner, east_corner)
