"""
Tests für Flachdach-Erkennung und Blechrand.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.facade.flat_roof import FlatRoofRimBuilder, is_flat_roof

TRIS = np.array([[0, 1, 2], [0, 2, 3]])


def _rect(x0, y0, x1, y1, z):
    return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=float)


def _wall(a, b, z0, z1):
    return (np.array([[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1], [a[0], a[1], z0]], float), TRIS)


def _box_walls(x0, y0, x1, y1, z1):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [_wall(corners[i], corners[(i + 1) % 4], 0.0, z1) for i in range(4)]


def _tilted(slope_deg):
    slope = np.radians(slope_deg)
    return np.array([[0, 0, 6], [10, 0, 6], [10, 6 * np.cos(slope), 6 + 6 * np.sin(slope)], [0, 6 * np.cos(slope), 6 + 6 * np.sin(slope)]])


def _normals(mesh):
    return [
        np.cross(mesh.vertices[j] - mesh.vertices[i], mesh.vertices[k] - mesh.vertices[i]) for i, j, k in mesh.faces
    ]


@pytest.mark.parametrize("slope, flat", [(0, True), (4, True), (4.9, True), (5.1, False), (6, False), (30, False)])
def test_flat_roof_detection(slope, flat):
    assert is_flat_roof(_tilted(slope)) is flat


def test_rectangular_flat_roof_gets_six_triangles_per_edge():
    building = {"walls": _box_walls(0, 0, 10, 6, 6), "roofs": [(_rect(0, 0, 10, 6, 6), TRIS)]}

    mesh = FlatRoofRimBuilder().build(building)

    assert len(mesh.faces) == 4 * 6


@pytest.mark.parametrize("reverse", [False, True])
def test_rim_faces_point_outward_inward_and_up_regardless_of_ring_direction(reverse):
    ring = _rect(0, 0, 10, 6, 6)
    ring = ring[::-1].copy() if reverse else ring
    building = {"walls": _box_walls(0, 0, 10, 6, 6), "roofs": [(ring, TRIS)]}

    mesh = FlatRoofRimBuilder().build(building)

    centre = np.array([5.0, 3.0])
    up = out = inward = 0
    for face, normal in zip(mesh.faces, _normals(mesh)):
        if np.linalg.norm(normal) < 1e-12:
            continue
        direction = normal / np.linalg.norm(normal)
        if direction[2] > 0.99:
            up += 1
        else:
            position = mesh.vertices[face].mean(axis=0)[:2]
            outward_side = float(direction[:2] @ (position - centre)) > 0
            # Außenfläche liegt weiter außen als die Innenfläche derselben Kante, Normale zeigt jeweils weg vom Rand
            out += outward_side
            inward += not outward_side
    assert up == 8 and out == 8 and inward == 8


def test_rim_sits_on_the_roof_edge_with_the_configured_height():
    building = {"walls": _box_walls(0, 0, 10, 6, 6), "roofs": [(_rect(0, 0, 10, 6, 6), TRIS)]}

    mesh = FlatRoofRimBuilder().build(building)

    assert mesh.vertices[:, 2].min() == pytest.approx(6.0)
    assert mesh.vertices[:, 2].max() == pytest.approx(6.0 + config.FLAT_ROOF_EDGE_HEIGHT_M)


def test_no_rim_on_sloped_roofs():
    building = {"walls": _box_walls(0, 0, 10, 6, 6), "roofs": [(_tilted(30), TRIS)]}

    assert len(FlatRoofRimBuilder().build(building).faces) == 0


def test_no_rim_on_edges_shared_with_another_roof():
    # Zwei nebeneinanderliegende Flachdachpolygone: die gemeinsame Kante bekommt keinen Rand
    building = {
        "walls": _box_walls(0, 0, 20, 6, 6),
        "roofs": [(_rect(0, 0, 10, 6, 6), TRIS), (_rect(10, 0, 20, 6, 6), TRIS)],
    }

    mesh = FlatRoofRimBuilder().build(building)

    assert len(mesh.faces) == (3 + 3) * 6


def test_no_rim_where_a_higher_wall_continues():
    # Nordwand (y = 6) reicht auf 9 m -> Brandwand über dem 6-m-Flachdach
    walls = _box_walls(0, 0, 10, 6, 6)
    walls[2] = _wall((10, 6), (0, 6), 0.0, 9.0)
    building = {"walls": walls, "roofs": [(_rect(0, 0, 10, 6, 6), TRIS)]}

    mesh = FlatRoofRimBuilder().build(building)

    assert len(mesh.faces) == 3 * 6
    assert mesh.vertices[:, 1].max() < 6.0 + 1e-9 + config.FLAT_ROOF_EDGE_THICKNESS_M  # kein Rand an y = 6 (nur Eckenüberstand)


def test_building_without_roof_has_no_rim():
    mesh = FlatRoofRimBuilder().build({"walls": _box_walls(0, 0, 10, 6, 6), "roofs": []})

    assert len(mesh.faces) == 0 and len(mesh.vertices) == 0
