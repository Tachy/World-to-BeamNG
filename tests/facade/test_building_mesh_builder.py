"""
Integration: BuildingMeshBuilder verteilt Wände auf Putz + Fenster, Schrägdächer auf Biberschwanz + Überstand,
Flachdächer auf Kies + Blechrand.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.builders import BuildingMeshBuilder
from world_to_beamng.facade.facade_styles import plaster_index
from world_to_beamng.facade.material_names import (
    FLAT_ROOF_MATERIAL,
    ROOF_EDGE_MATERIAL,
    ROOF_MATERIAL,
    ROOF_TRIM_MATERIAL,
    WALL_MATERIALS,
    WINDOW_MATERIAL,
)

TRIS = np.array([[0, 1, 2], [0, 2, 3]])


def _walls(width=10.0, depth=8.0, height=6.0):
    corners = [(0, 0), (width, 0), (width, depth), (0, depth)]
    walls = []
    for i in range(4):
        (ax, ay), (bx, by) = corners[i], corners[(i + 1) % 4]
        ring = np.array([[ax, ay, 0], [bx, by, 0], [bx, by, height], [ax, ay, height], [ax, ay, 0]], float)
        walls.append((ring, TRIS))
    return walls


def _building(roof_ring, key="DEBW_TEST_1"):
    return {"id": key, "walls": _walls(), "roofs": [(np.array(roof_ring, float), TRIS)], "bounds": (0, 0, 0, 10, 8, 6)}


FLAT = [[0, 0, 6], [10, 0, 6], [10, 8, 6], [0, 8, 6]]
SLOPED = [[0, 0, 6], [10, 0, 6], [10, 4, 9], [0, 4, 9]]  # Neigung ~37°, First bei y = 4 ohne Wand darunter


def _build(building):
    meshes = BuildingMeshBuilder().with_buildings([building]).build()
    assert len(meshes) == 1
    return meshes[0]


def _assert_consistent(mesh):
    count = len(mesh["vertices"])
    assert len(mesh["uvs"]) == count
    for faces in mesh["faces"].values():
        for face in faces:
            assert len(face) == 3 and all(0 <= i < count for i in face)


def test_flat_roof_gets_gravel_and_sheet_metal_rim():
    faces = _build(_building(FLAT))["faces"]

    assert faces[FLAT_ROOF_MATERIAL] and len(faces[ROOF_EDGE_MATERIAL]) == 4 * 6
    assert ROOF_MATERIAL not in faces and ROOF_TRIM_MATERIAL not in faces


def test_sloped_roof_keeps_the_tile_material_gets_an_overhang_and_no_rim():
    faces = _build(_building(SLOPED))["faces"]

    assert faces[ROOF_MATERIAL] and faces[ROOF_TRIM_MATERIAL]
    assert FLAT_ROOF_MATERIAL not in faces and ROOF_EDGE_MATERIAL not in faces


def test_sloped_roof_projects_beyond_the_walls():
    mesh = _build(_building(SLOPED))

    roof = sorted({i for face in mesh["faces"][ROOF_MATERIAL] for i in face})
    xs, ys = mesh["vertices"][roof][:, 0], mesh["vertices"][roof][:, 1]
    assert ys.min() == pytest.approx(-config.ROOF_EAVE_OVERHANG_M)
    assert xs.min() == pytest.approx(-config.ROOF_VERGE_OVERHANG_M)
    assert xs.max() == pytest.approx(10.0 + config.ROOF_VERGE_OVERHANG_M)


def test_walls_use_one_plaster_material_per_building_plus_windows():
    key = "DEBW_TEST_1"
    faces = _build(_building(FLAT, key))["faces"]

    plaster = [name for name in WALL_MATERIALS if name in faces]
    assert plaster == [WALL_MATERIALS[plaster_index(key)]]
    assert faces[WINDOW_MATERIAL]
    assert "lod2_wall_white" not in faces and "lod2_wall_facade" not in faces


def test_roof_uv_of_a_steep_roof_is_metric_including_the_overhang():
    mesh = _build(_building(SLOPED))

    roof = sorted({i for face in mesh["faces"][ROOF_MATERIAL] for i in face})
    uvs = mesh["uvs"][roof]
    eave_extension = config.ROOF_EAVE_OVERHANG_M / 0.8  # waagerechtes Maß -> in der Dachebene (cos 37° = 0,8)
    assert (uvs[:, 1].max() - uvs[:, 1].min()) * config.ROOF_REPEAT_M == pytest.approx(5.0 + eave_extension)


def test_all_faces_reference_valid_vertices():
    for roof in (FLAT, SLOPED):
        _assert_consistent(_build(_building(roof)))


def test_building_without_roof_still_gets_walls():
    building = _building(FLAT)
    building["roofs"] = []

    mesh = _build(building)

    assert any(name in mesh["faces"] for name in WALL_MATERIALS)
    assert ROOF_MATERIAL not in mesh["faces"] and FLAT_ROOF_MATERIAL not in mesh["faces"]
    _assert_consistent(mesh)
