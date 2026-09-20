"""
Tests für die Wände: fugenloser Putz, Fenster von oben nach unten, erhöhter Keller.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.facade.facade_mapper import FacadeMapper
from world_to_beamng.facade.facade_styles import PLASTER_COLORS, plaster_index, stable_hash
from world_to_beamng.facade.window_atlas import WindowAtlasLayout, WindowSprite

LAYOUT = WindowAtlasLayout()
TRIS = np.array([[0, 1, 2], [0, 2, 3]])
DOORS = (WindowSprite.DOOR_WOOD, WindowSprite.DOOR_WHITE)
BASEMENT = (WindowSprite.BASEMENT_PLAIN, WindowSprite.BASEMENT_BARS)


def _wall(a, b, z0, z1):
    """Rechteckwand von a nach b (xy); Ring mit Schlusspunkt, Normale nach rechts der Richtung a->b."""
    (ax, ay), (bx, by) = a, b
    ring = np.array([[ax, ay, z0], [bx, by, z0], [bx, by, z1], [ax, ay, z1], [ax, ay, z0]], dtype=float)
    return ring, TRIS


def _polygon_wall(points_xyz):
    ring = np.array(points_xyz + [points_xyz[0]], dtype=float)
    return ring, np.array([[0, i, i + 1] for i in range(1, len(ring) - 2)])


def _house(key="H", width=10.0, depth=8.0, height=6.0, rotation=0.0, inward=False, floor_z=(0.0, 0.0, 0.0, 0.0)):
    """Quader mit vier Wänden (nach außen orientiert, von oben gegen den Uhrzeigersinn) und Flachdach in Traufhöhe."""
    corners = [(0, 0), (width, 0), (width, depth), (0, depth)]
    walls = []
    for i in range(4):
        ring, faces = _wall(corners[i], corners[(i + 1) % 4], floor_z[i], height)
        ring = _rotate(ring, rotation)
        walls.append((ring[::-1].copy() if inward else ring, faces))
    roof = _rotate(np.array([[0, 0, height], [width, 0, height], [width, depth, height], [0, depth, height]], float), rotation)
    return {"id": key, "walls": walls, "roofs": [(roof, TRIS)], "bounds": (0, 0, 0, width, depth, height)}


def _rotate(points, degrees):
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return points @ rot.T


def _windows(mesh):
    """Fensterflächen als [(Sprite, [BL, BR, TR, TL])]; die Vertices je Fenster liegen als Vierergruppe hintereinander."""
    indices = sorted({i for face in mesh.window_faces for i in face})
    result = []
    for start in range(0, len(indices), 4):
        quad = indices[start : start + 4]
        u, v = mesh.uvs[quad[0]]
        sprite = next(s for s in WindowSprite if abs(LAYOUT.uv_rect(s)[0] - u) < 1e-9 and abs(LAYOUT.uv_rect(s)[1] - v) < 1e-9)
        result.append((sprite, mesh.vertices[quad]))
    return result


def _sills(mesh, exclude=()):
    return sorted({round(float(q[0][2]), 3) for s, q in _windows(mesh) if s not in exclude})


def _area(mesh, faces):
    total = 0.0
    for i, j, k in faces:
        total += np.linalg.norm(np.cross(mesh.vertices[j] - mesh.vertices[i], mesh.vertices[k] - mesh.vertices[i])) / 2
    return total


# ---------------------------------------------------------------- Putz


def test_every_wall_is_one_uncut_polygon():
    mesh = FacadeMapper().map_building(_house(width=25.0, depth=14.0, height=9.0))

    assert len(mesh.wall_faces) == 4 * 2  # zwei Dreiecke je Rechteckwand, keine Zellen
    plaster_vertices = {i for face in mesh.wall_faces for i in face}
    assert len(plaster_vertices) == 4 * 4  # nur die Eckpunkte der Wände


def test_plaster_area_is_preserved_including_a_concave_wall():
    house = _house()
    notch = _polygon_wall([[0, 0, 0], [10, 0, 0], [10, 0, 6], [6, 0, 6], [6, 0, 3], [4, 0, 3], [4, 0, 6], [0, 0, 6]])
    house["walls"] = [notch]

    mesh = FacadeMapper().map_building(house)

    assert _area(mesh, mesh.wall_faces) == pytest.approx(10 * 6 - 2 * 3)


@pytest.mark.parametrize("wall_index", [0, 1])  # 0: entlang x, 1: entlang y (Regression: alter Code hatte dort kein U)
def test_plaster_uvs_are_metric(wall_index):
    house = _house()
    house["walls"] = [house["walls"][wall_index]]

    mesh = FacadeMapper().map_building(house)

    used = sorted({i for face in mesh.wall_faces for i in face})
    for a in used:
        for b in used:
            world = np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])
            texture = np.linalg.norm(mesh.uvs[a] - mesh.uvs[b]) * config.FACADE_PLASTER_REPEAT_M
            assert texture == pytest.approx(world, abs=1e-6)


def test_plaster_triangles_keep_the_winding_of_the_source_rings():
    for inward in (False, True):
        house = _house(inward=inward)
        mesh = FacadeMapper().map_building(house)
        ring_normals = [np.cross(ring[1] - ring[0], ring[2] - ring[1]) for ring, _ in house["walls"]]

        for i, j, k in mesh.wall_faces:
            normal = np.cross(mesh.vertices[j] - mesh.vertices[i], mesh.vertices[k] - mesh.vertices[i])
            assert any(np.linalg.norm(np.cross(normal, r)) < 1e-9 and normal @ r > 0 for r in ring_normals)


# ---------------------------------------------------------------- Fenster


def test_windows_are_counted_from_the_eave_downwards():
    mesh = FacadeMapper().map_building(_house(height=6.0))

    # zwei Geschosse: Böden bei 3,0 und 0,0 -> Fensterunterkante 0,9 m darüber
    assert _sills(mesh, exclude=DOORS) == [0.9, 3.9]


def test_a_taller_house_keeps_the_storeys_aligned_to_the_eave():
    mesh = FacadeMapper().map_building(_house(height=8.0))

    # Traufe 8,0: Böden bei 5,0 und 2,0; darunter (2,0 m) ein erhöhter Keller
    assert _sills(mesh, exclude=DOORS + BASEMENT) == [2.9, 5.9]


def test_the_remainder_below_the_storeys_becomes_a_raised_basement_with_low_windows():
    mesh = FacadeMapper().map_building(_house(height=8.0))

    basement = [q for s, q in _windows(mesh) if s in BASEMENT]
    assert basement, "kein Kellerfenster"
    assert {round(float(q[0][2]), 3) for q in basement} == {config.FACADE_BASEMENT_SILL_M}  # Bodennähe
    assert all(q[3][2] < 2.0 for q in basement)  # bleiben unter dem Erdgeschossboden


def test_no_basement_when_the_storeys_fill_the_wall():
    mesh = FacadeMapper().map_building(_house(height=6.0))

    assert not [s for s, _ in _windows(mesh) if s in BASEMENT]


def test_basement_windows_only_where_enough_wall_is_visible():
    # Nordwand (Index 2) steht am Hang 1,5 m höher: dort ist der Keller kaum sichtbar
    house = _house(height=8.0, floor_z=(0.0, 0.0, 1.5, 0.0))

    mesh = FacadeMapper().map_building(house)

    north_basement = [q for s, q in _windows(mesh) if s in BASEMENT and np.allclose(q[:, 1], 8.0, atol=0.1)]
    south_basement = [q for s, q in _windows(mesh) if s in BASEMENT and np.allclose(q[:, 1], 0.0, atol=0.1)]
    assert south_basement and not north_basement


def test_a_door_only_where_the_ground_floor_is_at_ground_level():
    flush = FacadeMapper().map_building(_house(height=6.0))
    raised = FacadeMapper().map_building(_house(height=8.0))  # Erdgeschoss 2 m über dem Gelände

    assert [s for s, _ in _windows(flush) if s in DOORS]
    assert not [s for s, _ in _windows(raised) if s in DOORS]


def test_exactly_one_door_and_it_sits_on_the_longest_wall():
    mesh = FacadeMapper().map_building(_house(width=14.0, depth=6.0, height=6.0))

    doors = [q for s, q in _windows(mesh) if s in DOORS]
    assert len(doors) == 1
    assert np.allclose(doors[0][:, 1], 0.0, atol=0.1) or np.allclose(doors[0][:, 1], 6.0, atol=0.1)  # lange Wand
    assert doors[0][0][2] == pytest.approx(0.0)  # Schwelle = Erdgeschossboden


def test_windows_stand_in_front_of_the_wall():
    house = _house(height=6.0)
    house["walls"] = [house["walls"][0]]  # Südwand y = 0, Normale -y

    mesh = FacadeMapper().map_building(house)

    ys = np.array([v[1] for _, quad in _windows(mesh) for v in quad])
    assert np.allclose(ys, -config.FACADE_WINDOW_OFFSET_M)


def test_window_triangles_face_outwards():
    for inward in (False, True):
        house = _house(inward=inward)
        mesh = FacadeMapper().map_building(house)
        ring_normals = [np.cross(ring[1] - ring[0], ring[2] - ring[1]) for ring, _ in house["walls"]]

        for i, j, k in mesh.window_faces:
            normal = np.cross(mesh.vertices[j] - mesh.vertices[i], mesh.vertices[k] - mesh.vertices[i])
            assert any(np.linalg.norm(np.cross(normal, r)) < 1e-9 and normal @ r > 0 for r in ring_normals)


def test_windows_never_reach_over_the_wall_edge():
    mesh = FacadeMapper().map_building(_house(width=9.0, depth=7.0, height=9.0))

    for _, quad in _windows(mesh):
        assert quad[:, 2].min() >= -1e-9 and quad[:, 2].max() <= 9.0 + 1e-9
        assert quad[:, 0].min() >= -config.FACADE_WINDOW_OFFSET_M - 1e-9 and quad[:, 0].max() <= 9.0 + config.FACADE_WINDOW_OFFSET_M + 1e-9


def test_narrow_walls_get_no_windows():
    house = _house()
    house["walls"] = [_wall((0, 0), (1.5, 0), 0.0, 6.0)]

    assert not FacadeMapper().map_building(house).window_faces


def test_no_windows_in_the_gable_above_the_eave():
    house = _house(height=6.0)
    gable = _polygon_wall([[0, 0, 0], [8, 0, 0], [8, 0, 6], [4, 0, 9], [0, 0, 6]])
    house["walls"] = [gable]
    house["roofs"] = [(np.array([[0, 0, 6], [8, 0, 6], [4, 4, 9]], float), np.array([[0, 1, 2]]))]

    mesh = FacadeMapper().map_building(house)

    assert _windows(mesh)
    assert all(quad[:, 2].max() <= 6.0 for _, quad in _windows(mesh))


def test_low_buildings_without_a_full_storey_get_no_windows():
    mesh = FacadeMapper().map_building(_house(height=1.8))

    assert not mesh.window_faces


def test_empty_building():
    mesh = FacadeMapper().map_building({"id": "E", "walls": [], "roofs": [], "bounds": (0, 0, 0, 1, 1, 1)})

    assert not mesh.wall_faces and not mesh.window_faces


# ---------------------------------------------------------------- Farbe und Determinismus


def test_mapping_is_deterministic():
    house = _house("DEBW_X")

    first, second = FacadeMapper().map_building(house), FacadeMapper().map_building(house)

    assert first.plaster == second.plaster
    assert np.array_equal(first.uvs, second.uvs) and first.window_faces == second.window_faces


def test_hash_is_reproducible_across_processes():
    # Fester Wert: crc32 ist prozessübergreifend konstant, Python-hash() (gesalzen) wäre es nicht
    assert stable_hash("DEBWL0010000abcd") == 858753638
    assert plaster_index("DEBWL0010000abcd") == plaster_index("DEBWL0010000abcd")


def test_plaster_colours_follow_the_weights():
    counts = np.zeros(len(PLASTER_COLORS))
    total = 20000
    for i in range(total):
        counts[plaster_index(f"DEBW{i:08d}")] += 1
    share = counts / total

    for color, measured in zip(PLASTER_COLORS, share):
        assert measured == pytest.approx(color.weight / 1000, abs=0.012)
    assert share[0] > 0.5  # vorwiegend weiß
    assert share[-2:].sum() < 0.08  # Rottöne ganz vereinzelt
