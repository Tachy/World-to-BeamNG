"""
Church towers: detection (OSM church + tall walls) and effect in the mapper (no windows, tower clock on the front).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.facade.church_towers import ChurchTowerFinder, is_bell_tower, is_church
from world_to_beamng.facade.facade_mapper import FacadeMapper
from world_to_beamng.facade.window_atlas import WindowAtlasLayout, WindowSprite

LAYOUT = WindowAtlasLayout()
TRIS = np.array([[0, 1, 2], [0, 2, 3]])
CLOCK_UV = LAYOUT.uv_rect(WindowSprite.TOWER_CLOCK)


def _box_walls(x0, y0, x1, y1, z0, z1):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = []
    for i in range(4):
        (ax, ay), (bx, by) = corners[i], corners[(i + 1) % 4]
        walls.append((np.array([[ax, ay, z0], [bx, by, z0], [bx, by, z1], [ax, ay, z1], [ax, ay, z0]], float), TRIS))
    return walls


def _church(tower_side=5.0, tower_height=24.0, nave_height=8.0):
    """Nave (x 0..12) with a tower on the west side (x -tower_side..0), a building like in the LOD2 data."""
    walls = _box_walls(0, 0, 12, 8, 0, nave_height) + _box_walls(-tower_side, 1.5, 0, 1.5 + tower_side, 0, tower_height)
    roofs = [
        (np.array([[0, 0, nave_height], [12, 0, nave_height], [12, 8, nave_height], [0, 8, nave_height]], float), TRIS),
        (np.array([[-tower_side, 1.5, tower_height]] * 4, float), TRIS),
    ]
    return {"id": "DEBW_CHURCH", "walls": walls, "roofs": roofs, "bounds": (-tower_side, 0, 0, 12, 8, tower_height)}


CHURCH_AREA = box(-6, -1, 13, 9)


def _sprites(mesh):
    indices = sorted({i for face in mesh.window_faces for i in face})
    found = []
    for start in range(0, len(indices), 4):
        quad = indices[start : start + 4]
        u, v = mesh.uvs[quad[0]]
        found.append((abs(CLOCK_UV[0] - u) < 1e-9 and abs(CLOCK_UV[1] - v) < 1e-9, mesh.vertices[quad]))
    return found


# ---------------------------------------------------------------- Detection


def test_tags():
    assert is_church({"building": "church"}) and is_church({"amenity": "place_of_worship"}) and is_church({"building": "chapel"})
    assert not is_church({"building": "house"})
    assert is_bell_tower({"man_made": "tower", "tower:type": "bell_tower"})
    assert not is_bell_tower({"man_made": "tower", "tower:type": "communication"})


def test_the_tall_walls_of_a_church_are_the_tower():
    church = _church()
    finder = ChurchTowerFinder([CHURCH_AREA], [])

    assert finder.mark([church]) == 1
    assert church["tower_walls"] == [4, 5, 6, 7]  # the four tower walls, not the nave


def test_a_tall_building_without_a_church_polygon_is_no_tower():
    church = _church()

    assert ChurchTowerFinder([], []).mark([church]) == 0
    assert ChurchTowerFinder([box(100, 100, 120, 120)], []).mark([church]) == 0
    assert "tower_walls" not in church


def test_a_church_without_a_tower_rising_out_of_it_is_left_alone():
    church = _church(tower_height=10.0, nave_height=8.0)  # only 2 m taller than the nave

    assert ChurchTowerFinder([CHURCH_AREA], []).mark([church]) == 0


def test_walls_inside_an_osm_bell_tower_polygon_count_as_tower():
    church = _church(tower_height=9.0)  # too low for the height rule
    tower_area = box(-5.5, 1.0, 0.5, 7.0)

    ChurchTowerFinder([CHURCH_AREA], [tower_area]).mark([church])

    assert {4, 5, 6, 7} <= set(church["tower_walls"])  # (plus, if applicable, the shared wall at x = 0 in the synthetic building)


def test_a_building_covered_by_a_bell_tower_polygon_is_a_free_standing_tower():
    tower = {"id": "T", "walls": _box_walls(0, 0, 5, 5, 0, 20), "roofs": [], "bounds": (0, 0, 0, 5, 5, 20)}

    assert ChurchTowerFinder([], [box(-1, -1, 6, 6)]).mark([tower]) == 1
    assert tower["tower_walls"] == [0, 1, 2, 3]


def test_osm_elements_are_turned_into_local_polygons():
    from world_to_beamng.geometry.coordinates import transformer_to_wgs84
    from world_to_beamng.osm.landuse_polygons import make_local_transform

    offset = (401000.0, 5298000.0)

    def way(points_local, tags):
        geometry = []
        for x, y in points_local:
            lon, lat = transformer_to_wgs84.transform(offset[0] + x, offset[1] + y)
            geometry.append({"lat": lat, "lon": lon})
        return {"type": "way", "id": 1, "tags": tags, "geometry": geometry}

    church = _church()
    osm = [way([(-6, -1), (13, -1), (13, 9), (-6, 9), (-6, -1)], {"building": "church"}), way([(0, 0), (1, 0), (1, 1), (0, 0)], {"building": "house"})]

    assert ChurchTowerFinder.from_osm(osm, make_local_transform(offset)).mark([church]) == 1


# ---------------------------------------------------------------- Mapper


def test_tower_walls_get_no_windows_but_the_nave_keeps_them():
    church = _church()
    ChurchTowerFinder([CHURCH_AREA], []).mark([church])

    mesh = FacadeMapper().map_building(church)

    tower_windows = [q for is_clock, q in _sprites(mesh) if not is_clock and q[:, 0].min() < -0.5]  # tower: x from -5 to 0
    nave_windows = [q for is_clock, q in _sprites(mesh) if not is_clock and q[:, 0].min() >= -0.5]
    assert not tower_windows
    assert nave_windows


def test_the_clock_sits_on_the_front_of_the_tower_facing_away_from_the_nave():
    church = _church()
    ChurchTowerFinder([CHURCH_AREA], []).mark([church])

    clocks = [q for is_clock, q in _sprites(FacadeMapper().map_building(church)) if is_clock]

    assert len(clocks) == 1
    clock = clocks[0]
    assert np.allclose(clock[:, 0], -5.0 - config.FACADE_WINDOW_OFFSET_M)  # west wall of the tower (x = -5), slightly in front of it
    width, height = 2.0, 2.0
    assert abs(clock[1][1] - clock[0][1]) == pytest.approx(width) and clock[2][2] - clock[1][2] == pytest.approx(height)
    assert clock[:, 1].mean() == pytest.approx(1.5 + 2.5)  # centered on the wall (y 1.5 .. 6.5)
    assert clock[:, 2].min() >= config.CHURCH_CLOCK_MIN_HEIGHT_M
    assert clock[:, 2].max() <= 24.0 - config.CHURCH_CLOCK_BELOW_TOP_M + 1.0 + 1e-9  # below the tower head


def test_a_free_standing_tower_gets_its_clock_on_the_west_wall():
    tower = {"id": "T", "walls": _box_walls(0, 0, 5, 5, 0, 20), "roofs": [], "bounds": (0, 0, 0, 5, 5, 20)}
    ChurchTowerFinder([], [box(-1, -1, 6, 6)]).mark([tower])

    clocks = [q for is_clock, q in _sprites(FacadeMapper().map_building(tower)) if is_clock]

    assert len(clocks) == 1 and np.allclose(clocks[0][:, 0], -config.FACADE_WINDOW_OFFSET_M)


def test_a_tower_too_narrow_for_the_clock_gets_none():
    church = _church(tower_side=2.7)
    ChurchTowerFinder([CHURCH_AREA], []).mark([church])

    assert not [q for is_clock, q in _sprites(FacadeMapper().map_building(church)) if is_clock]


def test_buildings_without_tower_walls_are_unchanged():
    house = {"id": "H", "walls": _box_walls(0, 0, 10, 8, 0, 6), "roofs": [], "bounds": (0, 0, 0, 10, 8, 6)}

    assert not [q for is_clock, q in _sprites(FacadeMapper().map_building(house)) if is_clock]
