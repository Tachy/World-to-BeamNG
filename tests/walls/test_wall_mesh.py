"""Tests für world_to_beamng.walls.wall_mesh: Bruchsteinmauern aus OSM-Linien (barrier=wall) mit Höhenangabe.

Nur Mauern mit `height`-Tag werden gebaut. Die Mauer ist ein Quader-Band von ~50 cm Dicke entlang der Linie, das dem
Gelände folgt: Oberkante = Boden an der Mittellinie + Höhe, Unterkante beidseitig unter dem Boden (kein Spalt).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from world_to_beamng.walls.wall_mesh import build_wall_mesh, build_walls, parse_wall_height, select_walls

MATERIAL = "rubble_stone_wall"


def _flat(z=100.0):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _way(way_id, points, **tags):
    return {"type": "way", "id": way_id, "tags": {"barrier": "wall", **tags}, "geometry": [{"lon": x, "lat": y} for x, y in points]}


def _to_local(points):
    return [(p["lon"], p["lat"]) for p in points]


# --- Höhenangabe ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [("1.50", 1.5), ("3", 3.0), ("1.8", 1.8), ("2 m", 2.0), ("2,5", 2.5), (" 1.2m", 1.2)],
)
def test_wall_height_is_parsed_in_metres(value, expected):
    assert parse_wall_height(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "", "abc", "0", "-1", "50", "1..2"])
def test_missing_or_implausible_heights_are_rejected(value):
    assert parse_wall_height(value) is None


# --- Auswahl -------------------------------------------------------------------------------------------------------


def test_only_walls_with_a_height_are_selected():
    ways = [
        _way(1, [(0, 0), (10, 0)], height="1.5", material="stone"),
        _way(2, [(0, 5), (10, 5)]),  # keine Höhe
        _way(3, [(0, 8), (10, 8)], height="kaputt"),
        _way(4, [(0, 9), (10, 9)], height="2"),  # kein material-Tag: Stein angenommen
    ]

    selected = select_walls(ways, _to_local)

    assert [w["osm_id"] for w in selected] == [1, 4]
    assert selected[0]["height"] == 1.5 and selected[0]["coords"] == [(0, 0), (10, 0)]


def test_other_materials_and_other_barriers_are_not_rubble_walls():
    ways = [
        _way(1, [(0, 0), (10, 0)], height="2", material="brick"),
        _way(2, [(0, 0), (10, 0)], height="2", material="concrete"),
        {**_way(3, [(0, 0), (10, 0)], height="2"), "tags": {"barrier": "fence", "height": "2"}},
        {**_way(4, [(0, 0), (10, 0)], height="2"), "type": "node"},
        _way(5, [(0, 0)], height="2"),  # zu kurz
        _way(6, [(0, 0), (10, 0)], height="2", material="stone"),
        {**_way(7, [(0, 0), (10, 0)], height="2"), "tags": {"barrier": "retaining_wall", "height": "3"}},
    ]

    assert [w["osm_id"] for w in select_walls(ways, _to_local)] == [6, 7]


# --- Geometrie -----------------------------------------------------------------------------------------------------


def _mesh(coords, height=2.0, ground=None, **kwargs):
    kwargs.setdefault("thickness", 0.5)
    kwargs.setdefault("sink", 0.3)
    kwargs.setdefault("max_step", 1.0)
    kwargs.setdefault("tile_m", 1.2)
    return build_wall_mesh(coords, height, ground or _flat(), **kwargs)


def _triangles(mesh):
    return np.array([[mesh["vertices"][i] for i in face] for face in mesh["faces"]])


def test_straight_wall_is_half_a_metre_thick_and_as_high_as_tagged():
    mesh = _mesh([(0.0, 0.0), (10.0, 0.0)], height=2.0)
    v = mesh["vertices"]

    assert v[:, 1].min() == pytest.approx(-0.25) and v[:, 1].max() == pytest.approx(0.25)  # 50 cm dick
    assert v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(10.0)
    assert v[:, 2].max() == pytest.approx(102.0)  # Boden 100 + Höhe 2
    assert v[:, 2].min() == pytest.approx(99.7)  # 30 cm unter dem Boden, kein Spalt am Fuß


def test_faces_point_outwards_and_match_the_stored_normals():
    mesh = _mesh([(0.0, 0.0), (10.0, 0.0)])
    tris, normals = _triangles(mesh), mesh["normals"]

    for face, tri in zip(mesh["faces"], tris):
        geometric = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        assert np.dot(geometric, normals[face[0]]) > 0  # Umlaufsinn passt zur Normalen (nach außen)
    # es gibt Flächen zu beiden Seiten, oben und an den Enden
    directions = {tuple(np.round(n, 2)) for n in normals}
    assert (0.0, 1.0, 0.0) in directions and (0.0, -1.0, 0.0) in directions and (0.0, 0.0, 1.0) in directions
    assert (1.0, 0.0, 0.0) in directions and (-1.0, 0.0, 0.0) in directions


def test_wall_follows_the_terrain_and_sinks_below_the_ground_on_both_sides():
    slope = lambda x, y: 100.0 + 0.1 * np.asarray(x, float) + 0.2 * np.asarray(y, float)  # auch quer geneigt
    mesh = _mesh([(0.0, 0.0), (10.0, 0.0)], height=2.0, ground=slope)
    v = mesh["vertices"]

    top = v[v[:, 2] > 100.0 + 1.5]
    assert np.allclose(top[:, 2], slope(top[:, 0], 0.0) + 2.0, atol=1e-6)  # Oberkante = Boden Mitte + Höhe
    bottom = v[v[:, 2] < slope(v[:, 0], v[:, 1]) + 0.05]
    assert np.allclose(bottom[:, 2], slope(bottom[:, 0], bottom[:, 1]) - 0.3, atol=1e-6)  # jede Seite unter ihrem Boden


def test_long_walls_are_split_into_short_segments_so_they_can_follow_the_ground():
    mesh = _mesh([(0.0, 0.0), (10.0, 0.0)], max_step=1.0)

    xs = np.unique(np.round(mesh["vertices"][:, 0], 6))
    assert len(xs) >= 11 and np.diff(xs).max() <= 1.0 + 1e-6


def test_corner_is_mitred_without_gap_or_overlap():
    coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    mesh = _mesh(coords)
    top = [t for t, n in zip(_triangles(mesh), mesh["normals"][[f[0] for f in mesh["faces"]]]) if n[2] > 0.99]

    area = unary_union([Polygon(t[:, :2]).buffer(1e-6) for t in top if Polygon(t[:, :2]).area > 1e-9])
    expected = LineString(coords).buffer(0.25, cap_style="flat", join_style="mitre")
    assert area.symmetric_difference(expected).area < 0.02 * expected.area


def test_closed_ring_has_no_end_caps():
    ring = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0), (0.0, 0.0)]
    open_line = ring[:-1]

    closed_mesh, open_mesh = _mesh(ring), _mesh(open_line)

    # 24 Segmente à 1 m, je Segment 3 Vierecke (links, rechts, oben) = 6 Dreiecke; die offene Linie hat zusätzlich 2 Stirnflächen
    assert len(closed_mesh["faces"]) == 6 * 24
    assert len(open_mesh["faces"]) == 6 * 18 + 4
    v = closed_mesh["vertices"]
    assert v[:, 0].min() == pytest.approx(-0.25) and v[:, 0].max() == pytest.approx(6.25)  # Außenkante der Ecken (Gehrung)


def test_uvs_are_metric_in_texture_tiles():
    mesh = _mesh([(0.0, 0.0), (12.0, 0.0)], tile_m=1.2)

    assert mesh["uvs"].shape == (len(mesh["vertices"]), 2)
    assert mesh["uvs"][:, 0].max() - mesh["uvs"][:, 0].min() == pytest.approx(12.0 / 1.2)  # 12 m = 10 Textur-Kacheln
    side = np.abs(mesh["normals"][:, 1]) > 0.99  # Längsseiten (Oberseite/Stirn haben eigene UVs)
    heights = mesh["uvs"][side, 1]
    assert heights.max() - heights.min() == pytest.approx((102.0 - 99.7) / 1.2)  # Höhe in Kacheln


def test_build_walls_returns_one_mesh_per_wall_with_the_rubble_material():
    ways = [_way(1, [(0, 0), (10, 0)], height="1.5", material="stone"), _way(2, [(0, 5), (10, 5)]), _way(3, [(0, 8), (10, 8)], height="3")]

    meshes, stats = build_walls(ways, _to_local, _flat(), MATERIAL)

    assert [m["id"] for m in meshes] == ["wall_1", "wall_3"]
    assert all(set(m["faces"]) == {MATERIAL} and len(m["faces"][MATERIAL]) > 0 for m in meshes)
    assert stats["built"] == 2 and stats["length"] == pytest.approx(20.0) and stats["without_height"] == 1
