"""Tests für world_to_beamng.terrain.road_embedding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.terrain.road_embedding import (
    embed_roads_into_heightmap,
    sample_heightmap_bilinear,
    build_road_embankment_profiles,
    apply_embankment_blend,
)


def _road(polygon_xy, centerline_xyz):
    return {
        "road_polygon": np.array(polygon_xy, dtype=np.float64),
        "trimmed_centerline": np.array(centerline_xyz, dtype=np.float64),
    }


def test_embed_roads_sets_exact_road_height_only_near_road():
    # 20x20 Heightmap, 1m/Zelle, überall 100m hoch
    size = 20
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # Eine flache Straße bei Z=95, Fläche x=[5,15], y=[5,15] (Centerline bei y=10)
    road = _road(
        polygon_xy=[[5, 5], [15, 5], [15, 15], [5, 15]],
        centerline_xyz=[[5, 10, 95], [15, 10, 95]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road])

    # Zellen unter der Straße müssen exakt auf Centerline-Höhe (95) gesetzt sein
    # - kein Sicherheitsabstand mehr, da DecalRoad direkt auf das Terrain
    # projiziert wird (siehe Modul-Docstring).
    assert np.isclose(result[10, 10], 95.0)

    # Zellen weit weg von der Straße müssen unverändert bei 100 bleiben
    assert result[1, 1] == 100.0
    assert result[18, 18] == 100.0

    # Original-Array darf nicht verändert worden sein (Funktion gibt Kopie zurück)
    assert heights[10, 10] == 100.0


def test_embed_roads_can_raise_terrain_above_surroundings():
    # Straße liegt HÖHER als natürliches Terrain (z.B. Damm/Brückenrampe) ->
    # anders als beim früheren Mesh-Ansatz DARF das Terrain jetzt angehoben
    # werden, weil es die sichtbare Straßenoberfläche selbst ist (DecalRoad
    # hat keine eigene Geometrie, die durchstoßen werden könnte).
    size = 10
    heights = np.full((size, size), 50.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road = _road(
        polygon_xy=[[2, 2], [8, 2], [8, 8], [2, 8]],
        centerline_xyz=[[2, 5, 200], [8, 5, 200]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road])

    assert np.isclose(result[5, 5], 200.0)
    # Ausserhalb der Straße unverändert
    assert result[0, 0] == 50.0


def test_embed_roads_follows_curved_centerline_height():
    # Gebogene Straße mit unterschiedlicher Höhe an beiden Enden -> die
    # Ziel-Höhe muss entlang der Centerline linear interpoliert werden, nicht
    # konstant sein.
    size = 20
    heights = np.zeros((size, size))
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road = _road(
        polygon_xy=[[0, 8], [20, 8], [20, 12], [0, 12]],
        centerline_xyz=[[0, 10, 0], [20, 10, 20]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road])

    assert np.isclose(result[10, 2], 2.0, atol=0.5)
    assert np.isclose(result[10, 18], 18.0, atol=0.5)
    assert result[10, 2] < result[10, 18]


# --- embed_roads_into_heightmap(clamp_to_max=True) - Brücken-Auflager -----------------------------------


def test_clamp_to_max_lowers_terrain_above_the_deck():
    size = 20
    heights = np.full((size, size), 120.0)  # Gelände ragt komplett über das Deck (100) hinaus
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road = _road(
        polygon_xy=[[5, 5], [15, 5], [15, 15], [5, 15]],
        centerline_xyz=[[5, 10, 100], [15, 10, 100]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road], clamp_to_max=True)

    assert np.isclose(result[10, 10], 100.0)  # auf Deck-Niveau gekappt
    assert result[1, 1] == 120.0  # außerhalb der Brückenbreite unverändert


def test_clamp_to_max_leaves_terrain_below_the_deck_untouched():
    size = 20
    heights = np.full((size, size), 40.0)  # Talboden weit UNTER dem Deck (100)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road = _road(
        polygon_xy=[[5, 5], [15, 5], [15, 15], [5, 15]],
        centerline_xyz=[[5, 10, 100], [15, 10, 100]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road], clamp_to_max=True)

    assert result[10, 10] == 40.0  # Talboden bleibt sichtbar, nicht auf Deck-Niveau angehoben
    assert heights[10, 10] == 40.0  # Original unverändert (Funktion gibt Kopie zurück)


def test_sample_heightmap_bilinear_matches_grid_points():
    heights = np.array([[0.0, 10.0], [20.0, 30.0]])
    # Exact grid points should return exact values
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = sample_heightmap_bilinear(heights, origin_x=0.0, origin_y=0.0, square_size=1.0, points_xy=points)
    assert np.allclose(result, [0.0, 10.0, 20.0, 30.0])

    # Midpoint should be the average of all 4 corners
    midpoint = np.array([[0.5, 0.5]])
    result_mid = sample_heightmap_bilinear(heights, 0.0, 0.0, 1.0, midpoint)
    assert np.isclose(result_mid[0], 15.0)


def test_build_road_embankment_profiles_straight_road():
    # Flat terrain at 100.0, road at 95.0 (5m cut) -> slope_width should reflect that
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # A straight road along x=20, from y=5 to y=35, road_z=95.0, width=6.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)
    poly = {"trimmed_centerline": centerline, "osm_tags": {"highway": "residential"}}

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    assert len(roads) == 1
    road = roads[0]
    # Road is at x=20, half_width=3.0 -> left edge at x=23, right edge at x=17
    assert np.allclose(road["left_edge_xyz"][:, 0], 23.0)
    assert np.allclose(road["right_edge_xyz"][:, 0], 17.0)
    # Natural height at edges should be 100.0 (flat terrain)
    assert np.allclose(road["left_natural_z"], 100.0)
    # height_diff = |100 - 95| = 5, tan(45deg) = 1 -> slope_width = 5.0
    assert np.allclose(road["left_slope_width"], 5.0)


def test_slope_width_override_replaces_the_computed_width_on_that_side_only():
    # Gleiches Setup wie test_build_road_embankment_profiles_straight_road(): Centerline entlang x=20,
    # Richtung +y. STANDARD-"links" (offset_points()-Konvention: Richtung +90 Grad gedreht) liegt hier bei
    # x=17 - das ist (siehe Docstring-Hinweis) genau "right_edge_xyz" dieser Funktion.
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}, "slope_width_override": {"left": 0.0}}
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    assert np.allclose(road["right_edge_xyz"][:, 0], 17.0)  # = STANDARD-"links", die überschriebene Seite
    assert np.allclose(road["right_slope_width"], 0.0)  # feste Breite (0) statt berechnet
    assert np.allclose(road["left_slope_width"], 5.0)  # andere Seite unverändert normal (berechnet)


def test_slope_width_override_can_set_both_sides_to_different_fixed_values():
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}, "slope_width_override": {"left": 0.0, "right": 5.0}}
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    assert np.allclose(road["right_slope_width"], 0.0)  # STANDARD "left" -> diese Funktion "right"
    assert np.allclose(road["left_slope_width"], 5.0)  # STANDARD "right" -> diese Funktion "left"


def test_no_override_leaves_both_sides_normal():
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}}
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    assert np.allclose(road["left_slope_width"], 5.0)
    assert np.allclose(road["right_slope_width"], 5.0)


def test_apply_embankment_blend_interpolates_correctly():
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # One "road" edge segment at x=20, y from 10 to 30, road_z=95, slope_width=5
    n = 21
    y_vals = np.linspace(10, 30, n)
    edge_xyz = np.column_stack([np.full(n, 20.0), y_vals, np.full(n, 95.0)])
    slope_width = np.full(n, 5.0)
    natural_z = np.full(n, 100.0)

    roads = [{
        "left_edge_xyz": edge_xyz,
        "right_edge_xyz": np.empty((0, 3)),
        "left_slope_width": slope_width,
        "right_slope_width": np.empty(0),
        "left_natural_z": natural_z,
        "right_natural_z": np.empty(0),
    }]

    result = apply_embankment_blend(heights, origin_x, origin_y, square_size, roads)

    # At the edge itself (x=20, dist~0): should be close to road height (95), not exactly
    # touched since dist>0 required, but very close cells should approach it
    near_edge_col = 21  # x=21, dist=1 from edge at x=20
    mid_row = 20
    val_near_edge = result[mid_row, near_edge_col]
    assert val_near_edge < 100.0  # pulled toward road height
    assert val_near_edge > 95.0   # not fully at road height (partway through blend)

    # Far beyond slope_width (x=30, dist=10 > slope_width=5): untouched
    far_col = 30
    assert result[mid_row, far_col] == 100.0

    # Original heights array must not be mutated
    assert heights[mid_row, near_edge_col] == 100.0


def test_apply_embankment_blend_handles_fill_and_cut():
    # Cut: road lower than terrain (tested above, terrain=100, road=95)
    # Fill: road higher than terrain
    size = 20
    heights = np.full((size, size), 50.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    n = 11
    y_vals = np.linspace(3, 13, n)
    edge_xyz = np.column_stack([np.full(n, 10.0), y_vals, np.full(n, 60.0)])  # road at 60, terrain at 50 -> fill
    slope_width = np.full(n, 3.0)
    natural_z = np.full(n, 50.0)

    roads = [{
        "left_edge_xyz": edge_xyz, "right_edge_xyz": np.empty((0, 3)),
        "left_slope_width": slope_width, "right_slope_width": np.empty(0),
        "left_natural_z": natural_z, "right_natural_z": np.empty(0),
    }]

    result = apply_embankment_blend(heights, origin_x, origin_y, square_size, roads)

    mid_row = 8
    # Near edge (x=11, dist=1): pulled UP toward road height (60), i.e. > natural 50
    near_val = result[mid_row, 11]
    assert near_val > 50.0
    assert near_val < 60.0


if __name__ == "__main__":
    test_embed_roads_sets_exact_road_height_only_near_road()
    print("[OK] test_embed_roads_sets_exact_road_height_only_near_road")
    test_embed_roads_can_raise_terrain_above_surroundings()
    print("[OK] test_embed_roads_can_raise_terrain_above_surroundings")
    test_embed_roads_follows_curved_centerline_height()
    print("[OK] test_embed_roads_follows_curved_centerline_height")
    test_sample_heightmap_bilinear_matches_grid_points()
    print("[OK] test_sample_heightmap_bilinear_matches_grid_points")
    test_build_road_embankment_profiles_straight_road()
    print("[OK] test_build_road_embankment_profiles_straight_road")
    test_apply_embankment_blend_interpolates_correctly()
    print("[OK] test_apply_embankment_blend_interpolates_correctly")
    test_apply_embankment_blend_handles_fill_and_cut()
    print("[OK] test_apply_embankment_blend_handles_fill_and_cut")
    print("Alle Tests bestanden.")


# --- Charakterisierung: Optimierung darf das Ergebnis nicht verändern ---------------------


def _diagonal_road(size=400, width=6.0):
    """Lange, gekrümmte Diagonal-Straße: riesige Bounding-Box, aber schmaler Streifen."""
    from shapely.geometry import LineString

    t = np.linspace(20.0, size - 20.0, 120)
    xy = np.column_stack([t, 0.9 * t + 12.0 * np.sin(t / 25.0)])
    z = 100.0 + 0.05 * t + 2.0 * np.sin(t / 40.0)
    polygon = np.array(LineString(xy).buffer(width / 2.0, cap_style=2).exterior.coords[:-1])
    return _road(polygon, np.column_stack([xy, z]))


def _reference_embed(heights, origin_x, origin_y, square_size, road):
    """Brute-Force-Referenz: volle Bounding-Box, kein Vorfiltern (alte Logik)."""
    from world_to_beamng.terrain.road_embedding import _points_in_polygon_2d, _project_onto_polyline

    result = heights.copy()
    polygon, centerline = road["road_polygon"], road["trimmed_centerline"]
    size_y, size_x = heights.shape
    col_start = max(0, int(np.floor((polygon[:, 0].min() - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((polygon[:, 0].max() - origin_x) / square_size)))
    row_start = max(0, int(np.floor((polygon[:, 1].min() - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((polygon[:, 1].max() - origin_y) / square_size)))
    gx, gy = np.meshgrid(
        origin_x + np.arange(col_start, col_end + 1) * square_size,
        origin_y + np.arange(row_start, row_end + 1) * square_size,
    )
    inside = _points_in_polygon_2d(gx, gy, polygon)
    target = _project_onto_polyline(gx, gy, centerline[:, 0], centerline[:, 1], centerline[:, 2])
    sub = result[row_start : row_end + 1, col_start : col_end + 1]
    sub[inside] = target[inside]
    return result


def test_embed_matches_brute_force_reference_on_long_diagonal_road():
    size = 400
    rng = np.random.RandomState(1)
    heights = 100.0 + rng.rand(size, size)
    road = _diagonal_road(size)

    result = embed_roads_into_heightmap(heights, 0.0, 0.0, 1.0, [road])
    reference = _reference_embed(heights, 0.0, 0.0, 1.0, road)

    assert np.count_nonzero(result != heights) > 500  # Straße wurde wirklich eingebettet
    np.testing.assert_array_equal(result, reference)


def test_embed_with_subcell_resolution_and_offset_origin_matches_reference():
    rng = np.random.RandomState(2)
    heights = 50.0 + rng.rand(300, 300)
    road = _diagonal_road(150)

    result = embed_roads_into_heightmap(heights, 3.3, -2.7, 0.5, [road])
    reference = _reference_embed(heights, 3.3, -2.7, 0.5, road)

    np.testing.assert_array_equal(result, reference)


# --- Optimierung: identisch zur einfachen Referenzschleife ----------------------------------


def _reference_project_onto_polyline(qx, qy, poly_x, poly_y, poly_z):
    """Die ursprüngliche Schleife über alle Segmente (Referenz)."""
    best_dist = np.full(qx.shape, np.inf)
    best_z = np.zeros(qx.shape)
    for i in range(len(poly_x) - 1):
        ax, ay, az = poly_x[i], poly_y[i], poly_z[i]
        bx, by, bz = poly_x[i + 1], poly_y[i + 1], poly_z[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            continue
        t = np.clip(((qx - ax) * dx + (qy - ay) * dy) / seg_len_sq, 0.0, 1.0)
        dist = np.hypot(qx - (ax + t * dx), qy - (ay + t * dy))
        z = az + t * (bz - az)
        better = dist < best_dist
        best_dist = np.where(better, dist, best_dist)
        best_z = np.where(better, z, best_z)
    return best_z


def _polylines():
    rng = np.random.RandomState(7)
    lines = {}
    t = np.linspace(0, 300, 200)
    lines["kurve"] = (t, 40 * np.sin(t / 30.0), 100 + 0.1 * t)
    # Haarnadel: die Linie läuft zurück und kommt sich selbst nahe
    u = np.linspace(0, 1, 120)
    lines["haarnadel"] = (np.concatenate([u * 200, 200 - u * 200]), np.concatenate([np.zeros(120), np.full(120, 6.0)]), np.linspace(50, 80, 240))
    # doppelte Punkte (Nulllängen-Segmente) und wenige Punkte
    lines["doppelte"] = (np.array([0, 0, 10, 10, 10, 30.0]), np.array([0, 0, 5, 5, 5, -4.0]), np.array([1, 2, 3, 4, 5, 6.0]))
    lines["zwei"] = (np.array([0.0, 50.0]), np.array([0.0, 20.0]), np.array([10.0, 30.0]))
    lines["zufall"] = (np.cumsum(rng.uniform(0.5, 4, 150)), np.cumsum(rng.uniform(-3, 3, 150)), rng.uniform(0, 100, 150))
    return lines


def test_project_onto_polyline_is_identical_to_the_reference_loop():
    from world_to_beamng.terrain.road_embedding import _project_onto_polyline

    rng = np.random.RandomState(3)
    for name, (px, py, pz) in _polylines().items():
        lo_x, hi_x, lo_y, hi_y = px.min() - 15, px.max() + 15, py.min() - 15, py.max() + 15
        qx = rng.uniform(lo_x, hi_x, 3000)
        qy = rng.uniform(lo_y, hi_y, 3000)
        # auch Punkte exakt auf Stützpunkten und in gleichem Abstand zu zwei Segmenten
        n_on = min(5, len(px))
        qx[:n_on], qy[:n_on] = px[:n_on], py[:n_on]

        np.testing.assert_array_equal(
            _project_onto_polyline(qx, qy, px, py, pz), _reference_project_onto_polyline(qx, qy, px, py, pz), err_msg=name
        )


def test_project_onto_polyline_keeps_the_shape_and_handles_no_valid_segment():
    from world_to_beamng.terrain.road_embedding import _project_onto_polyline

    qx, qy = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0])
    same = np.array([5.0, 5.0, 5.0])
    np.testing.assert_array_equal(_project_onto_polyline(qx, qy, same, same, same), np.zeros(3))
    assert _project_onto_polyline(np.empty(0), np.empty(0), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 2.0])).shape == (0,)


def _reference_blend_one_side(heights, origin_x, origin_y, square_size, size_x, size_y, edge_xyz, slope_width, natural_z):
    """Die ursprüngliche Variante: alle Zellen der Bounding Box abfragen (Referenz)."""
    from scipy.spatial import cKDTree

    if len(edge_xyz) == 0:
        return
    max_width = float(np.max(slope_width)) if len(slope_width) else 0.0
    if max_width <= 0:
        return
    min_x = float(np.min(edge_xyz[:, 0])) - max_width
    max_x = float(np.max(edge_xyz[:, 0])) + max_width
    min_y = float(np.min(edge_xyz[:, 1])) - max_width
    max_y = float(np.max(edge_xyz[:, 1])) + max_width
    col_start = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row_start = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))
    if col_start > col_end or row_start > row_end:
        return
    tree = cKDTree(edge_xyz[:, :2])
    gx, gy = np.meshgrid(origin_x + np.arange(col_start, col_end + 1) * square_size, origin_y + np.arange(row_start, row_end + 1) * square_size)
    query_points = np.column_stack([gx.ravel(), gy.ravel()])
    dist, idx = tree.query(query_points, distance_upper_bound=max_width + 1e-9)
    near = np.isfinite(dist)
    dist, idx = dist[near], idx[near]
    nearest_edge_z, nearest_slope_width, nearest_natural_z = edge_xyz[idx, 2], slope_width[idx], natural_z[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(nearest_slope_width > 0, dist / nearest_slope_width, 1.0)
    t = np.clip(t, 0.0, 1.0)
    blended = nearest_edge_z + (nearest_natural_z - nearest_edge_z) * t
    in_corridor = (dist > 0) & (dist <= nearest_slope_width)
    sub_shape = (row_end - row_start + 1, col_end - col_start + 1)
    sub = heights[row_start : row_end + 1, col_start : col_end + 1].reshape(-1)
    sub[np.flatnonzero(near)[in_corridor]] = blended[in_corridor]
    heights[row_start : row_end + 1, col_start : col_end + 1] = sub.reshape(sub_shape)


def test_blend_one_side_is_identical_to_the_full_bounding_box_reference():
    from world_to_beamng.terrain.road_embedding import _blend_one_side

    rng = np.random.RandomState(11)
    size = 500
    heights = 200.0 + rng.rand(size, size) * 5
    cases = []
    t = np.linspace(20, 470, 300)
    diagonal = np.column_stack([t, 0.9 * t + 12 * np.sin(t / 25.0), 100 + 0.05 * t])  # lange Diagonale
    cases.append((diagonal, rng.uniform(1.0, 9.0, len(t)), 200 + rng.rand(len(t))))
    short = np.column_stack([np.linspace(100, 130, 30), np.linspace(50, 60, 30), np.full(30, 210.0)])
    cases.append((short, np.full(30, 4.0), np.full(30, 205.0)))
    cases.append((diagonal[::-1].copy(), np.zeros(300), np.zeros(300)))  # Böschungsbreite 0
    cases.append((np.array([[-50.0, -50.0, 1.0], [-40.0, -45.0, 1.0]]), np.array([5.0, 5.0]), np.array([2.0, 2.0])))  # außerhalb

    for origin_x, origin_y, square in ((0.0, 0.0, 1.0), (3.3, -2.7, 0.5), (-10.0, 5.0, 2.0)):
        for edge, width, natural in cases:
            expected, actual = heights.copy(), heights.copy()
            _reference_blend_one_side(expected, origin_x, origin_y, square, size, size, edge, width, natural)
            _blend_one_side(actual, origin_x, origin_y, square, size, size, edge, width, natural)
            np.testing.assert_array_equal(actual, expected)
