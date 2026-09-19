"""Tests für world_to_beamng.forest.vineyard_generator.

Rebzeilen sind Forest-Items (grape_vine: ein 5,6 m langes Zeilensegment, X-Achse =
Zeilenrichtung). Sie werden in parallelen, geraden Zeilen ausgerichtet - standardmäßig
entlang der Falllinie (Steigungsgradient) - und folgen der Hangneigung.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from world_to_beamng.forest.vineyard_generator import (
    build_exclusion_geometry,
    compute_row_direction,
    generate_vineyard_instances,
    generate_vineyards,
    make_height_sampler,
    split_by_direction,
)

ROWS = {
    "item": "grape_vine",
    "orientation": "gradient",
    "row_spacing": 2.5,
    "segment_length": 5.6,
    "edge_margin": 0.0,
    "min_slope_percent": 2.0,
    "scale_range": [1.0, 1.0],
}


def _plane(slope_x=0.0, slope_y=0.0, base=100.0):
    return lambda x, y: base + slope_x * np.asarray(x, float) + slope_y * np.asarray(y, float)


def _matrix(instance):
    """
    rotationMatrix zeilenweise. BeamNG liest die Modellachsen als ZEILEN (X = Zeile 0):
    gemessen an BeamNGs eigenen Weinbergen (italy) - dort folgt Zeile 0 zu 98,5 % dem
    Geländegefälle, Spalte 0 dagegen ist negativ korreliert.
    """
    return np.array(instance["rotationMatrix"]).reshape(3, 3)


def _axis_dot(vec_a, vec_b):
    return abs(float(np.dot(vec_a, vec_b)))


def test_height_sampler_interpolates_bilinearly():
    heights = np.zeros((4, 4))
    heights[:, :] = np.arange(4)[None, :] * 2.0  # steigt mit x um 2 m je Zelle
    height_at = make_height_sampler(heights, origin_x=10.0, origin_y=20.0, square_size=1.0)

    assert height_at(11.0, 21.0) == pytest.approx(2.0)
    assert height_at(11.5, 21.0) == pytest.approx(3.0)
    assert height_at(np.array([10.0, 12.0]), np.array([20.0, 20.0])) == pytest.approx([0.0, 4.0])


def test_height_sampler_clamps_outside_the_grid():
    height_at = make_height_sampler(np.full((4, 4), 7.0), 0.0, 0.0, 1.0)

    assert height_at(-50.0, 500.0) == pytest.approx(7.0)


def test_row_direction_gradient_runs_down_the_fall_line():
    polygon = box(0, 0, 100, 60)

    u = compute_row_direction(polygon, _plane(slope_x=0.2), "gradient", 2.0)

    assert _axis_dot(u, [1, 0]) == pytest.approx(1.0, abs=1e-6)


def test_row_direction_gradient_follows_diagonal_slope():
    polygon = box(0, 0, 100, 100)

    u = compute_row_direction(polygon, _plane(slope_x=0.1, slope_y=0.1), "gradient", 2.0)

    assert _axis_dot(u, np.array([1.0, 1.0]) / np.sqrt(2)) == pytest.approx(1.0, abs=1e-6)


def test_row_direction_contour_runs_along_the_contour_lines():
    polygon = box(0, 0, 100, 60)

    u = compute_row_direction(polygon, _plane(slope_x=0.2), "contour", 2.0)

    assert _axis_dot(u, [0, 1]) == pytest.approx(1.0, abs=1e-6)


def test_row_direction_is_a_unit_vector():
    u = compute_row_direction(box(0, 0, 50, 50), _plane(slope_y=0.3), "gradient", 2.0)

    assert np.linalg.norm(u) == pytest.approx(1.0)


def test_row_direction_flat_terrain_falls_back_to_long_axis_of_polygon():
    long_in_y = box(0, 0, 20, 120)

    u = compute_row_direction(long_in_y, _plane(), "gradient", 2.0)

    assert _axis_dot(u, [0, 1]) == pytest.approx(1.0, abs=1e-6)


def test_row_direction_nearly_flat_below_threshold_uses_long_axis():
    long_in_x = box(0, 0, 120, 20)

    # 1 % Gefälle in y liegt unter der 2-%-Schwelle -> Längsachse (x), nicht die Falllinie (y)
    u = compute_row_direction(long_in_x, _plane(slope_y=0.01), "gradient", 2.0)

    assert _axis_dot(u, [1, 0]) == pytest.approx(1.0, abs=1e-6)


def test_row_direction_on_curved_terrain_ignores_gradient_sign_flips():
    # Kuppe: Gefälle zeigt links nach links, rechts nach rechts - beide gehören zur x-Achse
    polygon = box(-50, -30, 50, 30)
    dome = lambda x, y: 100.0 - 0.001 * (np.asarray(x, float) ** 2)

    u = compute_row_direction(polygon, dome, "gradient", 2.0)

    assert _axis_dot(u, [1, 0]) == pytest.approx(1.0, abs=1e-3)


def test_instances_have_the_expected_fields():
    instances = generate_vineyard_instances(box(0, 0, 60, 40), _plane(slope_x=0.15), ROWS)

    assert instances
    for instance in instances:
        assert instance["type"] == "grape_vine"
        assert len(instance["pos"]) == 3
        assert len(instance["rotationMatrix"]) == 9
        assert instance["scale"] == pytest.approx(1.0)


def test_all_rows_are_parallel_to_the_fall_line():
    instances = generate_vineyard_instances(box(0, 0, 80, 50), _plane(slope_x=0.2), ROWS)

    for instance in instances:
        forward_xy = _matrix(instance)[0, :2]  # erste Zeile = Modell-X-Achse = Zeilenrichtung
        assert _axis_dot(forward_xy / np.linalg.norm(forward_xy), [1, 0]) == pytest.approx(1.0, abs=1e-6)


def test_row_spacing_perpendicular_to_the_rows():
    instances = generate_vineyard_instances(box(0, 0, 80, 50), _plane(slope_x=0.2), ROWS)

    ys = sorted({round(i["pos"][1], 3) for i in instances})
    assert len(ys) > 3
    gaps = np.diff(ys)
    assert gaps == pytest.approx(np.full(len(gaps), ROWS["row_spacing"]), abs=1e-3)


def test_segments_abut_along_the_row():
    instances = generate_vineyard_instances(box(0, 0, 80, 20), _plane(slope_x=0.2), ROWS)

    one_row_y = instances[0]["pos"][1]
    xs = sorted(i["pos"][0] for i in instances if abs(i["pos"][1] - one_row_y) < 1e-6)
    assert len(xs) >= 5
    # Die Segmentmitten liegen im gleichmäßigen horizontalen Raster (Segmentlänge, auf die Zeilenlänge verteilt: höchstens
    # wenige Prozent Abweichung); das Modell wird nur geneigt.
    gaps = np.diff(xs)
    assert gaps == pytest.approx(np.full(len(gaps), gaps[0]), abs=1e-6)
    assert gaps[0] == pytest.approx(ROWS["segment_length"], rel=0.06)


def test_instances_stay_inside_the_polygon_with_edge_margin():
    polygon = Polygon([(0, 0), (90, 10), (80, 70), (5, 60)])
    rows = {**ROWS, "edge_margin": 1.5}
    instances = generate_vineyard_instances(polygon, _plane(slope_x=0.2, slope_y=0.05), rows)

    inner = polygon.buffer(-rows["edge_margin"] + 0.01)
    assert instances
    for instance in instances:
        assert inner.contains(Point(instance["pos"][0], instance["pos"][1]))


def _row_extents(instances, y, segment=ROWS["segment_length"]):
    """Ausdehnung (x_min, x_max) der Segmente einer Zeile (bei Gefälle nach x laufen die Zeilen entlang x)."""
    xs = [i["pos"][0] for i in instances if abs(i["pos"][1] - y) < 1e-6]
    return min(xs) - segment / 2, max(xs) + segment / 2


def test_rows_run_up_to_the_polygon_boundary():
    # Länge 57 m ist kein Vielfaches von 5,6 m: die Segmente werden auf die Zeile verteilt, statt Reste zu lassen
    instances = generate_vineyard_instances(box(0, 0, 57, 40), _plane(slope_x=0.2), ROWS)

    ys = sorted({round(i["pos"][1], 6) for i in instances})
    assert len(ys) > 5
    for y in ys:
        x0, x1 = _row_extents(instances, y)
        assert x0 == pytest.approx(0.0, abs=0.4)
        assert x1 == pytest.approx(57.0, abs=0.4)


def test_rows_end_at_the_boundary_for_an_awkward_polygon_length():
    for length in (23.0, 41.3, 62.9, 100.0):
        instances = generate_vineyard_instances(box(0, 0, length, 12), _plane(slope_x=0.2), ROWS)
        y = instances[0]["pos"][1]
        x0, x1 = _row_extents(instances, y)
        assert x0 == pytest.approx(0.0, abs=0.6) and x1 == pytest.approx(length, abs=0.6)


def test_rows_stop_at_the_road_margin_where_a_road_crosses_the_polygon():
    road_edge = box(30, -10, 36, 100)  # Straßenfläche quer durch den Weinberg
    exclusion = build_exclusion_geometry([road_edge], 2.0)  # 2 m Abstand zum Straßenrand

    instances = generate_vineyard_instances(box(0, 0, 80, 20), _plane(slope_x=0.2), ROWS, exclusion=exclusion)

    for y in sorted({round(i["pos"][1], 6) for i in instances}):
        left = [i["pos"][0] for i in instances if abs(i["pos"][1] - y) < 1e-6 and i["pos"][0] < 30]
        right = [i["pos"][0] for i in instances if abs(i["pos"][1] - y) < 1e-6 and i["pos"][0] > 36]
        assert max(left) + ROWS["segment_length"] / 2 == pytest.approx(28.0, abs=0.6)  # 30 - 2 m
        assert min(right) - ROWS["segment_length"] / 2 == pytest.approx(38.0, abs=0.6)  # 36 + 2 m


def test_exclusion_zone_keeps_instances_out_e_g_roads():
    road = box(30, -10, 36, 100)  # Weg quer durch den Weinberg
    instances = generate_vineyard_instances(box(0, 0, 80, 50), _plane(slope_y=0.2), ROWS, exclusion=road)

    assert instances
    assert not any(road.contains(Point(i["pos"][0], i["pos"][1])) for i in instances)


def test_rotation_matrix_is_a_proper_rotation_and_follows_the_slope():
    slope = 0.2
    instances = generate_vineyard_instances(box(0, 0, 60, 30), _plane(slope_x=slope), ROWS)

    for instance in instances:
        m = _matrix(instance)
        assert m @ m.T == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-9)
        forward = m[0]
        # Bergauf entlang +x: Modell-X-Achse zeigt mit der Steigung nach oben
        assert forward[2] == pytest.approx(math.sin(math.atan(slope)), abs=1e-6)
        assert m[2, 2] > 0.9  # Reben stehen aufrecht


def test_diagonal_slope_rows_point_along_the_fall_line_and_climb_with_it():
    # Diagonaler Hang: eine transponierte Matrix würde die Zeilenrichtung an der x-Achse
    # spiegeln (quer zum Hang) und die Neigung umkehren (Reben tauchen in den Boden).
    slope = 0.15
    instances = generate_vineyard_instances(box(0, 0, 80, 80), _plane(slope_x=slope, slope_y=slope), ROWS)

    assert instances
    fall_line = np.array([1.0, 1.0]) / np.sqrt(2)
    for instance in instances:
        forward = _matrix(instance)[0]
        assert np.dot(forward[:2] / np.linalg.norm(forward[:2]), fall_line) == pytest.approx(1.0, abs=1e-6)
        assert forward[2] == pytest.approx(np.sin(np.arctan(slope * np.sqrt(2))), abs=1e-6)


def test_vertical_axis_stays_upright_on_a_cross_slope():
    # Zeilen entlang y, Gefälle quer dazu (x): Seitenneigung entsteht nicht - die Rebe bleibt aufrecht
    rows = {**ROWS, "orientation": "contour"}
    instances = generate_vineyard_instances(box(0, 0, 60, 60), _plane(slope_x=0.2), rows)

    for instance in instances:
        m = _matrix(instance)
        assert m[2, 2] > 0.95
        assert m[0][2] == pytest.approx(0.0, abs=1e-6)  # entlang der Höhenlinie: keine Längsneigung


def test_same_input_gives_same_output():
    polygon = box(0, 0, 60, 40)

    a = generate_vineyard_instances(polygon, _plane(slope_x=0.2), {**ROWS, "scale_range": [0.97, 1.03]})
    b = generate_vineyard_instances(polygon, _plane(slope_x=0.2), {**ROWS, "scale_range": [0.97, 1.03]})

    assert a == b


def test_scale_stays_within_range():
    rows = {**ROWS, "scale_range": [0.97, 1.03]}

    instances = generate_vineyard_instances(box(0, 0, 60, 40), _plane(slope_x=0.2), rows)

    scales = [i["scale"] for i in instances]
    assert min(scales) >= 0.97 and max(scales) <= 1.03


def test_polygon_smaller_than_one_segment_yields_nothing():
    assert generate_vineyard_instances(box(0, 0, 4, 4), _plane(slope_x=0.2), ROWS) == []


def test_multipolygon_parts_get_their_own_row_direction():
    left = box(0, 0, 60, 60)  # fällt nach x
    right = box(200, 0, 260, 60)  # (gleiche Ebene) -> ebenfalls x, aber je Teil berechnet
    instances = generate_vineyard_instances(MultiPolygon([left, right]), _plane(slope_x=0.2), ROWS)

    xs = [i["pos"][0] for i in instances]
    assert any(x < 100 for x in xs) and any(x > 150 for x in xs)


def test_group_item_profile_uses_configured_dimensions():
    rows = {**ROWS, "item": "grape_vine_group", "segment_length": 16.0, "row_spacing": 7.5}

    instances = generate_vineyard_instances(box(0, 0, 100, 60), _plane(slope_x=0.2), rows)

    assert instances and all(i["type"] == "grape_vine_group" for i in instances)
    ys = sorted({round(i["pos"][1], 3) for i in instances})
    assert np.diff(ys) == pytest.approx(np.full(len(ys) - 1, 7.5), abs=1e-3)


def test_exclusion_geometry_is_none_without_shapes():
    assert build_exclusion_geometry([], 3.0) is None


def test_exclusion_geometry_unions_buffered_shapes():
    road, house = box(0, 0, 10, 2), box(50, 50, 55, 55)

    zone = build_exclusion_geometry([road, house], margin=3.0)

    assert zone.contains(Point(5, 4.5))  # 2,5 m neben der Straße, innerhalb von 3 m Puffer
    assert zone.contains(Point(52, 57.5))
    assert not zone.contains(Point(25, 25))


def _two_slopes(x, y):
    """Links fällt das Gelände nach x, rechts (ab x=100) nach y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.where(x < 100, 100.0 + 0.2 * x, 120.0 + 0.2 * y)


def test_split_by_direction_keeps_uniform_terrain_in_one_block():
    polygon = box(0, 0, 200, 100)

    blocks = split_by_direction(polygon, _plane(slope_x=0.2), max_spread_deg=20.0, min_area=1000.0)

    assert len(blocks) == 1
    assert blocks[0].equals(polygon)


def test_split_by_direction_splits_where_the_fall_line_changes():
    polygon = box(0, 0, 200, 100)

    blocks = split_by_direction(polygon, _two_slopes, max_spread_deg=20.0, min_area=1000.0)

    assert len(blocks) >= 2
    # jeder Block hat eine einheitliche Falllinie: links x, rechts y
    for block in blocks:
        u = compute_row_direction(block, _two_slopes, "gradient", 2.0)
        assert max(_axis_dot(u, [1, 0]), _axis_dot(u, [0, 1])) > 0.98


def test_split_blocks_cover_the_polygon_without_overlap():
    polygon = box(0, 0, 200, 100)

    blocks = split_by_direction(polygon, _two_slopes, max_spread_deg=20.0, min_area=1000.0)

    assert sum(b.area for b in blocks) == pytest.approx(polygon.area, rel=1e-6)
    assert all(b.geom_type == "Polygon" for b in blocks)


def test_split_by_direction_does_not_split_below_min_area():
    polygon = box(90, 0, 110, 40)  # 800 m², liegt genau auf dem Knick

    assert len(split_by_direction(polygon, _two_slopes, max_spread_deg=20.0, min_area=1000.0)) == 1


def test_split_by_direction_ignores_flat_terrain():
    polygon = box(0, 0, 200, 100)

    assert len(split_by_direction(polygon, _plane(), max_spread_deg=20.0, min_area=500.0)) == 1


def test_rows_in_each_block_follow_their_own_fall_line():
    rows = {**ROWS, "max_direction_spread_deg": 20.0, "min_block_area": 1000.0}

    instances = generate_vineyard_instances(box(0, 0, 200, 100), _two_slopes, rows)

    left = [i for i in instances if i["pos"][0] < 90]
    right = [i for i in instances if i["pos"][0] > 110]
    assert left and right
    for i in left:
        assert _axis_dot(_matrix(i)[0, :2] / np.linalg.norm(_matrix(i)[0, :2]), [1, 0]) > 0.98
    for i in right:
        assert _axis_dot(_matrix(i)[0, :2] / np.linalg.norm(_matrix(i)[0, :2]), [0, 1]) > 0.98


MAPPINGS = {
    "vineyard": {"osm_tags": {"landuse": ["vineyard"]}, "priority": 9, "internal_name": "mat_vineyard", "rows": ROWS},
    "meadow": {"osm_tags": {"landuse": ["meadow"]}, "priority": 4, "internal_name": "mat_grass"},
}


def test_generate_vineyards_only_uses_vineyard_polygons():
    polygons = [
        {"osm_tags": {"landuse": "vineyard"}, "geometry": box(0, 0, 60, 40)},
        {"osm_tags": {"landuse": "meadow"}, "geometry": box(100, 0, 160, 40)},
    ]

    instances = generate_vineyards(polygons, MAPPINGS, _plane(slope_x=0.2))

    assert instances
    assert all(i["pos"][0] < 100 for i in instances)


def test_generate_vineyards_returns_nothing_without_rows_config():
    mappings = {"vineyard": {"osm_tags": {"landuse": ["vineyard"]}, "priority": 9}}
    polygons = [{"osm_tags": {"landuse": "vineyard"}, "geometry": box(0, 0, 60, 40)}]

    assert generate_vineyards(polygons, mappings, _plane(slope_x=0.2)) == []


def test_generate_vineyards_stays_inside_the_terrain_bounds():
    # Die OSM-Abfrage reicht über das Terrain hinaus; außerhalb gibt es keine Höhendaten
    # (die Heightmap klemmt am Rand) - dort würden Reben in der Luft schweben.
    polygons = [{"osm_tags": {"landuse": "vineyard"}, "geometry": box(-100, 0, 100, 60)}]
    terrain = box(-20, -20, 40, 100)

    instances = generate_vineyards(polygons, MAPPINGS, _plane(slope_x=0.2), bounds=terrain)

    assert instances
    assert all(terrain.contains(Point(i["pos"][0], i["pos"][1])) for i in instances)


def test_generate_vineyards_skips_polygons_completely_outside_the_bounds():
    polygons = [{"osm_tags": {"landuse": "vineyard"}, "geometry": box(500, 500, 600, 560)}]

    assert generate_vineyards(polygons, MAPPINGS, _plane(slope_x=0.2), bounds=box(0, 0, 100, 100)) == []


def test_generate_vineyards_passes_exclusion_through():
    polygons = [{"osm_tags": {"landuse": "vineyard"}, "geometry": box(0, 0, 80, 50)}]
    road = box(30, -10, 36, 100)

    instances = generate_vineyards(polygons, MAPPINGS, _plane(slope_y=0.2), exclusion=road)

    assert not any(road.contains(Point(i["pos"][0], i["pos"][1])) for i in instances)
