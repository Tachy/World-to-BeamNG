"""Poisson-Disk-Sampling und JSONL-Ausgabe der Wald-Pipeline (Verhalten bleibt bei Geschwindigkeitsumbauten gleich)."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, box

from world_to_beamng.forest.forest_json_writer import ForestJSONWriter
from world_to_beamng.forest.forest_point_generator import ForestPointGenerator


def _generator(seed=1):
    np.random.seed(seed)
    return ForestPointGenerator(min_distance=5.0, max_attempts=30)


def _min_pair_distance(points):
    pts = np.asarray(points)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist[np.diag_indices(len(pts))] = np.inf
    return float(dist.min())


def test_points_keep_the_minimum_distance_and_stay_inside_the_polygon():
    polygon = box(0, 0, 120, 90)
    points = _generator().generate_points(polygon, tree_density=1.0)

    assert len(points) > 100
    assert _min_pair_distance(points) >= 5.0 - 1e-9
    assert all(polygon.contains(Point(x, y)) for x, y in points)


def test_density_scales_the_spacing():
    polygon = box(0, 0, 200, 200)
    sparse = _generator().generate_points(polygon, tree_density=0.25)  # Abstand 10 m
    dense = _generator().generate_points(polygon, tree_density=1.0)  # Abstand 5 m

    assert _min_pair_distance(sparse) >= 10.0 - 1e-9
    assert 3.0 < len(dense) / len(sparse) < 5.0  # ~4x mehr Punkte bei halbem Abstand


def test_packing_is_dense_like_bridson():
    points = _generator().generate_points(box(0, 0, 300, 300), tree_density=1.0)

    per_cell = len(points) * 5.0**2 / (300 * 300)
    assert 0.55 < per_cell < 0.85  # Poisson-Disk füllt ca. 0,7 Punkte je r²


def test_points_respect_holes_and_concave_shapes():
    ring = Polygon(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        holes=[[(30, 30), (70, 30), (70, 70), (30, 70)]],
    )
    l_shape = Polygon([(0, 0), (80, 0), (80, 30), (30, 30), (30, 80), (0, 80)])
    for polygon in (ring, l_shape):
        points = _generator().generate_points(polygon, tree_density=1.0)
        assert len(points) > 50
        assert all(polygon.contains(Point(x, y)) for x, y in points)


def test_road_buffer_removes_points_inside_it():
    generator = _generator()
    road = box(40, -10, 55, 200)
    generator.set_road_buffer(road)

    points = generator.generate_points(box(0, 0, 100, 100), tree_density=1.0)

    assert points
    assert not any(road.intersects(Point(x, y)) for x, y in points)
    assert any(x < 40 for x, _ in points) and any(x > 55 for x, _ in points)


def test_tiny_polygon_and_zero_density_yield_nothing():
    generator = _generator()
    assert generator.generate_points(box(0, 0, 0.5, 0.5), tree_density=1.0) == []
    assert generator.generate_points(box(0, 0, 50, 50), tree_density=0.0) == []


def test_line_points_skip_the_row_exclusion():
    generator = _generator()
    generator.set_row_exclusion(box(40, -5, 60, 5))
    from shapely.geometry import LineString

    points = generator.generate_points_along_line(LineString([(0, 0), (100, 0)]), spacing=10.0, jitter=0.0)

    assert points
    assert not any(40 <= x <= 60 for x, _ in points)


def test_forest_json_is_written_as_jsonl_with_ctxid(tmp_path):
    writer = ForestJSONWriter(tmp_path)
    instances = [
        {"type": "oak", "pos": [1.5, 2.25, 3.0], "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "scale": 1.1},
        {"type": "grüne_birke", "pos": [4.0, 5.0, 6.0], "rotationMatrix": [0, 1, 0, 1, 0, 0, 0, 0, 1], "scale": 0.9},
    ]

    result = writer.write_forest_json(instances)

    assert result["status"] == "success" and result["tree_count"] == 2
    text = (tmp_path / "forest.forest4.json").read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[-1] == "" and len(lines) == 3  # jede Instanz eine Zeile, abschließender Zeilenumbruch
    assert lines[0] == '{"type":"oak","pos":[1.5,2.25,3.0],"rotationMatrix":[1,0,0,0,1,0,0,0,1],"scale":1.1,"ctxid":0}'
    assert json.loads(lines[1])["type"] == "grüne_birke"  # kein ASCII-Escaping


def test_forest_json_handles_no_instances(tmp_path):
    result = ForestJSONWriter(tmp_path).write_forest_json([])

    assert result["status"] == "success" and result["tree_count"] == 0
    assert (tmp_path / "forest.forest4.json").read_text(encoding="utf-8") == ""


def test_thin_diagonal_polygon_is_filled_at_the_normal_density():
    # 1500 m lange, 12 m breite Diagonale: die Bounding Box (1060 x 1060 m) ist zu >99 % leer
    strip = Polygon([(0, 0), (1060, 1030), (1060, 1060), (0, 30)])
    points = _generator().generate_points(strip, tree_density=1.0)

    per_cell = len(points) * 5.0**2 / strip.area
    assert 0.5 < per_cell < 0.85
    assert _min_pair_distance(points) >= 5.0 - 1e-9
    assert all(strip.contains(Point(x, y)) for x, y in points)


def test_many_small_far_apart_parts_are_all_planted():
    parts = MultiPolygon([box(i * 300, 0, i * 300 + 40, 40) for i in range(8)])
    generator = _generator()
    per_part = [len(generator.generate_points(part, 1.0)) for part in parts.geoms]
    assert min(per_part) > 20 and max(per_part) < 2 * min(per_part)
