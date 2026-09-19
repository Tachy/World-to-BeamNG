"""Vereinigte Straßenflächen für Ausschlusszonen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from shapely.geometry import LineString, Point

from world_to_beamng.geometry.road_surfaces import union_road_surfaces


def _road(points, width=6.0):
    polygon = LineString(points).buffer(width / 2.0, cap_style=2)
    return {"road_polygon": np.array(polygon.exterior.coords[:-1])}


def test_no_roads_gives_none():
    assert union_road_surfaces(None) is None
    assert union_road_surfaces([]) is None
    assert union_road_surfaces([{"road_polygon": np.array([[0, 0], [1, 1]])}]) is None


def test_crossing_roads_are_merged_into_one_surface():
    surface = union_road_surfaces([_road([(0, 0), (100, 0)]), _road([(50, -50), (50, 50)])])

    assert surface.geom_type == "Polygon"
    assert surface.contains(Point(50, 0)) and surface.contains(Point(10, 0)) and surface.contains(Point(50, 30))
    assert not surface.contains(Point(10, 20))


def test_simplification_stays_within_the_tolerance():
    # dicht aufgelöste Kurve: viele Eckpunkte
    t = np.linspace(0, np.pi, 400)
    road = _road(list(zip(100 * np.cos(t), 100 * np.sin(t))))
    exact = union_road_surfaces([road], tolerance=0.0)
    simple = union_road_surfaces([road], tolerance=0.1)

    assert len(simple.exterior.coords) < len(exact.exterior.coords) / 3
    assert exact.hausdorff_distance(simple) <= 0.11


def test_self_intersecting_polygon_is_repaired():
    bowtie = {"road_polygon": np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=float)}

    surface = union_road_surfaces([bowtie, _road([(30, 0), (60, 0)])])

    assert surface is not None and surface.contains(Point(45, 0))
