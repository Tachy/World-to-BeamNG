"""Tests for the terrain exception and the DecalRoad exclusion of bridges/tunnels/galleries
(TerrainWorkflow.process_tile() wiring and export_decal_roads())."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.geometry.road_structures import split_by_structure_type
from world_to_beamng.terrain.road_embedding import embed_roads_into_heightmap
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


def _poly(road_id, structure_type, z):
    coords = np.array([[0.0, 5.0, z], [10.0, 5.0, z]])
    polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return {"road_id": road_id, "road_polygon": polygon, "trimmed_centerline": coords, "osm_tags": {}, "structure_type": structure_type}


def test_bridge_and_tunnel_polygons_are_not_embedded_into_the_heightmap():
    roads = [_poly(1, "surface", z=150.0), _poly(2, "bridge", z=200.0), _poly(3, "tunnel", z=90.0)]
    heights = np.full((20, 20), 100.0)

    surface_roads, structure_roads = split_by_structure_type(roads)
    assert {r["road_id"] for r in structure_roads} == {2, 3}

    result = embed_roads_into_heightmap(heights, 0.0, 0.0, 1.0, surface_roads)

    assert result[5, 5] == 150.0  # only the surface road (1) was embedded


class _RecordingItems:
    def __init__(self):
        self.roads = {}

    def add_decal_road(self, name, nodes, material, **extra):
        self.roads[name] = {"nodes": nodes, "material": material, **extra}


def test_export_decal_roads_skips_bridges_tunnels_and_galleries():
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}))
    polys = [
        {"road_id": 1, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "residential"}, "structure_type": "surface"},
        {"road_id": 2, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "primary", "bridge": "yes"}, "structure_type": "bridge"},
        {"road_id": 3, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "trunk", "tunnel": "yes"}, "structure_type": "tunnel"},
        {"road_id": 4, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"tunnel": "avalanche_protector"}, "structure_type": "gallery"},
    ]

    count = TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})

    assert count == 1
    assert list(stub.items.roads) == ["road_1"]


def test_export_decal_roads_still_works_without_a_structure_type_field():
    # Regression: existing callers/tests that do not set "structure_type" stay unchanged (surface default).
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}))
    polys = [{"road_id": 1, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "residential"}}]

    count = TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})

    assert count == 1
