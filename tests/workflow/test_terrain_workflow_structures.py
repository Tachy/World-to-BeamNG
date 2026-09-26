"""Tests for the terrain exception and the DecalRoad exclusion of bridges/tunnels/galleries
(TerrainWorkflow.process_tile() wiring and export_decal_roads())."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

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


def _structure_polys():
    line = np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]])
    return [
        {"road_id": 1, "trimmed_centerline": line, "osm_tags": {"highway": "residential"}, "structure_type": "surface"},
        {"road_id": 2, "trimmed_centerline": line + [0, 20, 30], "osm_tags": {"highway": "primary", "bridge": "yes"}, "structure_type": "bridge"},
        {"road_id": 3, "trimmed_centerline": line + [0, 40, -30], "osm_tags": {"highway": "trunk", "tunnel": "yes"}, "structure_type": "tunnel"},
        {"road_id": 4, "trimmed_centerline": line + [0, 60, 0], "osm_tags": {"highway": "primary", "tunnel": "avalanche_protector"}, "structure_type": "gallery"},
        # path "tunnels" get no tube (config.TUNNEL_EXCLUDED_HIGHWAYS) - a decal there would land on the terrain
        {"road_id": 5, "trimmed_centerline": line + [0, 80, -50], "osm_tags": {"highway": "path", "tunnel": "yes"}, "structure_type": "tunnel"},
    ]


def _stub():
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}), structure_markings=None)
    stub._export_structure_road_assets = lambda lines: setattr(stub, "structure_markings", lines)
    return stub


def test_structure_roads_get_an_invisible_ai_decal_road_at_structure_height(monkeypatch):
    from world_to_beamng import config

    monkeypatch.setattr(config, "STRUCTURE_AI_ROADS", True)
    stub = _stub()

    TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": _structure_polys()})
    roads = stub.items.roads

    assert sorted(n for n in roads if n.startswith("road_")) == ["road_1", "road_2", "road_3", "road_4"]
    for name in ("road_2", "road_3", "road_4"):
        assert roads[name]["material"] == config.STRUCTURE_AI_ROAD_MATERIAL
        assert roads[name]["drivability"] > 0  # part of the AI road network
    assert roads["road_1"]["material"] != config.STRUCTURE_AI_ROAD_MATERIAL
    assert [n[2] for n in roads["road_2"]["nodes"]] == [130.0, 130.0]  # deck height, 1:1 the profile
    assert config.STRUCTURE_AI_ROAD_MATERIAL in stub.materials.materials


def test_structure_markings_are_handed_to_the_mesh_export_not_exported_as_decals(monkeypatch):
    from world_to_beamng import config

    monkeypatch.setattr(config, "STRUCTURE_AI_ROADS", True)
    stub = _stub()

    TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": _structure_polys()})

    assert not any(n.startswith(("marking_2_", "marking_3_", "marking_4_")) for n in stub.items.roads)
    assert {line["name"].split("_")[1] for line in stub.structure_markings} == {"2", "3", "4"}


def test_export_decal_roads_skips_structures_when_disabled(monkeypatch):
    from world_to_beamng import config

    monkeypatch.setattr(config, "STRUCTURE_AI_ROADS", False)
    stub = _stub()

    TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": _structure_polys()})

    assert list(stub.items.roads) == ["road_1"]


def test_export_decal_roads_still_works_without_a_structure_type_field():
    # Regression: existing callers/tests that do not set "structure_type" stay unchanged (surface default).
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}), _export_structure_road_assets=lambda lines: None)
    polys = [{"road_id": 1, "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]]), "osm_tags": {"highway": "residential"}}]

    count = TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})

    assert count == 1


def test_invisible_road_material_is_alpha_tested_like_vanilla_road_invisible():
    from world_to_beamng.workflow.terrain_workflow import _invisible_road_material

    entry = _invisible_road_material("structure_road_invisible")

    assert entry["name"] == entry["mapTo"] == "structure_road_invisible"
    assert entry["alphaTest"] is True and entry["alphaRef"] == 127
    assert entry["Stages"][0]["colorMap"].endswith("structure_road_invisible.png")
    # colorMap is only read by version-1 materials (like vanilla road_invisible, no "version" field); a 1.5 (PBR)
    # material ignores it, has no alpha to test and rendered black on the terrain above the tunnel (in game 2026-09-26)
    assert entry.get("version", 1) == 1
    assert entry["castShadows"] is False


def _line(*points):
    return {"trimmed_centerline": np.array([[x, y, 100.0] for x, y in points]), "osm_tags": {}}


def test_gallery_and_approach_road_end_flush_at_the_same_line():
    from world_to_beamng.workflow.terrain_workflow import _gallery_embankment_cuts

    road = _line((-30.0, 0.0), (0.0, 0.0))
    gallery = _line((0.0, 0.0), (50.0, 0.0))

    _gallery_embankment_cuts([road], [gallery], tol=0.5)

    assert road["embankment_cuts"] == [((0.0, 0.0), pytest.approx((1.0, 0.0)))]
    starts = {c[0]: c[1] for c in gallery["embankment_cuts"]}
    assert starts[(0.0, 0.0)] == pytest.approx((-1.0, 0.0))
    assert starts[(50.0, 0.0)] == pytest.approx((1.0, 0.0))  # free end (e.g. tunnel transition): perpendicular


def test_kinked_transition_is_cut_on_the_bisector_so_no_wedge_stays_open():
    from world_to_beamng.workflow.terrain_workflow import _gallery_embankment_cuts

    road = _line((-30.0, -30.0), (0.0, 0.0))  # reaches the gallery at 45 degrees
    gallery = _line((50.0, 0.0), (0.0, 0.0))  # digitized toward the road

    _gallery_embankment_cuts([road], [gallery], tol=0.5)

    (_, road_n), = road["embankment_cuts"]
    gallery_n = dict(gallery["embankment_cuts"])[(0.0, 0.0)]
    assert gallery_n == pytest.approx(tuple(-v for v in road_n))  # one shared line
    road_out, gallery_out = np.array([1.0, 1.0]) / np.sqrt(2.0), np.array([-1.0, 0.0])
    assert np.dot(road_n, road_out) == pytest.approx(-np.dot(road_n, gallery_out))  # bisector: same angle to both


def test_roads_away_from_galleries_keep_their_round_end_caps():
    from world_to_beamng.workflow.terrain_workflow import _gallery_embankment_cuts

    road = _line((100.0, 0.0), (130.0, 0.0))
    gallery = _line((0.0, 0.0), (50.0, 0.0))

    _gallery_embankment_cuts([road], [gallery], tol=0.5)

    assert "embankment_cuts" not in road
