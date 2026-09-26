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


def test_bridge_footprint_covers_deck_curbs_and_margin():
    from shapely.geometry import Polygon

    from world_to_beamng.workflow.terrain_workflow import _bridge_footprints

    bridge = {
        "road_polygon": np.array([[0.0, -3.25], [40.0, -3.25], [40.0, 3.25], [0.0, 3.25]]),
        "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [40.0, 0.0, 100.0]]),
    }

    (footprint,) = _bridge_footprints([bridge], extra=0.9)
    polygon = Polygon(footprint["road_polygon"])

    assert polygon.bounds[1] == pytest.approx(-4.15) and polygon.bounds[3] == pytest.approx(4.15)
    assert footprint["trimmed_centerline"] is bridge["trimmed_centerline"]


def test_gets_decal_road_skips_a_tunnel_piece_that_belonged_to_a_dropped_pass_through_chain():
    from world_to_beamng.workflow.terrain_workflow import _gets_decal_road

    road = {"structure_type": "tunnel", "road_id": 5, "osm_tags": {"highway": "trunk"}}

    assert not _gets_decal_road(road, dropped_ids={5})
    assert _gets_decal_road(road, dropped_ids={6})  # a different piece was dropped, this one wasn't
    assert _gets_decal_road(road)  # default: nothing dropped


def test_dropped_tunnel_road_ids_is_every_piece_of_a_chain_that_did_not_survive_filtering():
    from world_to_beamng.workflow.terrain_workflow import _dropped_tunnel_road_ids

    all_ids = {1, 2, 3, 4}  # chain A = pieces 1+2 (dropped), chain B = piece 3 (kept), piece 4 (kept, single)
    kept_plans = [{"id": 3, "piece_ids": [3]}, {"id": 4, "piece_ids": [4]}]

    assert _dropped_tunnel_road_ids(all_ids, kept_plans) == frozenset({1, 2})


def test_bridge_photo_areas_cover_deck_curbs_and_the_photo_margin(monkeypatch):
    from shapely.geometry import Polygon

    from world_to_beamng import config
    from world_to_beamng.workflow.terrain_workflow import _bridge_photo_areas

    monkeypatch.setattr(config, "BRIDGE_CURB_WIDTH", 0.4)
    monkeypatch.setattr(config, "BRIDGE_PHOTO_FILL_MARGIN", 2.0)
    bridge = {
        "structure_type": "bridge",
        "road_polygon": np.array([[0.0, -3.25], [40.0, -3.25], [40.0, 3.25], [0.0, 3.25]]),
        "trimmed_centerline": np.array([[0.0, 0.0, 100.0], [40.0, 0.0, 100.0]]),
    }
    tunnel = {**bridge, "structure_type": "tunnel"}

    (area,) = _bridge_photo_areas([bridge, tunnel])

    assert area.shape[1] == 2
    assert Polygon(area).bounds[1] == pytest.approx(-5.65) and Polygon(area).bounds[3] == pytest.approx(5.65)


def test_road_width_specs_start_a_split_branch_with_its_slot_width():
    from world_to_beamng import config
    from world_to_beamng.workflow.terrain_workflow import _road_width_specs

    def poly(road_id, coords, **extra):
        centerline = np.array([(x, y, 100.0) for x, y in coords])
        return {"road_id": road_id, "trimmed_centerline": centerline, "structure_type": "surface",
                "osm_tags": {"highway": "primary", "lanes": "2"}, **extra}

    width = config.OSM_MAPPER.get_road_properties({"highway": "primary", "lanes": "2"})["width"]
    trunk = poly(1, [(-40.0, 0.0), (0.0, 0.0)], lane_split_trunk={"end"})
    branch = poly(2, [(0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (60.0, 0.0)],
                  lane_split_branch={"end": "start", "slot_width": width - 2.0, "hold": 0.0, "length": 30.0})

    specs, node_lists = _road_width_specs([trunk, branch])

    assert node_lists[0][-1][3] == pytest.approx(width)
    assert node_lists[1][0][3] == pytest.approx(width - 2.0)
    assert node_lists[1][-1][3] == pytest.approx(width)


def test_bridge_groups_join_the_bridges_of_a_lane_split_and_their_nearby_continuations():
    from world_to_beamng.workflow.terrain_workflow import _bridge_groups

    def bridge(road_id, coords, **extra):
        return {"road_id": road_id, "structure_type": "bridge", "trimmed_centerline": np.array([(x, y, 100.0) for x, y in coords]), **extra}

    node = (0.0, 0.0)
    trunk = bridge(1, [(-40.0, 0.0), (0.0, 0.0)], lane_split_trunk_nodes=[node])
    main = bridge(2, [(0.0, 0.0), (15.0, 0.0)], lane_split_branch={"node": node})
    ramp = bridge(3, [(0.0, 4.9), (30.0, 12.0)], lane_split_branch={"node": node})
    main_next = bridge(4, [(15.0, 0.0), (60.0, 0.0)])  # continues the main road 15 m from the node: same structure
    far_away = bridge(5, [(60.0, 0.0), (120.0, 0.0)])  # joint 60 m from the node: its own bridge
    other = bridge(6, [(500.0, 0.0), (540.0, 0.0)])

    groups = _bridge_groups([trunk, main, ramp, main_next, far_away, other], reach=50.0)

    assert groups[1] == groups[2] == groups[3] == groups[4]
    assert 5 not in groups and 6 not in groups


def test_bridge_stems_only_where_the_trunk_itself_is_a_bridge():
    from world_to_beamng.workflow.terrain_workflow import _bridge_stems

    node = (0.0, 0.0)
    mark = {"node": node, "axis": (1.0, 0.0), "left_normal": (0.0, 1.0), "hold": 30.0, "trunk_width": 13.0}
    trunk = {"road_id": 1, "osm_tags": {"highway": "primary", "lanes": "4"}, "lane_split_trunk_nodes": [node]}
    branch = {"road_id": 2, "osm_tags": {"highway": "primary_link"}, "lane_split_branch": mark}
    lonely = {"road_id": 3, "osm_tags": {"highway": "primary_link"}, "lane_split_branch": {**mark, "node": (500.0, 0.0)}}

    stems = _bridge_stems([trunk, branch, lonely])

    assert stems[2]["width"] == 13.0 and stems[2]["hold"] == 30.0 and stems[2]["node"] == node
    assert stems[2]["deck_material"].endswith("_structure")
    assert 3 not in stems  # its trunk is no bridge: the split lies on the ground
