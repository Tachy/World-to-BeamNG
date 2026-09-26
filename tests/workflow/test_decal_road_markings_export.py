"""Tests for the DecalRoad export with smooth width transitions and marking lines
(TerrainWorkflow.export_decal_roads())."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _RecordingItems:
    def __init__(self):
        self.roads = {}

    def add_decal_road(self, name, nodes, material, **extra):
        self.roads[name] = {"nodes": nodes, "material": material, **extra}


def _export(polys):
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}), _export_structure_road_assets=lambda lines: None)
    count = TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})
    return count, stub.items.roads, stub.materials.materials


def _poly(road_id, points, **tags):
    coords = np.array([[float(x), float(y), 100.0] for x, y in points])
    return {"road_id": road_id, "trimmed_centerline": coords, "osm_tags": tags}


def _markings(roads):
    return {name: road for name, road in roads.items() if name.startswith("marking_")}


def test_two_lane_primary_gets_two_edge_lines_and_a_dashed_divider():
    count, roads, materials = _export([_poly(1, [(0, 0), (10, 0), (20, 0), (30, 0)], highway="primary", lanes="2")])

    assert count == 1  # the return value still counts only carriageways
    markings = _markings(roads)
    assert sorted(markings) == ["marking_1_0_0", "marking_1_1_0", "marking_1_2_0"]
    left, right, divider = markings["marking_1_0_0"], markings["marking_1_1_0"], markings["marking_1_2_0"]
    assert left["material"] == right["material"] == config.ROAD_MARKING_EDGE_MATERIAL
    assert divider["material"] == config.ROAD_MARKING_DIVIDER_MATERIAL
    assert [n[1] for n in left["nodes"]] == pytest.approx([3.0] * 4)  # 6.5 m / 2 - 0.25 m
    assert [n[1] for n in right["nodes"]] == pytest.approx([-3.0] * 4)
    assert [n[1] for n in divider["nodes"]] == pytest.approx([0.0] * 4)
    assert all(n[3] == config.ROAD_MARKING_LINE_WIDTH for n in left["nodes"])
    assert left["drivability"] == -1
    assert left["renderPriority"] == config.ROAD_MARKING_RENDER_PRIORITY
    assert divider["textureLength"] == pytest.approx(
        config.OSM_MAPPER.road_markings[config.ROAD_MARKING_DIVIDER_MATERIAL]["textureLength"]
    )
    assert {config.ROAD_MARKING_EDGE_MATERIAL, config.ROAD_MARKING_DIVIDER_MATERIAL} <= set(materials)


def test_single_lane_link_gets_edge_lines_only():
    _, roads, _ = _export([_poly(7, [(0, 0), (10, 0), (20, 0)], highway="primary_link")])

    assert sorted(_markings(roads)) == ["marking_7_0_0", "marking_7_1_0"]


@pytest.mark.parametrize(
    "tags",
    [
        {"highway": "residential"},
        {"highway": "track"},
        {"highway": "secondary", "lanes": "2", "lane_markings": "no"},
        {"highway": "secondary", "lanes": "2", "surface": "sett"},
    ],
)
def test_unmarked_roads_get_no_markings(tags):
    _, roads, materials = _export([_poly(1, [(0, 0), (10, 0), (20, 0)], **tags)])

    assert _markings(roads) == {}
    assert config.ROAD_MARKING_EDGE_MATERIAL not in materials


def test_markings_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "ROAD_MARKINGS_ENABLED", False)
    _, roads, _ = _export([_poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2")])

    assert _markings(roads) == {}


def test_width_transition_is_applied_to_road_and_followed_by_edge_line():
    polys = [
        _poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2"),
        _poly(2, [(20, 0), (30, 0), (40, 0)], highway="primary", lanes="3"),
    ]
    _, roads, _ = _export(polys)

    assert roads["road_1"]["nodes"][-1][3] == pytest.approx(8.125)  # mean of 6.5 and 9.75 m
    assert roads["road_1"]["nodes"][0][3] == pytest.approx(6.5)
    assert roads["road_2"]["nodes"][0][3] == pytest.approx(8.125)
    assert roads["marking_1_0_0"]["nodes"][-1][1] == pytest.approx(8.125 / 2 - 0.25)


def test_marking_nodes_keep_min_spacing():
    polys = [_poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2"),
             _poly(2, [(20, 0), (30, 0), (40, 0)], highway="primary", lanes="3")]
    _, roads, _ = _export(polys)

    for road in roads.values():
        pts = np.array(road["nodes"])[:, :2]
        assert np.linalg.norm(np.diff(pts, axis=0), axis=1).min() >= config.DECAL_ROAD_MIN_NODE_SPACING


def test_side_road_interrupts_main_edge_line_but_not_divider():
    polys = [
        _poly(1, [(-40, 0), (-20, 0), (0, 0)], highway="primary", lanes="2"),
        _poly(2, [(0, 0), (20, 0), (40, 0)], highway="primary", lanes="2"),
        _poly(3, [(0, 0), (0, 20), (0, 40)], highway="secondary", lanes="2"),
    ]
    _, roads, _ = _export(polys)

    clearance = 6.5 / 2 + config.ROAD_MARKING_JUNCTION_CLEARANCE  # 3.75 m
    assert roads["marking_1_0_0"]["nodes"][-1][0] == pytest.approx(-clearance, abs=0.02)  # left ends before the T-junction
    assert "marking_1_0_1" not in roads
    assert roads["marking_1_1_0"]["nodes"][-1][0] == pytest.approx(0.0)  # right edge line runs through
    assert roads["marking_1_2_0"]["nodes"][-1][0] == pytest.approx(0.0)  # guide line runs through
    assert roads["marking_2_0_0"]["nodes"][0][0] == pytest.approx(clearance, abs=0.02)
    for side_edge in ("marking_3_0_0", "marking_3_1_0"):
        assert roads[side_edge]["nodes"][0][1] == pytest.approx(clearance, abs=0.02)  # side road starts at the edge


def test_track_junction_does_not_interrupt_edge_line():
    polys = [
        _poly(1, [(-40, 0), (0, 0), (40, 0)], highway="primary", lanes="2"),
        _poly(2, [(0, 0), (0, 20), (0, 40)], highway="track"),
    ]
    _, roads, _ = _export(polys)

    assert "marking_1_0_1" not in roads
    assert roads["marking_1_0_0"]["nodes"][-1][0] == pytest.approx(40.0)


def test_edge_lines_meet_at_kinked_continuation():
    kink = np.radians(12.0)
    polys = [
        _poly(1, [(-40, 0), (-20, 0), (0, 0)], highway="primary", lanes="2"),
        _poly(2, [(0, 0), (20 * np.cos(kink), 20 * np.sin(kink)), (40 * np.cos(kink), 40 * np.sin(kink))], highway="primary", lanes="2"),
    ]
    _, roads, _ = _export(polys)

    for line_idx in range(3):
        end = roads[f"marking_1_{line_idx}_0"]["nodes"][-1]
        start = roads[f"marking_2_{line_idx}_0"]["nodes"][0]
        assert end[:2] == pytest.approx(start[:2], abs=1e-6)


def test_oblique_side_road_does_not_cut_divider():
    a = np.radians(30.0)
    polys = [
        _poly(1, [(-40, 0), (-20, 0), (0, 0)], highway="primary", lanes="2"),
        _poly(2, [(0, 0), (20, 0), (40, 0)], highway="primary", lanes="2"),
        _poly(3, [(0, 0), (20 * np.cos(a), 20 * np.sin(a)), (40 * np.cos(a), 40 * np.sin(a))], highway="secondary", lanes="2"),
    ]
    _, roads, _ = _export(polys)

    assert roads["marking_1_2_0"]["nodes"][-1][0] == pytest.approx(0.0)
    assert roads["marking_2_2_0"]["nodes"][0][0] == pytest.approx(0.0)
    assert "marking_2_2_1" not in roads
    assert "marking_2_1_1" not in roads  # right (opposite) edge line continuous


def test_markings_and_better_surfaces_are_drawn_on_top():
    # DecalRoads are drawn in descending renderPriority: smaller value = later = on top
    polys = [
        _poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2"),
        _poly(2, [(0, 20), (10, 20), (20, 20)], highway="track"),
    ]
    _, roads, _ = _export(polys)

    asphalt, dirt = roads["road_1"]["renderPriority"], roads["road_2"]["renderPriority"]
    assert asphalt < dirt  # asphalt above dirt track (at T-junctions the ends overlap)
    assert roads["marking_1_0_0"]["renderPriority"] < asphalt  # lines above the asphalt


def test_long_road_decal_is_split_into_chunks_within_the_area_budget():
    # BeamNG draws only a limited amount of geometry per DecalRoad (see geometry/decal_chunks.py)
    xs = [float(x) for x in range(0, 201, 1)]
    count, roads, _ = _export([_poly(1, [(x, 0) for x in xs], highway="primary", lanes="2")])

    chunks = sorted((name for name in roads if name.startswith("road_1_")), key=lambda n: int(n.rsplit("_", 1)[1]))
    assert len(chunks) >= 2 and "road_1" not in roads
    assert count == len(chunks)
    for name in chunks:
        nodes = np.array(roads[name]["nodes"])
        area = float((np.linalg.norm(np.diff(nodes[:, :2], axis=0), axis=1) * nodes[1:, 3]).sum())
        assert area <= config.ROAD_DECAL_MAX_AREA + 1e-6
    for first, second in zip(chunks, chunks[1:]):
        assert roads[first]["nodes"][-1] == roads[second]["nodes"][0]
    assert roads[chunks[0]]["nodes"][0][0] == 0.0 and roads[chunks[-1]]["nodes"][-1][0] == 200.0
    assert roads["marking_1_0_0"]["nodes"][-1][0] == pytest.approx(200.0)  # lines stay in one piece


def test_short_road_keeps_its_single_decal_name():
    _, roads, _ = _export([_poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2")])

    assert "road_1" in roads and not any(name.startswith("road_1_") for name in roads)


def test_road_decals_overlap_at_kinked_continuation_but_markings_still_meet():
    kink = np.radians(15.0)
    polys = [
        _poly(1, [(-40, 0), (-20, 0), (0, 0)], highway="primary", lanes="2"),
        _poly(2, [(0, 0), (20 * np.cos(kink), 20 * np.sin(kink)), (40 * np.cos(kink), 40 * np.sin(kink))], highway="primary", lanes="2"),
    ]
    _, roads, _ = _export(polys)

    road_1 = sorted(name for name in roads if name == "road_1" or name.startswith("road_1_"))
    assert roads[road_1[-1]]["nodes"][-1][0] > 0.5  # extended beyond the joint point (closes the outer wedge)
    assert roads["marking_1_0_0"]["nodes"][-1][:2] == pytest.approx(roads["marking_2_0_0"]["nodes"][0][:2], abs=1e-6)


def test_three_lane_two_way_primary_gets_a_solid_double_centre_line():
    points = [(x, 0) for x in range(0, 31, 10)]
    _, roads, _ = _export([_poly(1, points, highway="primary", lanes="3", **{"lanes:forward": "1", "lanes:backward": "2"})])

    solid = [r for r in _markings(roads).values() if r["material"] == config.ROAD_MARKING_CENTER_MATERIAL]
    dashed = [r for r in _markings(roads).values() if r["material"] == config.ROAD_MARKING_DIVIDER_MATERIAL]
    width = 3 * 3.25
    boundary = -width / 2.0 + width / 3.0  # after the one forward lane (right-hand traffic: right side)
    shift = (config.ROAD_MARKING_CENTER_GAP + config.ROAD_MARKING_LINE_WIDTH) / 2.0
    centre_ys = sorted(r["nodes"][0][1] for r in solid if abs(r["nodes"][0][1]) < width / 2.0 - 0.5)
    assert centre_ys == pytest.approx([boundary - shift, boundary + shift])
    assert len(dashed) == 1  # between the two backward lanes


def _widths(roads, road_id):
    """x -> width over all DecalRoad pieces of a road (long carriageways are split, see decal_chunks.py)."""
    names = [n for n in roads if n == f"road_{road_id}" or n.startswith(f"road_{road_id}_")]
    return {round(node[0]): node[3] for name in names for node in roads[name]["nodes"]}


def test_lane_change_from_two_to_three_lanes_blends_over_100_m():
    a = _poly(1, [(x, 0) for x in range(-200, 1)], highway="primary", lanes="2")
    b = _poly(2, [(x, 0) for x in range(0, 201)], highway="primary", lanes="3")

    _, roads, _ = _export([a, b])

    widths_a = _widths(roads, 1)
    assert widths_a[-50] == pytest.approx(6.5) and widths_a[-25] > 6.5  # zone starts 50 m before the joint


def test_width_change_toward_a_bridge_lies_on_the_road(monkeypatch):
    monkeypatch.setattr(config, "STRUCTURE_AI_ROADS", True)
    road = _poly(1, [(x, 0) for x in range(-200, 1)], highway="primary", lanes="3")
    bridge = dict(_poly(2, [(x, 0) for x in range(0, 201)], highway="primary", lanes="2", bridge="yes"), structure_type="bridge")

    _, roads, _ = _export([road, bridge])

    widths = _widths(roads, 1)
    assert widths[-100] == pytest.approx(9.75) and widths[0] == pytest.approx(6.5)
    assert all(w == pytest.approx(6.5) for w in _widths(roads, 2).values())


def _export_lines(polys):
    captured = {}
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}),
                           _export_structure_road_assets=lambda lines: captured.setdefault("lines", lines))
    TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})
    return captured.get("lines", [])


@pytest.mark.parametrize("structure, expect_double", [("tunnel", True), ("gallery", True), ("bridge", False)])
def test_two_lane_tunnels_and_galleries_get_a_double_centre_line_bridges_do_not(monkeypatch, structure, expect_double):
    monkeypatch.setattr(config, "STRUCTURE_AI_ROADS", True)
    poly = dict(_poly(9, [(x, 0) for x in range(0, 61, 10)], highway="primary", lanes="2"), structure_type=structure)

    lines = _export_lines([poly])

    solid = [l for l in lines if l["material"] == config.ROAD_MARKING_CENTER_MATERIAL and abs(l["nodes"][0][1]) < 1.0]
    dashed = [l for l in lines if l["material"] == config.ROAD_MARKING_DIVIDER_MATERIAL]
    assert (len(solid) == 2) == expect_double
    assert (len(dashed) == 0) == expect_double


def test_two_lane_surface_road_keeps_its_dashed_centre_line():
    _, roads, _ = _export([_poly(1, [(0, 0), (10, 0), (20, 0)], highway="primary", lanes="2")])

    assert any(r["material"] == config.ROAD_MARKING_DIVIDER_MATERIAL for r in _markings(roads).values())
