"""Tests für drop_close_nodes() und den DecalRoad-Export mit zu nah beieinander
liegenden Knoten.

Hintergrund (Eichgasse, OSM-Way 33870636): Ein DecalRoad, dessen erstes Segment
nur 0,10 m lang war (Rest vom Junction-Schnitt neben einem Resample-Punkt),
wurde von BeamNG überhaupt nicht gezeichnet. Nach dem Entfernen dieses einen
Knotens erschien das Decal - verifiziert im Spiel. Ein Segment von 0,32 m
Länge (Nachbarstück) funktionierte dagegen.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.geometry.polygon import drop_close_nodes
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


def _nodes(xs, width=5.0):
    return [[float(x), 0.0, 100.0 + i, width] for i, x in enumerate(xs)]


def _min_gap(nodes):
    pts = np.array(nodes)[:, :2]
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).min())


def test_drop_close_nodes_keeps_well_spaced_nodes_unchanged():
    nodes = _nodes([0.0, 0.8, 1.6, 2.4])

    assert drop_close_nodes(nodes, 0.5) == nodes


def test_drop_close_nodes_removes_second_node_when_start_segment_is_too_short():
    # Eichgasse-Fall: Startsegment 0,10 m
    nodes = _nodes([0.0, 0.10, 0.9, 1.7])

    result = drop_close_nodes(nodes, 0.5)

    assert result[0] == nodes[0]  # Startpunkt exakt erhalten (Junction-Anschluss)
    assert [n[0] for n in result] == [0.0, 0.9, 1.7]
    assert _min_gap(result) >= 0.5


def test_drop_close_nodes_removes_close_node_in_the_middle():
    nodes = _nodes([0.0, 0.8, 0.85, 1.65, 2.45])

    result = drop_close_nodes(nodes, 0.5)

    assert [n[0] for n in result] == [0.0, 0.8, 1.65, 2.45]


def test_drop_close_nodes_keeps_exact_end_point_when_last_segment_is_too_short():
    nodes = _nodes([0.0, 0.8, 1.6, 1.7])

    result = drop_close_nodes(nodes, 0.5)

    assert result[-1] == nodes[-1]  # Endpunkt exakt erhalten
    assert [n[0] for n in result] == [0.0, 0.8, 1.7]


def test_drop_close_nodes_preserves_z_and_width_of_kept_nodes():
    nodes = _nodes([0.0, 0.1, 1.0], width=6.5)

    result = drop_close_nodes(nodes, 0.5)

    assert result == [nodes[0], nodes[2]]
    assert all(n[3] == 6.5 for n in result)


def test_drop_close_nodes_returns_empty_for_unusably_short_road():
    assert drop_close_nodes(_nodes([0.0, 0.1, 0.2]), 0.5) == []
    assert drop_close_nodes(_nodes([0.0]), 0.5) == []
    assert drop_close_nodes([], 0.5) == []


def test_min_node_spacing_config_is_above_failing_and_below_typical_spacing():
    # 0,10 m fiel in BeamNG aus; typischer Abstand nach dem Resampling ~0,8 m.
    assert 0.10 < config.DECAL_ROAD_MIN_NODE_SPACING < 0.8


class _RecordingItems:
    def __init__(self):
        self.roads = {}

    def add_decal_road(self, name, nodes, material, **extra):
        self.roads[name] = {"nodes": nodes, "material": material, **extra}


def _export(polys):
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}))
    count = TerrainWorkflow.export_decal_roads(stub, {"road_slope_polygons_2d": polys})
    return count, stub.items.roads


def _poly(road_id, xs):
    coords = np.array([[x, 0.0, 100.0] for x in xs])
    return {"road_id": road_id, "trimmed_centerline": coords, "osm_tags": {"highway": "residential"}}


def test_export_decal_roads_filters_close_nodes():
    count, roads = _export([_poly(1, [0.0, 0.10, 0.9, 1.7, 2.5])])

    assert count == 1
    nodes = roads["road_1"]["nodes"]
    assert nodes[0][0] == 0.0 and nodes[-1][0] == 2.5
    assert _min_gap(nodes) >= config.DECAL_ROAD_MIN_NODE_SPACING


def test_export_decal_roads_skips_road_shorter_than_min_spacing():
    count, roads = _export([_poly(1, [0.0, 0.1, 0.2]), _poly(2, [0.0, 0.8, 1.6])])

    assert count == 1
    assert list(roads) == ["road_2"]
