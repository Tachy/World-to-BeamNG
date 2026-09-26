"""Guard rails in TerrainWorkflow.export_decal_roads(): planned on the finished heightmap, handed over as forest items."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.io.guardrail_assets import GUARDRAIL_ITEMS
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _RecordingItems:
    def __init__(self):
        self.roads = {}

    def add_decal_road(self, name, nodes, material, **extra):
        self.roads[name] = {"nodes": nodes, "material": material, **extra}


def _export(highway, monkeypatch, enabled=True):
    from world_to_beamng import config

    monkeypatch.setattr(config, "GUARDRAILS_ENABLED", enabled)
    heights = np.full((200, 200), 100.0)
    heights[:95, :] = 90.0  # 10 m lower south of y = 95 (the road runs at y = 100; probe 4 m beyond the edge at ~92.5)
    centerline = np.array([[x, 100.0, 100.0] for x in np.arange(20.0, 180.5, 1.0)])
    mesh_data = {
        "road_slope_polygons_2d": [{"road_id": 1, "trimmed_centerline": centerline, "osm_tags": {"highway": highway}}],
        "heightmap": heights,
        "terrain_origin_x": 0.0,
        "terrain_origin_y": 0.0,
    }
    stub = SimpleNamespace(items=_RecordingItems(), materials=SimpleNamespace(materials={}),
                           _export_structure_road_assets=lambda lines: None)
    TerrainWorkflow.export_decal_roads(stub, mesh_data)
    return mesh_data.get("guardrail_instances")


def test_road_above_a_drop_gets_guard_rail_forest_items(monkeypatch):
    items = _export("secondary", monkeypatch)

    types = [i["type"] for i in items]
    assert types.count(GUARDRAIL_ITEMS["start"]) == 1 and types.count(GUARDRAIL_ITEMS["end"]) == 1
    segments = [i for i in items if i["type"] == GUARDRAIL_ITEMS["segment"]]
    assert len(segments) >= 50  # the whole 160 m stretch
    assert all(i["pos"][1] < 100.0 - 3.25 for i in segments)  # on the drop side only


def test_paths_get_no_guard_rails(monkeypatch):
    assert not _export("footway", monkeypatch)


def test_guard_rails_can_be_switched_off(monkeypatch):
    assert not _export("secondary", monkeypatch, enabled=False)
