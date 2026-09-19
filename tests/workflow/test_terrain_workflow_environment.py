"""Tests: Nebelhöhe aus dem Gelände und Wasser-Raster (TerrainWorkflow)."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


class _Items:
    def __init__(self):
        self.base = {}
        self.objects = {}

    def set_base_line_fields(self, name, **fields):
        self.base.setdefault(name, {}).update(fields)

    def add_item(self, name, item_class="TSStatic", position=(0, 0, 0), scale=(1, 1, 1), **fields):
        self.objects[name] = {"class": item_class, "scale": list(scale), **fields}


def test_fog_atmosphere_height_is_the_highest_terrain_point_plus_margin():
    stub = SimpleNamespace(items=_Items())
    heights = np.full((8, 8), 300.0)
    heights[3, 4] = 689.4

    TerrainWorkflow._set_fog_height(stub, heights)

    assert stub.items.base["the_level_info"]["fogAtmosphereHeight"] == pytest.approx(689.4 + config.ENV_FOG_HEIGHT_MARGIN, abs=0.1)


def _pond(scale):
    return {"rivers": [], "ponds": [{"name": "pond_0", "blocks": [{"position": [0, 0, 100], "scale": scale, "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]}]}]}


def test_water_block_grid_is_never_larger_than_the_block():
    # BeamNG warnt sonst "gridElementSize 5 is larger than scale ..." und kürzt selbst
    stub = SimpleNamespace(items=_Items())

    TerrainWorkflow.export_water(stub, {"water": _pond([2.2, 1.1, 3.0])})
    TerrainWorkflow.export_water(stub, {"water": {"rivers": [], "ponds": [{"name": "pond_1", "blocks": [{"position": [0, 0, 100], "scale": [40.0, 30.0, 3.0]}]}]}})

    assert stub.items.objects["pond_0_0"]["gridElementSize"] == pytest.approx(1.1)
    assert stub.items.objects["pond_1_0"]["gridElementSize"] <= 5.0
    assert stub.items.objects["pond_1_0"]["gridElementSize"] > 1.1
