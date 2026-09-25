"""Tests for tools/level_viewer/level_data.py: items, assets, terrain, forest and debug network loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from tests.viewer.conftest import MAX_HEIGHT, ORIGIN, PADDING, TER_SIZE
from tools.level_viewer.level_data import load_debug_network, load_forest, load_level, load_terrain, resolve_asset


def test_items_of_both_files_get_a_global_index(level_dir):
    level = load_level(level_dir)

    assert [item.index for item in level.items] == list(range(len(level.items)))
    spawn = level.of("SpawnSphere")[0]
    assert spawn.name == "spawn_east" and spawn.index == len(level.items) - 1
    assert spawn.source.endswith("PlayerDropPoints/items.level.json")
    assert [r.name for r in level.of("DecalRoad")] == ["road_a", "marking_road_a_edge"]
    assert level.of("Zone")[0].rotation.shape == (3, 3)
    assert level.of("DecalRoad")[0].material == "asphalt_road_standard"


def test_resolve_asset_handles_level_paths_only(level_dir):
    assert resolve_asset(level_dir, "/levels/test_level/test_level.ter") == level_dir / "test_level.ter"
    assert resolve_asset(level_dir, "levels/test_level/test_level.ter") == level_dir / "test_level.ter"
    assert resolve_asset(level_dir, "/assets/materials/road/x.dds") is None
    assert resolve_asset(level_dir, "levels/test_level/missing.dae") is None
    assert resolve_asset(level_dir, "") is None


def test_minimap_placement_is_read_from_info_json(level_dir):
    level = load_level(level_dir)

    assert level.minimap.x_min == ORIGIN[0] and level.minimap.y_max == ORIGIN[1] + TER_SIZE
    assert level.minimap.size_x == TER_SIZE


def test_terrain_is_cropped_to_the_non_hole_area_and_rises_to_the_north(level_dir):
    grid = load_terrain(load_level(level_dir), step=4)

    solid = TER_SIZE - PADDING
    assert grid.x[0] == ORIGIN[0] and grid.x[-1] == ORIGIN[0] + solid - 1  # last column kept despite the step
    assert grid.y[-1] == ORIGIN[1] + solid - 1
    assert not grid.hole.any()
    assert np.argmax(grid.z[:, 0]) == len(grid.y) - 1  # row 0 = south, heights grow with y
    assert grid.z[0, 0] == pytest.approx(ORIGIN[2])
    assert grid.z[1, 0] == pytest.approx(ORIGIN[2] + 4 * 400 * MAX_HEIGHT / 65536, rel=1e-5)


def test_terrain_height_and_material_readout(level_dir):
    grid = load_terrain(load_level(level_dir), step=4)

    assert grid.height_at(ORIGIN[0] + 5, ORIGIN[1] + 10.5) == pytest.approx(ORIGIN[2] + 10.5 * 400 * MAX_HEIGHT / 65536)
    assert grid.material_at(ORIGIN[0] + 15, ORIGIN[1] + 10) == "dirt"
    assert grid.material_at(ORIGIN[0] + 15, ORIGIN[1] + 50) == "grass"
    assert grid.material_at(ORIGIN[0] + TER_SIZE - 2, ORIGIN[1] + 5) == "(hole)"
    assert grid.height_at(ORIGIN[0] + TER_SIZE - 2, ORIGIN[1] + 5) is None
    assert grid.height_at(ORIGIN[0] - 10, ORIGIN[1]) is None


def test_forest_instances_are_read_from_jsonl(level_dir):
    forest = load_forest(load_level(level_dir))

    assert forest.types == ["oak", "pine"]
    assert forest.type_idx.tolist() == [0, 1, 0]
    assert forest.positions.shape == (3, 3)


def test_missing_debug_network_gives_an_empty_dict(tmp_path):
    assert load_debug_network(tmp_path / "missing.json") == {}
    assert load_debug_network(None) == {}
