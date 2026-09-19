"""
Tests für den Gebäude-Export als EIN Objekt auf der Gesamtfläche (statt 500-m-Kacheln).

Wie die Straßen (DecalRoads) über die ganze Fläche laufen, gibt es jetzt EINE Gebäude-DAE und EIN TSStatic.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.managers.item_manager import ItemManager
from world_to_beamng.workflow.building_workflow import (
    SINGLE_BUILDINGS_NAME,
    BuildingWorkflow,
    group_buildings,
    remove_stale_building_daes,
)


def _building(x, y):
    # bounds = (min_x, min_y, min_z, max_x, max_y, max_z) wie im Exporter
    return {"bounds": (x - 5, y - 5, 0, x + 5, y + 5, 10)}


BUILDINGS = [_building(100, 100), _building(700, 100), _building(-900, 1500), _building(120, 130)]


def test_tiled_grouping_is_unchanged():
    groups = group_buildings(BUILDINGS, 500)

    assert sorted(groups) == [(-1000, 1500), (0, 0), (500, 0)]
    assert len(groups[(0, 0)]) == 2


def test_without_a_tile_size_all_buildings_form_one_group():
    groups = group_buildings(BUILDINGS, None)

    assert list(groups) == [(0, 0)]
    assert len(groups[(0, 0)]) == 4


def test_single_group_also_keeps_buildings_without_bounds():
    groups = group_buildings(BUILDINGS + [{"name": "ohne bounds"}], None)

    assert len(groups[(0, 0)]) == 5


def test_config_switches_to_one_object_by_default():
    assert config.BUILDINGS_AS_ONE_OBJECT is True


def test_item_and_shape_use_the_single_name(tmp_path):
    ItemManager.reset_instance()
    workflow = BuildingWorkflow.__new__(BuildingWorkflow)
    workflow.items = ItemManager.get_instance(tmp_path)

    workflow.add_items(BUILDINGS, 0, 0, name=SINGLE_BUILDINGS_NAME)

    assert set(workflow.items.items) == {"buildings"}
    item = workflow.items.items["buildings"]
    assert item["class"] == "TSStatic"
    assert item["shapeName"].endswith("/art/shapes/buildings/buildings.dae")
    assert item["position"] == [0, 0, 0]


def test_tiled_items_keep_their_old_names(tmp_path):
    ItemManager.reset_instance()
    workflow = BuildingWorkflow.__new__(BuildingWorkflow)
    workflow.items = ItemManager.get_instance(tmp_path)

    workflow.add_items(BUILDINGS, 500, 0)

    assert "buildings_tile_500_0" in workflow.items.items


def test_dae_is_written_under_the_single_name(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(config, "BEAMNG_DIR_BUILDINGS", tmp_path)
    import world_to_beamng.builders as builders

    class _Builder:
        def with_buildings(self, b):
            return self

        def with_bounds_filter(self, f):
            return self

        def build(self):
            return [object(), object()]

    monkeypatch.setattr(builders, "BuildingMeshBuilder", _Builder)
    workflow = BuildingWorkflow.__new__(BuildingWorkflow)
    workflow.dae = SimpleNamespace(export_multi_mesh=lambda output_path, meshes, with_uv: written.append(Path(output_path)))

    path = workflow.export_buildings(BUILDINGS, 0, 0, name=SINGLE_BUILDINGS_NAME)

    assert path == tmp_path / "buildings.dae" and written == [path]


def test_stale_tile_daes_are_removed_but_the_current_one_stays(tmp_path):
    for name in ("buildings_tile_0_0.dae", "buildings_tile_500_0.dae", "buildings_tile_500_0.cdae", "buildings.dae", "other.txt"):
        (tmp_path / name).write_bytes(b"x")

    removed = remove_stale_building_daes(tmp_path, keep={"buildings"})

    assert sorted(p.name for p in tmp_path.iterdir()) == ["buildings.dae", "other.txt"]
    assert removed == 3


def test_switching_back_to_tiles_removes_the_single_dae(tmp_path):
    for name in ("buildings.dae", "buildings_tile_0_0.dae"):
        (tmp_path / name).write_bytes(b"x")

    remove_stale_building_daes(tmp_path, keep={"buildings_tile_0_0"})

    assert sorted(p.name for p in tmp_path.iterdir()) == ["buildings_tile_0_0.dae"]
