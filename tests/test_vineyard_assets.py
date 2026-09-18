"""Tests für world_to_beamng.io.vineyard_assets.ensure_vineyard_assets().

Die Reben-Shapes (grape_vine, grape_vine_group) stammen aus BeamNGs italy-Level und
werden samt Materialien in den eigenen Level kopiert; die Forest-Items werden in
art/forest/managedItemData.json eingetragen.
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from world_to_beamng.io.vineyard_assets import ensure_vineyard_assets

ITALY_SHAPES = "levels/italy/art/shapes/trees/trees_italy/"
EAST_COAST_MATERIALS = "levels/east_coast_usa/art/shapes/trees/main.materials.json"


def _material(name):
    return {"name": name, "mapTo": name, "class": "Material", "Stages": [{}], "persistentId": f"orig-{name}"}


def _make_install(tmp_path, with_imposters=True):
    levels = tmp_path / "install" / "content" / "levels"
    levels.mkdir(parents=True)
    with zipfile.ZipFile(levels / "italy.zip", "w") as z:
        for stem in ("grape_vine", "grape_vine_group"):
            z.writestr(f"{ITALY_SHAPES}{stem}.dae", f"<dae {stem}/>")
            z.writestr(f"{ITALY_SHAPES}{stem}.cdae", f"cdae {stem}")
            if with_imposters:
                z.writestr(f"{ITALY_SHAPES}{stem}.dae.imposter.dds", "imp")
                z.writestr(f"{ITALY_SHAPES}{stem}.dae.imposter_normals.dds", "impn")
        z.writestr(
            f"{ITALY_SHAPES}main.materials.json",
            json.dumps({n: _material(n) for n in ("grape", "olive_trunk", "unrelated")}),
        )
    with zipfile.ZipFile(levels / "east_coast_usa.zip", "w") as z:
        z.writestr(EAST_COAST_MATERIALS, json.dumps({"leaves_strong": _material("leaves_strong")}))
    return tmp_path / "install"


def _level(tmp_path):
    level = tmp_path / "levels" / "world_to_beamng"
    level.mkdir(parents=True)
    return level


def test_copies_shapes_into_the_level(tmp_path):
    install, level = _make_install(tmp_path), _level(tmp_path)

    ensure_vineyard_assets(level, install)

    target = level / "art" / "shapes" / "vineyard"
    for stem in ("grape_vine", "grape_vine_group"):
        assert (target / f"{stem}.dae").read_text() == f"<dae {stem}/>"
        assert (target / f"{stem}.cdae").exists()
        assert (target / f"{stem}.dae.imposter.dds").exists()
        assert (target / f"{stem}.dae.imposter_normals.dds").exists()


def test_missing_optional_files_are_not_an_error(tmp_path):
    install, level = _make_install(tmp_path, with_imposters=False), _level(tmp_path)

    ensure_vineyard_assets(level, install)

    assert (level / "art" / "shapes" / "vineyard" / "grape_vine.dae").exists()


def test_writes_only_the_needed_materials(tmp_path):
    install, level = _make_install(tmp_path), _level(tmp_path)

    ensure_vineyard_assets(level, install)

    materials = json.loads((level / "art" / "shapes" / "vineyard" / "main.materials.json").read_text())
    assert set(materials) == {"grape", "olive_trunk", "leaves_strong"}  # "unrelated" nicht


def test_materials_already_defined_in_the_level_are_not_duplicated(tmp_path):
    install, level = _make_install(tmp_path), _level(tmp_path)
    trees = level / "art" / "shapes" / "trees"
    trees.mkdir(parents=True)
    (trees / "main.materials.json").write_text(json.dumps({"leaves_strong": _material("leaves_strong")}))

    ensure_vineyard_assets(level, install)

    materials = json.loads((level / "art" / "shapes" / "vineyard" / "main.materials.json").read_text())
    assert set(materials) == {"grape", "olive_trunk"}


def test_materials_get_stable_unique_persistent_ids(tmp_path):
    install, level = _make_install(tmp_path), _level(tmp_path)

    ensure_vineyard_assets(level, install)
    first = json.loads((level / "art" / "shapes" / "vineyard" / "main.materials.json").read_text())
    ensure_vineyard_assets(level, install)
    second = json.loads((level / "art" / "shapes" / "vineyard" / "main.materials.json").read_text())

    ids = [m["persistentId"] for m in first.values()]
    assert len(set(ids)) == len(ids)
    assert all(not i.startswith("orig-") for i in ids)  # nicht die IDs des Original-Levels
    assert first == second  # stabil über Exporte hinweg


def test_registers_forest_items_in_managed_item_data(tmp_path):
    install, level = _make_install(tmp_path), _level(tmp_path)

    result = ensure_vineyard_assets(level, install)

    data = json.loads((level / "art" / "forest" / "managedItemData.json").read_text())
    assert set(result["items"]) == {"grape_vine", "grape_vine_group"}
    for stem in ("grape_vine", "grape_vine_group"):
        item = data[stem]
        assert item["internalName"] == stem
        assert item["class"] == "TSForestItemData"
        assert item["shapeFile"] == f"levels/world_to_beamng/art/shapes/vineyard/{stem}.dae"
        assert item["persistentId"]


def test_existing_forest_items_are_preserved_when_merging(tmp_path):
    install, level = _make_install(tmp_path), _level(tmp_path)
    forest = level / "art" / "forest"
    forest.mkdir(parents=True)
    existing = {"oak": {"name": "oak", "internalName": "oak", "class": "ForestItemData", "shapeFile": "x.dae"}}
    (forest / "managedItemData.json").write_text(json.dumps(existing))

    ensure_vineyard_assets(level, install)

    data = json.loads((forest / "managedItemData.json").read_text())
    assert data["oak"] == existing["oak"]
    assert "grape_vine" in data


def test_is_idempotent(tmp_path):
    install, level = _make_install(tmp_path), _level(tmp_path)

    ensure_vineyard_assets(level, install)
    first = (level / "art" / "forest" / "managedItemData.json").read_text()
    ensure_vineyard_assets(level, install)
    second = (level / "art" / "forest" / "managedItemData.json").read_text()

    assert first == second


def test_missing_italy_level_raises_a_clear_error(tmp_path):
    level = _level(tmp_path)
    (tmp_path / "empty" / "content" / "levels").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="italy"):
        ensure_vineyard_assets(level, tmp_path / "empty")
