"""Tests for world_to_beamng/io/beamng_assets.py: tree shapes and stock textures from a (fake) BeamNG installation."""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.io.beamng_assets import ensure_shared_textures, ensure_tree_assets

LEVEL = "test_level"
TREES = "levels/east_coast_usa/art/shapes/trees/"


def _zip(path: Path, members: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


@pytest.fixture
def install(tmp_path):
    install_dir = tmp_path / "BeamNG"
    dae = '<COLLADA><image><init_from>levels/east_coast_usa/art/shapes/trees/trees_beech/bark.dds</init_from></image></COLLADA>'
    _zip(install_dir / "content" / "levels" / "east_coast_usa.zip", {
        TREES + "trees_beech/tree_beech_large_a.dae": dae,
        TREES + "trees_beech/tree_beech_large_a.cdae": b"binary",
        TREES + "trees_beech/main.materials.json": '{"m": {"colorMap": "levels/east_coast_usa/art/shapes/trees/trees_beech/bark.dds"}}',
        TREES + "ECA_coast_bush/cork_oak_bush_large.dae": "<COLLADA/>",
        TREES + "ECA_coast_bush/generibush.dae": "<COLLADA/>",
        "levels/east_coast_usa/art/shapes/trees/main.materials.json": json.dumps({"leaves_strong": {"class": "Material"}}),
    })
    _zip(install_dir / "content" / "assets" / "materials" / "decalroad.zip", {
        "assets/materials/decalroad/asphalt/asphalt_b.color.dds": b"dds-bytes",
    })
    return install_dir


def test_tree_assets_are_extracted_rewritten_and_registered(tmp_path, install):
    level_dir = tmp_path / "levels" / LEVEL

    result = ensure_tree_assets(level_dir, install, LEVEL)

    shapes = level_dir / "art" / "shapes" / "trees"
    assert (shapes / "trees_beech" / "tree_beech_large_a.dae").read_text().count(f"levels/{LEVEL}/art/shapes/trees") == 1
    assert "east_coast_usa" not in (shapes / "trees_beech" / "main.materials.json").read_text()
    assert not list(shapes.rglob("*.cdae"))
    assert not (shapes / "ECA_coast_bush" / "cork_oak_bush_large.dae").exists()  # non-native species excluded
    items = json.loads((level_dir / "art" / "forest" / "managedItemData.json").read_text())
    assert set(items) == {"tree_beech_large_a", "generibush"}
    assert items["tree_beech_large_a"]["shapeFile"] == f"levels/{LEVEL}/art/shapes/trees/trees_beech/tree_beech_large_a.dae"
    assert result == {"items": 2, "extracted": result["extracted"]} and result["extracted"] > 0


def test_second_run_changes_nothing_and_keeps_other_items(tmp_path, install):
    level_dir = tmp_path / "levels" / LEVEL
    ensure_tree_assets(level_dir, install, LEVEL)
    item_path = level_dir / "art" / "forest" / "managedItemData.json"
    data = json.loads(item_path.read_text())
    data["grape_vine"] = {"name": "grape_vine", "shapeFile": f"levels/{LEVEL}/art/shapes/vineyard/grape_vine.dae"}
    item_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stamp = item_path.stat().st_mtime_ns

    result = ensure_tree_assets(level_dir, install, LEVEL)

    assert result["extracted"] == 0
    assert item_path.stat().st_mtime_ns == stamp  # unchanged content is not rewritten (forest cache key)
    assert "grape_vine" in json.loads(item_path.read_text())


def test_deleted_tree_file_triggers_a_new_extraction(tmp_path, install):
    level_dir = tmp_path / "levels" / LEVEL
    ensure_tree_assets(level_dir, install, LEVEL)
    (level_dir / "art" / "shapes" / "trees" / "trees_beech" / "tree_beech_large_a.dae").unlink()

    assert ensure_tree_assets(level_dir, install, LEVEL)["extracted"] > 0
    assert (level_dir / "art" / "shapes" / "trees" / "trees_beech" / "tree_beech_large_a.dae").exists()


def test_missing_source_level_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="east_coast_usa"):
        ensure_tree_assets(tmp_path / "level", tmp_path / "no_install", LEVEL)


def test_stock_textures_are_copied_once_and_missing_ones_reported(tmp_path, install):
    level_dir = tmp_path / "levels" / LEVEL
    prefix = f"levels/{LEVEL}/art/shapes/assets/materials/"
    mapping = tmp_path / "osm_to_beamng.json"
    mapping.write_text(json.dumps({"roads": {"a": {"textures": {
        "baseColorMap": prefix + "decalroad/asphalt/asphalt_b.color.dds",
        "normalMap": prefix + "decalroad/asphalt/missing.normal.dds",
    }}}}), encoding="utf-8")

    first = ensure_shared_textures(level_dir, install, LEVEL, mapping)
    second = ensure_shared_textures(level_dir, install, LEVEL, mapping)

    target = level_dir / "art" / "shapes" / "assets" / "materials" / "decalroad" / "asphalt" / "asphalt_b.color.dds"
    assert target.read_bytes() == b"dds-bytes"
    assert first["copied"] == 1 and first["missing"] == [prefix + "decalroad/asphalt/missing.normal.dds"]
    assert second["copied"] == 0 and first["referenced"] == 2
