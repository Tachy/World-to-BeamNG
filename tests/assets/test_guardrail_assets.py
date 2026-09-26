"""Tests for world_to_beamng.io.guardrail_assets: stock guard rail shapes registered as forest items."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.io.guardrail_assets import GUARDRAIL_ITEMS, ensure_guardrail_assets


def _items(level_dir):
    return json.loads((level_dir / "art" / "forest" / "managedItemData.json").read_text(encoding="utf-8"))


def test_registers_segment_and_end_caps_with_the_global_stock_shapes(tmp_path):
    ensure_guardrail_assets(tmp_path, "world_to_beamng")
    data = _items(tmp_path)

    for name in GUARDRAIL_ITEMS.values():
        entry = data[name]
        assert entry["internalName"] == name and entry["class"] == "TSForestItemData"
        assert entry["shapeFile"] == f"art/shapes/objects/{name}.dae"  # global BeamNG asset, nothing is copied
        assert entry["annotation"] == "GUARD_RAIL"


def test_keeps_existing_items_and_does_not_rewrite_an_unchanged_file(tmp_path):
    path = tmp_path / "art" / "forest" / "managedItemData.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"oak": {"name": "oak", "shapeFile": "levels/x/oak.dae"}}), encoding="utf-8")

    ensure_guardrail_assets(tmp_path, "world_to_beamng")
    first = path.stat().st_mtime_ns
    ensure_guardrail_assets(tmp_path, "world_to_beamng")

    assert "oak" in _items(tmp_path)
    assert path.stat().st_mtime_ns == first
