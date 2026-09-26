"""
Guard rail assets: BeamNG's stock European guard rail (segment + end caps, as used by the italy level) registered as
forest items in art/forest/managedItemData.json, like the vanilla levels place their guard rails.

The shapes lie in BeamNG's global art/shapes/objects (content/art_shapes.zip) together with their material
(`italy_guardrails` in art/shapes/objects/main.materials.json) - every level can use them, nothing is copied.
Runs on every export (idempotent) after io/beamng_assets.py::ensure_tree_assets(), which keeps entries that are not
trees.
"""

import json
import uuid
from pathlib import Path
from typing import Dict

from ..logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

# role -> stock shape name (= forest item name)
GUARDRAIL_ITEMS = {
    "segment": "italy_guardrails_basic",  # 3 m, x -1.5..1.5, traffic face at y = -0.065
    "start": "italy_guardrails_basic_end_cw",  # reaches -x from its origin, flares to +y
    "end": "italy_guardrails_basic_end_ccw",  # reaches +x from its origin, flares to +y
}

# Values as in BeamNG's italy level (managedItemData.json)
_ITEM_DEFAULTS = {
    "class": "TSForestItemData",
    "annotation": "GUARD_RAIL",
    "dampingCoefficient": 3,
    "dynamicCubemap": "0",
    "planarReflection": "0",
    "radius": 0.9,
    "tightnessCoefficient": 1,
    "translucentBlendOp": "LerpAlpha",
}


def ensure_guardrail_assets(level_dir: Path, level_name: str = "world_to_beamng") -> Dict:
    """
    Adds the guard rail items to managedItemData.json (existing entries are kept; the file is only written when
    something changes - its timestamp is part of the forest cache key, see workflow/forest_workflow.py).

    Returns:
        {"items": [item names]}
    """
    item_path = Path(level_dir) / "art" / "forest" / "managedItemData.json"
    item_path.parent.mkdir(parents=True, exist_ok=True)
    existing_text = item_path.read_text(encoding="utf-8") if item_path.exists() else None
    data = json.loads(existing_text) if existing_text else {}
    for name in GUARDRAIL_ITEMS.values():
        data[name] = {
            "name": name,
            "internalName": name,
            "persistentId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"world_to_beamng/{level_name}/item/{name}")),
            "shapeFile": f"art/shapes/objects/{name}.dae",
            **_ITEM_DEFAULTS,
        }
    new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if new_text != existing_text:
        item_path.write_text(new_text, encoding="utf-8")
    return {"items": list(GUARDRAIL_ITEMS.values())}
