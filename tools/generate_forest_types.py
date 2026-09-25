"""
Regenerates "forest_type_templates" and "forest_mappings" in data/osm_to_beamng.json from the tree items of the
level (development tool; the export itself copies the tree assets automatically and only reads the JSON).

Run from the repository root: python tools/generate_forest_types.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config
from world_to_beamng.forest.forest_types import categorize_trees, generate_forest_mappings, generate_forest_types
from world_to_beamng.io.beamng_assets import ensure_tree_assets
from world_to_beamng.io.beamng_install import get_beamng_install_dir


def main() -> int:
    install_dir = get_beamng_install_dir()
    result = ensure_tree_assets(config.BEAMNG_DIR, install_dir, config.LEVEL_NAME)
    item_path = config.BEAMNG_DIR / "art" / "forest" / "managedItemData.json"
    items = json.loads(item_path.read_text(encoding="utf-8"))
    tree_prefix = f"levels/{config.LEVEL_NAME}/art/shapes/trees/"
    trees = {k: v for k, v in items.items() if str(v.get("shapeFile", "")).startswith(tree_prefix)}
    print(f"[INFO] {len(trees)} tree items ({result['extracted']} files extracted)")

    forest_types = generate_forest_types(categorize_trees(trees))
    forest_mappings = generate_forest_mappings(forest_types)

    mapping_path = config.OSM_MAPPING_JSON
    osm_config = json.loads(mapping_path.read_text(encoding="utf-8"))
    osm_config["forest_type_templates"] = forest_types
    osm_config["forest_mappings"] = forest_mappings
    mapping_path.write_text(json.dumps(osm_config, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] {mapping_path} updated: {len(forest_types)} forest types")
    for name in sorted(forest_types):
        print(f"       {name:30s} {len(forest_types[name].get('preferred_trees', {})):3d} trees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
