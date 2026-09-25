"""
Forest JSON Writer: writes forest.forest4.json for BeamNG.

Exports all collected tree instances in BeamNG's forest.forest4.json format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

WRITE_BLOCK = 20000  # instances per write block


class ForestJSONWriter:
    """
    Writes forest.forest4.json with all tree instances.

    Format: JSONL (newline-delimited JSON) - each instance on a separate line
    {"type": "oak", "pos": [145.2, 330.5, 42.12], "rotationMatrix": [1,0,0,0,1,0,0,0,1], "scale": 1.15, "ctxid": 0}
    {"type": "birch", "pos": [150.1, 332.2, 43.5], "rotationMatrix": [1,0,0,0,1,0,0,0,1], "scale": 1.2, "ctxid": 0}
    ...
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: directory for forest.forest4.json (e.g. levels/world_to_beamng/main/)
        """
        self.output_dir = Path(output_dir)

    def write_forest_json(self, tree_instances: List[Dict], filename: str = "forest.forest4.json") -> Dict:
        """
        Write forest.forest4.json in JSONL format (newline-delimited JSON).

        Each tree instance is written on a separate line - this is the
        format BeamNG expects for forest.forest4.json.

        Args:
            tree_instances: list of tree instance dicts
                           (with "type", "pos", "rotationMatrix", "scale")
            filename: optional - file name (default: "forest.forest4.json")

        Returns:
            {
                "status": "success" | "error",
                "filepath": str,
                "tree_count": int,
                "error": Optional[str]
            }
        """
        try:
            # Create the directory if it does not exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            filepath = self.output_dir / filename

            # Write in JSONL format: each instance as a separate line
            # json.dumps uses the C encoder (json.dump on a file, by contrast, uses the pure-Python encoder, ~5x
            # slower); the lines are collected and written in blocks.
            encode = json.JSONEncoder(separators=(",", ":"), ensure_ascii=False).encode
            with open(filepath, "w", encoding="utf-8") as f:
                for start in range(0, len(tree_instances), WRITE_BLOCK):
                    lines = []
                    for instance in tree_instances[start : start + WRITE_BLOCK]:
                        # Add ctxid:0 (BeamNG field for the forest context)
                        instance["ctxid"] = 0
                        lines.append(encode(instance))
                    # Line separator after each object
                    f.write("\n".join(lines) + "\n")

            logger.info(f"✓ forest.forest4.json (JSONL format) written: {filepath} ({len(tree_instances)} trees)")

            return {"status": "success", "filepath": str(filepath), "tree_count": len(tree_instances), "error": None}

        except Exception as e:
            logger.error(f"Error writing forest.forest4.json: {e}", exc_info=True)
            return {"status": "error", "filepath": "", "tree_count": 0, "error": str(e)}

    def get_statistics(self, tree_instances: List[Dict]) -> Dict:
        """
        Compute statistics about tree instances.

        Args:
            tree_instances: list of instances

        Returns:
            Dict with statistics
        """
        if not tree_instances:
            return {"total_trees": 0, "types": {}, "avg_scale": 0.0, "min_height": 0.0, "max_height": 0.0}

        # Count tree types
        type_counts = {}
        scales = []
        heights = []

        for instance in tree_instances:
            tree_type = instance.get("type", "unknown")
            type_counts[tree_type] = type_counts.get(tree_type, 0) + 1

            scale = instance.get("scale", 1.0)
            scales.append(scale)

            pos = instance.get("pos", [0, 0, 0])
            if len(pos) >= 3:
                heights.append(pos[2])

        return {
            "total_trees": len(tree_instances),
            "types": type_counts,
            "avg_scale": sum(scales) / len(scales) if scales else 0.0,
            "min_height": min(heights) if heights else 0.0,
            "max_height": max(heights) if heights else 0.0,
        }
