"""
Forest JSON Writer: Schreibt forest.forest4.json für BeamNG.

Exportiert alle gesammelten Tree-Instances in BeamNG's forest.forest4.json Format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

WRITE_BLOCK = 20000  # Instanzen je Schreibblock


class ForestJSONWriter:
    """
    Schreibt forest.forest4.json mit allen Baum-Instanzen.

    Format: JSONL (newline-delimited JSON) - jede Instanz auf separater Zeile
    {"type": "oak", "pos": [145.2, 330.5, 42.12], "rotationMatrix": [1,0,0,0,1,0,0,0,1], "scale": 1.15, "ctxid": 0}
    {"type": "birch", "pos": [150.1, 332.2, 43.5], "rotationMatrix": [1,0,0,0,1,0,0,0,1], "scale": 1.2, "ctxid": 0}
    ...
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Verzeichnis für forest.forest4.json (z.B. levels/world_to_beamng/main/)
        """
        self.output_dir = Path(output_dir)

    def write_forest_json(self, tree_instances: List[Dict], filename: str = "forest.forest4.json") -> Dict:
        """
        Schreibe forest.forest4.json im JSONL-Format (newline-delimited JSON).

        Jede Bauminstanz wird auf einer separaten Zeile geschrieben - das ist das
        Format, das BeamNG für forest.forest4.json erwartet.

        Args:
            tree_instances: Liste von Baum-Instance-Dicts
                           (mit "type", "pos", "rotationMatrix", "scale")
            filename: Optional - Dateiname (default: "forest.forest4.json")

        Returns:
            {
                "status": "success" | "error",
                "filepath": str,
                "tree_count": int,
                "error": Optional[str]
            }
        """
        try:
            # Erstelle Verzeichnis falls nicht vorhanden
            self.output_dir.mkdir(parents=True, exist_ok=True)

            filepath = self.output_dir / filename

            # Schreibe im JSONL-Format: jede Instanz als separate Zeile
            # json.dumps nutzt den C-Encoder (json.dump auf einer Datei dagegen den reinen Python-Encoder, ~5x
            # langsamer); die Zeilen werden blockweise gesammelt und geschrieben.
            encode = json.JSONEncoder(separators=(",", ":"), ensure_ascii=False).encode
            with open(filepath, "w", encoding="utf-8") as f:
                for start in range(0, len(tree_instances), WRITE_BLOCK):
                    lines = []
                    for instance in tree_instances[start : start + WRITE_BLOCK]:
                        # Füge ctxid:0 hinzu (BeamNG-Feld für Forest-Context)
                        instance["ctxid"] = 0
                        lines.append(encode(instance))
                    # Zeilentrennung nach jedem Objekt
                    f.write("\n".join(lines) + "\n")

            logger.info(f"✓ forest.forest4.json (JSONL format) written: {filepath} ({len(tree_instances)} trees)")

            return {"status": "success", "filepath": str(filepath), "tree_count": len(tree_instances), "error": None}

        except Exception as e:
            logger.error(f"Error writing forest.forest4.json: {e}", exc_info=True)
            return {"status": "error", "filepath": "", "tree_count": 0, "error": str(e)}

    def get_statistics(self, tree_instances: List[Dict]) -> Dict:
        """
        Berechne Statistiken über Baum-Instanzen.

        Args:
            tree_instances: Liste von Instanzen

        Returns:
            Dict mit Statistiken
        """
        if not tree_instances:
            return {"total_trees": 0, "types": {}, "avg_scale": 0.0, "min_height": 0.0, "max_height": 0.0}

        # Zähle Tree-Types
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
