"""
Forest JSON Writer: Schreibt forest.json für BeamNG.

Exportiert alle gesammelten Tree-Instances in BeamNG's forest.json Format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ForestJSONWriter:
    """
    Schreibt forest.json mit allen Baum-Instanzen.

    Format:
    {
        "formatVersion": 1,
        "trees": [
            {
                "type": "oak",
                "pos": [145.2, 330.5, 42.12],
                "rot": [0, 0, 0.382, 0.923],
                "scale": 1.15
            },
            ...
        ]
    }
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Verzeichnis für forest.json (z.B. levels/world_to_beamng/main/)
        """
        self.output_dir = Path(output_dir)

    def write_forest_json(self, tree_instances: List[Dict], filename: str = "forest.json") -> Dict:
        """
        Schreibe forest.json.

        Args:
            tree_instances: Liste von Baum-Instance-Dicts
                           (mit "type", "pos", "rot", "scale")
            filename: Optional - Dateiname (default: "forest.json")

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

            # BeamNG forest.json Format
            forest_data = {"formatVersion": 1, "trees": tree_instances}

            # Schreibe JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(forest_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ forest.json geschrieben: {filepath} ({len(tree_instances)} Bäume)")

            return {"status": "success", "filepath": str(filepath), "tree_count": len(tree_instances), "error": None}

        except Exception as e:
            logger.error(f"Fehler beim Schreiben von forest.json: {e}", exc_info=True)
            return {"status": "error", "filepath": "", "tree_count": 0, "error": str(e)}

    def append_to_forest_json(self, new_instances: List[Dict], filename: str = "forest.json") -> Dict:
        """
        Füge neue Instanzen zu existierendem forest.json hinzu.

        Nützlich falls forest.json inkrementell geschrieben werden soll.

        Args:
            new_instances: Neue Baum-Instanzen
            filename: Dateiname

        Returns:
            Status-Dict
        """
        try:
            filepath = self.output_dir / filename

            # Lade existierende Daten
            existing_instances = []
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    existing_instances = data.get("trees", [])

            # Kombiniere
            all_instances = existing_instances + new_instances

            # Schreibe kombinierte Daten
            return self.write_forest_json(all_instances, filename)

        except Exception as e:
            logger.error(f"Fehler beim Anhängen zu forest.json: {e}", exc_info=True)
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
