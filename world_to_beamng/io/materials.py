"""
Hilfsfunktionen zum Umgang mit main.materials.json.

Strategie:
- Am Start des Laufs materials.json löschen (frischer Zustand).
- In den Generierungs-Abschnitten Material-Einträge sammeln und sequentiell anhängen.
- Format bleibt ein Dict: {"material_name": { ...eintrag... }}
"""

import json
import os
from typing import Iterable, Dict, Any


def reset_materials_json(output_path: str) -> None:
    """Löscht materials.json zu Beginn des Laufs, falls vorhanden."""
    if os.path.exists(output_path):
        os.remove(output_path)


def append_materials_to_json(material_entries: Iterable[Dict[str, Any]], output_path: str) -> None:
    """Hängt Material-Einträge (Dict-Format) an main.materials.json an."""
    materials: Dict[str, Any] = {}

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                materials = json.load(f)
            except json.JSONDecodeError:
                materials = {}

    for entry in material_entries:
        name = entry.get("name")
        if not name:
            continue
        materials[name] = entry

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(materials, f, indent=2, ensure_ascii=False)
