"""
Merge-Funktionen für Materials und Items JSON (Multi-Tile-System).

Beim Merge neuer Tiles mit bestehenden Daten:
- add_new mode: Nur neue Keys hinzufügen, bestehende Keys nicht überschreiben
- Damit werden keine alten Materials/Items gelöscht, nur neue hinzugefügt
"""

import os
import json


def merge_materials_json(output_path, new_materials, mode="add_new"):
    """
    Merged neue Material-Definitionen mit bestehenden.

    Args:
        output_path: Pfad zur Ziel-materials.json
        new_materials: Neue Material-Daten (Dict)
        mode: 'add_new' (default) = nur neue Keys, 'overwrite' = überschreiben

    Returns:
        Dict: Merged materials data
    """
    merged = {}

    # Lade bestehende Materials, wenn vorhanden
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                merged = json.load(f)
            print(f"  [OK] {len(merged)} bestehende Materials geladen")
        except Exception as e:
            print(f"  [!] Fehler beim Laden bestehender Materials: {e}")
            merged = {}

    # Merge neue Materials
    added_count = 0
    skipped_count = 0

    if new_materials:
        for key, value in new_materials.items():
            if key in merged:
                if mode == "add_new":
                    skipped_count += 1
                else:  # overwrite
                    merged[key] = value
                    added_count += 1
            else:
                merged[key] = value
                added_count += 1

    print(f"  [OK] Materials: {added_count} neue Keys hinzugefügt, {skipped_count} existierende Keys behalten")

    return merged


def merge_items_json(output_path, new_items, mode="add_new"):
    """
    Merged neue Item-Definitionen mit bestehenden.

    Args:
        output_path: Pfad zur Ziel-items.json
        new_items: Neue Item-Daten (Dict)
        mode: 'add_new' (default) = nur neue Keys, 'overwrite' = überschreiben

    Returns:
        Dict: Merged items data
    """
    merged = {}

    # Lade bestehende Items, wenn vorhanden
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                merged = json.load(f)
            print(f"  [OK] {len(merged)} bestehende Items geladen")
        except Exception as e:
            print(f"  [!] Fehler beim Laden bestehender Items: {e}")
            merged = {}

    # Merge neue Items
    added_count = 0
    skipped_count = 0

    if new_items:
        for key, value in new_items.items():
            if key in merged:
                if mode == "add_new":
                    skipped_count += 1
                else:  # overwrite
                    merged[key] = value
                    added_count += 1
            else:
                merged[key] = value
                added_count += 1

    print(f"  [OK] Items: {added_count} neue Keys hinzugefügt, {skipped_count} existierende Keys behalten")

    return merged


def save_materials_json(output_path, materials):
    """Speichert Materials JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(materials, f, indent=2)
        print(f"  [OK] {len(materials)} Materials in {os.path.basename(output_path)} gespeichert")
    except Exception as e:
        print(f"  [!] Fehler beim Speichern von Materials: {e}")


def save_items_json(output_path, items):
    """Speichert Items JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        print(f"  [OK] {len(items)} Items in {os.path.basename(output_path)} gespeichert")
    except Exception as e:
        print(f"  [!] Fehler beim Speichern von Items: {e}")
