"""
Cache-Management für OSM und Elevation-Daten.
"""

import os
import json
import hashlib

from .. import config


def get_bbox_hash(bbox):
    """Erstellt einen eindeutigen Hash für eine BBox zur Cache-Identifikation."""
    bbox_str = f"{bbox[0]:.6f}_{bbox[1]:.6f}_{bbox[2]:.6f}_{bbox[3]:.6f}"
    return hashlib.md5(bbox_str.encode()).hexdigest()[:12]


def get_cache_path(bbox, data_type):
    """Gibt den Pfad zur Cache-Datei zurück."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    bbox_hash = get_bbox_hash(bbox)
    return os.path.join(config.CACHE_DIR, f"{data_type}_{bbox_hash}.json")


def load_from_cache(bbox, data_type):
    """Lädt Daten aus dem Cache, falls vorhanden."""
    cache_path = get_cache_path(bbox, data_type)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  ✓ {data_type.upper()}-Daten aus Cache geladen ({cache_path})")
                return data
        except Exception as e:
            print(f"  ⚠ Fehler beim Laden des Caches: {e}")
    return None


def save_to_cache(bbox, data_type, data):
    """Speichert Daten im Cache."""
    cache_path = get_cache_path(bbox, data_type)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ {data_type.upper()}-Daten im Cache gespeichert ({cache_path})")
    except Exception as e:
        print(f"  ⚠ Fehler beim Speichern des Caches: {e}")
