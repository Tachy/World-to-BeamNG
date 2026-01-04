"""
Cache-Management fuer OSM und Elevation-Daten.
"""

import os
import json
import hashlib

from .. import config


def get_bbox_hash(bbox):
    """Erstellt einen eindeutigen Hash fuer eine BBox zur Cache-Identifikation."""
    bbox_str = f"{bbox[0]:.6f}_{bbox[1]:.6f}_{bbox[2]:.6f}_{bbox[3]:.6f}"
    return hashlib.md5(bbox_str.encode()).hexdigest()[:12]


def get_cache_path(bbox, data_type, height_hash=None):
    """Gibt den Pfad zur Cache-Datei zurueck.

    Args:
        bbox: Bounding Box
        data_type: Typ der Daten (osm_all, elevations, etc.)
        height_hash: Optional - Height-Data-Hash fuer Cache-Konsistenz

    Wenn height_hash gegeben ist, wird dieser fuer osm_all und elevations verwendet
    fuer garantierte Konsistenz bei Height-Daten-Aenderungen.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Fuer osm_all und elevations: height_hash verwenden (wenn vorhanden)
    # Sonst: BBox-Hash verwenden (fallback fuer alte Caches)
    if height_hash and data_type in ["osm_all", "elevations"]:
        file_hash = height_hash
    else:
        file_hash = get_bbox_hash(bbox)

    return os.path.join(config.CACHE_DIR, f"{data_type}_{file_hash}.json")


def load_from_cache(bbox, data_type):
    """Lädt Daten aus dem Cache, falls vorhanden.

    Für osm_all und elevations wird HEIGHT_HASH verwendet (falls vorhanden)
    für garantierte Konsistenz bei Höhendaten-Änderungen.
    """
    if config.HEIGHT_HASH and data_type in ["osm_all", "elevations"]:
        cache_path = get_cache_path(bbox, data_type, config.HEIGHT_HASH)
    else:
        cache_path = get_cache_path(bbox, data_type)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  [OK] {data_type.upper()}-Daten aus Cache geladen ({cache_path})")
                return data
        except Exception as e:
            print(f"  [!] Fehler beim Laden des Caches: {e}")
    return None


def save_to_cache(bbox, data_type, data):
    """Speichert Daten im Cache.

    Für osm_all und elevations wird HEIGHT_HASH verwendet (falls vorhanden)
    für garantierte Konsistenz bei Höhendaten-Änderungen.
    """
    if config.HEIGHT_HASH and data_type in ["osm_all", "elevations"]:
        cache_path = get_cache_path(bbox, data_type, config.HEIGHT_HASH)
    else:
        cache_path = get_cache_path(bbox, data_type)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  [OK] {data_type.upper()}-Daten im Cache gespeichert ({cache_path})")
    except Exception as e:
        print(f"  [!] Fehler beim Speichern des Caches: {e}")
