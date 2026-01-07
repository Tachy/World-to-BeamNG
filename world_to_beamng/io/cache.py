"""
Cache-Management fuer OSM und Elevation-Daten.

Für Multi-Tile-Systeme:
- height_data_hash.txt speichert Hashes pro Datei zur Invalidierung
- Format: "filename: hash" (z.B. "dgm1_4658000_5394000.xyz.zip: abc123")
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


def load_from_cache(bbox, data_type, height_hash=None):
    """Lädt Daten aus dem Cache, falls vorhanden.

    Für osm_all und elevations wird height_hash verwendet (falls vorhanden)
    für garantierte Konsistenz bei Höhendaten-Änderungen.

    Args:
        bbox: Bounding Box
        data_type: Typ der Daten (osm_all, elevations, etc.)
        height_hash: Optional - Hash für tile-spezifische Cache-Identifikation
    """
    # Verwende übergebenes height_hash oder fallback auf config.HEIGHT_HASH
    effective_hash = height_hash or (config.HEIGHT_HASH if hasattr(config, "HEIGHT_HASH") else None)

    if effective_hash and data_type in ["osm_all", "elevations"]:
        cache_path = get_cache_path(bbox, data_type, effective_hash)
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


def save_to_cache(bbox, data_type, data, height_hash=None):
    """Speichert Daten im Cache.

    Für osm_all und elevations wird height_hash verwendet (falls vorhanden)
    für garantierte Konsistenz bei Höhendaten-Änderungen.

    Args:
        bbox: Bounding Box
        data_type: Typ der Daten (osm_all, elevations, etc.)
        data: Zu speichernde Daten
        height_hash: Optional - Hash für tile-spezifische Cache-Identifikation
    """
    # Verwende übergebenes height_hash oder fallback auf config.HEIGHT_HASH
    effective_hash = height_hash or (config.HEIGHT_HASH if hasattr(config, "HEIGHT_HASH") else None)

    if effective_hash and data_type in ["osm_all", "elevations"]:
        cache_path = get_cache_path(bbox, data_type, effective_hash)
    else:
        cache_path = get_cache_path(bbox, data_type)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  [OK] {data_type.upper()}-Daten im Cache gespeichert ({cache_path})")
    except Exception as e:
        print(f"  [!] Fehler beim Speichern des Caches: {e}")


def load_height_hashes():
    """
    Lädt die Hash-Registry für Height-Daten (Multi-Tile-System).

    Format der height_data_hash.txt:
    dgm1_4658000_5394000.xyz.zip: abc123def456
    dgm1_4660000_5394000.xyz.zip: xyz789

    Returns:
        Dict: {filename: hash_value}
    """
    hash_file = os.path.join(config.CACHE_DIR, "height_data_hash.txt")
    hashes = {}

    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    filename, hash_value = line.split(":", 1)
                    hashes[filename.strip()] = hash_value.strip()
            print(f"  [OK] {len(hashes)} Height-Data-Hashes geladen aus height_data_hash.txt")
        except Exception as e:
            print(f"  [!] Fehler beim Laden von height_data_hash.txt: {e}")

    return hashes


def save_height_hashes(hashes):
    """
    Speichert die Hash-Registry für Height-Daten.

    Args:
        hashes: Dict {filename: hash_value}
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    hash_file = os.path.join(config.CACHE_DIR, "height_data_hash.txt")

    try:
        with open(hash_file, "w", encoding="utf-8") as f:
            for filename in sorted(hashes.keys()):
                f.write(f"{filename}: {hashes[filename]}\n")
        print(f"  [OK] {len(hashes)} Height-Data-Hashes in height_data_hash.txt gespeichert")
    except Exception as e:
        print(f"  [!] Fehler beim Speichern von height_data_hash.txt: {e}")


def calculate_file_hash(filepath, chunk_size=8192):
    """
    Berechnet MD5-Hash einer Datei.

    Args:
        filepath: Pfad zur Datei
        chunk_size: Größe der Chunks zum Lesen

    Returns:
        str: MD5-Hash (12 Zeichen)
    """
    hash_obj = hashlib.md5()

    try:
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_obj.update(chunk)
        return hash_obj.hexdigest()[:12]
    except Exception as e:
        print(f"  [!] Fehler beim Berechnen des File-Hash: {e}")
        return None
