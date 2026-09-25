"""
Cache-Management fuer OSM und Elevation-Daten.

Für Multi-Tile-Systeme:
- height_data_hash.txt speichert Hashes pro Datei zur Invalidierung
- Format: "filename: hash" (z.B. "dgm1_4658000_5394000.xyz.zip: abc123")
"""

import json
import hashlib
from pathlib import Path

from .. import config
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


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
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # For osm_all and elevations: height_hash verwenden (wenn vorhanden)
    # Sonst: BBox-Hash verwenden (fallback fuer alte Caches)
    if height_hash and data_type in ["osm_all", "elevations"]:
        file_hash = height_hash
    else:
        file_hash = get_bbox_hash(bbox)

    return config.CACHE_DIR / f"{data_type}_{file_hash}.json"


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

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"  [OK] {data_type.upper()} data loaded from cache ({cache_path})")
                return data
        except Exception as e:
            logger.error(f"  [!] Error loading the cache: {e}")
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
        logger.info(f"  [OK] {data_type.upper()} data saved to cache ({cache_path})")
    except Exception as e:
        logger.error(f"  [!] Error saving the cache: {e}")


def calculate_file_hash(filepath: Path, chunk_size=8192):
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
        with open(filepath, "rb") as f: # open can take Path objects directly
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_obj.update(chunk)
        return hash_obj.hexdigest()[:12]
    except Exception as e:
        logger.error(f"  [!] Error computing the file hash: {e}")
        return None


def calculate_global_tiles_hash(tiles):
    """
    Berechnet einen globalen Hash über alle Tiles.

    Dieser Hash ändert sich, wenn:
    - Tiles hinzugefügt oder entfernt werden
    - Die Reihenfolge sich ändert
    - Ein einzelnes Tile geändert wird

    Args:
        tiles: Liste von Tile-Dicts mit 'filename' und 'filepath'

    Returns:
        str: MD5-Hash (12 Zeichen)
    """
    # Sortiere nach Filename für konsistente Reihenfolge
    sorted_tiles = sorted(tiles, key=lambda t: t.get("filename", ""))

    # Kombiniere Filenames und Hashes
    hash_input = ""
    for tile in sorted_tiles:
        filename = tile.get("filename", "")
        filepath = tile.get("filepath", "")

        # Berechne Hash des einzelnen Tiles
        tile_hash = calculate_file_hash(filepath) or "none"
        hash_input += f"{filename}:{tile_hash};"

    # Berechne globalen Hash
    global_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    return global_hash
