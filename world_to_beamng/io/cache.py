"""
Cache management for OSM and elevation data.

For multi-tile systems:
- height_data_hash.txt stores hashes per file for invalidation
- Format: "filename: hash" (e.g. "dgm1_4658000_5394000.xyz.zip: abc123")
"""

import json
import hashlib
from pathlib import Path

from .. import config
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def get_bbox_hash(bbox):
    """Creates a unique hash for a bbox to identify the cache entry."""
    bbox_str = f"{bbox[0]:.6f}_{bbox[1]:.6f}_{bbox[2]:.6f}_{bbox[3]:.6f}"
    return hashlib.md5(bbox_str.encode()).hexdigest()[:12]


def get_cache_path(bbox, data_type, height_hash=None):
    """Returns the path to the cache file.

    Args:
        bbox: Bounding box
        data_type: Type of the data (osm_all, elevations, etc.)
        height_hash: Optional - height data hash for cache consistency

    If height_hash is given, it is used for osm_all and elevations
    to guarantee consistency when the height data changes.
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # For osm_all and elevations: use height_hash (if present)
    # Otherwise: use the bbox hash (fallback for old caches)
    if height_hash and data_type in ["osm_all", "elevations"]:
        file_hash = height_hash
    else:
        file_hash = get_bbox_hash(bbox)

    return config.CACHE_DIR / f"{data_type}_{file_hash}.json"


def load_from_cache(bbox, data_type, height_hash=None):
    """Loads data from the cache, if present.

    For osm_all and elevations, height_hash is used (if present)
    to guarantee consistency when the elevation data changes.

    Args:
        bbox: Bounding box
        data_type: Type of the data (osm_all, elevations, etc.)
        height_hash: Optional - hash for tile-specific cache identification
    """
    # Use the passed height_hash or fall back to config.HEIGHT_HASH
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
    """Saves data to the cache.

    For osm_all and elevations, height_hash is used (if present)
    to guarantee consistency when the elevation data changes.

    Args:
        bbox: Bounding box
        data_type: Type of the data (osm_all, elevations, etc.)
        data: Data to save
        height_hash: Optional - hash for tile-specific cache identification
    """
    # Use the passed height_hash or fall back to config.HEIGHT_HASH
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
    Computes the MD5 hash of a file.

    Args:
        filepath: Path to the file
        chunk_size: Chunk size for reading

    Returns:
        str: MD5 hash (12 characters)
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
    Computes a global hash over all tiles.

    This hash changes when:
    - Tiles are added or removed
    - The order changes
    - A single tile is changed

    Args:
        tiles: List of tile dicts with 'filename' and 'filepath'

    Returns:
        str: MD5 hash (12 characters)
    """
    # Sort by filename for a consistent order
    sorted_tiles = sorted(tiles, key=lambda t: t.get("filename", ""))

    # Combine filenames and hashes
    hash_input = ""
    for tile in sorted_tiles:
        filename = tile.get("filename", "")
        filepath = tile.get("filepath", "")

        # Compute the hash of the individual tile
        tile_hash = calculate_file_hash(filepath) or "none"
        hash_input += f"{filename}:{tile_hash};"
        if tile.get("reproject_from") is not None:  # same file, different target CRS -> different points
            hash_input += f"epsg{tile['reproject_from']}->{tile['target_epsg']};"

    # Compute the global hash
    global_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    return global_hash
