"""
Elevation data management (loading, caching, interpolation).
"""

import hashlib
import json
import numpy as np

from .. import config
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def get_height_data_hash():
    """Creates a hash based on the files in the data/height folder.

    If height_data_hash.txt is missing or different, all old cache files
    are deleted (forces regeneration).
    """
    xyz_files = sorted(config.HEIGHT_DATA_DIR.glob("*.xyz"))
    zip_files = sorted(config.HEIGHT_DATA_DIR.glob("*.zip"))
    tif_files = sorted(config.HEIGHT_DATA_DIR.glob("*.tif")) + sorted(config.HEIGHT_DATA_DIR.glob("*.tiff"))
    all_files = xyz_files + zip_files + tif_files

    if not all_files:
        return None

    # Hash based on file names and modification times
    hash_input = ""
    for file in all_files:
        mtime = file.stat().st_mtime
        hash_input += f"{file.name}_{mtime}_"

    new_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

    # Check whether height_data_hash.txt exists and contains a DIFFERENT hash
    hash_file = config.CACHE_DIR / "height_data_hash.txt"
    old_hash = None

    if hash_file.exists():
        try:
            with open(hash_file, "r") as f:
                old_hash = f.read().strip()
        except:
            pass

    # If the hash has changed or the file is missing: cleanup
    if old_hash != new_hash:
        if old_hash is None:
            logger.debug(f"  [i] height_data_hash.txt missing - deleting old cache files...")
        else:
            logger.debug(f"  [i] Height data changed ({old_hash} -> {new_hash}) - deleting old cache files...")

        # Delete all old cache files (if old_hash is known)
        if old_hash:
            for pattern in [
                f"height_raw_{old_hash}.npz",
                f"grid_v3_{old_hash}_*.npz",
                f"osm_all_{old_hash}.json",
                f"elevations_{old_hash}.json",
            ]:
                for old_file in config.CACHE_DIR.glob(pattern):
                    try:
                        old_file.unlink() # old_file is already a Path object from glob
                        logger.info(f"    • Deleted: {old_file.name}")
                    except Exception as e:
                        logger.error(f"    [!] Error deleting {old_file.name}: {e}")
        else:
            # If old_hash is empty/None: delete ALL potential old caches (safety measure)
            logger.info(f"    Deleting all _*.npz and _*.json cache files...")
            for pattern in ["height_raw_*.npz", "grid_v3_*.npz", "osm_all_*.json", "elevations_*.json"]:
                for old_file in config.CACHE_DIR.glob(pattern):
                    try:
                        old_file.unlink()
                        logger.info(f"    • Deleted: {old_file.name}")
                    except Exception as e:
                        logger.error(f"    [!] Error deleting {old_file.name}: {e}")

        # Also delete the generated DAE tiles in the BeamNG directory
        logger.info(f"    Deleting terrain tiles in the BeamNG directory...")
        beamng_shapes = config.BEAMNG_DIR_SHAPES
        if beamng_shapes.exists():
            for file_path in beamng_shapes.glob("*.dae"):
                try:
                    file_path.unlink()
                    logger.info(f"    • Deleted: {file_path.name}")
                except Exception as e:
                    logger.error(f"    [!] Error deleting {file_path.name}: {e}")
            # Also delete the DAE index file if present
            for meta_file_name in ["index.json", "manifest.json"]:
                meta_path = beamng_shapes / meta_file_name
                if meta_path.exists():
                    try:
                        meta_path.unlink()
                        logger.info(f"    • Deleted: {meta_path.name}")
                    except Exception as e:
                        logger.error(f"    [!] Error deleting {meta_path.name}: {e}")

        # Also delete texture tiles
        logger.info(f"    Deleting texture tiles in the BeamNG directory...")
        beamng_textures = config.BEAMNG_DIR_TEXTURES
        if beamng_textures.exists():
            for file_path in beamng_textures.glob("tile*"):
                try:
                    file_path.unlink()
                    logger.info(f"    • Deleted: {file_path.name}")
                except Exception as e:
                    logger.error(f"    [!] Error deleting {file_path.name}: {e}")

        # Save the new hash
        try:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(hash_file, "w") as f:
                f.write(new_hash)
        except:
            pass

    return new_hash


def get_elevation_cache(bbox, height_hash=None):
    """Loads the elevation cache for a BBox (coordinate -> elevation).

    Args:
        bbox: Bounding Box
        height_hash: Optional - tile_hash for cache consistency
    """
    from ..io.cache import get_cache_path

    # Use the passed height_hash or fall back to config (if present)
    effective_hash = height_hash or (config.HEIGHT_HASH if hasattr(config, "HEIGHT_HASH") else None)

    if effective_hash:
        cache_path = get_cache_path(bbox, "elevations", effective_hash)
    else:
        cache_path = get_cache_path(bbox, "elevations")

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                # Check the cache version (v2 = normalized Z values)
                if cache_data.get("_cache_version") == 2:
                    logger.info(f"  [OK] Elevation cache loaded: {len(cache_data)-1} coordinates")
                    return cache_data
                else:
                    logger.debug(f"  [i] Old cache format detected, ignored")
        except:
            pass
    return {"_cache_version": 2}


def save_elevation_cache(bbox, cache_data, height_hash=None):
    """Saves the elevation cache.

    Args:
        bbox: Bounding Box
        cache_data: Cache data to save
        height_hash: Optional - tile_hash for cache consistency
    """
    from ..io.cache import get_cache_path

    # Use the passed height_hash or fall back to config (if present)
    effective_hash = height_hash or (config.HEIGHT_HASH if hasattr(config, "HEIGHT_HASH") else None)

    if effective_hash:
        cache_path = get_cache_path(bbox, "elevations", effective_hash)
    else:
        cache_path = get_cache_path(bbox, "elevations")

    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        coord_count = len(cache_data) - 1  # -1 for _cache_version
        logger.info(f"  [OK] Elevation cache saved: {coord_count} coordinates")
    except Exception as e:
        logger.error(f"  [!] Error saving the elevation cache: {e}")


def get_elevations_for_points(pts, bbox, height_points, height_elevations, global_offset, height_hash=None):
    """Gets elevation data for coordinates - from the cache or by interpolation from local data.

    Args:
        pts: Coordinates (lat, lon) in WGS84
        bbox: Bounding Box
        height_points: Elevation data points (XY) - LOCAL, already normalized!
        height_elevations: Z values - LOCAL, already normalized!
        global_offset: (origin_x, origin_y) for the WGS84->UTM->local transformation
        height_hash: Optional - tile_hash for cache consistency
    """
    from ..geometry.coordinates import transformer_to_utm
    from scipy.interpolate import griddata

    # Load the existing cache
    elevation_cache = get_elevation_cache(bbox, height_hash=height_hash)

    # Find missing coordinates
    missing_pts = []
    missing_indices = []

    for idx, pt in enumerate(pts):
        # Create a unique key for the coordinate (rounded to 6 decimal places)
        coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
        if coord_key not in elevation_cache:
            missing_pts.append(pt)
            missing_indices.append(idx)

    # Compute missing elevations by interpolation
    if missing_pts:
        logger.info(f"  Interpolating {len(missing_pts)} height values...")

        # Convert WGS84 to UTM and then to local using global_offset
        ox, oy = global_offset

        missing_pts_local = []
        for pt in missing_pts:
            x_utm, y_utm = transformer_to_utm.transform(pt[1], pt[0])  # lon, lat -> x, y
            # Transform to local coordinates
            x = x_utm - ox
            y = y_utm - oy
            missing_pts_local.append([x, y])

        missing_pts_local = np.array(missing_pts_local)

        # Interpolate elevations (nearest neighbor for faster computation)
        # IMPORTANT: height_elevations is ALREADY normalized (local)!
        new_elevations = griddata(height_points, height_elevations, missing_pts_local, method="nearest")

        # Add to the cache
        for pt, elev in zip(missing_pts, new_elevations):
            coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
            elevation_cache[coord_key] = float(elev)

        # Save the updated cache (with version)
        save_elevation_cache(bbox, elevation_cache, height_hash=height_hash)

    # Build the elevation array for all points
    elevations = []
    for pt in pts:
        coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
        elevations.append(elevation_cache.get(coord_key, 0))

    return elevations
