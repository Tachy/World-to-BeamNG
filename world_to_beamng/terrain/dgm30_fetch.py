"""
Automatic download of missing Copernicus DEM GLO-30 tiles (30 m elevation data for the
horizon background) from the public, unauthenticated AWS S3 bucket
(`config.DGM30_S3_BUCKET`).

Analogous to the existing automatic OSM Overpass download in `world_to_beamng/osm/downloader.py`
(retry with exponential backoff, streamed progress logging) - the approach is deliberately
duplicated locally here instead of being factored out into a shared helper (smaller, lower-risk
step, see SDD plan).

The tiles are named after a 1°x1° degree grid (southwest corner), e.g.
`Copernicus_DSM_COG_10_N47_00_E007_00_DEM`. `horizon_area_wgs84()` in horizon_image.py provides
the BBox in WGS84 that this module needs for tile selection.
"""

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .. import config
from ..logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

_NOT_FOUND_CACHE_FILENAME = ".not_found_cache.json"


def copernicus_tile_id(lat_deg: int, lon_deg: int) -> str:
    """
    Copernicus DEM GLO-30 tile ID for the tile with southwest corner point (lat_deg, lon_deg).

    Args:
        lat_deg: Integer latitude of the southwest corner (e.g. 47; negative = south of the equator)
        lon_deg: Integer longitude of the southwest corner (e.g. 7; negative = west of zero)
    """
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"


def required_tile_ids(bbox_wgs84) -> list:
    """
    All 1°x1° tiles that intersect the BBox. Pure geometry, no I/O.

    Args:
        bbox_wgs84: (lon_min, lat_min, lon_max, lat_max), see horizon_image.horizon_area_wgs84()
    """
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    lat_start, lat_stop = math.floor(lat_min), math.ceil(lat_max)
    lon_start, lon_stop = math.floor(lon_min), math.ceil(lon_max)
    return [
        copernicus_tile_id(lat, lon)
        for lat in range(lat_start, lat_stop)
        for lon in range(lon_start, lon_stop)
    ]


def _not_found_entry_is_recent(timestamp_iso: str, cutoff: datetime) -> bool:
    """True if the not-found cache entry is still within the TTL. A broken/
    unreadable timestamp does NOT count as recently confirmed - the tile is then retried
    instead of being skipped permanently."""
    try:
        timestamp = datetime.fromisoformat(timestamp_iso)
    except (TypeError, ValueError):
        return False
    return timestamp > cutoff


def missing_tile_ids(bbox_wgs84, dgm30_dir, not_found_cache: dict) -> list:
    """
    `required_tile_ids(bbox_wgs84)` minus already existing .tif files in `dgm30_dir` and
    tiles that were recently (within config.DGM30_NOT_FOUND_CACHE_TTL_DAYS) confirmed as "not found".
    Pure function; apart from the directory listing of `dgm30_dir` there is no I/O.

    Args:
        bbox_wgs84: (lon_min, lat_min, lon_max, lat_max)
        dgm30_dir: Directory with already existing tiles (does not have to exist)
        not_found_cache: {tile_id: iso8601_timestamp_str}
    """
    dgm30_dir = Path(dgm30_dir)
    existing_files = (
        list(dgm30_dir.glob("*.tif")) + list(dgm30_dir.glob("*.tiff")) if dgm30_dir.is_dir() else []
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.DGM30_NOT_FOUND_CACHE_TTL_DAYS)

    missing = []
    for tile_id in required_tile_ids(bbox_wgs84):
        if any(tile_id in f.name for f in existing_files):
            continue
        cached_at = not_found_cache.get(tile_id)
        if cached_at is not None and _not_found_entry_is_recent(cached_at, cutoff):
            continue
        missing.append(tile_id)
    return missing


def _cache_path(dgm30_dir: Path) -> Path:
    return dgm30_dir / _NOT_FOUND_CACHE_FILENAME


def _load_not_found_cache(dgm30_dir: Path) -> dict:
    """Loads the not-found cache; a missing or broken file never causes a crash, only
    an empty cache (with a log message if the file is broken)."""
    path = _cache_path(dgm30_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"  [!] Not-found cache {path} is corrupt ({e}) - ignored")
        return {}


def _save_not_found_cache(dgm30_dir: Path, not_found_cache: dict) -> None:
    """Saves the not-found cache atomically (part file + os.replace) so that a crash during the
    next tile download does not lose already confirmed 404s."""
    path = _cache_path(dgm30_dir)
    part_path = path.with_name(path.name + ".part")
    part_path.write_text(json.dumps(not_found_cache, indent=2), encoding="utf-8")
    os.replace(part_path, path)


def _write_with_progress(response, part_path: Path, log_every_bytes: int = 2 * 1024 * 1024) -> None:
    """Writes a streamed response chunk by chunk to `part_path` and logs the progress
    every `log_every_bytes` (analogous to osm/downloader._download_with_progress(), simpler here because
    it writes directly to a file instead of into memory)."""
    downloaded = 0
    next_log_at = log_every_bytes
    with open(part_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=256 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_log_at:
                logger.info(f"    ... {downloaded / (1024 * 1024):.1f} MB received")
                next_log_at += log_every_bytes


def _download_one_tile(tile_id: str, url: str, dgm30_dir: Path) -> str:
    """Downloads a single tile with retry + exponential backoff (analogous to get_osm_data() in
    osm/downloader.py). Returns: "downloaded", "not_found" or "failed"."""
    part_path = dgm30_dir / f"{tile_id}.tif.part"
    final_path = dgm30_dir / f"{tile_id}.tif"

    for attempt in range(config.DGM30_FETCH_MAX_RETRIES):
        try:
            response = requests.get(url, stream=True, timeout=config.DGM30_FETCH_TIMEOUT_S)

            if response.status_code == 404:
                # Expected behavior for sea tiles (no land area = no DEM tile) -
                # no retry, no error log.
                logger.info(f"  [i] {tile_id}: no data on S3 (404, probably a sea tile)")
                return "not_found"

            response.raise_for_status()
            _write_with_progress(response, part_path)
            os.replace(part_path, final_path)  # atomic: a crash midway never leaves a .tif
            logger.info(f"  [OK] {tile_id} downloaded")
            return "downloaded"

        except requests.exceptions.Timeout:
            logger.info(f"  [x] {tile_id}: timeout (attempt {attempt + 1}/{config.DGM30_FETCH_MAX_RETRIES})")
        except requests.exceptions.HTTPError as e:
            logger.info(f"  [x] {tile_id}: HTTP error {e} (attempt {attempt + 1}/{config.DGM30_FETCH_MAX_RETRIES})")
        except Exception as e:
            logger.info(f"  [x] {tile_id}: error {e} (attempt {attempt + 1}/{config.DGM30_FETCH_MAX_RETRIES})")

        if attempt < config.DGM30_FETCH_MAX_RETRIES - 1:
            wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s, ...
            logger.info(f"  Waiting {wait_time}s before retrying...")
            time.sleep(wait_time)

    logger.error(f"  [x] {tile_id}: all {config.DGM30_FETCH_MAX_RETRIES} attempts failed")
    # An aborted stream may have left a .tif.part behind - clean up (analogous to the
    # total failure in sentinel2_fetch.fetch_eox_mosaic()).
    part_path.unlink(missing_ok=True)
    return "failed"


def download_dgm30_tiles(bbox_wgs84, dgm30_dir) -> dict:
    """
    Downloads all Copernicus DEM tiles missing for `bbox_wgs84` from S3 (see
    missing_tile_ids()).

    Args:
        bbox_wgs84: (lon_min, lat_min, lon_max, lat_max)
        dgm30_dir: Target directory for the .tif tiles (created if needed)

    Returns:
        {"downloaded": [...], "not_found": [...], "failed": [...]} (tile IDs per category)
    """
    dgm30_dir = Path(dgm30_dir)
    dgm30_dir.mkdir(parents=True, exist_ok=True)

    not_found_cache = _load_not_found_cache(dgm30_dir)
    missing = missing_tile_ids(bbox_wgs84, dgm30_dir, not_found_cache)
    if not missing:
        logger.debug("  DGM30: all required tiles already present or recently confirmed missing")
        return {"downloaded": [], "not_found": [], "failed": []}

    logger.info(f"  DGM30: {len(missing)} missing tile(s) will be loaded from S3: {', '.join(missing)}")

    result = {"downloaded": [], "not_found": [], "failed": []}
    for tile_id in missing:
        url = f"https://{config.DGM30_S3_BUCKET}.s3.{config.DGM30_S3_REGION}.amazonaws.com/{tile_id}/{tile_id}.tif"
        outcome = _download_one_tile(tile_id, url, dgm30_dir)

        if outcome == "not_found":
            not_found_cache[tile_id] = datetime.now(timezone.utc).isoformat()
            try:
                _save_not_found_cache(dgm30_dir, not_found_cache)  # save immediately: survives a crash on the next tile
            except OSError as e:
                # A single failed cache write attempt (disk full, no permissions, ...) should
                # not abort the rest of the batch - only this tile is detected as a 404
                # again on the next run instead of staying cached.
                logger.warning(f"  [!] Not-found cache could not be saved ({e}) - {tile_id} stays uncached")

        result[outcome].append(tile_id)

    return result


def ensure_dgm30_coverage(bbox_wgs84, dgm30_dir=config.DGM30_CACHE_DIR) -> dict:
    """
    Public entry point for horizon_workflow.py: ensures that all DGM30 tiles needed for `bbox_wgs84`
    are available locally (downloads missing ones). NEVER raises - network/DNS
    errors etc. are caught and logged so that the pipeline always keeps running (the
    existing fallback for tiles that are still missing applies afterwards anyway).

    Args:
        bbox_wgs84: (lon_min, lat_min, lon_max, lat_max)
        dgm30_dir: Target directory; default config.DGM30_CACHE_DIR

    Returns:
        {"downloaded": [...], "not_found": [...], "failed": [...]}
    """
    if not config.DGM30_AUTO_DOWNLOAD:
        return {"downloaded": [], "not_found": [], "failed": []}

    try:
        return download_dgm30_tiles(bbox_wgs84, Path(dgm30_dir))
    except Exception as e:
        logger.error(f"  [x] DGM30 auto-download failed: {e}")
        return {"downloaded": [], "not_found": [], "failed": []}
