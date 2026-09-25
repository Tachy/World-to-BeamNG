"""
Automatic download of a Sentinel-2 cloudless mosaic (EOX WMS service, `config.EOX_WMS_URL`)
for the horizon area. Two separate cache layers (both under cache/, named per area, safe to
delete at any time): the raw mosaic (config.EOX_MOSAIC_CACHE_DIR) and the finished texture
clipped from it via `horizon_image.build_horizon_image()` (config.EOX_TEXTURE_CACHE_DIR).
data/DOP300/ remains exclusively the manual override slot for a self-provided file - see
ensure_horizon_texture().

Analogous to the automatic DGM30 download in `world_to_beamng/terrain/dgm30_fetch.py` and the OSM
Overpass download in `world_to_beamng/osm/downloader.py` (retry with exponential backoff,
streamed progress logging) - the approach is deliberately duplicated locally here instead of
being extracted into a shared helper (same decision as in Task 2).

The EOX server does not document a size limit for WMS GetMap requests - the mosaic is therefore
split into tiles of at most `config.EOX_MAX_REQUEST_PX` edge length (tile_grid()) and written
tile by tile directly into the target GeoTIFF.
"""

import hashlib
import io
import os
import time
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from .. import config
from ..logging_config import LoggerConfig
from .horizon_image import build_horizon_image

logger = LoggerConfig.get_logger()

# Set to True exactly once per process run as soon as the EOX attribution has been logged -
# prevents duplicate/repeated messages if ensure_horizon_texture() is called several times.
_attribution_logged = False


class TileRequest(NamedTuple):
    """A single WMS GetMap request window within the full mosaic (see tile_grid())."""

    col_off: int
    row_off: int
    width: int
    height: int
    bbox: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in EPSG:3857


def _mercator_bbox_for_area(area_utm: tuple) -> Tuple[float, float, float, float]:
    """
    Transforms `area_utm` (x_min, x_max, y_min, y_max in the resolved source CRS, see
    horizon_image.horizon_area()) to EPSG:3857 and widens the result by
    config.EOX_FETCH_MARGIN_FACTOR around the center (against rounding gaps at the border).

    Returns:
        (minx, miny, maxx, maxy) in EPSG:3857 - NOTE: different tuple format than area_utm!
    """
    from rasterio.warp import transform_bounds

    from ..geometry.coordinates import get_source_crs_epsg

    x_min, x_max, y_min, y_max = area_utm
    left, bottom, right, top = transform_bounds(
        f"EPSG:{get_source_crs_epsg()}", "EPSG:3857", x_min, y_min, x_max, y_max
    )

    cx, cy = (left + right) / 2, (bottom + top) / 2
    half_w = (right - left) / 2 * config.EOX_FETCH_MARGIN_FACTOR
    half_h = (top - bottom) / 2 * config.EOX_FETCH_MARGIN_FACTOR
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _mosaic_pixel_size(bbox_3857: tuple) -> Tuple[int, int]:
    """Pixel size (w, h) of the full mosaic for `bbox_3857` at config.EOX_TARGET_RESOLUTION_M,
    capped at config.EOX_MOSAIC_MAX_PX per axis."""
    minx, miny, maxx, maxy = bbox_3857
    w = round((maxx - minx) / config.EOX_TARGET_RESOLUTION_M)
    h = round((maxy - miny) / config.EOX_TARGET_RESOLUTION_M)
    w = max(1, min(w, config.EOX_MOSAIC_MAX_PX))
    h = max(1, min(h, config.EOX_MOSAIC_MAX_PX))
    return w, h


def tile_grid(bbox_3857: tuple, w: int, h: int) -> List[TileRequest]:
    """
    Splits the w x h mosaic into TileRequest tiles of at most config.EOX_MAX_REQUEST_PX
    edge length (many WMS servers limit WIDTH/HEIGHT per request). Pure function, no network.

    Image convention: row 0 = top edge of the image = geographic north = maxy; the geographic
    Y coordinate decreases as row_off grows.
    """
    minx, miny, maxx, maxy = bbox_3857
    px_x = (maxx - minx) / w
    px_y = (maxy - miny) / h
    tiles = []
    for row_off in range(0, h, config.EOX_MAX_REQUEST_PX):
        tile_h = min(config.EOX_MAX_REQUEST_PX, h - row_off)
        for col_off in range(0, w, config.EOX_MAX_REQUEST_PX):
            tile_w = min(config.EOX_MAX_REQUEST_PX, w - col_off)
            tile_minx = minx + col_off * px_x
            tile_maxx = minx + (col_off + tile_w) * px_x
            tile_maxy = maxy - row_off * px_y
            tile_miny = maxy - (row_off + tile_h) * px_y
            tiles.append(TileRequest(col_off, row_off, tile_w, tile_h, (tile_minx, tile_miny, tile_maxx, tile_maxy)))
    return tiles


def _fetch_one_tile(tile: TileRequest) -> Optional[np.ndarray]:
    """Downloads a single WMS tile with retry + exponential backoff (analogous to
    dgm30_fetch._download_one_tile()). Returns: (H, W, 3) uint8 array, or None on final
    failure (timeout, 5xx, unexpected content type, decode error - whatever the reason,
    this tile simply stays black)."""
    params = {
        "service": "WMS",
        "version": config.EOX_WMS_VERSION,
        "request": "GetMap",
        "layers": config.EOX_WMS_LAYER,
        "styles": "",
        "SRS": "EPSG:3857",
        "bbox": f"{tile.bbox[0]},{tile.bbox[1]},{tile.bbox[2]},{tile.bbox[3]}",
        "width": tile.width,
        "height": tile.height,
        "format": config.EOX_WMS_FORMAT,
    }
    headers = {"User-Agent": config.EOX_USER_AGENT}

    for attempt in range(config.EOX_FETCH_MAX_RETRIES):
        try:
            response = requests.get(
                config.EOX_WMS_URL, params=params, headers=headers, timeout=config.EOX_FETCH_TIMEOUT_S
            )

            content_type = response.headers.get("Content-Type", "")
            if response.status_code == 200 and content_type.startswith("image/"):
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                return np.array(image, dtype=np.uint8)

            logger.info(
                f"  [x] Tile ({tile.col_off},{tile.row_off}): status {response.status_code}, "
                f"content type {content_type!r} (attempt {attempt + 1}/{config.EOX_FETCH_MAX_RETRIES})"
            )

        except requests.exceptions.Timeout:
            logger.info(
                f"  [x] Tile ({tile.col_off},{tile.row_off}): timeout "
                f"(attempt {attempt + 1}/{config.EOX_FETCH_MAX_RETRIES})"
            )
        except Exception as e:
            logger.info(
                f"  [x] Tile ({tile.col_off},{tile.row_off}): error {e} "
                f"(attempt {attempt + 1}/{config.EOX_FETCH_MAX_RETRIES})"
            )

        if attempt < config.EOX_FETCH_MAX_RETRIES - 1:
            wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s, ...
            logger.info(f"  Waiting {wait_time}s before retrying...")
            time.sleep(wait_time)

    logger.warning(
        f"  [x] Tile ({tile.col_off},{tile.row_off}): all {config.EOX_FETCH_MAX_RETRIES} "
        f"attempts failed - stays black"
    )
    return None


def fetch_eox_mosaic(area_utm: tuple, dest_path) -> Tuple[bool, int]:
    """
    Downloads the complete Sentinel-2 cloudless mosaic for `area_utm` from EOX (tiled) and writes
    it as a single GeoTIFF to `dest_path` (EPSG:3857).

    Args:
        area_utm: (x_min, x_max, y_min, y_max) in the resolved source CRS, see
            horizon_image.horizon_area()
        dest_path: Target file (only created if at least one tile succeeded)

    Returns:
        (success, failed_count):
        - success: True if at least one tile was downloaded successfully (dest_path then exists,
          failed tiles stay black); False if ALL tiles failed (dest_path then does NOT exist).
        - failed_count: Number of tiles that stayed black (after exhausting all retries or due to
          a wrong image size). > 0 with success=True means: the mosaic is only PARTIALLY
          complete - the caller must then not cache it permanently (see
          ensure_horizon_texture()).
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.windows import Window

    dest_path = Path(dest_path)
    bbox_3857 = _mercator_bbox_for_area(area_utm)
    w, h = _mosaic_pixel_size(bbox_3857)
    tiles = tile_grid(bbox_3857, w, h)

    minx, miny, maxx, maxy = bbox_3857
    transform = from_bounds(minx, miny, maxx, maxy, w, h)
    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }

    temp_path = dest_path.with_name(dest_path.name + ".part")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    any_success = False
    failed_count = 0
    logger.info(f"  EOX mosaic: loading {w}x{h} px in {len(tiles)} tile(s)...")
    with rasterio.open(temp_path, "w", **profile) as dst:
        for tile in tiles:
            arr = _fetch_one_tile(tile)
            if arr is None:
                failed_count += 1
                continue  # tile stays black (a freshly created GeoTIFF is zero-initialized)
            if arr.shape[:2] != (tile.height, tile.width):
                # The server returned status 200 + Content-Type image/*, but the decoded image
                # does not have the requested pixel size (a realistic failure mode with an
                # external WMS server) - dst.write() with a wrong window size would raise an
                # exception and abort the WHOLE mosaic run. Instead: treat it like a decode
                # failure, the tile stays black, continue with the next tile.
                logger.warning(
                    f"  [x] Tile ({tile.col_off},{tile.row_off}): image size {arr.shape[1]}x{arr.shape[0]} "
                    f"does not match the requested size {tile.width}x{tile.height} - stays black"
                )
                failed_count += 1
                continue
            dst.write(np.moveaxis(arr, 2, 0), window=Window(tile.col_off, tile.row_off, tile.width, tile.height))
            any_success = True

    if any_success:
        os.replace(temp_path, dest_path)  # atomic: a crash midway never leaves a mosaic behind
        if failed_count:
            logger.warning(f"  [!] EOX mosaic: {failed_count}/{len(tiles)} tile(s) stayed black")
        else:
            logger.info(f"  [OK] EOX mosaic written: {dest_path}")
        return True, failed_count

    temp_path.unlink(missing_ok=True)
    logger.error("  [x] EOX mosaic: all tiles failed - no mosaic created")
    return False, failed_count


def ensure_horizon_texture(area_utm: tuple, size_px=None, resampling: str = "bilinear") -> Optional[Path]:
    """
    Public entry point: ensures a horizon texture is available for `area_utm` and returns its
    path. Fully automatic - no manual override file anymore (the former fixed
    `data/DOP300/horizon_temp.tif` could not be reliably distinguished per area, see git
    history). The texture ends up in config.EOX_TEXTURE_CACHE_DIR, with a file name that depends
    on area + target size + resampling + WMS layer - a cache file that is safe to delete and
    regenerate at any time (like the raw mosaic). This makes switching the source area (e.g.
    trying a different region) automatically correct: every area gets its own cache file.

    Args:
        area_utm: (x_min, x_max, y_min, y_max) in the resolved source CRS, see
            horizon_image.horizon_area()
        size_px: Edge length of the target texture; default config.HORIZON_IMAGE_SIZE_PX
        resampling: rasterio resampling name, see build_horizon_image()

    Returns:
        Path to the horizon texture under config.EOX_TEXTURE_CACHE_DIR, or None if no download
        was attempted (config.EOX_AUTO_DOWNLOAD == False), it failed completely, or an
        unexpected error occurred. NEVER raises - analogous to dgm30_fetch.ensure_dgm30_coverage()
        this entry point must simply return None on any error (network, file system, CRS
        transform, ...) so that horizon_workflow.py can call it without its own error handling.
    """
    global _attribution_logged

    try:
        if not config.EOX_AUTO_DOWNLOAD:
            return None

        size_px = size_px if size_px is not None else config.HORIZON_IMAGE_SIZE_PX

        # Cache key for the raw mosaic: independent of size_px/resampling (the mosaic is
        # independent of the target size), area_utm values rounded to 1 m + layer name.
        area_sig = f"{round(area_utm[0])}_{round(area_utm[1])}_{round(area_utm[2])}_{round(area_utm[3])}_{config.EOX_WMS_LAYER}"
        mosaic_cache_key = hashlib.sha1(area_sig.encode("utf-8")).hexdigest()[:16]
        mosaic_path = config.EOX_MOSAIC_CACHE_DIR / f"eox_mosaic_{mosaic_cache_key}.tif"

        # Cache key for the FINISHED, clipped texture: additionally depends on size_px/resampling,
        # since these change the result of build_horizon_image().
        texture_cache_key = hashlib.sha1(f"{area_sig}_{size_px}_{resampling}".encode("utf-8")).hexdigest()[:16]
        texture_path = config.EOX_TEXTURE_CACHE_DIR / f"horizon_texture_{texture_cache_key}.tif"

        if texture_path.exists():
            # Already built for exactly this area/size/resampling - neither network nor
            # another clip needed.
            if not _attribution_logged:
                logger.info(config.EOX_ATTRIBUTION_NOTICE)
                _attribution_logged = True
            return texture_path

        # Only > 0 after a fresh (partial) fetch in this call - a mosaic cache hit does not
        # reload the mosaic and therefore cannot have any new failed tiles either.
        failed_count = 0
        if not mosaic_path.exists():
            config.EOX_MOSAIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            success, failed_count = fetch_eox_mosaic(area_utm, mosaic_path)
            if not success:
                return None
            if failed_count:
                # Partial success: still build a texture for the CURRENT level (below), but cache
                # neither the raw mosaic NOR the texture built from it permanently - otherwise a
                # one-off network hiccup bakes a black tile into the horizon forever
                # (analogous to the "otherwise the error stays stuck in the cache" logic in
                # osm/downloader.get_osm_data() and to the failed/not_found distinction in
                # dgm30_fetch.download_dgm30_tiles()). The next run then sees neither a
                # mosaic NOR a texture cache hit and retries the missing tiles.
                logger.warning(
                    f"  [!] EOX mosaic incomplete ({failed_count} tile(s) black) - NOT "
                    f"cached permanently ({mosaic_path}). The next run retries the missing "
                    f"tiles automatically."
                )

        # The imagery license (CC BY-NC-SA) ties the attribution requirement to the USE of the
        # imagery, not to the download - hence logged here (on every call that actually uses
        # EOX imagery), not only in the cache-miss branch above.
        if not _attribution_logged:
            logger.info(config.EOX_ATTRIBUTION_NOTICE)
            _attribution_logged = True

        config.EOX_TEXTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        build_horizon_image(mosaic_path, texture_path, area=area_utm, size_px=size_px, resampling=resampling)

        if failed_count:
            # Do not leave an incomplete texture under the canonical cache name - otherwise a later
            # run would wrongly accept it as a complete cache hit. It remains usable for THIS run
            # nonetheless: renamed to a distinct name: the caller reads exactly the path returned
            # here right afterwards.
            partial_path = texture_path.with_name(f"{texture_path.stem}_partial.tif")
            os.replace(texture_path, partial_path)
            mosaic_path.unlink(missing_ok=True)
            return partial_path

        if not config.EOX_KEEP_RAW_MOSAIC:
            mosaic_path.unlink(missing_ok=True)

        return texture_path if texture_path.exists() else None

    except Exception as e:
        logger.error(f"  [x] Horizon texture auto-download failed: {e}")
        return None
