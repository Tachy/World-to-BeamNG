"""
Scanner for elevation data tiles (data/height).

Each file (loose GeoTIFF OR ZIP) is recognized by its ACTUAL content - see
terrain.elevation_io.read_elevation_tile() for the two supported formats (ASCII-XYZ point cloud,
GeoTIFF raster). The file name is irrelevant for this (only for informative display in the log); there is
no preferred special path for a particular naming scheme such as that of LGL Baden-Württemberg.

Each tile is cached through the same file-based cache as workflow/tile_processor.py (key
"height_raw_<hash>"), so that a file is only actually parsed once, no matter whether it is scanned
first or loaded first.
"""

import logging
from pathlib import Path
from typing import Dict, List

from .. import config
from ..core.cache_manager import CacheManager
from ..terrain.elevation_io import read_elevation_tile_cached

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".zip", ".tif", ".tiff")


def scan_elevation_tiles(dgm_dir, cache_dir=None) -> List[Dict]:
    """
    Scans dgm_dir for elevation data files and fully reads each one (cached) to determine its true
    BBox and CRS - independent of the file name or a fixed tile size.

    Args:
        dgm_dir: directory with elevation data (loose GeoTIFFs and/or ZIPs, mixed arbitrarily)
        cache_dir: cache directory (default: config.CACHE_DIR) - same cache key scheme as
            workflow/tile_processor.py, so that a file is only parsed once

    Returns:
        List of tile metadata dicts, sorted by file name:
        [{"filename", "filepath", "bbox_utm": (x_min, x_max, y_min, y_max),
          "easting", "northing", "tile_x", "tile_y", "tile_size", "crs_epsg"}, ...]
        easting/northing/tile_x/tile_y/tile_size are derived from bbox_utm (compatibility with
        existing callers); bbox_utm is the authoritative, possibly non-square area.
        crs_epsg is None for formats without an embedded CRS (ASCII-XYZ) - see
        resolve_source_crs_epsg().
    """
    if not Path(dgm_dir).exists():
        logger.warning(f"[WARNING] Height data directory not found: {dgm_dir}")
        return []

    cache = CacheManager(cache_dir or config.CACHE_DIR)
    tiles = []

    candidates = sorted(p for p in Path(dgm_dir).iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    for filepath in candidates:
        points, _elevations, crs_epsg, bbox_utm = read_elevation_tile_cached(filepath, cache)
        if points is None or len(points) == 0 or bbox_utm is None:
            logger.warning(f"[WARNING] No height data in {filepath.name} - skipped")
            continue

        x_min, x_max, y_min, y_max = bbox_utm

        tiles.append(
            {
                "filename": filepath.name,
                "filepath": filepath,
                "bbox_utm": (x_min, x_max, y_min, y_max),
                "easting": x_min,
                "northing": y_min,
                "tile_x": x_min,
                "tile_y": y_min,
                "tile_size": max(x_max - x_min, y_max - y_min),
                "crs_epsg": crs_epsg,
            }
        )

    if not tiles:
        logger.warning(f"[WARNING] No height data files found in: {dgm_dir}")
    else:
        logger.info(f"[INFO] {len(tiles)} height data tiles found")
        for tile in tiles:
            x0, x1, y0, y1 = tile["bbox_utm"]
            logger.debug(f"  - {tile['filename']} → X={x0:.0f}..{x1:.0f}, Y={y0:.0f}..{y1:.0f}")

    return tiles


def resolve_source_crs_epsg(tiles: List[Dict]) -> int:
    """
    Determines the common source CRS of all tiles.

    Tiles without their own CRS (ASCII-XYZ, e.g. LGL Baden-Württemberg) say nothing about the CRS
    - config.SOURCE_CRS_EPSG applies for them. Tiles WITH their own CRS (GeoTIFF) must all
    agree; otherwise it is unclear in which CRS the whole area should be processed.

    Raises:
        ValueError: if tiles with different CRS are mixed (mixing different
            DGM CRS is a deliberate scope limit, not a supported case)
    """
    detected = {t["crs_epsg"] for t in tiles if t.get("crs_epsg") is not None}
    if len(detected) > 1:
        raise ValueError(
            f"Height data tiles with different CRS found (EPSG {sorted(detected)}) - "
            "mixing different DGM CRS is not supported."
        )
    if detected:
        return detected.pop()
    return config.SOURCE_CRS_EPSG


def compute_global_bbox(tiles):
    """
    Computes the global bounding box over all tiles.

    Args:
        tiles: result of scan_elevation_tiles()

    Returns:
        Tuple: (min_x, max_x, min_y, max_y) in UTM coordinates
    """
    if not tiles:
        return None

    min_x = min(t["bbox_utm"][0] for t in tiles)
    max_x = max(t["bbox_utm"][1] for t in tiles)
    min_y = min(t["bbox_utm"][2] for t in tiles)
    max_y = max(t["bbox_utm"][3] for t in tiles)

    return (min_x, max_x, min_y, max_y)


def compute_global_center(tiles):
    """
    Computes the global center point over all tiles.

    Args:
        tiles: result of scan_elevation_tiles()

    Returns:
        Tuple: (center_x, center_y) in UTM coordinates
    """
    bbox = compute_global_bbox(tiles)
    if bbox is None:
        return (0.0, 0.0)

    min_x, max_x, min_y, max_y = bbox
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    return (center_x, center_y)
