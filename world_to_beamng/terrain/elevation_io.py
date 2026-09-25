"""
Unified reader for elevation data tiles.

Detects the content of a tile file (loose file OR ZIP) by its ACTUAL format, not by its file
name - there are only two format branches, no preferred LGL special path:

- ASCII XYZ point cloud (e.g. LGL Baden-Württemberg: ZIP with *.xyz files). Never has an
  embedded CRS.
- GeoTIFF raster (loose *.tif/*.tiff file OR GeoTIFF member in a ZIP). Has an embedded CRS,
  which is read automatically.

Both branches return the same result format (points, elevations, crs_epsg, bbox_utm), so that
tile_scanner.py and workflow/tile_processor.py do not have to distinguish between formats.

IMPORTANT regarding bbox_utm: points/pixels represent CELL CENTERS (for GeoTIFF: pixel centers,
for XYZ: the raster cell the point stands for). The area actually covered therefore extends
half a cell size beyond the outermost point - bbox_utm is NOT simply points.min()/max(),
otherwise the tile would be computed half a cell too small (before this fix, that shifted the
aerial photo tile boundaries by 0.5 m relative to the real terrain area).
"""

import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from world_to_beamng.logging_config import LoggerConfig

from .. import config

logger = LoggerConfig.get_logger()

RASTER_EXTENSIONS = (".tif", ".tiff")
CACHE_KEY_PREFIX = "height_raw_"

BBox = Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
ReadResult = Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[BBox]]


def read_elevation_tile_cached(filepath, cache_manager, tile_hash: Optional[str] = None) -> ReadResult:
    """
    Like read_elevation_tile(), but with the same file-based cache (CacheManager, key
    "height_raw_<hash>") that workflow/tile_processor.py and the scan phase
    (utils/tile_scanner.py::scan_elevation_tiles()) also use - a file is thereby effectively parsed
    only once, no matter how often/where it is needed.

    Args:
        filepath: see read_elevation_tile()
        cache_manager: core.cache_manager.CacheManager
        tile_hash: optional, already computed file hash (saves hashing again)

    Returns:
        same as read_elevation_tile()
    """
    filepath = Path(filepath)
    if tile_hash is None:
        tile_hash = cache_manager.hash_file(filepath)
    cache_key = f"{CACHE_KEY_PREFIX}{tile_hash}"

    cached = cache_manager.get_npz(cache_key)
    # "bbox_utm" is missing in cache entries from before this function existed (only
    # points/elevations, possibly crs_epsg) - such an incomplete hit is treated like a cache miss
    # (read fresh and rewritten in the full format), instead of returning bbox_utm=None and
    # thereby silently making the tile unusable (see utils.tile_scanner).
    if cached is not None and "bbox_utm" in cached:
        crs_epsg = int(cached["crs_epsg"]) if "crs_epsg" in cached else -1
        bbox_utm = tuple(cached["bbox_utm"].tolist())
        return cached["points"], cached["elevations"], (crs_epsg if crs_epsg >= 0 else None), bbox_utm

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(filepath)
    if points is not None and elevations is not None:
        cache_manager.set_npz(
            cache_key,
            points=points,
            elevations=elevations,
            crs_epsg=np.array(crs_epsg if crs_epsg is not None else -1),
            bbox_utm=np.array(bbox_utm, dtype=np.float64),
        )
    return points, elevations, crs_epsg, bbox_utm


def read_elevation_tile(filepath) -> ReadResult:
    """
    Reads a single elevation data file completely.

    Args:
        filepath: loose GeoTIFF file OR ZIP (with XYZ point files OR GeoTIFF member)

    Returns:
        (points Nx2, elevations N, crs_epsg, bbox_utm) - crs_epsg is None for ASCII XYZ (never has
        an embedded CRS, regardless of the file name -> the config.SOURCE_CRS_EPSG fallback
        applies), otherwise the EPSG read from the GeoTIFF (None if the CRS is not an EPSG code -
        known limitation, see plan). bbox_utm = area actually covered (x_min, x_max, y_min,
        y_max), see module docstring. (None, None, None, None) on errors or unknown format.
    """
    filepath = Path(filepath)
    try:
        suffix = filepath.suffix.lower()
        if suffix == ".zip":
            return _read_zip(filepath)
        if suffix in RASTER_EXTENSIONS:
            return _read_raster(str(filepath))
        logger.error(f"  [!] Unknown height data format: {filepath.name}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"  [!] Error loading {filepath}: {e}")
        return None, None, None, None


def _read_zip(zip_path: Path) -> ReadResult:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        xyz_members = [n for n in names if n.lower().endswith(".xyz")]
        raster_members = [n for n in names if n.lower().endswith(RASTER_EXTENSIONS)]

        if xyz_members:
            points, elevations = _read_xyz_members(zf, xyz_members)
            if points is None:
                return None, None, None, None
            return points, elevations, None, _point_grid_bbox(points)

        if raster_members:
            all_points, all_elevations, all_bbox = [], [], []
            crs_epsg = None
            for member in raster_members:
                points, elevations, epsg, bbox = _read_raster(f"/vsizip/{zip_path}/{member}")
                if points is None:
                    continue
                all_points.append(points)
                all_elevations.append(elevations)
                all_bbox.append(bbox)
                if crs_epsg is None:
                    crs_epsg = epsg
            if not all_points:
                return None, None, None, None
            combined_bbox = (
                min(b[0] for b in all_bbox),
                max(b[1] for b in all_bbox),
                min(b[2] for b in all_bbox),
                max(b[3] for b in all_bbox),
            )
            return np.vstack(all_points), np.hstack(all_elevations), crs_epsg, combined_bbox

        logger.error(f"  [!] Neither XYZ nor GeoTIFF data found in {zip_path.name}")
        return None, None, None, None


def _read_xyz_members(zf: zipfile.ZipFile, members) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """ASCII XYZ point files from a ZIP (LGL format: X Y Z, space-separated)."""
    all_points, all_elevations = [], []
    for name in members:
        with zf.open(name) as f:
            data = np.loadtxt(f, delimiter=" ", dtype=float)
            if data.size == 0:
                continue
            all_points.append(data[:, :2])
            all_elevations.append(data[:, 2])
    if not all_points:
        return None, None
    return np.vstack(all_points), np.hstack(all_elevations)


def _point_grid_bbox(points: np.ndarray) -> BBox:
    """
    Estimates the actual coverage area of a regular point cloud of cell centers
    (e.g. LGL DGM1: 1x1m cells). Each point stands for a cell with a half grid spacing of border
    around it - the plain point bbox (min/max) would be half a cell too small on each side.

    The grid spacing is derived from the data itself (smallest positive distance between
    distinct X or Y values), not assumed from config.GRID_SPACING - so it works
    for any native point density.
    """
    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
    margin_x = _half_spacing(points[:, 0])
    margin_y = _half_spacing(points[:, 1])
    return x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y


def _half_spacing(values: np.ndarray) -> float:
    unique = np.unique(values)
    if unique.size < 2:
        return 0.0
    diffs = np.diff(unique)
    positive = diffs[diffs > 1e-9]
    if positive.size == 0:
        return 0.0
    return float(np.min(positive)) / 2.0


def _read_raster(path_or_vsi: str) -> ReadResult:
    """
    Single-band raster (GeoTIFF or similar) via rasterio.

    If the native resolution is finer than config.GRID_SPACING, GDAL already reads it
    downscaled (average resampling) - avoids needlessly large point clouds and keeps the
    target resolution constant regardless of the source (see plan, section 2). A native resolution
    coarser than or equal to GRID_SPACING is read unchanged; the downstream
    NearestNDInterpolator resampling in terrain/grid.py handles the upscaling there as before.

    NoData pixels are removed before returning (via src.nodata or NaN/Inf). bbox_utm is the
    REAL raster coverage (from transform+shape, rasterio.transform.array_bounds) - not derived from
    the (NoData-cleaned) pixel centers, so it stays correct even with holes at the edge.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds

    with rasterio.open(path_or_vsi) as src:
        crs_epsg = src.crs.to_epsg() if src.crs else None

        res_x, res_y = src.res
        target = float(config.GRID_SPACING)
        if res_x < target or res_y < target:
            out_w = max(1, round(src.width * (res_x / target)))
            out_h = max(1, round(src.height * (res_y / target)))
            band = src.read(1, out_shape=(1, out_h, out_w), resampling=Resampling.average, masked=True)
            transform = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
        else:
            out_h, out_w = src.height, src.width
            band = src.read(1, masked=True)
            transform = src.transform

        left, bottom, right, top = array_bounds(out_h, out_w, transform)
        bbox_utm = (left, right, bottom, top)

        data = np.ma.filled(np.ma.asarray(band), np.nan).astype(np.float64)
        if data.ndim == 3:
            data = data[0]

        valid = np.isfinite(data)
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            return None, None, crs_epsg, bbox_utm

        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        # Pixel centers, vectorized (no Python loop over rasterio.transform.xy() - too slow for
        # millions of pixels)
        xs = a * (cols + 0.5) + b * (rows + 0.5) + c
        ys = d * (cols + 0.5) + e * (rows + 0.5) + f
        points = np.column_stack([xs, ys])
        elevations = data[rows, cols]

    return points, elevations, crs_epsg, bbox_utm
