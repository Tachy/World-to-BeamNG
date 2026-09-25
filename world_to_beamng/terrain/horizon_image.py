"""
Horizon area and horizon image.

The horizon covers config.HORIZON_HALF_SIZE_M in every direction around the area center.
`build_horizon_image()` clips the area from any georeferenced RGB image (e.g. the automatically
downloaded Sentinel-2 raw mosaic, see terrain/sentinel2_fetch.py) and reprojects it into the
pipeline's resolved source CRS (default EPSG:25832/UTM32N, automatically detected for
GeoTIFF elevation data, see geometry.coordinates).
"""

import logging
from pathlib import Path
from typing import Sequence, Tuple

from .. import config

logger = logging.getLogger(__name__)

# From this share of the horizon area covered by the source image on, there is no warning
_FULL_COVERAGE = 0.995


def _dst_crs() -> str:
    """Target CRS of the horizon image: the pipeline's resolved source CRS. A function instead of a module
    constant, because the value is only known after the tile scan (geometry.coordinates.set_source_crs())."""
    from ..geometry.coordinates import get_source_crs_epsg

    return f"EPSG:{get_source_crs_epsg()}"


def horizon_area(global_offset: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    Horizon area in UTM 32N.

    Args:
        global_offset: area center (x, y[, z]) in UTM

    Returns:
        (x_min, x_max, y_min, y_max)
    """
    ox, oy = global_offset[0], global_offset[1]
    half = config.HORIZON_HALF_SIZE_M
    return ox - half, ox + half, oy - half, oy + half


def horizon_area_wgs84(global_offset: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    Horizon area in WGS84 (lon/lat) - basis for the Copernicus DEM tile selection
    (dgm30_fetch.py), whose tiles are named by degree grid instead of by UTM.

    Args:
        global_offset: area center (x, y[, z]) in UTM, as in horizon_area()

    Returns:
        (lon_min, lat_min, lon_max, lat_max)
    """
    from rasterio.warp import transform_bounds

    x_min, x_max, y_min, y_max = horizon_area(global_offset)
    return transform_bounds(_dst_crs(), "EPSG:4326", x_min, y_min, x_max, y_max)


def build_horizon_image(
    source: Path,
    output: Path,
    area: Tuple[float, float, float, float],
    size_px: int = config.HORIZON_IMAGE_SIZE_PX,
    resampling: str = "bilinear",
) -> float:
    """
    Clips `area` from a georeferenced RGB image and writes it as GeoTIFF in EPSG:25832.

    Args:
        source: source image (GeoTIFF with coordinate system, at least 3 bands = RGB, any CRS)
        output: target file
        area: (x_min, x_max, y_min, y_max) in UTM 32N, see horizon_area()
        size_px: edge length of the result (square)
        resampling: rasterio resampling name ("bilinear", "cubic", "average", ...)

    Returns:
        Share of the area (0..1) that the source image covers

    Raises:
        ValueError: source image without coordinate system, fewer than 3 bands or no overlap with `area`
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds

    x_min, x_max, y_min, y_max = area
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError(f"{source} has no coordinate system (not georeferenced)")
        if src.count < 3:
            raise ValueError(f"{source} has {src.count} band(s), 3 are required (RGB)")

        coverage = _coverage(src, area, transform_bounds)
        if coverage <= 0:
            raise ValueError(f"{source} does not cover the horizon area (X {x_min:.0f}..{x_max:.0f}, Y {y_min:.0f}..{y_max:.0f})")
        if coverage < _FULL_COVERAGE:
            logger.warning(f"  [!] The source image covers only {coverage:.0%} of the horizon area - the rest stays black")

        dst_crs = _dst_crs()
        transform = from_bounds(x_min, y_min, x_max, y_max, size_px, size_px)
        profile = {
            "driver": "GTiff",
            "width": size_px,
            "height": size_px,
            "count": 3,
            "dtype": src.dtypes[0],  # 8 or 16 bit is preserved; the horizon loader normalizes to 0..255
            "crs": dst_crs,
            "transform": transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            for band in (1, 2, 3):  # Band by band: the source image is never loaded into memory completely
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=getattr(Resampling, resampling),
                )
    return coverage


def _coverage(src, area, transform_bounds) -> float:
    """Share of the horizon area that lies within the source image (rectangles in the source image's CRS)."""
    x_min, x_max, y_min, y_max = area
    left, bottom, right, top = transform_bounds(_dst_crs(), src.crs, x_min, y_min, x_max, y_max)
    overlap_w = max(0.0, min(right, src.bounds.right) - max(left, src.bounds.left))
    overlap_h = max(0.0, min(top, src.bounds.top) - max(bottom, src.bounds.bottom))
    total = (right - left) * (top - bottom)
    return (overlap_w * overlap_h) / total if total > 0 else 0.0
