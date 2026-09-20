"""
Horizont-Fläche und Horizont-Bild.

Der Horizont deckt config.HORIZON_HALF_SIZE_M in jede Richtung um die Gebietsmitte ab. Das Satellitenbild dafür
(data/DOP300/<config.SENTINEL2_FILE>) muss GENAU diese Fläche als GeoTIFF in UTM 32N (EPSG:25832) zeigen.
`build_horizon_image()` schneidet es aus einem beliebigen georeferenzierten RGB-Bild (z. B. Sentinel-2 in Web-Mercator)
zu und projiziert es um; das Kommandozeilenwerkzeug dafür ist tools/make_horizon_image.py.
"""

import logging
from pathlib import Path
from typing import Sequence, Tuple

from .. import config

logger = logging.getLogger(__name__)

UTM_CRS = "EPSG:25832"
# Ab diesem Anteil der Horizont-Fläche, der vom Quellbild abgedeckt wird, gibt es keine Warnung
_FULL_COVERAGE = 0.995


def horizon_area(global_offset: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    Horizont-Fläche in UTM 32N.

    Args:
        global_offset: Gebietsmitte (x, y[, z]) in UTM

    Returns:
        (x_min, x_max, y_min, y_max)
    """
    ox, oy = global_offset[0], global_offset[1]
    half = config.HORIZON_HALF_SIZE_M
    return ox - half, ox + half, oy - half, oy + half


def build_horizon_image(
    source: Path,
    output: Path,
    area: Tuple[float, float, float, float],
    size_px: int = config.HORIZON_IMAGE_SIZE_PX,
    resampling: str = "bilinear",
) -> float:
    """
    Schneidet `area` aus einem georeferenzierten RGB-Bild aus und schreibt es als GeoTIFF in EPSG:25832.

    Args:
        source: Quellbild (GeoTIFF mit Koordinatensystem, mindestens 3 Bänder = RGB, beliebiges CRS)
        output: Zieldatei
        area: (x_min, x_max, y_min, y_max) in UTM 32N, siehe horizon_area()
        size_px: Kantenlänge des Ergebnisses (quadratisch)
        resampling: rasterio-Resampling-Name ("bilinear", "cubic", "average", ...)

    Returns:
        Anteil der Fläche (0..1), den das Quellbild abdeckt

    Raises:
        ValueError: Quellbild ohne Koordinatensystem, weniger als 3 Bänder oder keine Überdeckung mit `area`
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds

    x_min, x_max, y_min, y_max = area
    with rasterio.open(source) as src:
        if src.crs is None:
            raise ValueError(f"{source} hat kein Koordinatensystem (nicht georeferenziert)")
        if src.count < 3:
            raise ValueError(f"{source} hat {src.count} Band/Bänder, gebraucht werden 3 (RGB)")

        coverage = _coverage(src, area, transform_bounds)
        if coverage <= 0:
            raise ValueError(f"{source} überdeckt die Horizont-Fläche nicht (X {x_min:.0f}..{x_max:.0f}, Y {y_min:.0f}..{y_max:.0f})")
        if coverage < _FULL_COVERAGE:
            logger.warning(f"  [!] Das Quellbild deckt nur {coverage:.0%} der Horizont-Fläche ab - der Rest bleibt schwarz")

        transform = from_bounds(x_min, y_min, x_max, y_max, size_px, size_px)
        profile = {
            "driver": "GTiff",
            "width": size_px,
            "height": size_px,
            "count": 3,
            "dtype": src.dtypes[0],  # 8 oder 16 Bit bleibt erhalten; der Horizont-Loader normiert auf 0..255
            "crs": UTM_CRS,
            "transform": transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            for band in (1, 2, 3):  # Band für Band: das Quellbild wird nie komplett in den Speicher geladen
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=UTM_CRS,
                    resampling=getattr(Resampling, resampling),
                )
    return coverage


def _coverage(src, area, transform_bounds) -> float:
    """Anteil der Horizont-Fläche, der im Quellbild liegt (Rechtecke im Koordinatensystem des Quellbilds)."""
    x_min, x_max, y_min, y_max = area
    left, bottom, right, top = transform_bounds(UTM_CRS, src.crs, x_min, y_min, x_max, y_max)
    overlap_w = max(0.0, min(right, src.bounds.right) - max(left, src.bounds.left))
    overlap_h = max(0.0, min(top, src.bounds.top) - max(bottom, src.bounds.bottom))
    total = (right - left) * (top - bottom)
    return (overlap_w * overlap_h) / total if total > 0 else 0.0
