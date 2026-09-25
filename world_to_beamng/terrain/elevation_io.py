"""
Einheitlicher Reader für Höhendaten-Kacheln.

Erkennt den Inhalt einer Kachel-Datei (lose Datei ODER ZIP) am TATSÄCHLICHEN Format, nicht am
Dateinamen - es gibt nur zwei Format-Zweige, kein bevorzugter LGL-Sonderpfad:

- ASCII-XYZ-Punktwolke (z.B. LGL Baden-Württemberg: ZIP mit *.xyz-Dateien). Hat nie ein
  eingebettetes CRS.
- GeoTIFF-Raster (lose *.tif/*.tiff-Datei ODER GeoTIFF-Member in einem ZIP). Hat ein eingebettetes
  CRS, das automatisch gelesen wird.

Beide Zweige liefern dasselbe Rückgabeformat (points, elevations, crs_epsg, bbox_utm), damit
tile_scanner.py und workflow/tile_processor.py nicht zwischen Formaten unterscheiden müssen.

WICHTIG zu bbox_utm: Punkte/Pixel repräsentieren ZELLMITTELPUNKTE (bei GeoTIFF: Pixel-Mittelpunkte,
bei XYZ: die Rasterzelle, für die der Punkt steht). Die tatsächlich abgedeckte Fläche reicht daher
eine halbe Zellgröße über den äußersten Punkt hinaus - bbox_utm ist NICHT einfach points.min()/max(),
sonst würde die Kachel um eine halbe Zelle zu klein berechnet (das hat vor dieser Korrektur die
Luftbild-Kachelgrenzen um 0.5 m gegenüber der echten Terrain-Fläche verschoben).
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
    Wie read_elevation_tile(), aber mit demselben dateibasierten Cache (CacheManager, Key
    "height_raw_<hash>"), den auch workflow/tile_processor.py und die Scan-Phase
    (utils/tile_scanner.py::scan_elevation_tiles()) nutzen - eine Datei wird dadurch effektiv nur
    einmal geparst, egal wie oft/wo sie gebraucht wird.

    Args:
        filepath: siehe read_elevation_tile()
        cache_manager: core.cache_manager.CacheManager
        tile_hash: optionaler, bereits berechneter Datei-Hash (spart ein erneutes Hashen)

    Returns:
        wie read_elevation_tile()
    """
    filepath = Path(filepath)
    if tile_hash is None:
        tile_hash = cache_manager.hash_file(filepath)
    cache_key = f"{CACHE_KEY_PREFIX}{tile_hash}"

    cached = cache_manager.get_npz(cache_key)
    # "bbox_utm" fehlt in Cache-Einträgen aus einer Zeit vor dieser Funktion (nur points/elevations,
    # evtl. crs_epsg) - so ein unvollständiger Treffer wird wie ein Cache-Miss behandelt (frisch
    # gelesen und mit dem vollständigen Format neu geschrieben), statt bbox_utm=None zurückzugeben
    # und die Kachel dadurch stillschweigend unbrauchbar zu machen (siehe utils.tile_scanner).
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
    Liest eine einzelne Höhendaten-Datei vollständig ein.

    Args:
        filepath: lose GeoTIFF-Datei ODER ZIP (mit XYZ-Punktdateien ODER GeoTIFF-Member)

    Returns:
        (points Nx2, elevations N, crs_epsg, bbox_utm) - crs_epsg ist None bei ASCII-XYZ (nie
        eingebettetes CRS vorhanden, unabhängig vom Dateinamen -> config.SOURCE_CRS_EPSG-Fallback
        gilt), sonst das aus dem GeoTIFF gelesene EPSG (None, falls das CRS kein EPSG-Code ist -
        bekannte Grenze, siehe Plan). bbox_utm = tatsächlich abgedeckte Fläche (x_min, x_max, y_min,
        y_max), siehe Modul-Docstring. (None, None, None, None) bei Fehlern oder unbekanntem Format.
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
    """ASCII-XYZ-Punktdateien aus einem ZIP (LGL-Format: X Y Z, leerzeichengetrennt)."""
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
    Schätzt die tatsächliche Abdeckungsfläche einer regelmäßigen Punktwolke aus Zellmittelpunkten
    (z.B. LGL-DGM1: 1x1m-Zellen). Jeder Punkt vertritt eine Zelle mit einer halben Gitterweite Rand
    um sich herum - die reine Punkt-BBox (min/max) wäre um eine halbe Zelle je Seite zu klein.

    Die Gitterweite wird aus den Daten selbst abgeleitet (kleinster positiver Abstand zwischen
    unterschiedlichen X- bzw. Y-Werten), nicht aus config.GRID_SPACING angenommen - funktioniert
    damit für jede native Punktdichte.
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
    Einzelband-Raster (GeoTIFF o.ä.) via rasterio.

    Wird die native Auflösung feiner als config.GRID_SPACING gelesen, liest GDAL bereits
    verkleinert (average-Resampling) - vermeidet unnötig große Punktwolken und hält die
    Zielauflösung unabhängig von der Quelle konstant (siehe Plan, Abschnitt 2). Native Auflösung
    gröber oder gleich GRID_SPACING wird unverändert gelesen; das nachgelagerte
    NearestNDInterpolator-Resampling in terrain/grid.py übernimmt dort wie bisher das Hochskalieren.

    NoData-Pixel werden vor der Rückgabe entfernt (per src.nodata bzw. NaN/Inf). bbox_utm ist die
    ECHTE Raster-Abdeckung (aus Transform+Shape, rasterio.transform.array_bounds) - nicht aus den
    (NoData-bereinigten) Pixel-Mittelpunkten abgeleitet, bleibt also auch bei Löchern am Rand korrekt.
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
        # Pixel-Mittelpunkte, vektorisiert (kein Python-Loop über rasterio.transform.xy() - bei
        # Millionen Pixeln zu langsam)
        xs = a * (cols + 0.5) + b * (rows + 0.5) + c
        ys = d * (cols + 0.5) + e * (rows + 0.5) + f
        points = np.column_stack([xs, ys])
        elevations = data[rows, cols]

    return points, elevations, crs_epsg, bbox_utm
