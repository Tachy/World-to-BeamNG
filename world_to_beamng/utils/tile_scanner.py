"""
Scanner für Höhendaten-Kacheln (data/height).

Jede Datei (lose GeoTIFF ODER ZIP) wird am TATSÄCHLICHEN Inhalt erkannt - siehe
terrain.elevation_io.read_elevation_tile() für die zwei unterstützten Formate (ASCII-XYZ-Punktwolke,
GeoTIFF-Raster). Der Dateiname ist dafür irrelevant (nur zur informativen Anzeige im Log); es gibt
keinen bevorzugten Sonderpfad für ein bestimmtes Namensschema wie das der LGL Baden-Württemberg.

Jede Kachel wird gecacht über denselben dateibasierten Cache wie workflow/tile_processor.py (Key
"height_raw_<hash>"), damit eine Datei nur einmal tatsächlich geparst wird, egal ob sie zuerst
gescannt oder zuerst geladen wird.
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
    Scannt dgm_dir nach Höhendaten-Dateien und liest jede (gecacht) vollständig ein, um ihre echte
    BBox und ihr CRS zu bestimmen - unabhängig vom Dateinamen oder einer festen Kachelgröße.

    Args:
        dgm_dir: Verzeichnis mit Höhendaten (lose GeoTIFFs und/oder ZIPs, beliebig gemischt)
        cache_dir: Cache-Verzeichnis (Default: config.CACHE_DIR) - dasselbe Cache-Key-Schema wie
            workflow/tile_processor.py, damit eine Datei nur einmal geparst wird

    Returns:
        Liste von Tile-Metadaten-Dicts, sortiert nach Dateiname:
        [{"filename", "filepath", "bbox_utm": (x_min, x_max, y_min, y_max),
          "easting", "northing", "tile_x", "tile_y", "tile_size", "crs_epsg"}, ...]
        easting/northing/tile_x/tile_y/tile_size sind aus bbox_utm abgeleitet (Kompatibilität zu
        bestehenden Aufrufern); bbox_utm ist die maßgebliche, ggf. nicht-quadratische Fläche.
        crs_epsg ist None bei Formaten ohne eingebettetes CRS (ASCII-XYZ) - siehe
        resolve_source_crs_epsg().
    """
    if not Path(dgm_dir).exists():
        logger.warning(f"[WARNUNG] Höhendaten-Verzeichnis nicht gefunden: {dgm_dir}")
        return []

    cache = CacheManager(cache_dir or config.CACHE_DIR)
    tiles = []

    candidates = sorted(p for p in Path(dgm_dir).iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    for filepath in candidates:
        points, _elevations, crs_epsg, bbox_utm = read_elevation_tile_cached(filepath, cache)
        if points is None or len(points) == 0 or bbox_utm is None:
            logger.warning(f"[WARNUNG] Keine Höhendaten in {filepath.name} - übersprungen")
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
        logger.warning(f"[WARNUNG] Keine Höhendaten-Dateien gefunden in: {dgm_dir}")
    else:
        logger.info(f"[INFO] {len(tiles)} Höhendaten-Kacheln gefunden")
        for tile in tiles:
            x0, x1, y0, y1 = tile["bbox_utm"]
            logger.info(f"  - {tile['filename']} → X={x0:.0f}..{x1:.0f}, Y={y0:.0f}..{y1:.0f}")

    return tiles


def resolve_source_crs_epsg(tiles: List[Dict]) -> int:
    """
    Bestimmt die gemeinsame Quell-CRS aller Kacheln.

    Kacheln ohne eigenes CRS (ASCII-XYZ, z.B. LGL Baden-Württemberg) sagen nichts über die CRS aus
    - dafür gilt config.SOURCE_CRS_EPSG. Kacheln MIT eigenem CRS (GeoTIFF) müssen sich alle einig
    sein; sonst ist unklar, in welcher CRS die Gesamtfläche verarbeitet werden soll.

    Raises:
        ValueError: wenn Kacheln mit unterschiedlichem CRS gemischt sind (Mischung verschiedener
            DGM-CRS ist eine bewusste Scope-Grenze, kein unterstützter Fall)
    """
    detected = {t["crs_epsg"] for t in tiles if t.get("crs_epsg") is not None}
    if len(detected) > 1:
        raise ValueError(
            f"Höhendaten-Kacheln mit unterschiedlichem CRS gefunden (EPSG {sorted(detected)}) - "
            "Mischung verschiedener DGM-CRS wird nicht unterstützt."
        )
    if detected:
        return detected.pop()
    return config.SOURCE_CRS_EPSG


def compute_global_bbox(tiles):
    """
    Berechnet die globale Bounding Box über alle Tiles.

    Args:
        tiles: Ergebnis von scan_elevation_tiles()

    Returns:
        Tuple: (min_x, max_x, min_y, max_y) in UTM-Koordinaten
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
    Berechnet den globalen Center-Punkt über alle Tiles.

    Args:
        tiles: Ergebnis von scan_elevation_tiles()

    Returns:
        Tuple: (center_x, center_y) in UTM-Koordinaten
    """
    bbox = compute_global_bbox(tiles)
    if bbox is None:
        return (0.0, 0.0)

    min_x, max_x, min_y, max_y = bbox
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    return (center_x, center_y)
