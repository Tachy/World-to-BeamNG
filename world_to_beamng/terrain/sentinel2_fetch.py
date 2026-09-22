"""
Automatischer Download eines Sentinel-2-cloudless-Mosaiks (EOX WMS-Dienst, `config.EOX_WMS_URL`)
für die Horizont-Fläche, als Rohmosaik gecacht und über `horizon_image.build_horizon_image()` zur
finalen Horizont-Textur zugeschnitten/reprojiziert.

Analog zum automatischen DGM30-Download in `world_to_beamng/terrain/dgm30_fetch.py` und zum OSM-
Overpass-Download in `world_to_beamng/osm/downloader.py` (Retry mit exponentiellem Backoff,
gestreamtes Fortschritts-Logging) - der Ansatz ist hier bewusst lokal dupliziert statt in einen
gemeinsamen Helfer ausgelagert (gleiche Entscheidung wie in Task 2).

Der EOX-Server begrenzt WMS-GetMap-Requests nicht dokumentiert in der Größe - das Mosaik wird
deshalb in Kacheln von höchstens `config.EOX_MAX_REQUEST_PX` Kantenlänge zerlegt (tile_grid()) und
Kachel für Kachel direkt in das Ziel-GeoTIFF geschrieben.
"""

import hashlib
import io
import logging
import os
import time
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from .. import config
from .horizon_image import build_horizon_image

logger = logging.getLogger(__name__)

# Wird genau einmal pro Prozesslauf auf True gesetzt, sobald die EOX-Attribution geloggt wurde -
# verhindert doppelte/mehrfache Meldung, falls ensure_horizon_texture() mehrfach aufgerufen wird.
_attribution_logged = False


class TileRequest(NamedTuple):
    """Ein einzelnes WMS-GetMap-Request-Fenster innerhalb des Gesamtmosaiks (siehe tile_grid())."""

    col_off: int
    row_off: int
    width: int
    height: int
    bbox: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in EPSG:3857


def _mercator_bbox_for_area(area_utm: tuple) -> Tuple[float, float, float, float]:
    """
    Transformiert `area_utm` (x_min, x_max, y_min, y_max in der aufgelösten Quell-CRS, siehe
    horizon_image.horizon_area()) nach EPSG:3857 und weitet das Ergebnis um
    config.EOX_FETCH_MARGIN_FACTOR um den Mittelpunkt auf (gegen Rundungslücken am Rand).

    Returns:
        (minx, miny, maxx, maxy) in EPSG:3857 - ACHTUNG anderes Tupel-Format als area_utm!
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
    """Pixelgröße (w, h) des Gesamtmosaiks für `bbox_3857` bei config.EOX_TARGET_RESOLUTION_M,
    gedeckelt auf config.EOX_MOSAIC_MAX_PX je Achse."""
    minx, miny, maxx, maxy = bbox_3857
    w = round((maxx - minx) / config.EOX_TARGET_RESOLUTION_M)
    h = round((maxy - miny) / config.EOX_TARGET_RESOLUTION_M)
    w = max(1, min(w, config.EOX_MOSAIC_MAX_PX))
    h = max(1, min(h, config.EOX_MOSAIC_MAX_PX))
    return w, h


def tile_grid(bbox_3857: tuple, w: int, h: int) -> List[TileRequest]:
    """
    Zerlegt das w x h-Mosaik in TileRequest-Kacheln mit höchstens config.EOX_MAX_REQUEST_PX
    Kantenlänge (viele WMS-Server begrenzen WIDTH/HEIGHT je Request). Reine, netzwerklose Funktion.

    Bild-Konvention: Zeile 0 = Bildoberkante = geografisch Norden = maxy; mit wachsendem row_off
    sinkt die geografische Y-Koordinate.
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
    """Lädt eine einzelne WMS-Kachel mit Retry + exponentiellem Backoff (analog
    dgm30_fetch._download_one_tile()). Rückgabe: (H, W, 3)-uint8-Array oder None bei endgültigem
    Fehlschlag (Timeout, 5xx, unerwarteter Content-Type, Dekodier-Fehler - egal welcher Grund,
    diese Kachel bleibt dann einfach schwarz)."""
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
                f"  [x] Kachel ({tile.col_off},{tile.row_off}): Status {response.status_code}, "
                f"Content-Type {content_type!r} (Versuch {attempt + 1}/{config.EOX_FETCH_MAX_RETRIES})"
            )

        except requests.exceptions.Timeout:
            logger.info(
                f"  [x] Kachel ({tile.col_off},{tile.row_off}): Timeout "
                f"(Versuch {attempt + 1}/{config.EOX_FETCH_MAX_RETRIES})"
            )
        except Exception as e:
            logger.info(
                f"  [x] Kachel ({tile.col_off},{tile.row_off}): Fehler {e} "
                f"(Versuch {attempt + 1}/{config.EOX_FETCH_MAX_RETRIES})"
            )

        if attempt < config.EOX_FETCH_MAX_RETRIES - 1:
            wait_time = 2**attempt  # Exponentielles Backoff: 1s, 2s, 4s, ...
            logger.info(f"  Warte {wait_time}s vor erneutem Versuch...")
            time.sleep(wait_time)

    logger.warning(
        f"  [x] Kachel ({tile.col_off},{tile.row_off}): alle {config.EOX_FETCH_MAX_RETRIES} "
        f"Versuche fehlgeschlagen - bleibt schwarz"
    )
    return None


def fetch_eox_mosaic(area_utm: tuple, dest_path) -> bool:
    """
    Lädt das komplette Sentinel-2-cloudless-Mosaik für `area_utm` von EOX (gekachelt) und schreibt
    es als ein GeoTIFF nach `dest_path` (EPSG:3857).

    Args:
        area_utm: (x_min, x_max, y_min, y_max) in der aufgelösten Quell-CRS, siehe
            horizon_image.horizon_area()
        dest_path: Zieldatei (wird nur bei mindestens einer erfolgreichen Kachel erzeugt)

    Returns:
        True, wenn mindestens eine Kachel erfolgreich geladen wurde (dest_path existiert dann,
        fehlgeschlagene Kacheln bleiben schwarz); False, wenn ALLE Kacheln fehlgeschlagen sind
        (dest_path existiert dann NICHT).
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
    logger.info(f"  EOX-Mosaik: {w}x{h} px in {len(tiles)} Kachel(n) laden...")
    with rasterio.open(temp_path, "w", **profile) as dst:
        for tile in tiles:
            arr = _fetch_one_tile(tile)
            if arr is None:
                continue  # Kachel bleibt schwarz (frisch angelegtes GeoTIFF ist nullinitialisiert)
            if arr.shape[:2] != (tile.height, tile.width):
                # Der Server hat Status 200 + Content-Type image/* geliefert, aber das dekodierte
                # Bild hat nicht die angefragte Pixelgröße (realistischer Fehlermodus bei einem
                # externen WMS-Server) - dst.write() mit falscher Fenstergröße würde eine Exception
                # werfen und den GANZEN Mosaik-Lauf abbrechen. Stattdessen: wie ein Dekodier-
                # Fehlschlag behandeln, Kachel bleibt schwarz, weiter mit der nächsten Kachel.
                logger.warning(
                    f"  [x] Kachel ({tile.col_off},{tile.row_off}): Bildgröße {arr.shape[1]}x{arr.shape[0]} "
                    f"passt nicht zur angefragten Größe {tile.width}x{tile.height} - bleibt schwarz"
                )
                continue
            dst.write(np.moveaxis(arr, 2, 0), window=Window(tile.col_off, tile.row_off, tile.width, tile.height))
            any_success = True

    if any_success:
        os.replace(temp_path, dest_path)  # atomar: ein Absturz mittendrin hinterlässt nie ein Mosaik
        logger.info(f"  [OK] EOX-Mosaik geschrieben: {dest_path}")
        return True

    temp_path.unlink(missing_ok=True)
    logger.error("  [x] EOX-Mosaik: alle Kacheln fehlgeschlagen - kein Mosaik erzeugt")
    return False


def ensure_horizon_texture(area_utm: tuple, dest=None, size_px=None, resampling: str = "bilinear") -> Optional[Path]:
    """
    Öffentlicher Einstiegspunkt: stellt sicher, dass die Horizont-Textur (config.SENTINEL2_FILE)
    vorhanden ist - lädt bei Bedarf das EOX-Rohmosaik (gecacht, siehe config.EOX_MOSAIC_CACHE_DIR)
    und schneidet/reprojiziert es per horizon_image.build_horizon_image() zurecht.

    Args:
        area_utm: (x_min, x_max, y_min, y_max) in der aufgelösten Quell-CRS, siehe
            horizon_image.horizon_area()
        dest: Zieldatei; Default config.DOP300_DATA_DIR / config.SENTINEL2_FILE
        size_px: Kantenlänge der Zieltextur; Default config.HORIZON_IMAGE_SIZE_PX
        resampling: rasterio-Resampling-Name, siehe build_horizon_image()

    Returns:
        Pfad zur Horizont-Textur, oder None, wenn kein Download versucht wurde
        (config.EOX_AUTO_DOWNLOAD == False), der Download vollständig fehlgeschlagen ist, oder ein
        unerwarteter Fehler auftrat. Wirft NIE - analog zu dgm30_fetch.ensure_dgm30_coverage() muss
        dieser Einstiegspunkt bei jedem Fehler (Netzwerk, Dateisystem, CRS-Transform, ...) einfach
        None liefern, damit horizon_workflow.py ihn ohne eigene Fehlerbehandlung aufrufen kann.
    """
    global _attribution_logged

    dest = Path(dest) if dest is not None else config.DOP300_DATA_DIR / config.SENTINEL2_FILE
    if dest.exists():
        # Vorhandene, manuell abgelegte Dateien werden nie angetastet - egal welchen Wert
        # config.EOX_AUTO_DOWNLOAD hat.
        return dest

    if not config.EOX_AUTO_DOWNLOAD:
        return None

    try:
        size_px = size_px if size_px is not None else config.HORIZON_IMAGE_SIZE_PX

        # Cache-Schlüssel fürs Rohmosaik: unabhängig von size_px (das Mosaik ist unabhängig von
        # der Zielgröße), auf 1 m gerundete area_utm-Werte + Layer-Name (Jahr wechselt ggf.).
        sig = f"{round(area_utm[0])}_{round(area_utm[1])}_{round(area_utm[2])}_{round(area_utm[3])}_{config.EOX_WMS_LAYER}"
        cache_key = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
        mosaic_path = config.EOX_MOSAIC_CACHE_DIR / f"eox_mosaic_{cache_key}.tif"

        if not mosaic_path.exists():
            config.EOX_MOSAIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if not _attribution_logged:
                logger.info(config.EOX_ATTRIBUTION_NOTICE)
                _attribution_logged = True
            if not fetch_eox_mosaic(area_utm, mosaic_path):
                return None

        build_horizon_image(mosaic_path, dest, area=area_utm, size_px=size_px, resampling=resampling)

        if not config.EOX_KEEP_RAW_MOSAIC:
            mosaic_path.unlink(missing_ok=True)

        return dest if dest.exists() else None

    except Exception as e:
        logger.error(f"  [x] Horizont-Textur-Auto-Download fehlgeschlagen: {e}")
        return None
