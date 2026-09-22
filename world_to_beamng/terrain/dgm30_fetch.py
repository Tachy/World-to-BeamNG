"""
Automatischer Download fehlender Copernicus-DEM-GLO-30-Kacheln (30 m Höhendaten für den
Horizont-Hintergrund) von der öffentlichen, unauthentifizierten AWS-S3-Bucket
(`config.DGM30_S3_BUCKET`).

Analog zum bestehenden automatischen OSM-Overpass-Download in `world_to_beamng/osm/downloader.py`
(Retry mit exponentiellem Backoff, gestreamtes Fortschritts-Logging) - der Ansatz ist hier
bewusst lokal dupliziert statt in einen gemeinsamen Helfer ausgelagert (kleinerer, risikoärmerer
Schritt, siehe SDD-Plan).

Die Kacheln sind nach einem 1°x1°-Grad-Gitter benannt (Südwest-Ecke), z. B.
`Copernicus_DSM_COG_10_N47_00_E007_00_DEM`. `horizon_area_wgs84()` in horizon_image.py liefert
die BBox in WGS84, die dieses Modul zur Kachelauswahl braucht.
"""

import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .. import config

logger = logging.getLogger(__name__)

_NOT_FOUND_CACHE_FILENAME = ".not_found_cache.json"


def copernicus_tile_id(lat_deg: int, lon_deg: int) -> str:
    """
    Copernicus-DEM-GLO-30-Kachel-ID für die Kachel mit Südwest-Eckpunkt (lat_deg, lon_deg).

    Args:
        lat_deg: Ganzzahlige Breite der Südwest-Ecke (z. B. 47; negativ = südlich des Äquators)
        lon_deg: Ganzzahlige Länge der Südwest-Ecke (z. B. 7; negativ = westlich von Null)
    """
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"


def required_tile_ids(bbox_wgs84) -> list:
    """
    Alle 1°x1°-Kacheln, die die BBox schneiden. Reine Geometrie, kein I/O.

    Args:
        bbox_wgs84: (lon_min, lat_min, lon_max, lat_max), siehe horizon_image.horizon_area_wgs84()
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
    """True, wenn der Not-Found-Cache-Eintrag noch innerhalb der TTL liegt. Ein kaputter/
    unlesbarer Zeitstempel zählt NICHT als kürzlich bestätigt - die Kachel wird dann erneut
    versucht statt dauerhaft übersprungen zu werden."""
    try:
        timestamp = datetime.fromisoformat(timestamp_iso)
    except (TypeError, ValueError):
        return False
    return timestamp > cutoff


def missing_tile_ids(bbox_wgs84, dgm30_dir, not_found_cache: dict) -> list:
    """
    `required_tile_ids(bbox_wgs84)` abzüglich bereits vorhandener .tif-Dateien in `dgm30_dir` und
    Kacheln, die kürzlich (innerhalb config.DGM30_NOT_FOUND_CACHE_TTL_DAYS) als "nicht gefunden"
    bestätigt wurden. Reine Funktion, außer dem Directory-Listing von `dgm30_dir` kein I/O.

    Args:
        bbox_wgs84: (lon_min, lat_min, lon_max, lat_max)
        dgm30_dir: Verzeichnis mit bereits vorhandenen Kacheln (muss nicht existieren)
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
    """Lädt den Not-Found-Cache; eine fehlende oder kaputte Datei führt nie zum Absturz, nur zu
    einem leeren Cache (mit Log-Hinweis bei kaputter Datei)."""
    path = _cache_path(dgm30_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"  [!] Not-Found-Cache {path} ist beschädigt ({e}) - wird ignoriert")
        return {}


def _save_not_found_cache(dgm30_dir: Path, not_found_cache: dict) -> None:
    """Speichert den Not-Found-Cache atomar (Part-Datei + os.replace), damit ein Absturz beim
    nächsten Kachel-Download nicht bereits bestätigte 404s verliert."""
    path = _cache_path(dgm30_dir)
    part_path = path.with_name(path.name + ".part")
    part_path.write_text(json.dumps(not_found_cache, indent=2), encoding="utf-8")
    os.replace(part_path, path)


def _write_with_progress(response, part_path: Path, log_every_bytes: int = 2 * 1024 * 1024) -> None:
    """Schreibt eine gestreamte Response Chunk für Chunk in `part_path` und loggt den Fortschritt
    alle `log_every_bytes` (analog osm/downloader._download_with_progress(), hier einfacher weil
    direkt in eine Datei statt in den Speicher geschrieben wird)."""
    downloaded = 0
    next_log_at = log_every_bytes
    with open(part_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=256 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_log_at:
                logger.info(f"    ... {downloaded / (1024 * 1024):.1f} MB empfangen")
                next_log_at += log_every_bytes


def _download_one_tile(tile_id: str, url: str, dgm30_dir: Path) -> str:
    """Lädt eine einzelne Kachel mit Retry + exponentiellem Backoff (analog get_osm_data() in
    osm/downloader.py). Rückgabe: "downloaded", "not_found" oder "failed"."""
    part_path = dgm30_dir / f"{tile_id}.tif.part"
    final_path = dgm30_dir / f"{tile_id}.tif"

    for attempt in range(config.DGM30_FETCH_MAX_RETRIES):
        try:
            response = requests.get(url, stream=True, timeout=config.DGM30_FETCH_TIMEOUT_S)

            if response.status_code == 404:
                # Erwartetes Verhalten für Meereskacheln (keine Landfläche = keine DEM-Kachel) -
                # kein Retry, kein error-Log.
                logger.info(f"  [i] {tile_id}: keine Daten auf S3 (404, vermutlich Meereskachel)")
                return "not_found"

            response.raise_for_status()
            _write_with_progress(response, part_path)
            os.replace(part_path, final_path)  # atomar: ein Absturz mittendrin hinterlässt nie eine .tif
            logger.info(f"  [OK] {tile_id} heruntergeladen")
            return "downloaded"

        except requests.exceptions.Timeout:
            logger.info(f"  [x] {tile_id}: Timeout (Versuch {attempt + 1}/{config.DGM30_FETCH_MAX_RETRIES})")
        except requests.exceptions.HTTPError as e:
            logger.info(f"  [x] {tile_id}: HTTP-Fehler {e} (Versuch {attempt + 1}/{config.DGM30_FETCH_MAX_RETRIES})")
        except Exception as e:
            logger.info(f"  [x] {tile_id}: Fehler {e} (Versuch {attempt + 1}/{config.DGM30_FETCH_MAX_RETRIES})")

        if attempt < config.DGM30_FETCH_MAX_RETRIES - 1:
            wait_time = 2**attempt  # Exponentielles Backoff: 1s, 2s, 4s, ...
            logger.info(f"  Warte {wait_time}s vor erneutem Versuch...")
            time.sleep(wait_time)

    logger.error(f"  [x] {tile_id}: alle {config.DGM30_FETCH_MAX_RETRIES} Versuche fehlgeschlagen")
    # Ein abgebrochener Stream kann eine .tif.part hinterlassen haben - aufräumen (analog zum
    # totalen Fehlschlag in sentinel2_fetch.fetch_eox_mosaic()).
    part_path.unlink(missing_ok=True)
    return "failed"


def download_dgm30_tiles(bbox_wgs84, dgm30_dir) -> dict:
    """
    Lädt alle für `bbox_wgs84` fehlenden Copernicus-DEM-Kacheln von S3 herunter (siehe
    missing_tile_ids()).

    Args:
        bbox_wgs84: (lon_min, lat_min, lon_max, lat_max)
        dgm30_dir: Zielverzeichnis für die .tif-Kacheln (wird bei Bedarf angelegt)

    Returns:
        {"downloaded": [...], "not_found": [...], "failed": [...]} (Kachel-IDs je Kategorie)
    """
    dgm30_dir = Path(dgm30_dir)
    dgm30_dir.mkdir(parents=True, exist_ok=True)

    not_found_cache = _load_not_found_cache(dgm30_dir)
    missing = missing_tile_ids(bbox_wgs84, dgm30_dir, not_found_cache)
    if not missing:
        logger.debug("  DGM30: alle benötigten Kacheln bereits vorhanden oder kürzlich als fehlend bestätigt")
        return {"downloaded": [], "not_found": [], "failed": []}

    logger.info(f"  DGM30: {len(missing)} fehlende Kachel(n) werden von S3 geladen: {', '.join(missing)}")

    result = {"downloaded": [], "not_found": [], "failed": []}
    for tile_id in missing:
        url = f"https://{config.DGM30_S3_BUCKET}.s3.{config.DGM30_S3_REGION}.amazonaws.com/{tile_id}/{tile_id}.tif"
        outcome = _download_one_tile(tile_id, url, dgm30_dir)

        if outcome == "not_found":
            not_found_cache[tile_id] = datetime.now(timezone.utc).isoformat()
            try:
                _save_not_found_cache(dgm30_dir, not_found_cache)  # sofort speichern: übersteht einen Absturz im nächsten Tile
            except OSError as e:
                # Ein einzelner fehlgeschlagener Cache-Schreibversuch (voll, keine Rechte, ...) soll
                # nicht den Rest des Batches abbrechen - nur diese Kachel wird beim nächsten Lauf
                # erneut als 404 erkannt statt gecacht zu bleiben.
                logger.warning(f"  [!] Not-Found-Cache konnte nicht gespeichert werden ({e}) - {tile_id} bleibt ungecacht")

        result[outcome].append(tile_id)

    return result


def ensure_dgm30_coverage(bbox_wgs84, dgm30_dir=config.DGM30_DATA_DIR) -> dict:
    """
    Öffentlicher Einstiegspunkt für horizon_workflow.py: stellt sicher, dass alle für `bbox_wgs84`
    benötigten DGM30-Kacheln lokal vorhanden sind (lädt fehlende nach). Wirft NIE - Netzwerk-/DNS-
    Fehler o.ä. werden abgefangen und geloggt, damit die Pipeline immer weiterläuft (der
    bestehende Fallback bei weiterhin fehlenden Kacheln greift danach ohnehin).

    Args:
        bbox_wgs84: (lon_min, lat_min, lon_max, lat_max)
        dgm30_dir: Zielverzeichnis; Default config.DGM30_DATA_DIR

    Returns:
        {"downloaded": [...], "not_found": [...], "failed": [...]}
    """
    if not config.DGM30_AUTO_DOWNLOAD:
        return {"downloaded": [], "not_found": [], "failed": []}

    try:
        return download_dgm30_tiles(bbox_wgs84, Path(dgm30_dir))
    except Exception as e:
        logger.error(f"  [x] DGM30-Auto-Download fehlgeschlagen: {e}")
        return {"downloaded": [], "not_found": [], "failed": []}
