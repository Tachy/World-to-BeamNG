"""
OSM Daten Download via Overpass API.
"""

import json
import requests
import threading
import time

from .. import config
from ..io.cache import load_from_cache, save_to_cache
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def _log_waiting_heartbeat(stop_event, endpoint_label, interval=10):
    """
    Läuft in einem Hintergrund-Thread, während auf die Antwort-Header eines
    Overpass-Requests gewartet wird. Overpass liefert selbst keinen
    Fortschritt für die serverseitige Abfrageausführung - das hier ist nur
    ein "läuft noch"-Lebenszeichen, damit ein langsamer Server nicht wie ein
    Hänger aussieht.
    """
    waited = 0
    while not stop_event.wait(interval):
        waited += interval
        logger.info(f"    ... warte noch auf Antwort von {endpoint_label} ({waited}s)")


def _download_with_progress(response, log_every_bytes=2 * 1024 * 1024):
    """
    Liest eine gestreamte Response in Chunks und loggt den Fortschritt in MB
    (mit Prozentanzeige, falls der Server Content-Length sendet). Overpass
    nutzt teils Chunked-Transfer-Encoding ohne Content-Length - dann wird nur
    die kumulierte Menge geloggt.
    """
    total_bytes = response.headers.get("Content-Length")
    total_bytes = int(total_bytes) if total_bytes else None

    chunks = []
    downloaded = 0
    next_log_at = log_every_bytes

    for chunk in response.iter_content(chunk_size=256 * 1024):
        if not chunk:
            continue
        chunks.append(chunk)
        downloaded += len(chunk)
        if downloaded >= next_log_at:
            downloaded_mb = downloaded / (1024 * 1024)
            if total_bytes:
                pct = downloaded / total_bytes * 100
                total_mb = total_bytes / (1024 * 1024)
                logger.info(f"    ... {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({pct:.0f}%) empfangen")
            else:
                logger.info(f"    ... {downloaded_mb:.1f} MB empfangen")
            next_log_at += log_every_bytes

    return b"".join(chunks)


def get_osm_data(bbox, height_hash=None):
    """Holt ALLE OSM-Daten fuer eine BBox von der Overpass API oder aus dem Cache.

    Args:
        bbox: (lat_min, lon_min, lat_max, lon_max) Bounding Box
        height_hash: Optional - tile_hash für Cache-Konsistenz
    """
    # Pruefe Cache zuerst
    cached_data = load_from_cache(bbox, "osm_all", height_hash=height_hash)
    if cached_data is not None:
        return cached_data

    logger.info(f"Abfrage aller OSM-Daten fuer BBox {bbox}...")
    query = f"""
    [out:json][timeout:90];
    (
      node({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      way({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      relation({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out geom;
    """

    # Mehrere Overpass-Server lehnen Anfragen ohne aussagekräftigen User-Agent
    # ab (406 Not Acceptable) oder drosseln sie eher (429) - siehe config.OVERPASS_USER_AGENT.
    headers = {
        "User-Agent": config.OVERPASS_USER_AGENT,
        "Accept": "application/json",
    }

    # Versuche alle Endpoints mit Retry-Logik
    for endpoint_idx, overpass_url in enumerate(config.OVERPASS_ENDPOINTS):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                endpoint_label = f"Server {endpoint_idx + 1}/{len(config.OVERPASS_ENDPOINTS)}"
                logger.info(f"  Versuch {attempt + 1}/{max_retries} mit {endpoint_label}...")

                # Heartbeat, solange auf die Antwort-Header gewartet wird (die
                # eigentliche Overpass-Abfrageausführung liefert selbst keinen
                # Fortschritt - siehe _log_waiting_heartbeat()-Docstring).
                stop_heartbeat = threading.Event()
                heartbeat = threading.Thread(
                    target=_log_waiting_heartbeat, args=(stop_heartbeat, endpoint_label), daemon=True
                )
                heartbeat.start()
                try:
                    response = requests.get(
                        overpass_url, params={"data": query}, headers=headers, timeout=120, stream=True
                    )
                finally:
                    stop_heartbeat.set()

                response.raise_for_status()

                raw_body = _download_with_progress(response)
                elements = json.loads(raw_body).get("elements", [])
                logger.info(f"  [OK] Erfolgreich! {len(elements)} OSM-Elemente gefunden.")

                if not elements:
                    # Ein "erfolgreiches" 0-Elemente-Ergebnis ist für ein besiedeltes
                    # Gebiet höchst verdächtig (eher ein stilles Server-Problem als ein
                    # wirklich leeres Gebiet) - NICHT cachen, sonst bleibt der Fehler
                    # dauerhaft im Cache hängen und wird bei jedem Lauf wiederholt.
                    logger.warning(
                        f"  [!] Server lieferte 0 Elemente für BBox {bbox} - wird NICHT "
                        f"gecacht (vermutlich ein transientes Server-Problem statt eines "
                        f"wirklich leeren Gebiets)"
                    )
                    return elements

                # Im Cache speichern
                save_to_cache(bbox, "osm_all", elements, height_hash=height_hash)
                return elements

            except requests.exceptions.Timeout:
                logger.info(f"  [x] Timeout bei Server {endpoint_idx + 1}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponentielles Backoff: 1s, 2s, 4s
                    logger.info(f"  Warte {wait_time}s vor erneutem Versuch...")
                    time.sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                logger.error(f"  [x] HTTP-Fehler: {e}")
                break  # Bei HTTP-Fehler zum nächsten Server wechseln

            except Exception as e:
                logger.error(f"  [x] Fehler: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

    logger.error("Alle Versuche fehlgeschlagen.")
    return []
