"""
OSM data download via the Overpass API.
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
    Runs in a background thread while waiting for the response headers of an
    Overpass request. Overpass itself reports no progress for the
    server-side query execution - this is only a "still running" sign of life
    so that a slow server does not look like a hang.
    """
    waited = 0
    while not stop_event.wait(interval):
        waited += interval
        logger.info(f"    ... still waiting for a response from {endpoint_label} ({waited}s)")


def _download_with_progress(response, log_every_bytes=2 * 1024 * 1024):
    """
    Reads a streamed response in chunks and logs the progress in MB
    (with a percentage if the server sends Content-Length). Overpass
    sometimes uses chunked transfer encoding without Content-Length - in that case only
    the cumulative amount is logged.
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
                logger.info(f"    ... {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({pct:.0f}%) received")
            else:
                logger.info(f"    ... {downloaded_mb:.1f} MB received")
            next_log_at += log_every_bytes

    return b"".join(chunks)


def get_osm_data(bbox, height_hash=None):
    """Fetches ALL OSM data for a BBox from the Overpass API or from the cache.

    Args:
        bbox: (lat_min, lon_min, lat_max, lon_max) bounding box
        height_hash: Optional - tile_hash for cache consistency
    """
    # Check the cache first
    cached_data = load_from_cache(bbox, "osm_all", height_hash=height_hash)
    if cached_data is not None:
        return cached_data

    logger.info(f"Querying all OSM data for bbox {bbox}...")
    query = f"""
    [out:json][timeout:90];
    (
      node({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      way({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      relation({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out geom;
    """

    # Several Overpass servers reject requests without a meaningful User-Agent
    # (406 Not Acceptable) or are more likely to throttle them (429) - see config.OVERPASS_USER_AGENT.
    headers = {
        "User-Agent": config.OVERPASS_USER_AGENT,
        "Accept": "application/json",
    }

    # Try all endpoints with retry logic
    for endpoint_idx, overpass_url in enumerate(config.OVERPASS_ENDPOINTS):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                endpoint_label = f"Server {endpoint_idx + 1}/{len(config.OVERPASS_ENDPOINTS)}"
                logger.info(f"  Attempt {attempt + 1}/{max_retries} with {endpoint_label}...")

                # Heartbeat while waiting for the response headers (the
                # actual Overpass query execution reports no progress
                # itself - see the _log_waiting_heartbeat() docstring).
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
                logger.info(f"  [OK] Success! {len(elements)} OSM elements found.")

                if not elements:
                    # A "successful" 0-element result is highly suspicious for a populated
                    # area (more likely a silent server problem than a truly empty
                    # area) - do NOT cache it, otherwise the error stays in the cache
                    # permanently and is repeated on every run.
                    logger.warning(
                        f"  [!] Server returned 0 elements for bbox {bbox} - NOT "
                        f"cached (probably a transient server problem rather than "
                        f"a truly empty area)"
                    )
                    return elements

                # Save to the cache
                save_to_cache(bbox, "osm_all", elements, height_hash=height_hash)
                return elements

            except requests.exceptions.Timeout:
                logger.info(f"  [x] Timeout at server {endpoint_idx + 1}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"  Waiting {wait_time}s before retrying...")
                    time.sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                logger.error(f"  [x] HTTP error: {e}")
                break  # On an HTTP error, switch to the next server

            except Exception as e:
                logger.error(f"  [x] Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

    logger.error("All attempts failed.")
    return []
