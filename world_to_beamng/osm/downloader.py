"""
OSM Daten Download via Overpass API.
"""

import requests
import time

from .. import config
from ..io.cache import load_from_cache, save_to_cache


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

    print(f"Abfrage aller OSM-Daten fuer BBox {bbox}...")
    query = f"""
    [out:json][timeout:90];
    (
      node({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      way({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      relation({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out geom;
    """

    # Versuche alle Endpoints mit Retry-Logik
    for endpoint_idx, overpass_url in enumerate(config.OVERPASS_ENDPOINTS):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(
                    f"  Versuch {attempt + 1}/{max_retries} mit Server {endpoint_idx + 1}/{len(config.OVERPASS_ENDPOINTS)}..."
                )
                response = requests.get(
                    overpass_url, params={"data": query}, timeout=120
                )
                response.raise_for_status()
                elements = response.json().get("elements", [])
                print(f"  [OK] Erfolgreich! {len(elements)} OSM-Elemente gefunden.")

                # Im Cache speichern
                save_to_cache(bbox, "osm_all", elements, height_hash=height_hash)
                return elements

            except requests.exceptions.Timeout:
                print(f"  [x] Timeout bei Server {endpoint_idx + 1}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponentielles Backoff: 1s, 2s, 4s
                    print(f"  Warte {wait_time}s vor erneutem Versuch...")
                    time.sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                print(f"  [x] HTTP-Fehler: {e}")
                break  # Bei HTTP-Fehler zum nächsten Server wechseln

            except Exception as e:
                print(f"  [x] Fehler: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

    print("Alle Versuche fehlgeschlagen.")
    return []
