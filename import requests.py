import requests
import json
import numpy as np
from pyproj import Transformer

# --- KONFIGURATION ---
# Beispiel: Ein kleiner Ausschnitt (Stelvio Pass / Stilfser Joch)
# Format: [min_lat, min_lon, max_lat, max_lon]
BBOX = [46.5270, 10.4500, 46.5350, 10.4650]
ROAD_WIDTH = 7.0
LEVEL_NAME = "osm_generated_map"

# Transformer: GPS (WGS84) -> UTM Zone 32N (Metrisch für Mitteleuropa)
# Copilot-Tipp: Erweitere dies später um eine automatische UTM-Zonen-Erkennung
transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)


def get_osm_roads(bbox):
    """Extrahiert Straßendaten mit Geometrie aus der Overpass API."""
    print(f"Abfrage der OSM-Daten für BBox {bbox}...")
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    way["highway"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    out geom;
    """
    try:
        response = requests.get(overpass_url, params={"data": query}, timeout=30)
        response.raise_for_status()
        return response.json().get("elements", [])
    except Exception as e:
        print(f"Fehler beim OSM-Abruf: {e}")
        return []


def get_elevation_data(pts):
    """Holt Höhendaten via Open-Meteo API."""
    if not pts:
        return []
    lats = ",".join([str(p[0]) for p in pts])
    lons = ",".join([str(p[1]) for p in pts])
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    try:
        res = requests.get(url, timeout=20)
        return res.json().get("elevation", [0] * len(pts))
    except Exception as e:
        print(f"Fehler beim Höhen-Abruf: {e}")
        return [0] * len(pts)


def process_road_geometry(way_id, coords, elevations):
    """
    Berechnet die finale Geometrie.
    Hier wird die Straße 90 Grad zur Richtung geebnet.
    """
    nodes = []
    num_pts = len(coords)

    for i in range(num_pts):
        lat, lon = coords[i]
        z = elevations[i]

        # Umrechnung in UTM-Meter (X, Y)
        x, y = transformer.transform(lon, lat)

        # Richtungsvektor berechnen für die 90°-Ebnung
        # (Wichtig für spätere Mesh-Generierung oder Banking-Korrektur)
        if i < num_pts - 1:
            next_x, next_y = transformer.transform(coords[i + 1][1], coords[i + 1][0])
            direction = np.array([next_x - x, next_y - y])
        else:
            prev_x, prev_y = transformer.transform(coords[i - 1][1], coords[i - 1][0])
            direction = np.array([x - prev_x, y - prev_y])

        # Normale (90° zur Straße) berechnen
        if np.linalg.norm(direction) > 0:
            norm = np.array([-direction[1], direction[0]])
            norm = norm / np.linalg.norm(norm)
        else:
            norm = np.array([1, 0])

        # BeamNG MeshRoad Node: [X, Y, Z, Breite]
        # Das Z bleibt hier fix pro Querschnitt -> Straße ist 90° zur Richtung eben.
        nodes.append([x, y, z, ROAD_WIDTH])

    return {
        "name": f"road_{way_id}",
        "class": "MeshRoad",
        "persistentId": str(way_id),
        "nodes": nodes,
        "topMaterial": "asphalt_01",
        "bottomMaterial": "asphalt_01",
        "sideMaterial": "asphalt_01",
        "breakAngle": 3,
        "textureLength": 5,
    }


def main():
    # 1. Daten holen
    osm_elements = get_osm_roads(BBOX)
    if not osm_elements:
        print("Keine Daten gefunden.")
        return

    beamng_items = []

    # 2. Jedes Straßensegment einzeln verarbeiten
    for way in osm_elements:
        if "geometry" not in way:
            continue

        pts = [[p["lat"], p["lon"]] for p in way["geometry"]]
        elevations = get_elevation_data(pts)

        road_item = process_road_geometry(way["id"], pts, elevations)
        beamng_items.append(road_item)

    # 3. Speichern der main.items.json
    output_filename = "main.items.json"
    with open(output_filename, "w") as f:
        json.dump(beamng_items, f, indent=4)

    print(f"\nGenerator beendet. {len(beamng_items)} Straßen erstellt.")
    print(f"Datei gespeichert als: {output_filename}")


if __name__ == "__main__":
    main()
