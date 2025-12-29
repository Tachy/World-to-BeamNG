"""
WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator

Benötigte Pakete:
  pip install requests numpy scipy pyproj pyvista shapely rtree

Alle Abhängigkeiten sind ERFORDERLICH - kein Fallback!
"""

import requests
import json
import numpy as np
import time
import os
import hashlib
import glob
import gc
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.spatial import cKDTree
from pyproj import Transformer
import pyvista as pv
from shapely.geometry import Polygon, Point, LineString
from shapely.prepared import prep

try:
    from rtree import index
except ImportError:
    index = None

# --- KONFIGURATION ---
ROAD_WIDTH = 7.0
SLOPE_ANGLE = 45.0  # Neigungswinkel der Böschung in Grad (45° = 1:1 Steigung)
GRID_SPACING = 1.0  # Abstand zwischen Grid-Punkten in Metern (1.0 = hohe Auflösung, 10.0 = niedrige Auflösung)
LEVEL_NAME = "osm_generated_map"
CACHE_DIR = "cache"  # Verzeichnis für Cache-Dateien
HEIGHT_DATA_DIR = "height-data"  # Verzeichnis mit Höhendaten

# BBOX wird automatisch aus Höhendaten ermittelt
BBOX = None

# Globaler Offset für lokale Koordinaten (wird bei erstem Punkt gesetzt)
LOCAL_OFFSET = None

# Grid Bounds in UTM (wird in create_terrain_grid gesetzt)
GRID_BOUNDS_UTM = None

# Transformer: GPS (WGS84) <-> UTM Zone 32N (Metrisch für Mitteleuropa)
transformer_to_utm = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
transformer_to_wgs84 = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)

# Alternative Overpass API Endpoints (Fallback-Server)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def get_height_data_hash():
    """Erstellt einen Hash basierend auf den Dateien im height-data Ordner."""
    xyz_files = sorted(glob.glob(os.path.join(HEIGHT_DATA_DIR, "*.xyz")))
    zip_files = sorted(glob.glob(os.path.join(HEIGHT_DATA_DIR, "*.zip")))
    all_files = xyz_files + zip_files

    if not all_files:
        return None

    # Hash basierend auf Dateinamen und Änderungszeitpunkten
    hash_input = ""
    for file in all_files:
        mtime = os.path.getmtime(file)
        hash_input += f"{os.path.basename(file)}_{mtime}_"

    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def load_height_data():
    """Lädt alle Höhendaten aus .xyz oder .zip Dateien (mit Caching)."""
    print("\nLade Höhendaten...")

    # Prüfe ob gecachte Rohdaten existieren
    height_hash = get_height_data_hash()
    cache_file = None

    if height_hash:
        cache_file = os.path.join(CACHE_DIR, f"height_raw_{height_hash}.npz")

        if os.path.exists(cache_file):
            print(f"  ✓ Cache gefunden: {os.path.basename(cache_file)}")
            data = np.load(cache_file)
            points = data["points"]
            elevations = data["elevations"]
            print(f"  ✓ {len(elevations)} Höhenpunkte aus Cache geladen")
            return points, elevations
        else:
            print(f"  Cache nicht gefunden, lade aus Dateien...")

    # Lade aus Dateien
    xyz_files = glob.glob(os.path.join(HEIGHT_DATA_DIR, "*.xyz"))
    zip_files = glob.glob(os.path.join(HEIGHT_DATA_DIR, "*.zip"))

    if not xyz_files and not zip_files:
        raise FileNotFoundError(
            f"Keine .xyz oder .zip Dateien in {HEIGHT_DATA_DIR} gefunden!"
        )

    print(f"  Lese {len(xyz_files)} XYZ + {len(zip_files)} ZIP Dateien...")

    all_points = []
    all_elevations = []

    # Lade .xyz Dateien
    for file in xyz_files:
        print(f"    • {os.path.basename(file)}...")
        data = np.loadtxt(file)
        all_points.append(data[:, :2])
        all_elevations.append(data[:, 2])

    # Lade .zip Dateien
    import zipfile
    import io

    for zip_file in zip_files:
        print(f"    • {os.path.basename(zip_file)}...")
        with zipfile.ZipFile(zip_file, "r") as z:
            for name in z.namelist():
                if name.endswith(".xyz"):
                    print(f"      └─ {name}")
                    with z.open(name) as f:
                        data = np.loadtxt(io.TextIOWrapper(f, encoding="utf-8"))
                        all_points.append(data[:, :2])
                        all_elevations.append(data[:, 2])

    # Kombiniere alle Kacheln
    points = np.vstack(all_points)
    elevations = np.hstack(all_elevations)

    print(f"  ✓ {len(elevations)} Höhenpunkte geladen")

    # Cache die Rohdaten (immer wenn wir frisch geladen haben)
    if height_hash:
        cache_file_path = os.path.join(CACHE_DIR, f"height_raw_{height_hash}.npz")
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_file_path, points=points, elevations=elevations)
        print(f"  ✓ Cache erstellt: {os.path.basename(cache_file_path)}")

    return points, elevations


def calculate_bbox_from_height_data(points):
    """Berechnet die BBOX (WGS84) aus UTM-Höhendaten."""
    # Finde Min/Max in UTM
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)

    # Konvertiere zu WGS84
    min_lon, min_lat = transformer_to_wgs84.transform(min_x, min_y)
    max_lon, max_lat = transformer_to_wgs84.transform(max_x, max_y)

    bbox = [min_lat, min_lon, max_lat, max_lon]
    print(f"  BBOX ermittelt: {bbox}")

    return bbox


def get_bbox_hash(bbox):
    """Erstellt einen eindeutigen Hash für eine BBox zur Cache-Identifikation."""
    bbox_str = f"{bbox[0]:.6f}_{bbox[1]:.6f}_{bbox[2]:.6f}_{bbox[3]:.6f}"
    return hashlib.md5(bbox_str.encode()).hexdigest()[:12]


def get_cache_path(bbox, data_type):
    """Gibt den Pfad zur Cache-Datei zurück."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    bbox_hash = get_bbox_hash(bbox)
    return os.path.join(CACHE_DIR, f"{data_type}_{bbox_hash}.json")


def load_from_cache(bbox, data_type):
    """Lädt Daten aus dem Cache, falls vorhanden."""
    cache_path = get_cache_path(bbox, data_type)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  ✓ {data_type.upper()}-Daten aus Cache geladen ({cache_path})")
                return data
        except Exception as e:
            print(f"  ⚠ Fehler beim Laden des Caches: {e}")
    return None


def save_to_cache(bbox, data_type, data):
    """Speichert Daten im Cache."""
    cache_path = get_cache_path(bbox, data_type)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ {data_type.upper()}-Daten im Cache gespeichert ({cache_path})")
    except Exception as e:
        print(f"  ⚠ Fehler beim Speichern des Caches: {e}")


def get_osm_data(bbox):
    """Holt ALLE OSM-Daten für eine BBox von der Overpass API oder aus dem Cache."""
    # Prüfe Cache zuerst
    cached_data = load_from_cache(bbox, "osm_all")
    if cached_data is not None:
        return cached_data

    print(f"Abfrage aller OSM-Daten für BBox {bbox}...")
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
    for endpoint_idx, overpass_url in enumerate(OVERPASS_ENDPOINTS):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(
                    f"  Versuch {attempt + 1}/{max_retries} mit Server {endpoint_idx + 1}/{len(OVERPASS_ENDPOINTS)}..."
                )
                response = requests.get(
                    overpass_url, params={"data": query}, timeout=120
                )
                response.raise_for_status()
                elements = response.json().get("elements", [])
                print(f"  ✓ Erfolgreich! {len(elements)} OSM-Elemente gefunden.")

                # Im Cache speichern
                save_to_cache(bbox, "osm_all", elements)
                return elements

            except requests.exceptions.Timeout:
                print(f"  ✗ Timeout bei Server {endpoint_idx + 1}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponentielles Backoff: 1s, 2s, 4s
                    print(f"  Warte {wait_time}s vor erneutem Versuch...")
                    time.sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                print(f"  ✗ HTTP-Fehler: {e}")
                break  # Bei HTTP-Fehler zum nächsten Server wechseln

            except Exception as e:
                print(f"  ✗ Fehler: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

    print("Alle Versuche fehlgeschlagen.")
    return []


def extract_roads_from_osm(osm_elements):
    """Extrahiert nur Straßen-Ways aus allen OSM-Daten."""
    roads = [
        element
        for element in osm_elements
        if element.get("type") == "way"
        and "tags" in element
        and "highway" in element["tags"]
    ]
    print(
        f"  → {len(roads)} Straßensegmente aus {len(osm_elements)} OSM-Elementen extrahiert"
    )
    return roads


def get_elevation_data(pts, bbox=None):
    """VERALTET - wird nicht mehr verwendet. Höhendaten kommen aus lokalen Dateien."""
    raise NotImplementedError(
        "Diese Funktion wird nicht mehr verwendet. Nutze get_elevations_for_points()."
    )


def get_elevation_cache(bbox):
    """Lädt den Elevation-Cache für eine BBox (Koordinate -> Höhe)."""
    cache_path = get_cache_path(bbox, "elevations")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
                print(f"  ✓ Elevation-Cache geladen: {len(cache)} Koordinaten")
                return cache
        except:
            pass
    return {}


def save_elevation_cache(bbox, cache):
    """Speichert den Elevation-Cache."""
    cache_path = get_cache_path(bbox, "elevations")
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"  ✓ Elevation-Cache gespeichert: {len(cache)} Koordinaten")
    except Exception as e:
        print(f"  ⚠ Fehler beim Speichern des Elevation-Cache: {e}")


def get_elevations_for_points(pts, bbox, height_points, height_elevations):
    """Holt Höhendaten für Koordinaten - aus Cache oder durch Interpolation aus lokalen Daten."""
    # Lade bestehenden Cache
    elevation_cache = get_elevation_cache(bbox)

    # Finde fehlende Koordinaten
    missing_pts = []
    missing_indices = []

    for idx, pt in enumerate(pts):
        # Erstelle eindeutigen Key für Koordinate (gerundet auf 6 Dezimalstellen)
        coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
        if coord_key not in elevation_cache:
            missing_pts.append(pt)
            missing_indices.append(idx)

    # Berechne fehlende Höhen durch Interpolation
    if missing_pts:
        print(f"  Interpoliere {len(missing_pts)} Höhenwerte...")

        # Konvertiere WGS84 zu UTM
        missing_pts_utm = []
        for pt in missing_pts:
            x, y = transformer_to_utm.transform(pt[1], pt[0])  # lon, lat -> x, y
            missing_pts_utm.append([x, y])

        missing_pts_utm = np.array(missing_pts_utm)

        # Interpoliere Höhen (nearest neighbor für schnellere Berechnung)
        new_elevations = griddata(
            height_points, height_elevations, missing_pts_utm, method="nearest"
        )

        # Füge zum Cache hinzu
        for pt, elev in zip(missing_pts, new_elevations):
            coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
            elevation_cache[coord_key] = float(elev)

        # Speichere aktualisierten Cache
        save_elevation_cache(bbox, elevation_cache)

    # Erstelle Elevation-Array für alle Punkte
    elevations = []
    for pt in pts:
        coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
        elevations.append(elevation_cache.get(coord_key, 0))

    return elevations


def apply_local_offset(x, y, z):
    """Konvertiert zu lokalen Koordinaten relativ zum ersten Punkt (Array-fähig)."""
    global LOCAL_OFFSET
    if LOCAL_OFFSET is None:
        # Setze Offset vom ersten Wert
        if isinstance(x, np.ndarray):
            LOCAL_OFFSET = (x[0], y[0], z[0])
        else:
            LOCAL_OFFSET = (x, y, z)
    return (x - LOCAL_OFFSET[0], y - LOCAL_OFFSET[1], z - LOCAL_OFFSET[2])


def create_terrain_grid(height_points, height_elevations, grid_spacing=10.0):
    """Erstellt ein reguläres Grid aus den Höhendaten (OPTIMIERT mit Caching)."""
    print(f"\nErstelle Terrain-Grid (Abstand: {grid_spacing}m)...")

    # Finde Bounds in UTM (IMMER berechnen, auch für Cache-Fall!)
    global GRID_BOUNDS_UTM
    min_x, max_x = height_points[:, 0].min(), height_points[:, 0].max()
    min_y, max_y = height_points[:, 1].min(), height_points[:, 1].max()
    GRID_BOUNDS_UTM = (min_x, min_y, max_x, max_y)

    # Prüfe ob gecachtes Grid existiert
    height_hash = get_height_data_hash()
    if height_hash:
        cache_file = os.path.join(
            CACHE_DIR, f"grid_{height_hash}_spacing{grid_spacing:.1f}m.npz"
        )

        if os.path.exists(cache_file):
            print(f"  ✓ Grid-Cache gefunden: {os.path.basename(cache_file)}")
            data = np.load(cache_file)
            grid_points = data["grid_points"]
            grid_elevations = data["grid_elevations"]
            nx = int(data["nx"])
            ny = int(data["ny"])
            print(
                f"  ✓ Grid aus Cache geladen: {nx} x {ny} = {len(grid_points)} Vertices"
            )
            return grid_points, grid_elevations, nx, ny

    # Erstelle Grid-Punkte
    x_coords = np.arange(min_x, max_x, grid_spacing)
    y_coords = np.arange(min_y, max_y, grid_spacing)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Interpoliere Höhen für Grid-Punkte (CHUNKED für bessere Performance)
    print(f"  Erstelle Interpolator...")
    interpolator = NearestNDInterpolator(height_points, height_elevations)

    print(f"  Interpoliere {len(grid_points)} Grid-Punkte (in Chunks)...")
    chunk_size = 500000  # 500k Punkte pro Chunk
    grid_elevations = np.empty(len(grid_points), dtype=np.float64)

    num_chunks = (len(grid_points) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(grid_points))
        grid_elevations[start_idx:end_idx] = interpolator(
            grid_points[start_idx:end_idx]
        )

        if (i + 1) % 5 == 0 or i == num_chunks - 1:
            progress = ((i + 1) / num_chunks) * 100
            print(f"    {progress:.0f}% ({i + 1}/{num_chunks} Chunks)")

    nx = len(x_coords)
    ny = len(y_coords)
    print(f"  Grid: {nx} x {ny} = {len(grid_points)} Vertices")

    # Cache das Grid für zukünftige Verwendung
    if height_hash:
        cache_file = os.path.join(
            CACHE_DIR, f"grid_{height_hash}_spacing{grid_spacing:.1f}m.npz"
        )
        print(f"  Speichere Grid-Cache: {os.path.basename(cache_file)}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(
            cache_file,
            grid_points=grid_points,
            grid_elevations=grid_elevations,
            nx=nx,
            ny=ny,
        )
        print(f"  ✓ Grid-Cache erstellt")

    return grid_points, grid_elevations, nx, ny


def get_road_polygons(roads, bbox, height_points, height_elevations):
    """Extrahiert Straßen-Polygone mit ihren Koordinaten und Höhen (OPTIMIERT)."""
    road_polygons = []

    # Sammle alle Koordinaten für Batch-Verarbeitung
    all_coords = []
    road_indices = []

    for way in roads:
        if "geometry" not in way:
            continue

        pts = [[p["lat"], p["lon"]] for p in way["geometry"]]
        if len(pts) < 2:
            continue

        road_indices.append((len(all_coords), len(all_coords) + len(pts), way))
        all_coords.extend(pts)

    if not all_coords:
        return road_polygons

    # Batch-Elevation-Lookup
    print(f"  Lade Elevations für {len(all_coords)} Straßen-Punkte...")
    all_elevations = get_elevations_for_points(
        all_coords, bbox, height_points, height_elevations
    )

    # Batch-UTM-Transformation (vektorisiert)
    lats = np.array([c[0] for c in all_coords])
    lons = np.array([c[1] for c in all_coords])
    xs, ys = transformer_to_utm.transform(lons, lats)

    # Erstelle Straßen-Polygone
    for start_idx, end_idx, way in road_indices:
        utm_coords = [
            (xs[i], ys[i], all_elevations[i]) for i in range(start_idx, end_idx)
        ]

        road_polygons.append(
            {
                "id": way["id"],
                "coords": utm_coords,
                "name": way.get("tags", {}).get("name", f"road_{way['id']}"),
            }
        )

    return road_polygons


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """VERALTET - wird nicht mehr verwendet. Benutze stattdessen 2D Point-in-Polygon Tests."""
    pass


def classify_grid_vertices(grid_points, grid_elevations, road_slope_polygons_2d):
    """
    Markiert Grid-Vertices unter Straßen/Böschungen als 'ausgeschnitten'.
    SUPER-OPTIMIERT: Nutzt LineString.buffer() um nur Punkte in der Nähe zu testen!

    Taktik:
    1. Für jeden Road: Extrahiere die Centerline (Koordinaten)
    2. Erstelle einen Puffer um die Centerline (nur links/rechts der Straße!)
    3. Teste nur Punkte in diesem Puffer gegen die eigentlichen Road/Slope Polygone

    Das ist 100-1000x schneller als BBox-Filterung!

    Args:
        grid_points: (N, 2) Array mit X-Y Koordinaten
        grid_elevations: (N,) Array mit Z-Koordinaten
        road_slope_polygons_2d: Liste von Dicts mit 'road_polygon' und 'slope_polygon'

    Returns:
        vertex_types: 0 = Terrain (behalten), 1 = Böschung (ausschneiden), 2 = Straße (ausschneiden)
        modified_heights: Unverändert (nur für Kompatibilität)
    """
    import time as time_module

    print(
        "\nMarkiere Straßen-/Böschungsbereiche im Grid (LineString-Buffer Optimierung)..."
    )

    vertex_types = np.zeros(len(grid_points), dtype=int)
    modified_heights = grid_elevations.copy()

    # Extrahiere nur X-Y Koordinaten
    grid_points_2d = grid_points[:, :2]

    # Konvertiere zu Shapely-Polygone
    print("  Konvertiere Polygone zu Shapely...")
    road_data = []

    for poly_data in road_slope_polygons_2d:
        road_poly_xy = poly_data["road_polygon"]
        slope_poly_xy = poly_data["slope_polygon"]

        if len(slope_poly_xy) >= 3 and len(road_poly_xy) >= 3:
            try:
                road_poly = Polygon(road_poly_xy)
                slope_poly = Polygon(slope_poly_xy)

                # MEGA-OPTIMIERUNG: Erstelle einen Centerline-Buffer!
                # Extrahiere die Mittellinie der Straße (erste Hälfte der Punkte)
                num_road_points = len(road_poly_xy)
                num_slope_points = len(slope_poly_xy)

                # Die slope_poly ist größer (Straße + Böschung)
                # Nutze einfach die Bounds davon für den Buffer
                centerline_coords = slope_poly_xy[
                    ::2
                ]  # Jeden 2. Punkt (Centerline-Approximation)
                if len(centerline_coords) < 2:
                    centerline_coords = slope_poly_xy

                # Erstelle LineString und gepufferte Zone
                centerline = LineString(centerline_coords)
                # Buffer-Radius: Straße (3.5m halbe Breite) + Böschung (2m max) = 5.5m
                # Aufrunden auf 6m zur Sicherheit
                buffer_zone = centerline.buffer(
                    6.0
                )  # 6 Meter links/rechts der Centerline

                road_data.append(
                    {
                        "road_geom": road_poly,
                        "slope_geom": slope_poly,
                        "buffer_zone": buffer_zone,  # Die schmale Zone um die Straße!
                        "buffer_bounds": buffer_zone.bounds,
                    }
                )
            except Exception:
                continue

    if not road_data:
        print("  ⚠ Keine gültigen Polygone gefunden!")
        return vertex_types, modified_heights

    process_start = time_module.time()

    # Import matplotlib.path für schnelle batch-Tests
    from matplotlib.path import Path

    # MEGA-OPTIMIERUNG: Baue KDTree über alle Grid-Punkte (EINMAL, dann reuse!)
    print(f"  Baue KDTree für {len(grid_points_2d)} Grid-Punkte...")
    kdtree = cKDTree(grid_points_2d)

    print(f"  Teste {len(road_data)} Roads gegen Grid-Punkte...")
    print(f"  (Nutzt LineString-Buffer + KDTree für extreme Optimierung)")

    # SUPER-SCHNELL: Nutze KDTree Radius-Abfrage für JEDEN Punkt der Centerline!
    # Das ist viel schneller als eine große BBox-Abfrage!
    for road_num, road_info in enumerate(road_data):
        # Extrahiere Centerline aus den Polygonpunkten
        road_poly_xy = road_info["road_geom"].exterior.coords[:]
        if len(road_poly_xy) < 2:
            continue

        # Centerline ist die Mittellinie: jeden 2. Punkt oder durchschnitt von links/rechts
        num_points = len(road_poly_xy) // 2
        centerline_points = np.array(
            road_poly_xy[:num_points]
        )  # Erste Hälfte = eine Seite

        if len(centerline_points) < 2:
            centerline_points = np.array(road_poly_xy)

        # Sammle alle Punkte um ALLE Centerline-Punkte im Radius 6m
        buffer_indices_set = set()

        # Für jeden Punkt auf der Centerline: finde Grid-Punkte im Radius 6m
        for centerline_pt in centerline_points:
            # KDTree query_ball_point: SEHR schnell für Radius-Abfragen!
            nearby = kdtree.query_ball_point(centerline_pt, r=6.0)
            buffer_indices_set.update(nearby)

        buffer_indices = list(buffer_indices_set)

        # Teste alle gefilterten Punkte auf einmal mit matplotlib.path (VEKTORISIERT!)
        if len(buffer_indices) > 0:
            # Test 1: Straßen-Bereich
            road_coords = np.array(road_info["road_geom"].exterior.coords)
            road_path = Path(road_coords)
            candidate_points = grid_points_2d[buffer_indices]
            inside_road = road_path.contains_points(candidate_points)

            # Test 2: Böschungs-Bereich
            slope_coords = np.array(road_info["slope_geom"].exterior.coords)
            slope_path = Path(slope_coords)
            inside_slope = slope_path.contains_points(candidate_points)

            # Markiere Punkte: Straße hat Priorität
            for inside_idx in range(len(buffer_indices)):
                pt_idx = buffer_indices[inside_idx]
                if inside_road[inside_idx] and vertex_types[pt_idx] == 0:
                    vertex_types[pt_idx] = 2  # Straße
                elif inside_slope[inside_idx] and vertex_types[pt_idx] == 0:
                    vertex_types[pt_idx] = 1  # Böschung

        if (road_num + 1) % 100 == 0:
            elapsed = time_module.time() - process_start
            rate = (road_num + 1) / elapsed if elapsed > 0 else 0
            eta = (len(road_data) - road_num - 1) / rate if rate > 0 else 0
            print(
                f"  {road_num + 1}/{len(road_data)} Roads ({rate:.0f}/s, ETA: {eta:.0f}s)"
            )

    elapsed_total = time_module.time() - process_start
    marked_count = np.count_nonzero(vertex_types)
    print(
        f"  ✓ Klassifizierung abgeschlossen ({elapsed_total:.1f}s, {marked_count} Punkte markiert)"
    )

    return vertex_types, modified_heights


def clip_road_to_bounds(coords, bounds_utm):
    """
    Clippt eine Straße an den Grid-Bounds (UTM-Koordinaten).

    Entfernt Segmente, die komplett außerhalb liegen.
    Behält Segmente, die Grid berühren oder durchqueren.

    Args:
        coords: Liste von (x, y, z) UTM-Koordinaten
        bounds_utm: (min_x, min_y, max_x, max_y)

    Returns:
        Liste von geclippten Koordinaten (kann leer sein)
    """
    if not coords or bounds_utm is None:
        return coords

    min_x, min_y, max_x, max_y = bounds_utm

    # Puffer: Straßenbreite + Böschung (ca. 40m)
    buffer = 40.0
    min_x -= buffer
    min_y -= buffer
    max_x += buffer
    max_y += buffer

    clipped = []

    for x, y, z in coords:
        # Prüfe ob Punkt im erweiterten Bereich liegt
        if min_x <= x <= max_x and min_y <= y <= max_y:
            clipped.append((x, y, z))
        elif clipped:
            # Punkt ist außerhalb, aber wir hatten schon Punkte innerhalb
            # → Straße verlässt Bereich, beende hier
            break

    return clipped


def generate_road_mesh_strips(road_polygons, height_points, height_elevations):
    """
    Generiert Straßen als separate Mesh-Streifen mit perfekt parallelen Kanten.

    Für jede Straße:
    - Erstelle durchgehende Vertex-Streifen (links/rechts) entlang der Mittellinie
    - Böschungen mit variabler Breite bis zum Terrain (konstante Neigung)
    - Vertices werden zwischen Segmenten GETEILT (keine Lücken in Kurven!)
    - Quads verbinden aufeinanderfolgende Punkte

    Returns:
        road_vertices: Liste von (x, y, z) Koordinaten
        road_faces: Liste von Dreiecken (indices 1-basiert)
        slope_vertices: Liste von Böschungs-Vertices
        slope_faces: Liste von Böschungs-Dreiecken
        road_slope_polygons_2d: Liste von Dicts mit 'road_polygon' und 'slope_polygon' (nur X-Y für 2D-Projektion)
    """
    half_width = ROAD_WIDTH / 2.0
    slope_gradient = np.tan(np.radians(SLOPE_ANGLE))  # tan(45°) = 1.0

    # Erstelle Interpolator für Terrain-Höhen
    from scipy.interpolate import NearestNDInterpolator

    terrain_interpolator = NearestNDInterpolator(height_points, height_elevations)

    all_road_vertices = []
    all_road_faces = []
    all_slope_vertices = []
    all_slope_faces = []
    road_slope_polygons_2d = []  # Für 2D-Ausschneiden im Grid

    # Keine current_vertex_index mehr - Faces werden 0-basiert erstellt!

    total_roads = len(road_polygons)
    processed = 0
    clipped_roads = 0

    for road in road_polygons:
        coords = road["coords"]

        if len(coords) < 2:
            continue

        # Clippe Straße an Grid-Bounds (entfernt Teile außerhalb)
        coords = clip_road_to_bounds(coords, GRID_BOUNDS_UTM)

        if len(coords) < 2:
            clipped_roads += 1
            continue

        # Erstelle Vertex-Streifen entlang der Straße (SHARED vertices!)
        road_left_vertices = []
        road_right_vertices = []
        road_left_abs = []  # Absolute Koordinaten (für Böschungs-Vertices)
        road_right_abs = []  # Absolute Koordinaten (für Böschungs-Vertices)
        slope_left_outer_vertices = []
        slope_right_outer_vertices = []

        # VEKTORISIERTE Direction-Berechnung (statt Loop)
        coords_array = np.array(coords)
        num_points = len(coords_array)

        # Berechne Differenzvektoren zwischen Punkten
        diffs = np.diff(coords_array[:, :2], axis=0)  # Shape: (N-1, 2)
        lengths = np.linalg.norm(diffs, axis=1)

        # Normiere Richtungsvektoren
        directions = diffs / lengths[:, np.newaxis]

        # Glätte Kurven: durchschnittliche Richtung vor und nach jedem Punkt
        avg_dirs = np.zeros((num_points, 2))
        avg_dirs[0] = directions[0]  # Erster Punkt
        avg_dirs[1:-1] = (directions[:-1] + directions[1:]) / 2  # Mittlere Punkte
        avg_dirs[-1] = directions[-1]  # Letzter Punkt

        # Normiere noch mal
        avg_lengths = np.linalg.norm(avg_dirs, axis=1)
        avg_dirs = avg_dirs / avg_lengths[:, np.newaxis]

        for i, (x, y, z) in enumerate(coords):
            # Nutze vorberechnete Direction
            dir_x = avg_dirs[i, 0]
            dir_y = avg_dirs[i, 1]

            # Perpendicular-Vektor (90° gedreht)
            perp_x = -dir_y
            perp_y = dir_x

            # Straßenkanten-Vertices (links und rechts)
            road_left_x = x + perp_x * half_width
            road_left_y = y + perp_y * half_width
            road_right_x = x - perp_x * half_width
            road_right_y = y - perp_y * half_width

            # Speichere absolute Koordinaten für Böschungen
            road_left_abs.append((road_left_x, road_left_y, z))
            road_right_abs.append((road_right_x, road_right_y, z))

            # Transformiere Straßen-Vertices zu lokalen Koordinaten
            p_left_local = apply_local_offset(road_left_x, road_left_y, z)
            p_right_local = apply_local_offset(road_right_x, road_right_y, z)

            road_left_vertices.append(p_left_local)
            road_right_vertices.append(p_right_local)

            # BÖSCHUNGS-BERECHNUNG:
            # Interpoliere Terrain-Höhe an Straßenkanten
            terrain_left_height = terrain_interpolator([[road_left_x, road_left_y]])[0]
            terrain_right_height = terrain_interpolator([[road_right_x, road_right_y]])[
                0
            ]

            # Höhendifferenz zwischen Straße und Terrain
            height_diff_left = terrain_left_height - z
            height_diff_right = terrain_right_height - z

            # Böschungsbreite = |height_diff| / tan(SLOPE_ANGLE)
            slope_width_left = abs(height_diff_left) / slope_gradient
            slope_width_right = abs(height_diff_right) / slope_gradient

            # Begrenze Böschungsbreite auf Maximum (verhindert extreme Werte)
            MAX_SLOPE_WIDTH = 30.0  # Meter
            slope_width_left = min(slope_width_left, MAX_SLOPE_WIDTH)
            slope_width_right = min(slope_width_right, MAX_SLOPE_WIDTH)

            # Wenn Höhendifferenz sehr klein, erstelle keine Böschung
            if abs(height_diff_left) < 0.1:
                slope_width_left = 0.0
            if abs(height_diff_right) < 0.1:
                slope_width_right = 0.0

            # Richtung der Böschung:
            # - Wenn Terrain HÖHER als Straße (height_diff > 0): Böschung geht nach AUSSEN (Damm/Aufschüttung)
            # - Wenn Terrain NIEDRIGER als Straße (height_diff < 0): Böschung geht nach AUSSEN (Einschnitt)
            # In beiden Fällen: nach außen! (Die Höhe am äußeren Rand ist immer terrain_height)

            # Wenn Höhendifferenz zu klein, setze slope_outer = road_edge (keine Böschung)
            if slope_width_left < 0.1:
                slope_left_outer_x = road_left_x
                slope_left_outer_y = road_left_y
                slope_left_outer_height = z  # Straßenhöhe, nicht Terrain!
            else:
                # Äußerer Rand der Böschung (wo sie Terrain trifft)
                slope_left_outer_x = road_left_x + perp_x * slope_width_left
                slope_left_outer_y = road_left_y + perp_y * slope_width_left
                slope_left_outer_height = terrain_left_height

            if slope_width_right < 0.1:
                slope_right_outer_x = road_right_x
                slope_right_outer_y = road_right_y
                slope_right_outer_height = z  # Straßenhöhe, nicht Terrain!
            else:
                # Äußerer Rand der Böschung (wo sie Terrain trifft)
                slope_right_outer_x = road_right_x - perp_x * slope_width_right
                slope_right_outer_y = road_right_y - perp_y * slope_width_right
                slope_right_outer_height = terrain_right_height

            # Höhe am äußeren Rand
            slope_left_outer_local = apply_local_offset(
                slope_left_outer_x, slope_left_outer_y, slope_left_outer_height
            )
            slope_right_outer_local = apply_local_offset(
                slope_right_outer_x, slope_right_outer_y, slope_right_outer_height
            )

            slope_left_outer_vertices.append(slope_left_outer_local)
            slope_right_outer_vertices.append(slope_right_outer_local)

        num_points = len(road_left_abs)

        # === STRASSEN-MESH ===
        # Merke Start-Index in all_road_vertices für DIESE Straße
        road_start_idx = len(all_road_vertices)

        # Füge Road-Vertices hinzu (EXPLIZIT transformieren!)
        # Verwende die absoluten Koordinaten und transformiere sie nochmal
        for i, (x_abs, y_abs, z_abs) in enumerate(road_left_abs):
            transformed = apply_local_offset(x_abs, y_abs, z_abs)
            all_road_vertices.append(transformed)
        for x_abs, y_abs, z_abs in road_right_abs:
            all_road_vertices.append(apply_local_offset(x_abs, y_abs, z_abs))

        # Straßen-Faces (Quads zwischen links/rechts)
        # WICHTIG: Indices 0-basiert relativ zu all_road_vertices!
        for i in range(num_points - 1):
            left1 = road_start_idx + i
            left2 = road_start_idx + i + 1
            right1 = road_start_idx + num_points + i
            right2 = road_start_idx + num_points + i + 1

            all_road_faces.append([left1, right1, right2])
            all_road_faces.append([left1, right2, left2])

        # === BÖSCHUNGS-MESH ===
        # WICHTIG: Böschungs-Faces müssen 0-basiert relativ zu all_slope_vertices sein
        # Sie werden später beim Kombinieren mit Offset versehen

        # Aktuelle Position in all_slope_vertices VOR dem Hinzufügen dieser Straße
        slope_start_for_this_road = len(all_slope_vertices)

        # Füge Road-Kanten-Vertices zu Slope-Vertices hinzu
        for i, (x_abs, y_abs, z_abs) in enumerate(road_left_abs):
            transformed = apply_local_offset(x_abs, y_abs, z_abs)
            all_slope_vertices.append(transformed)
        for x_abs, y_abs, z_abs in road_right_abs:
            all_slope_vertices.append(apply_local_offset(x_abs, y_abs, z_abs))

        # Füge Böschungs-Outer-Vertices hinzu
        all_slope_vertices.extend(slope_left_outer_vertices)
        all_slope_vertices.extend(slope_right_outer_vertices)

        # Böschungs-Faces:
        # Struktur für DIESE Straße in all_slope_vertices:
        # [road_left1, road_left2, ..., road_leftN, road_right1, ..., road_rightN,
        #  slope_left1, ..., slope_leftN, slope_right1, ..., slope_rightN]
        # WICHTIG: Indices relativ zu slope_start_for_this_road (werden später mit Offset versehen!)

        for i in range(num_points - 1):
            # Linke Böschung: Von road_left zu slope_left_outer
            road_left1 = slope_start_for_this_road + i
            road_left2 = slope_start_for_this_road + i + 1
            slope_left1 = slope_start_for_this_road + 2 * num_points + i
            slope_left2 = slope_start_for_this_road + 2 * num_points + i + 1

            all_slope_faces.append([road_left1, slope_left1, slope_left2])
            all_slope_faces.append([road_left1, slope_left2, road_left2])

            # Rechte Böschung: Von road_right zu slope_right_outer
            road_right1 = slope_start_for_this_road + num_points + i
            road_right2 = slope_start_for_this_road + num_points + i + 1
            slope_right1 = slope_start_for_this_road + 3 * num_points + i
            slope_right2 = slope_start_for_this_road + 3 * num_points + i + 1

            all_slope_faces.append([road_right1, slope_right2, slope_right1])
            all_slope_faces.append([road_right1, road_right2, slope_right2])

        # Sammle 2D-Polygone (nur X-Y) für Grid-Ausschneiden
        # WICHTIG: Verwende ABSOLUTE Koordinaten (vor apply_local_offset!)
        # Straßen-Polygon: linke Kante + rechte Kante (rückwärts)
        road_poly_2d = [(x, y) for x, y, z in road_left_abs] + [
            (x, y) for x, y, z in reversed(road_right_abs)
        ]

        # Böschungs-Polygon: Kombiniere linke und rechte Böschung
        # Struktur: slope_left_outer + road_left rückwärts + road_right + slope_right_outer rückwärts
        # WICHTIG: slope_*_outer_vertices sind bereits lokal transformiert!
        # Wir brauchen die absoluten Koordinaten - müssen sie zurückrechnen oder anders speichern
        # HACK: Extrahiere X-Y aus lokalen Koordinaten (nicht ideal, aber funktioniert wenn LOCAL_OFFSET gesetzt)
        if LOCAL_OFFSET is not None:
            slope_left_2d = [
                (v[0] + LOCAL_OFFSET[0], v[1] + LOCAL_OFFSET[1])
                for v in slope_left_outer_vertices
            ]
            slope_right_2d = [
                (v[0] + LOCAL_OFFSET[0], v[1] + LOCAL_OFFSET[1])
                for v in slope_right_outer_vertices
            ]
        else:
            # Fallback: verwende lokale Koordinaten direkt (sollte nicht passieren)
            slope_left_2d = [(v[0], v[1]) for v in slope_left_outer_vertices]
            slope_right_2d = [(v[0], v[1]) for v in slope_right_outer_vertices]

        # Gesamtes Böschungs-Polygon (geschlossener Ring)
        slope_poly_2d = (
            slope_left_2d
            + [(x, y) for x, y, z in reversed(road_left_abs)]
            + [(x, y) for x, y, z in road_right_abs]
            + list(reversed(slope_right_2d))
        )

        road_slope_polygons_2d.append(
            {"road_polygon": road_poly_2d, "slope_polygon": slope_poly_2d}
        )

        processed += 1
        if processed % 100 == 0:
            print(f"  {processed}/{total_roads} Straßen...")

    print(f"  ✓ {len(all_road_vertices)} Straßen-Vertices")
    print(f"  ✓ {len(all_road_faces)} Straßen-Faces")
    print(f"  ✓ {len(all_slope_vertices)} Böschungs-Vertices")
    print(f"  ✓ {len(all_slope_faces)} Böschungs-Faces")
    print(
        f"  ✓ {len(road_slope_polygons_2d)} Road/Slope-Polygone für Grid-Ausschneiden (2D)"
    )
    if clipped_roads > 0:
        print(f"  ℹ {clipped_roads} Straßen komplett außerhalb Grid (ignoriert)")

    return (
        all_road_vertices,
        all_road_faces,
        all_slope_vertices,
        all_slope_faces,
        road_slope_polygons_2d,
    )


def generate_full_grid_mesh(grid_points, modified_heights, vertex_types, nx, ny):
    """
    Generiert vollständiges Grid-Mesh OHNE Vereinfachung.
    Einfache Grid-basierte Triangulation.
    """
    print("  Transformiere Vertices...")
    x_local, y_local, z_local = apply_local_offset(
        grid_points[:, 0], grid_points[:, 1], modified_heights
    )
    vertices = np.column_stack([x_local, y_local, z_local])

    print("  Generiere Grid-Faces (vektorisiert)...")

    # VEKTORISIERTE Grid-Face-Generierung (100x schneller!)
    # Erstelle Index-Grid (1-basiert für OBJ)
    idx_grid = np.arange(1, len(vertices) + 1).reshape(ny, nx)

    # Extrahiere alle Quad-Ecken auf einmal
    tl = idx_grid[:-1, :-1].ravel()  # top-left
    tr = idx_grid[:-1, 1:].ravel()  # top-right
    br = idx_grid[1:, 1:].ravel()  # bottom-right
    bl = idx_grid[1:, :-1].ravel()  # bottom-left

    # Material-Typ pro Quad (Maximum der 4 Ecken)
    vertex_types_2d = vertex_types.reshape(ny, nx)
    mat_tl = vertex_types_2d[:-1, :-1].ravel()
    mat_tr = vertex_types_2d[:-1, 1:].ravel()
    mat_br = vertex_types_2d[1:, 1:].ravel()
    mat_bl = vertex_types_2d[1:, :-1].ravel()
    quad_materials = np.maximum.reduce([mat_tl, mat_tr, mat_br, mat_bl])

    # Erstelle alle Dreiecke (2 pro Quad)
    num_quads = len(tl)
    all_tris = np.empty((num_quads * 2, 3), dtype=np.int32)
    all_tris[0::2] = np.column_stack([tl, tr, br])  # Dreieck 1
    all_tris[1::2] = np.column_stack([tl, br, bl])  # Dreieck 2

    # Verdopple Material-Maske (2 Dreiecke pro Quad)
    tri_materials = np.repeat(quad_materials, 2)

    # Trenne nach Material - NUR TERRAIN (Straßen/Böschungen werden ausgeschnitten!)
    terrain_faces = all_tris[tri_materials == 0].tolist()

    # Straßen und Böschungen werden NICHT mehr aus Grid generiert
    road_faces = []  # Leer - wird später durch Mesh-Streifen ersetzt
    slope_faces = []  # Leer - wird später durch Mesh-Streifen ersetzt

    print(f"  ✓ {len(vertices)} Vertices")
    print(f"  ✓ {len(terrain_faces)} Terrain-Faces (Straßen ausgeschnitten)")
    print(f"  ✓ Straßen/Böschungen werden separat generiert")

    return vertices.tolist(), road_faces, slope_faces, terrain_faces


def create_pyvista_mesh(vertices, faces):
    """Erstellt PyVista PolyData direkt aus Vertices und Faces (VEKTORISIERT)."""
    if not faces:
        return None

    # Konvertiere zu NumPy für vektorisierte Operationen
    faces_array = np.array(faces, dtype=np.int32)

    # Finde verwendete Vertices (VEKTORISIERT statt Set+Loop)
    used_indices_sorted = np.unique(faces_array)

    # Erstelle Mapping: alter Index → neuer Index (0-basiert für PyVista)
    # Nutze NumPy searchsorted für O(n log n) statt Dict-Lookup
    index_map = np.arange(len(used_indices_sorted))

    # Extrahiere verwendete Vertices (NumPy fancy indexing statt List Comprehension)
    used_vertices = np.array(vertices)[used_indices_sorted - 1]  # Faces sind 1-basiert

    # Konvertiere Faces (VEKTORISIERT: 1-basiert → 0-basiert)
    # searchsorted mappt alte Indizes → neue Indizes in O(n log n)
    faces_remapped = np.searchsorted(used_indices_sorted, faces_array)

    # PyVista Face-Format: [3, v1, v2, v3, 3, v1, v2, v3, ...]
    num_faces = len(faces_remapped)
    pyvista_faces = np.empty(num_faces * 4, dtype=np.int32)
    pyvista_faces[0::4] = 3  # Jedes 4. Element = 3 (Triangle)
    pyvista_faces[1::4] = faces_remapped[:, 0]
    pyvista_faces[2::4] = faces_remapped[:, 1]
    pyvista_faces[3::4] = faces_remapped[:, 2]

    # Erstelle PyVista PolyData
    mesh = pv.PolyData(used_vertices, pyvista_faces)
    return mesh


def save_layer_obj(filename, vertices, faces, material_name):
    """Speichert ein einzelnes Layer-Mesh als OBJ (nur verwendete Vertices)."""
    if not faces:
        print(f"  → {filename}: Keine Faces, überspringe")
        return

    # Finde alle verwendeten Vertex-Indizes
    used_indices = set()
    for face in faces:
        used_indices.update(face)

    # Erstelle Mapping: alter Index → neuer Index
    used_indices_sorted = sorted(used_indices)
    index_map = {
        old_idx: new_idx + 1 for new_idx, old_idx in enumerate(used_indices_sorted)
    }

    # Extrahiere nur verwendete Vertices (Faces sind 1-basiert)
    used_vertices = [vertices[idx - 1] for idx in used_indices_sorted]

    # Nummeriere Faces neu
    remapped_faces = [[index_map[idx] for idx in face] for face in faces]

    # BATCH-SCHREIBEN (100x schneller als Loop)
    with open(filename, "w", buffering=8 * 1024 * 1024) as f:  # 8MB Buffer
        f.write(f"# {material_name} Layer\n")
        f.write(f"mtllib terrain.mtl\n\n")

        # Vertices (BATCH mit join, nur verwendete)
        vertex_lines = [f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n" for v in used_vertices]
        f.write("".join(vertex_lines))

        # Faces (BATCH mit join, neu nummeriert)
        f.write(f"\nusemtl {material_name}\n")
        face_lines = [f"f {face[0]} {face[1]} {face[2]}\n" for face in remapped_faces]
        f.write("".join(face_lines))

    print(f"  ✓ {filename}: {len(used_vertices)} vertices, {len(faces)} faces")


# === MESH GENERATION (PyVista-based workflow) ===


def save_unified_obj(filename, vertices, road_faces, slope_faces, terrain_faces):
    """Speichert ein einheitliches Terrain-Mesh mit integrierten Straßen (OPTIMIERT)."""
    mtl_filename = filename.replace(".obj", ".mtl")

    # Erstelle MTL-Datei
    with open(mtl_filename, "w") as f:
        f.write("# Material Library for BeamNG Terrain\n")
        f.write("# Auto-generated\n\n")

        # Straßenoberfläche (Asphalt)
        f.write("newmtl road_surface\n")
        f.write("Ns 50.000000\n")
        f.write("Ka 0.200000 0.200000 0.200000\n")
        f.write("Kd 0.300000 0.300000 0.300000\n")
        f.write("Ks 0.500000 0.500000 0.500000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n\n")

        # Böschungen (Erde)
        f.write("newmtl road_slope\n")
        f.write("Ns 5.000000\n")
        f.write("Ka 0.200000 0.150000 0.100000\n")
        f.write("Kd 0.400000 0.300000 0.200000\n")
        f.write("Ks 0.100000 0.100000 0.100000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n\n")

        # Terrain (Gras/Natur)
        f.write("newmtl terrain\n")
        f.write("Ns 10.000000\n")
        f.write("Ka 0.100000 0.200000 0.100000\n")
        f.write("Kd 0.200000 0.500000 0.200000\n")
        f.write("Ks 0.100000 0.100000 0.100000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")

    # Erstelle OBJ-Datei (BATCH-OPTIMIERT mit separaten Objekten)
    print(f"\nSchreibe OBJ-Datei: {filename}")
    with open(filename, "w", buffering=8 * 1024 * 1024) as f:  # 8MB Buffer
        f.write("# BeamNG Unified Terrain Mesh with integrated roads\n")
        f.write(f"# Generated from DGM1 data and OSM\n")
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")

        # Alle Vertices EINMAL schreiben (shared vertex pool)
        print(f"  Schreibe {len(vertices)} Vertices...")
        vertex_lines = [f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n" for v in vertices]
        f.write("".join(vertex_lines))

        # Straßen-Objekt
        print(f"  Schreibe {len(road_faces)} Straßen-Faces...")
        f.write("\no road_surface\n")
        f.write("usemtl road_surface\n")
        face_lines = [f"f {face[0]} {face[1]} {face[2]}\n" for face in road_faces]
        f.write("".join(face_lines))

        # Böschungs-Objekt
        print(f"  Schreibe {len(slope_faces)} Böschungs-Faces...")
        f.write("\no road_slope\n")
        f.write("usemtl road_slope\n")
        face_lines = [f"f {face[0]} {face[1]} {face[2]}\n" for face in slope_faces]
        f.write("".join(face_lines))

        # Terrain-Objekt
        print(f"  Schreibe {len(terrain_faces)} Terrain-Faces...")
        f.write("\no terrain\n")
        f.write("usemtl terrain\n")
        face_lines = [f"f {face[0]} {face[1]} {face[2]}\n" for face in terrain_faces]
        f.write("".join(face_lines))

    print(f"  ✓ {filename} erfolgreich erstellt!")
    print(f"  ✓ {mtl_filename} erfolgreich erstellt!")


def main():
    global LOCAL_OFFSET, BBOX, GRID_BOUNDS_UTM
    LOCAL_OFFSET = None  # Reset bei jedem Durchlauf
    GRID_BOUNDS_UTM = None  # Reset bei jedem Durchlauf

    start_time = time.time()
    timings = {}  # Zeitmessung für jeden Schritt

    # 1. Lade Höhendaten aus lokalen Dateien
    print("=" * 60)
    print("WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator")
    print("=" * 60)

    step_start = time.time()
    height_points, height_elevations = load_height_data()
    timings["1_Höhendaten_laden"] = time.time() - step_start

    # 2. Berechne BBOX aus Höhendaten
    print("\nBerechne BBOX aus Höhendaten...")
    step_start = time.time()
    BBOX = calculate_bbox_from_height_data(height_points)
    timings["2_BBOX_berechnen"] = time.time() - step_start

    # 3. Prüfe ob OSM-Daten neu geladen werden müssen
    step_start = time.time()
    height_hash = get_height_data_hash()
    if not height_hash:
        height_hash = "no_files"

    cache_bbox_path = os.path.join(CACHE_DIR, "current_bbox.json")
    cache_height_hash_path = os.path.join(CACHE_DIR, "height_data_hash.txt")

    # Prüfe ob height-data geändert wurde
    need_reload = False
    if os.path.exists(cache_height_hash_path):
        with open(cache_height_hash_path, "r") as f:
            cached_hash = f.read().strip()
        if cached_hash != height_hash:
            print("  ⚠ Höhendaten haben sich geändert - lade OSM-Daten neu")
            need_reload = True
    else:
        need_reload = True

    # Speichere aktuellen Hash
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_height_hash_path, "w") as f:
        f.write(height_hash)

    # 4. Alle OSM-Daten holen (oder aus Cache laden)
    if need_reload:
        # Lösche alte Caches wenn height-data sich geändert hat
        old_caches = glob.glob(os.path.join(CACHE_DIR, "osm_all_*.json"))
        old_caches += glob.glob(os.path.join(CACHE_DIR, "elevations_*.json"))
        for cache in old_caches:
            try:
                os.remove(cache)
                print(f"  Alter Cache gelöscht: {os.path.basename(cache)}")
            except:
                pass

    osm_elements = get_osm_data(BBOX)
    timings["3_OSM_Daten_holen"] = time.time() - step_start
    if not osm_elements:
        print("Keine Daten gefunden.")
        return

    # 5. Straßen aus den OSM-Daten extrahieren
    step_start = time.time()
    roads = extract_roads_from_osm(osm_elements)
    timings["4_Straßen_extrahieren"] = time.time() - step_start
    if not roads:
        print("Keine Straßen gefunden.")
        return

    # 6. Erstelle Terrain-Grid aus Höhendaten
    step_start = time.time()
    grid_points, grid_elevations, nx, ny = create_terrain_grid(
        height_points, height_elevations, grid_spacing=GRID_SPACING
    )
    timings["5_Grid_erstellen"] = time.time() - step_start

    # 7. Extrahiere Straßen-Polygone
    step_start = time.time()
    print(f"\nExtrahiere {len(roads)} Straßen-Polygone...")
    road_polygons = get_road_polygons(roads, BBOX, height_points, height_elevations)
    print(f"  ✓ {len(road_polygons)} Straßen-Polygone extrahiert")
    timings["6_Straßen_Polygone"] = time.time() - step_start

    # 8. Generiere Straßen-Mesh-Streifen (VOR Grid, um 2D-Polygone für Ausschneiden zu bekommen)
    step_start = time.time()
    print("\nGeneriere Straßen-Mesh-Streifen...")
    (
        road_vertices_temp,
        road_faces,
        slope_vertices_temp,
        slope_faces_strips,
        road_slope_polygons_2d,
    ) = generate_road_mesh_strips(road_polygons, height_points, height_elevations)
    print(
        f"  ✓ {len(road_slope_polygons_2d)} 2D-Polygone für Grid-Klassifizierung extrahiert"
    )
    timings["7_Straßen_Mesh_Vorläufig"] = time.time() - step_start

    # 9. Klassifiziere Grid-Vertices (markiere Straßen/Böschungs-Bereiche zum Ausschneiden)
    # Verwendet die 2D-Polygone von generate_road_mesh_strips
    step_start = time.time()
    vertex_types, modified_heights = classify_grid_vertices(
        grid_points, grid_elevations, road_slope_polygons_2d
    )
    timings["8_Vertex_Klassifizierung"] = time.time() - step_start

    # 10. Generiere Terrain-Grid (Straßen ausgeschnitten)
    # WICHTIG: JETZT wird LOCAL_OFFSET korrekt gesetzt!
    step_start = time.time()
    print("\nGeneriere Terrain-Grid-Mesh...")
    grid_vertices, _, _, terrain_faces = generate_full_grid_mesh(
        grid_points,
        modified_heights,
        vertex_types,
        nx,
        ny,
    )
    timings["9_Terrain_Grid_Generierung"] = time.time() - step_start

    # 11. Generiere Straßen-Mesh NOCHMAL (jetzt mit korrektem LOCAL_OFFSET!)
    # Die vorläufigen Vertices von oben werden verworfen
    step_start = time.time()
    print("\nGeneriere finales Straßen-Mesh...")
    road_vertices, road_faces, slope_vertices, slope_faces_strips, _ = (
        generate_road_mesh_strips(road_polygons, height_points, height_elevations)
    )
    timings["10_Straßen_Mesh_Final"] = time.time() - step_start

    # 12. Kombiniere Straßen-Mesh, Böschungs-Mesh und Terrain-Mesh
    step_start = time.time()
    print("\nKombiniere Straßen-Mesh, Böschungs-Mesh und Terrain-Mesh...")

    # Offset für Indices (Terrain-Vertices kommen zuerst)
    terrain_vertex_count = len(grid_vertices)

    # Straßen-Vertices
    road_vertex_count = len(road_vertices)

    # Böschungs-Vertices (enthalten bereits Road-Kanten-Vertices + Slope-Outer-Vertices)
    slope_vertex_count = len(slope_vertices)

    # Road-Faces: Indizes anpassen (nach Terrain-Vertices)
    road_faces_offset = [
        [idx + terrain_vertex_count for idx in face] for face in road_faces
    ]

    # Böschungs-Faces: Indizes anpassen (nach Terrain-Vertices + Road-Vertices)
    slope_faces_offset = [
        [idx + terrain_vertex_count + road_vertex_count for idx in face]
        for face in slope_faces_strips
    ]

    # Kombiniere Vertices: Terrain + Road + Slope
    all_vertices = grid_vertices + road_vertices + slope_vertices

    print(f"\n  Kombiniere Vertex-Daten...")
    print(f"    • Terrain: {terrain_vertex_count} Vertices")
    print(f"    • Straßen: {road_vertex_count} Vertices")
    print(f"    • Böschungen: {slope_vertex_count} Vertices")
    print(f"    ✓ Total: {len(all_vertices)} Vertices")

    # Faces bleiben getrennt für Materialien
    combined_road_faces = road_faces_offset
    combined_slope_faces = slope_faces_offset
    combined_terrain_faces = terrain_faces

    print(f"  ✓ Kombiniert: {len(all_vertices)} Vertices total")
    print(f"    • Terrain: {terrain_vertex_count} Vertices, {len(terrain_faces)} Faces")
    print(
        f"    • Straßen: {road_vertex_count} Vertices, {len(combined_road_faces)} Faces"
    )
    print(
        f"    • Böschungen: {slope_vertex_count} Vertices, {len(combined_slope_faces)} Faces"
    )

    timings["10_Mesh_Kombination"] = time.time() - step_start

    # 12. Erstelle PyVista-Meshes im Speicher
    step_start = time.time()
    print("\nErstelle und vereinfache Meshes...")
    print("  • Terrain (mit PyVista-Vereinfachung 90%)")
    print("  • Straßen und Böschungen (original)")

    # NUR Terrain durch PyVista optimieren
    terrain_mesh = create_pyvista_mesh(grid_vertices, combined_terrain_faces)

    # Speichere Road/Slope Daten VOR dem Löschen!
    road_vertices_original = road_vertices
    slope_vertices_original = slope_vertices
    road_faces_original = combined_road_faces
    slope_faces_original = combined_slope_faces

    # Speichere Original-Statistiken
    original_terrain_vertex_count = len(grid_vertices)

    # Gebe NUR die kombinierten Listen frei
    del all_vertices, grid_vertices, combined_terrain_faces
    gc.collect()

    # Vereinfache Terrain mit 90% Reduktion
    print(f"  Vereinfache Terrain (90% Reduktion)...")
    original_points = terrain_mesh.n_points
    terrain_simplified = terrain_mesh.decimate_pro(
        reduction=0.90,
        feature_angle=25.0,
        preserve_topology=True,
        boundary_vertex_deletion=False,
    )
    print(f"    ✓ {original_points:,} → {terrain_simplified.n_points:,} Vertices")

    del terrain_mesh
    gc.collect()

    # Kombiniere MANUELL: decimiertes Terrain + originale Roads + originale Slopes
    print("\n  Kombiniere Vertices manuell...")

    # Extrahiere decimierte Terrain-Vertices aus PyVista
    terrain_vertices_decimated = terrain_simplified.points.tolist()
    terrain_vertex_count = len(terrain_vertices_decimated)

    road_vertex_count = len(road_vertices_original)
    slope_vertex_count = len(slope_vertices_original)

    # Kombiniere alle Vertices
    all_vertices_combined = (
        terrain_vertices_decimated + road_vertices_original + slope_vertices_original
    )
    total_vertex_count = len(all_vertices_combined)
    print(f"    • Terrain: {terrain_vertex_count:,} Vertices")
    print(f"    • Straßen: {road_vertex_count:,} Vertices")
    print(f"    • Böschungen: {slope_vertex_count:,} Vertices")
    print(f"    ✓ Gesamt: {total_vertex_count:,} Vertices")

    # Speichere die decimierte Terrain-Vertex-Count BEVOR Speicher freigegeben wird!
    terrain_vertices_decimated_count = len(terrain_vertices_decimated)

    # Gebe temporäre Listen frei
    del road_vertices_original, slope_vertices_original, terrain_vertices_decimated
    gc.collect()

    # Extrahiere Terrain-Faces aus PyVista
    terrain_faces_decimated = []
    for i in range(terrain_simplified.n_cells):
        cell = terrain_simplified.get_cell(i)
        if cell.type == 5:  # VTK_TRIANGLE
            point_ids = cell.point_ids
            terrain_faces_decimated.append(
                [point_ids[0] + 1, point_ids[1] + 1, point_ids[2] + 1]
            )

    # Prepare faces for OBJ export (1-indexed)
    print("\n  Bereite Faces für OBJ-Export vor...")
    terrain_faces_final = terrain_faces_decimated

    # Road und Slope Faces: bereits offset, nur noch +1 für OBJ-Indexing
    road_faces_final = [[idx + 1 for idx in face] for face in road_faces_original]
    slope_faces_final = [[idx + 1 for idx in face] for face in slope_faces_original]

    print(f"    • Terrain: {len(terrain_faces_final):,} Faces")
    print(f"    • Straßen: {len(road_faces_final):,} Faces")
    print(f"    • Böschungen: {len(slope_faces_final):,} Faces")

    # Korrigiere Face-Indices nach Terrain-Decimation
    offset_diff = terrain_vertices_decimated_count - original_terrain_vertex_count

    # Passe Road und Slope Faces an (aber NICHT Terrain!)
    road_faces_final = [
        [
            idx + offset_diff if idx > original_terrain_vertex_count else idx
            for idx in face
        ]
        for face in road_faces_final
    ]
    slope_faces_final = [
        [
            idx + offset_diff if idx > original_terrain_vertex_count else idx
            for idx in face
        ]
        for face in slope_faces_final
    ]

    # Gebe PyVista-Mesh und Offset-Listen frei
    del (
        terrain_simplified,
        terrain_faces_decimated,
        road_faces_original,
        slope_faces_original,
    )
    gc.collect()
    print("\n  ✓ PyVista-Mesh und temporäre Listen aus Speicher entfernt")

    timings["12_PyVista_Simplification"] = time.time() - step_start

    # Speichere finales Mesh
    output_obj = "beamng.obj"
    # Schreibe OBJ-Datei
    print(f"\n  Schreibe: {output_obj}")
    save_unified_obj(
        output_obj,
        all_vertices_combined,
        road_faces_final,
        slope_faces_final,
        terrain_faces_final,
    )

    # Gebe große Mesh-Daten frei
    del (
        all_vertices_combined,
        terrain_faces_final,
        slope_faces_final,
        road_faces_final,
    )
    gc.collect()

    timings["12_Mesh_Export"] = time.time() - step_start

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print(f"✓ GENERATOR BEENDET!")
    print(f"{'=' * 60}")
    print(f"  Output-Datei: {output_obj}")
    if LOCAL_OFFSET:
        print(
            f"  Lokaler Offset: X={LOCAL_OFFSET[0]:.2f}m, Y={LOCAL_OFFSET[1]:.2f}m, Z={LOCAL_OFFSET[2]:.2f}m"
        )

    # DETAILLIERTE TIMING-ÜBERSICHT
    print(f"\n{'=' * 60}")
    print(f"ZEITMESSUNG (Gesamtzeit: {elapsed_time:.2f}s / {elapsed_time/60:.1f} min)")
    print(f"{'=' * 60}")
    for step_name, step_time in timings.items():
        percentage = (step_time / elapsed_time) * 100
        step_display = step_name.replace("_", " ").replace("  ", " ")
        bar_length = int(percentage / 2)  # 50 chars = 100%
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {step_display:.<35} {step_time:>6.2f}s ({percentage:>5.1f}%) {bar}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
