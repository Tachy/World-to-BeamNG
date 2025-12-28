"""
WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator

Benötigte Pakete:
  pip install requests numpy scipy pyproj pyvista

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

# --- KONFIGURATION ---
ROAD_WIDTH = 7.0
SLOPE_HEIGHT = 3.0  # Höhe der Böschung in Metern
LEVEL_NAME = "osm_generated_map"
CACHE_DIR = "cache"  # Verzeichnis für Cache-Dateien
HEIGHT_DATA_DIR = "height-data"  # Verzeichnis mit Höhendaten

# BBOX wird automatisch aus Höhendaten ermittelt
BBOX = None

# Globaler Offset für lokale Koordinaten (wird bei erstem Punkt gesetzt)
LOCAL_OFFSET = None

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
    print(f"  DEBUG: height_hash = {height_hash}")
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
    print(f"  DEBUG: Versuche Cache zu speichern, height_hash = {height_hash}")
    if height_hash:
        cache_file_path = os.path.join(CACHE_DIR, f"height_raw_{height_hash}.npz")
        print(f"  Speichere Höhendaten-Cache: {cache_file_path}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_file_path, points=points, elevations=elevations)
        print(f"  ✓ Cache erstellt: {os.path.basename(cache_file_path)}")
    else:
        print(f"  ⚠ Kein Cache gespeichert (height_hash ist None)")

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

    # Finde Bounds in UTM
    min_x, max_x = height_points[:, 0].min(), height_points[:, 0].max()
    min_y, max_y = height_points[:, 1].min(), height_points[:, 1].max()

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
    """Berechnet den Abstand eines Punktes zu einem Liniensegment."""
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2), proj_x, proj_y, t


def classify_grid_vertices(grid_points, grid_elevations, road_polygons):
    """Klassifiziert Grid-Vertices: Straße, Böschung oder Terrain (ULTRA-OPTIMIERT)."""
    print("\nKlassifiziere Grid-Vertices (spatial indexing)...")

    vertex_types = np.zeros(len(grid_points), dtype=int)
    modified_heights = grid_elevations.copy()

    half_width = ROAD_WIDTH / 2.0
    slope_width = half_width + SLOPE_HEIGHT

    # Baue KD-Tree für schnelle räumliche Suchen (einmalig!)
    print("  Erstelle räumlichen Index...")
    tree = cKDTree(grid_points)

    total_segments = sum(len(road["coords"]) - 1 for road in road_polygons)
    processed = 0

    for road in road_polygons:
        coords = road["coords"]

        # Verarbeite jedes Straßen-Segment
        for i in range(len(coords) - 1):
            x1, y1, z1 = coords[i]
            x2, y2, z2 = coords[i + 1]

            # Mittelpunkt und maximale Reichweite des Segments
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            seg_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            search_radius = seg_len / 2 + slope_width

            # Finde alle Punkte im Radius (KD-Tree: O(log n) statt O(n)!)
            nearby_indices = tree.query_ball_point([mid_x, mid_y], search_radius)

            if len(nearby_indices) == 0:
                continue

            nearby_indices = np.array(nearby_indices)
            nearby_points = grid_points[nearby_indices]

            # Vektorisierte Distanzberechnung
            dx = x2 - x1
            dy = y2 - y1
            len_sq = dx**2 + dy**2

            if len_sq < 1e-10:
                continue

            # Projektion auf Liniensegment (vektorisiert)
            t = np.clip(
                ((nearby_points[:, 0] - x1) * dx + (nearby_points[:, 1] - y1) * dy)
                / len_sq,
                0,
                1,
            )
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

            # Distanzen
            dists = np.sqrt(
                (nearby_points[:, 0] - proj_x) ** 2
                + (nearby_points[:, 1] - proj_y) ** 2
            )

            # Interpolierte Straßenhöhen
            road_heights = z1 + t * (z2 - z1)

            # Straßenoberfläche
            road_mask = dists <= half_width
            road_idx = nearby_indices[road_mask]
            vertex_types[road_idx] = 2
            modified_heights[road_idx] = road_heights[road_mask]

            # Böschung
            slope_mask = (
                (dists > half_width)
                & (dists <= slope_width)
                & (vertex_types[nearby_indices] < 1)
            )
            slope_idx = nearby_indices[slope_mask]
            blend = (dists[slope_mask] - half_width) / SLOPE_HEIGHT
            modified_heights[slope_idx] = (
                road_heights[slope_mask] * (1 - blend)
                + grid_elevations[slope_idx] * blend
            )
            vertex_types[slope_idx] = 1

            processed += 1
            if processed % 1000 == 0:
                print(f"  {processed}/{total_segments} Segmente verarbeitet...")

    print(f"  Straßen-Vertices: {np.sum(vertex_types == 2)}")
    print(f"  Böschungs-Vertices: {np.sum(vertex_types == 1)}")
    print(f"  Terrain-Vertices: {np.sum(vertex_types == 0)}")

    return vertex_types, modified_heights


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

    # Trenne nach Material
    road_faces = all_tris[tri_materials == 2].tolist()
    slope_faces = all_tris[tri_materials == 1].tolist()
    terrain_faces = all_tris[tri_materials == 0].tolist()

    print(f"  ✓ {len(vertices)} Vertices")
    print(f"  ✓ {len(road_faces)} Straßen-Faces")
    print(f"  ✓ {len(slope_faces)} Böschungs-Faces")
    print(f"  ✓ {len(terrain_faces)} Terrain-Faces")

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

    # Erstelle OBJ-Datei (BATCH-OPTIMIERT)
    print(f"\nSchreibe OBJ-Datei: {filename}")
    with open(filename, "w", buffering=8 * 1024 * 1024) as f:  # 8MB Buffer
        f.write("# BeamNG Unified Terrain Mesh with integrated roads\n")
        f.write(f"# Generated from DGM1 data and OSM\n")
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")
        f.write("o terrain\n")

        # Vertices (BATCH mit join - 100x schneller!)
        print(f"  Schreibe {len(vertices)} Vertices...")
        vertex_lines = [f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n" for v in vertices]
        f.write("".join(vertex_lines))

        # Straßen-Faces (BATCH)
        print(f"  Schreibe {len(road_faces)} Straßen-Faces...")
        f.write("\nusemtl road_surface\n")
        face_lines = [f"f {face[0]} {face[1]} {face[2]}\n" for face in road_faces]
        f.write("".join(face_lines))

        # Böschungs-Faces (BATCH)
        print(f"  Schreibe {len(slope_faces)} Böschungs-Faces...")
        f.write("\nusemtl road_slope\n")
        face_lines = [f"f {face[0]} {face[1]} {face[2]}\n" for face in slope_faces]
        f.write("".join(face_lines))

        # Terrain-Faces (BATCH)
        print(f"  Schreibe {len(terrain_faces)} Terrain-Faces...")
        f.write("\nusemtl terrain\n")
        face_lines = [f"f {face[0]} {face[1]} {face[2]}\n" for face in terrain_faces]
        f.write("".join(face_lines))

    print(f"  ✓ {filename} erfolgreich erstellt!")
    print(f"  ✓ {mtl_filename} erfolgreich erstellt!")


def main():
    global LOCAL_OFFSET, BBOX
    LOCAL_OFFSET = None  # Reset bei jedem Durchlauf

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
        height_points, height_elevations, grid_spacing=1.0
    )
    timings["5_Grid_erstellen"] = time.time() - step_start

    # 7. Extrahiere Straßen-Polygone
    step_start = time.time()
    print(f"\nExtrahiere {len(roads)} Straßen-Polygone...")
    road_polygons = get_road_polygons(roads, BBOX, height_points, height_elevations)
    print(f"  ✓ {len(road_polygons)} Straßen-Polygone extrahiert")
    timings["6_Straßen_Polygone"] = time.time() - step_start

    # 8. Klassifiziere Grid-Vertices und integriere Straßen
    step_start = time.time()
    vertex_types, modified_heights = classify_grid_vertices(
        grid_points, grid_elevations, road_polygons
    )
    timings["7_Vertex_Klassifizierung"] = time.time() - step_start

    # 9. Generiere VOLLSTÄNDIGES Mesh (OHNE Vereinfachung)
    step_start = time.time()
    print("\nGeneriere vollständiges Mesh...")
    vertices, road_faces, slope_faces, terrain_faces = generate_full_grid_mesh(
        grid_points,
        modified_heights,
        vertex_types,
        nx,
        ny,
    )
    timings["8_Mesh_Generierung"] = time.time() - step_start

    # 10. Erstelle PyVista-Meshes direkt im Speicher (OHNE Disk I/O!)
    step_start = time.time()
    print("\nErstelle PyVista-Meshes im Speicher...")

    print("  Erstelle Terrain-Mesh...")
    terrain_mesh = create_pyvista_mesh(vertices, terrain_faces)
    print(f"    ✓ {terrain_mesh.n_points:,} Vertices, {terrain_mesh.n_cells:,} Faces")

    print("  Erstelle Böschungs-Mesh...")
    slopes_mesh = create_pyvista_mesh(vertices, slope_faces)
    print(f"    ✓ {slopes_mesh.n_points:,} Vertices, {slopes_mesh.n_cells:,} Faces")

    print("  Erstelle Straßen-Mesh...")
    roads_mesh = create_pyvista_mesh(vertices, road_faces)
    print(f"    ✓ {roads_mesh.n_points:,} Vertices, {roads_mesh.n_cells:,} Faces")

    # Gebe Original-Listen frei (können mehrere GB sein!)
    del vertices, road_faces, slope_faces, terrain_faces
    gc.collect()
    print("  ✓ Original-Mesh-Listen aus Speicher entfernt")

    timings["9_PyVista_Meshes_erstellen"] = time.time() - step_start

    # 11. PyVista-Vereinfachung (DIREKT im Speicher)
    step_start = time.time()
    print("\nVereinfache Meshes mit PyVista...")

    # Vereinfachung (unterschiedliche Reduktion pro Layer)
    print(
        f"  Vereinfache Terrain ({terrain_mesh.n_points:,} Vertices, 90% Reduktion)..."
    )
    terrain_simplified = terrain_mesh.decimate_pro(
        reduction=0.90,
        feature_angle=25.0,
        preserve_topology=True,
        boundary_vertex_deletion=False,
    )
    print(f"    ✓ {terrain_mesh.n_points:,} → {terrain_simplified.n_points:,} Vertices")

    print(
        f"  Vereinfache Böschungen ({slopes_mesh.n_points:,} Vertices, 50% Reduktion)..."
    )
    slopes_simplified = slopes_mesh.decimate_pro(
        reduction=0.50,
        feature_angle=15.0,
        preserve_topology=True,
        boundary_vertex_deletion=False,
    )
    print(f"    ✓ {slopes_mesh.n_points:,} → {slopes_simplified.n_points:,} Vertices")

    print(f"  Straßen ({roads_mesh.n_points:,} Vertices, keine Vereinfachung)...")
    roads_simplified = roads_mesh

    # Gebe un-vereinfachte Meshes frei (nur noch simplified-Versionen behalten)
    del terrain_mesh, slopes_mesh, roads_mesh
    gc.collect()
    print("  ✓ Un-vereinfachte Meshes aus Speicher entfernt")

    # Kombiniere vereinfachte Meshes
    print("\n  Kombiniere vereinfachte Meshes...")
    combined = terrain_simplified + slopes_simplified + roads_simplified
    print(f"    Kombiniert: {combined.n_points:,} Vertices, {combined.n_cells:,} Faces")

    # Speichere finales Mesh (OPTIMIERT mit save_unified_obj statt PyVista save)
    output_obj = "beamng.obj"
    print(f"\n  Schreibe finales Mesh: {output_obj}")

    # Extrahiere Vertices und Faces von SEPARATEN Meshes (um Materialien zu erhalten)
    print("  Extrahiere Faces nach Material...")

    # Alle Vertices vom kombinierten Mesh
    vertices = combined.points.tolist()

    # Extrahiere Faces von jedem Layer separat
    def extract_faces_from_mesh(mesh, vertex_offset=0):
        """Extrahiert Faces aus PyVista Mesh und fügt Vertex-Offset hinzu."""
        faces_raw = mesh.faces
        faces = []
        i = 0
        while i < len(faces_raw):
            n = faces_raw[i]
            # +1 für OBJ-Format (1-basiert), + vertex_offset für korrekte Indizes
            face = [
                faces_raw[i + 1] + 1 + vertex_offset,
                faces_raw[i + 2] + 1 + vertex_offset,
                faces_raw[i + 3] + 1 + vertex_offset,
            ]
            faces.append(face)
            i += n + 1
        return faces

    # Berechne Vertex-Offsets (wie PyVista die Meshes kombiniert)
    terrain_offset = 0
    slopes_offset = terrain_simplified.n_points
    roads_offset = slopes_offset + slopes_simplified.n_points

    # Extrahiere Faces mit korrekten Offsets
    terrain_faces_final = extract_faces_from_mesh(terrain_simplified, terrain_offset)
    slope_faces_final = extract_faces_from_mesh(slopes_simplified, slopes_offset)
    road_faces_final = extract_faces_from_mesh(roads_simplified, roads_offset)

    print(f"    Terrain: {len(terrain_faces_final)} Faces")
    print(f"    Slopes: {len(slope_faces_final)} Faces")
    print(f"    Roads: {len(road_faces_final)} Faces")

    # Speichere mit korrekten Material-Zuweisungen
    save_unified_obj(
        output_obj, vertices, road_faces_final, slope_faces_final, terrain_faces_final
    )

    # Gebe große Mesh-Daten frei (nicht mehr benötigt)
    del combined, vertices, terrain_faces_final, slope_faces_final, road_faces_final
    del terrain_simplified, slopes_simplified, roads_simplified
    gc.collect()
    print("  ✓ Mesh-Daten aus Speicher entfernt")

    timings["10_PyVista_Vereinfachung"] = time.time() - step_start

    print(f"\n  ✓ Vereinfachtes Mesh gespeichert: {output_obj}")
    print(
        f"    Terrain: {terrain_mesh.n_points:,} → {terrain_simplified.n_points:,} Vertices"
    )
    print(
        f"    Böschungen: {slopes_mesh.n_points:,} → {slopes_simplified.n_points:,} Vertices"
    )
    print(f"    Straßen: {roads_mesh.n_points:,} Vertices (unverändert)")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print(f"✓ GENERATOR BEENDET!")
    print(f"{'=' * 60}")
    print(f"  Terrain-Vertices: {len(vertices)}")
    print(f"  Straßen-Faces: {len(road_faces)}")
    print(f"  Böschungs-Faces: {len(slope_faces)}")
    print(f"  Terrain-Faces: {len(terrain_faces)}")
    print(f"  Output-Datei: {output_obj}")
    print(f"  BBOX: {BBOX}")
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
