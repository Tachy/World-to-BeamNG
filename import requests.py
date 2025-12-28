import requests
import json
import numpy as np
import time
import os
import hashlib
import glob
from scipy.interpolate import griddata
from pyproj import Transformer  # type: ignore

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
    if not xyz_files:
        return None

    # Hash basierend auf Dateinamen und Änderungszeitpunkten
    hash_input = ""
    for file in xyz_files:
        mtime = os.path.getmtime(file)
        hash_input += f"{os.path.basename(file)}_{mtime}_"

    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def load_height_data():
    """Lädt alle XYZ-Höhendaten aus dem height-data Ordner."""
    xyz_files = glob.glob(os.path.join(HEIGHT_DATA_DIR, "*.xyz"))

    if not xyz_files:
        raise FileNotFoundError(f"Keine .xyz Dateien in {HEIGHT_DATA_DIR} gefunden!")

    print(f"Lade Höhendaten aus {len(xyz_files)} Kacheln...")

    all_points = []
    all_elevations = []

    for file in xyz_files:
        print(f"  Lade {os.path.basename(file)}...")
        data = np.loadtxt(file)
        all_points.append(data[:, :2])  # X, Y (UTM)
        all_elevations.append(data[:, 2])  # Z (Höhe)

    # Kombiniere alle Kacheln
    points = np.vstack(all_points)
    elevations = np.hstack(all_elevations)

    print(f"  ✓ {len(elevations)} Höhenpunkte geladen")

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
    """Konvertiert zu lokalen Koordinaten relativ zum ersten Punkt."""
    global LOCAL_OFFSET
    if LOCAL_OFFSET is None:
        LOCAL_OFFSET = (x, y, z)
    return (x - LOCAL_OFFSET[0], y - LOCAL_OFFSET[1], z - LOCAL_OFFSET[2])


def create_terrain_grid(height_points, height_elevations, grid_spacing=10.0):
    """Erstellt ein reguläres Grid aus den Höhendaten."""
    print(f"\nErstelle Terrain-Grid (Abstand: {grid_spacing}m)...")

    # Finde Bounds in UTM
    min_x, max_x = height_points[:, 0].min(), height_points[:, 0].max()
    min_y, max_y = height_points[:, 1].min(), height_points[:, 1].max()

    # Erstelle Grid-Punkte
    x_coords = np.arange(min_x, max_x, grid_spacing)
    y_coords = np.arange(min_y, max_y, grid_spacing)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Interpoliere Höhen für Grid-Punkte
    from scipy.interpolate import griddata

    grid_elevations = griddata(
        height_points, height_elevations, grid_points, method="linear", fill_value=0
    )

    print(f"  Grid: {len(x_coords)} x {len(y_coords)} = {len(grid_points)} Vertices")

    return grid_points, grid_elevations, len(x_coords), len(y_coords)


def get_road_polygons(roads, bbox, height_points, height_elevations):
    """Extrahiert Straßen-Polygone mit ihren Koordinaten und Höhen."""
    road_polygons = []

    for way in roads:
        if "geometry" not in way:
            continue

        pts = [[p["lat"], p["lon"]] for p in way["geometry"]]
        if len(pts) < 2:
            continue

        # Hole Elevations
        elevations = get_elevations_for_points(
            pts, bbox, height_points, height_elevations
        )

        # Konvertiere zu UTM
        utm_coords = []
        for i, (lat, lon) in enumerate(pts):
            x, y = transformer_to_utm.transform(lon, lat)
            utm_coords.append((x, y, elevations[i]))

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
    """Klassifiziert Grid-Vertices: Straße, Böschung oder Terrain."""
    print("\nKlassifiziere Grid-Vertices...")

    vertex_types = np.zeros(
        len(grid_points), dtype=int
    )  # 0=Terrain, 1=Böschung, 2=Straße
    vertex_heights = grid_elevations.copy()  # Original-Höhen als Referenz
    modified_heights = grid_elevations.copy()  # Modifizierte Höhen

    total_vertices = len(grid_points)
    processed = 0

    for road in road_polygons:
        coords = road["coords"]

        # Verarbeite jedes Straßen-Segment
        for i in range(len(coords) - 1):
            x1, y1, z1 = coords[i]
            x2, y2, z2 = coords[i + 1]

            # Prüfe alle Grid-Punkte
            for idx, (gx, gy) in enumerate(grid_points):
                dist, proj_x, proj_y, t = point_to_line_distance(gx, gy, x1, y1, x2, y2)

                # Interpoliere Straßenhöhe an der Projektionsstelle
                road_height = z1 + t * (z2 - z1)

                half_width = ROAD_WIDTH / 2.0
                slope_width = ROAD_WIDTH / 2.0 + SLOPE_HEIGHT

                # Straßenoberfläche
                if dist <= half_width:
                    vertex_types[idx] = 2
                    modified_heights[idx] = road_height

                # Böschung (45° Neigung)
                elif dist <= slope_width:
                    if vertex_types[idx] < 1:  # Nur wenn noch nicht als Straße markiert
                        vertex_types[idx] = 1
                        # Lineare Interpolation: von road_height bis original_height
                        blend = (dist - half_width) / SLOPE_HEIGHT
                        original_height = vertex_heights[idx]
                        modified_heights[idx] = (
                            road_height * (1 - blend) + original_height * blend
                        )

            processed += 1
            if processed % 50 == 0:
                print(
                    f"  {processed}/{len(road_polygons) * 10} Segmente verarbeitet..."
                )

    print(f"  Straßen-Vertices: {np.sum(vertex_types == 2)}")
    print(f"  Böschungs-Vertices: {np.sum(vertex_types == 1)}")
    print(f"  Terrain-Vertices: {np.sum(vertex_types == 0)}")

    return vertex_types, modified_heights


def generate_terrain_mesh(grid_points, modified_heights, vertex_types, nx, ny):
    """Generiert ein Terrain-Mesh mit integrierten Straßen."""
    print("\nGeneriere vereinheitlichtes Terrain-Mesh...")

    # Konvertiere Grid-Punkte in lokale Koordinaten
    vertices = []
    for i, (point, height) in enumerate(zip(grid_points, modified_heights)):
        x, y = point
        x_local, y_local, z_local = apply_local_offset(x, y, height)
        vertices.append([x_local, y_local, z_local])

    # Generiere Faces (Grid-Triangulation)
    road_faces = []
    slope_faces = []
    terrain_faces = []

    for row in range(ny - 1):
        for col in range(nx - 1):
            idx = row * nx + col

            # Vier Ecken des Quads
            v1 = idx + 1  # OBJ ist 1-basiert
            v2 = idx + 2
            v3 = idx + nx + 2
            v4 = idx + nx + 1

            # Bestimme Material basierend auf Vertex-Typen
            types = [
                vertex_types[idx],
                vertex_types[idx + 1],
                vertex_types[idx + nx],
                vertex_types[idx + nx + 1],
            ]
            max_type = max(types)

            # Zwei Dreiecke pro Quad
            tri1 = [v1, v2, v3]
            tri2 = [v1, v3, v4]

            if max_type == 2:  # Straße
                road_faces.extend([tri1, tri2])
            elif max_type == 1:  # Böschung
                slope_faces.extend([tri1, tri2])
            else:  # Terrain
                terrain_faces.extend([tri1, tri2])

        if (row + 1) % 50 == 0:
            print(f"  {row + 1}/{ny - 1} Zeilen verarbeitet...")

    print(f"  Straßen-Faces: {len(road_faces)}")
    print(f"  Böschungs-Faces: {len(slope_faces)}")
    print(f"  Terrain-Faces: {len(terrain_faces)}")

    return vertices, road_faces, slope_faces, terrain_faces


def save_unified_obj(filename, vertices, road_faces, slope_faces, terrain_faces):
    """Speichert ein einheitliches Terrain-Mesh mit integrierten Straßen."""
    mtl_filename = filename.replace(".obj", ".mtl")

    # Erstelle MTL-Datei
    with open(mtl_filename, "w") as f:
        f.write("# Material Library for BeamNG Terrain\n\n")

        # Straßenoberfläche (Asphalt)
        f.write("newmtl road_surface\n")
        f.write("Ka 0.2 0.2 0.2\n")
        f.write("Kd 0.3 0.3 0.3\n")
        f.write("Ks 0.1 0.1 0.1\n")
        f.write("Ns 10.0\n")
        f.write("d 1.0\n\n")

        # Böschungen (Erde)
        f.write("newmtl road_slope\n")
        f.write("Ka 0.2 0.15 0.1\n")
        f.write("Kd 0.4 0.3 0.2\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write("Ns 5.0\n")
        f.write("d 1.0\n\n")

        # Terrain (Gras/Natur)
        f.write("newmtl terrain\n")
        f.write("Ka 0.1 0.2 0.1\n")
        f.write("Kd 0.2 0.5 0.2\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write("Ns 5.0\n")
        f.write("d 1.0\n")

    # Erstelle OBJ-Datei
    print(f"\nSchreibe OBJ-Datei: {filename}")
    with open(filename, "w") as f:
        f.write("# BeamNG Unified Terrain Mesh with integrated roads\n")
        f.write(f"# Generated from DGM1 data and OSM\n")
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")
        f.write("o terrain\n")

        # Vertices
        print(f"  Schreibe {len(vertices)} Vertices...")
        for v in vertices:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")

        # Straßen-Faces
        print(f"  Schreibe {len(road_faces)} Straßen-Faces...")
        f.write("\nusemtl road_surface\n")
        for face in road_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        # Böschungs-Faces
        print(f"  Schreibe {len(slope_faces)} Böschungs-Faces...")
        f.write("\nusemtl road_slope\n")
        for face in slope_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        # Terrain-Faces
        print(f"  Schreibe {len(terrain_faces)} Terrain-Faces...")
        f.write("\nusemtl terrain\n")
        for face in terrain_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"  ✓ {filename} erfolgreich erstellt!")
    print(f"  ✓ {mtl_filename} erfolgreich erstellt!")


def main():
    global LOCAL_OFFSET, BBOX
    LOCAL_OFFSET = None  # Reset bei jedem Durchlauf

    # 1. Lade Höhendaten aus lokalen Dateien
    print("=" * 60)
    print("WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator")
    print("=" * 60)

    height_points, height_elevations = load_height_data()

    # 2. Berechne BBOX aus Höhendaten
    print("\nBerechne BBOX aus Höhendaten...")
    BBOX = calculate_bbox_from_height_data(height_points)

    # 3. Prüfe ob OSM-Daten neu geladen werden müssen
    height_hash = get_height_data_hash()
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
    if not osm_elements:
        print("Keine Daten gefunden.")
        return

    # 5. Straßen aus den OSM-Daten extrahieren
    roads = extract_roads_from_osm(osm_elements)
    if not roads:
        print("Keine Straßen gefunden.")
        return

    # 6. Erstelle Terrain-Grid aus Höhendaten
    grid_points, grid_elevations, nx, ny = create_terrain_grid(
        height_points, height_elevations, grid_spacing=10.0
    )

    # 7. Extrahiere Straßen-Polygone
    print(f"\nExtrahiere {len(roads)} Straßen-Polygone...")
    road_polygons = get_road_polygons(roads, BBOX, height_points, height_elevations)
    print(f"  ✓ {len(road_polygons)} Straßen-Polygone extrahiert")

    # 8. Klassifiziere Grid-Vertices und integriere Straßen
    vertex_types, modified_heights = classify_grid_vertices(
        grid_points, grid_elevations, road_polygons
    )

    # 9. Generiere vereinheitlichtes Terrain-Mesh
    vertices, road_faces, slope_faces, terrain_faces = generate_terrain_mesh(
        grid_points, modified_heights, vertex_types, nx, ny
    )

    # 10. Speichern als OBJ-Datei
    output_obj = "terrain.obj"
    save_unified_obj(output_obj, vertices, road_faces, slope_faces, terrain_faces)

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
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
