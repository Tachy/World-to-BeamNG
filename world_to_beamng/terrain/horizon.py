"""
Horizon Layer - Generiert niederauflösendes Horizont-Mesh aus DGM30 und Sentinel-2.

Pipeline:
1. Lade DGM30-Daten (30m Auflösung) für ±50km um Kerngebiet
   - Option A: Manuell heruntergeladene XYZ/ZIP Dateien
   - Option B: Automatischer Download via OpenTopography API
2. Lade Sentinel-2 RGB Satellitenbilder
3. Generiere 1km Horizon-Grid (1000m Zellgröße)
4. Texturiere mit Sentinel-2 RGB
5. Export als DAE mit Materials
"""

import os
import glob
import numpy as np
from pathlib import Path
import zipfile
import io
from PIL import Image
import json

from .. import config


def download_dgm30_from_opentopography(local_offset, tile_hash=None):
    """
    Lädt DGM30 (Copernicus 30m) automatisch via OpenTopography API.

    BBOX: 100km × 100km, zentriert um LOCAL_OFFSET
    - West:  LOCAL_OFFSET[0] - 50km
    - East:  LOCAL_OFFSET[0] + 50km
    - South: LOCAL_OFFSET[1] - 50km
    - North: LOCAL_OFFSET[1] + 50km

    Args:
        local_offset: (ox, oy, oz) in UTM Koordinaten
        tile_hash: Optional - Hash für Cache

    Returns:
        Tuple (height_points, height_elevations) oder (None, None) – immer in lokalen Koordinaten, falls local_offset gesetzt wurde
    """
    if not config.OPENTOPOGRAPHY_ENABLED:
        return None, None

    if not config.OPENTOPOGRAPHY_API_KEY or config.OPENTOPOGRAPHY_API_KEY == "YOUR_API_KEY_HERE":
        print("  [!] OpenTopography API-Key nicht konfiguriert")
        return None, None

    try:
        from bmi_topography import Topography
    except ImportError:
        print("  [!] bmi_topography nicht installiert. Install: pip install bmi-topography")
        return None, None

    ox, oy, oz = local_offset

    # Konvertiere UTM zu lat/lon
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

    # BBOX: ±50km um Origin
    utm_west = ox - 50000
    utm_east = ox + 50000
    utm_south = oy - 50000
    utm_north = oy + 50000

    # Konvertiere zu lat/lon
    lon_west, lat_south = transformer.transform(utm_west, utm_south)
    lon_east, lat_north = transformer.transform(utm_east, utm_north)

    print(f"  [i] Lade DGM30 via OpenTopography API")
    print(f"      BBOX (lat/lon): N={lat_north:.4f} S={lat_south:.4f} E={lon_east:.4f} W={lon_west:.4f}")

    # Erstelle Zielverzeichnis
    os.makedirs(config.DGM30_DATA_DIR, exist_ok=True)

    try:
        # Definiere Parameter mit explizitem Cache-Verzeichnis
        import tempfile

        cache_dir = config.DGM30_DATA_DIR

        params = {
            "dem_type": "COP30",
            "south": lat_south,
            "north": lat_north,
            "west": lon_west,
            "east": lon_east,
            "output_format": "GTiff",
            "api_key": config.OPENTOPOGRAPHY_API_KEY,
            "cache_dir": cache_dir,
        }

        # Download starten
        print(f"  [i] Starte Download nach {cache_dir}...")
        topo = Topography(**params)
        filepath = topo.fetch()

        if not filepath or not os.path.exists(filepath):
            print(f"  [!] Download fehlgeschlagen")
            return None, None

        # Verschiebe nach data/DGM30 falls nötig
        target_path = os.path.join(config.DGM30_DATA_DIR, "dgm30_copernicus.tif")
        if filepath != target_path:
            import shutil

            shutil.move(filepath, target_path)
            filepath = target_path

        print(f"  [OK] DGM30 heruntergeladen: {os.path.basename(filepath)}")

        # Konvertiere GeoTIFF zu 200m Grid
        return _load_geotiff_as_xyz(filepath, tile_hash, local_offset=local_offset)

    except Exception as e:
        print(f"  [!] OpenTopography API Fehler: {e}")
        return None, None


def _load_geotiff_as_xyz(geotiff_path, tile_hash=None, local_offset=None):
    """
    Konvertiert GeoTIFF zu XYZ Format (Koordinaten + Höhenwerte).

    Samplet 30m Auflösung auf 200m Grid herunter für schnellere Verarbeitung.

    Args:
        geotiff_path: Pfad zum GeoTIFF
        tile_hash: Optional - Hash für Cache
        local_offset: Optional (ox, oy, oz) – konvertiert Punkte/Höhen direkt in lokale Koordinaten

    Returns:
        Tuple (height_points, height_elevations)
    """
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError:
        print("  [!] rasterio nicht installiert. Install: pip install rasterio")
        return None, None

    try:
        with rasterio.open(geotiff_path) as src:
            # Prüfe CRS und reprojiziere falls nötig
            src_crs = src.crs

            # Ziel: UTM Zone 32N (EPSG:25832)
            dst_crs = "EPSG:25832"

            # Wenn Quell-CRS nicht UTM ist, reprojiziere
            if src_crs and src_crs.to_string() != dst_crs:
                print(f"  [i] Reprojiziere von {src_crs.to_string()} zu {dst_crs}")

                from rasterio.warp import calculate_default_transform, reproject, Resampling

                # Berechne neue Transform und Dimensionen
                transform, width, height = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds
                )

                # Erstelle temporäres Array für reprojizierte Daten
                dem_data = np.empty((height, width), dtype=np.float32)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=dem_data,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

                # Berechne neue Bounds in UTM
                from rasterio.transform import array_bounds

                bounds = array_bounds(height, width, transform)
                x_min, y_min, x_max, y_max = bounds[0], bounds[1], bounds[2], bounds[3]

            else:
                # Bereits UTM
                dem_data = src.read(1).astype(np.float32)
                transform = src.transform
                bounds = src.bounds
                x_min, y_min, x_max, y_max = bounds.left, bounds.bottom, bounds.right, bounds.top

            rows, cols = dem_data.shape

            # DEBUG: Zeige Bounds
            print(f"  [DEBUG] UTM Bounds: X=[{x_min:.2f}..{x_max:.2f}], Y=[{y_min:.2f}..{y_max:.2f}]")
            print(f"  [DEBUG] Breite: {x_max - x_min:.2f}m, Höhe: {y_max - y_min:.2f}m")

            # Erstelle 200m Grid
            grid_spacing = 200.0
            x_coords = np.arange(x_min, x_max + grid_spacing * 0.5, grid_spacing)
            y_coords = np.arange(y_min, y_max + grid_spacing * 0.5, grid_spacing)

            print(f"  [i] Sample {rows}×{cols} GeoTIFF (30m) auf {len(x_coords)}×{len(y_coords)} Grid (200m)")

            height_points = []
            height_elevations = []

            # Sample auf 200m Grid
            for y in y_coords:
                for x in x_coords:
                    # Konvertiere UTM zurück zu Pixel-Koordinaten
                    col, row = ~transform * (x, y)
                    col_int, row_int = int(col), int(row)

                    # Prüfe ob innerhalb Bounds
                    if 0 <= row_int < rows and 0 <= col_int < cols:
                        z = dem_data[row_int, col_int]

                        # Ignoriere NoData-Werte
                        if not np.isnan(z) and not np.isinf(z):
                            height_points.append([x, y])
                            height_elevations.append(z)

            if not height_points:
                print("  [!] Keine gültigen Höhenpunkte im GeoTIFF")
                return None, None

            height_points = np.array(height_points)
            height_elevations = np.array(height_elevations)

            print(f"  [OK] {len(height_elevations)} Höhenpunkte (200m Grid) aus GeoTIFF geladen")

            if local_offset is not None:
                # Speichere direkt in lokalen Koordinaten, damit spätere Verbraucher nichts mehr umrechnen müssen
                ox, oy, oz = local_offset
                height_points = height_points - np.array([ox, oy])
                height_elevations = height_elevations - oz

            # Cache speichern
            if tile_hash:
                cache_file = os.path.join(config.CACHE_DIR, f"dgm30_horizon_{tile_hash}.npz")
                os.makedirs(config.CACHE_DIR, exist_ok=True)
                np.savez_compressed(cache_file, points=height_points, elevations=height_elevations)
                print(f"  [OK] DGM30-Cache erstellt: {os.path.basename(cache_file)}")

            return height_points, height_elevations

    except Exception as e:
        print(f"  [!] Fehler beim Laden des GeoTIFF: {e}")
        return None, None


def load_dgm30_tiles(dgm30_dir, bbox_utm, local_offset=None, tile_hash=None):
    """
    Lädt DGM30-Höhendaten aus XYZ-Dateien oder ZIPs.

    Fallback-Strategie:
    1. Prüfe Cache
    2. Lade lokal gespeicherte Dateien (XYZ/ZIP)
    3. Falls OPENTOPOGRAPHY_ENABLED: Download via API

    DGM30 ist ähnlich strukturiert wie DGM1, aber mit 30m Auflösung statt 1m.

    Args:
        dgm30_dir: Verzeichnis mit DGM30 Dateien (z.B. data/DGM30/)
        bbox_utm: (min_x, max_x, min_y, max_y) in UTM Metern
        local_offset: (ox, oy, oz) Optional - für OpenTopography API
        tile_hash: Optional - Hash für Cache-Konsistenz

    Returns:
        Tuple (height_points, height_elevations) oder (None, None)
    """
    # Prüfe Cache zuerst (wir gehen davon aus, dass er bereits lokale Koordinaten enthält)
    if tile_hash:
        cache_file = os.path.join(config.CACHE_DIR, f"dgm30_horizon_{tile_hash}.npz")
        if os.path.exists(cache_file):
            print(f"  [OK] DGM30-Cache gefunden: {os.path.basename(cache_file)} (bereits lokal)")
            data = np.load(cache_file)
            return data["points"], data["elevations"]

    dgm30_path = Path(dgm30_dir)

    # Versuche lokal gespeicherte Dateien zu laden
    if dgm30_path.exists():
        height_points, height_elevations = _load_local_dgm30(dgm30_path, tile_hash, local_offset=local_offset)
        if height_points is not None:
            return height_points, height_elevations

    # Fallback: Lade via OpenTopography API
    if local_offset is not None:
        height_points, height_elevations = download_dgm30_from_opentopography(local_offset, tile_hash)
        if height_points is not None:
            return height_points, height_elevations

    print(f"  [!] DGM30 nicht gefunden und OpenTopography nicht aktiviert")
    return None, None


def _load_local_dgm30(dgm30_path, tile_hash=None, local_offset=None):
    """
    Lädt DGM30 aus lokal gespeicherten GeoTIFF Dateien.

    Args:
        dgm30_path: Path Objekt zum Verzeichnis
        tile_hash: Optional - Hash für Cache
        local_offset: Optional – speichere direkt in lokale Koordinaten

    Returns:
        Tuple (height_points, height_elevations) oder (None, None)
    """
    # Suche GeoTIFF Dateien
    tif_files = list(dgm30_path.glob("*.tif")) + list(dgm30_path.glob("*.tiff"))

    if not tif_files:
        print(f"  [i] Keine GeoTIFF Dateien in {dgm30_path} gefunden")
        return None, None

    print(f"  [i] Lade {len(tif_files)} GeoTIFF-Datei(en)...")

    # Lade erstes GeoTIFF (mehrere werden kombiniert)
    all_points = []
    all_elevations = []

    for tif_file in tif_files:
        print(f"    - {tif_file.name}")
        points, elevations = _load_geotiff_as_xyz(str(tif_file), tile_hash=None, local_offset=local_offset)

        if points is not None:
            all_points.append(points)
            all_elevations.append(elevations)

    if not all_points:
        print(f"  [!] Keine DGM30-Daten aus GeoTIFF geladen")
        return None, None

    # Kombiniere alle Daten
    height_points = np.vstack(all_points) if len(all_points) > 1 else all_points[0]
    height_elevations = np.concatenate(all_elevations) if len(all_elevations) > 1 else all_elevations[0]

    print(f"  [OK] {len(height_elevations)} Punkte (200m Grid) aus {len(tif_files)} GeoTIFF(s) geladen")

    # Cache speichern
    if tile_hash:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(config.CACHE_DIR, f"dgm30_horizon_{tile_hash}.npz")
        np.savez_compressed(cache_file, points=height_points, elevations=height_elevations)
        print(f"  [OK] DGM30-Cache erstellt: {os.path.basename(cache_file)}")

    return height_points, height_elevations


def load_sentinel2_geotiff(sentinel2_dir, bbox_utm, tile_hash=None):
    """
    Lädt Sentinel-2 RGB GeoTIFF Dateien mit Georeferenzierung.

    GeoTIFF muss georeferenziert sein (mit Metadaten für Koordinaten-Transformation).

    Args:
        sentinel2_dir: Verzeichnis mit Sentinel-2 GeoTIFF (z.B. data/DOP300/)
        bbox_utm: (min_x, max_x, min_y, max_y) in UTM Metern
        tile_hash: Optional - Hash für Cache

    Returns:
        Tuple (image_array, bounds_utm, transform) oder None
        image_array: (H, W, 3) numpy array RGB [0-255]
        bounds_utm: (x_min, y_min, x_max, y_max) in UTM
        transform: rasterio Affine Transform
    """
    try:
        import rasterio
    except ImportError:
        print("  [!] rasterio nicht installiert. Install: pip install rasterio")
        return None

    sentinel2_path = Path(sentinel2_dir)

    if not sentinel2_path.exists():
        print(f"  [i] Sentinel-2 Verzeichnis nicht gefunden: {sentinel2_dir}")
        return None

    # Suche GeoTIFF Dateien
    tif_files = list(sentinel2_path.glob("*.tif")) + list(sentinel2_path.glob("*.tiff"))

    if not tif_files:
        print(f"  [i] Keine GeoTIFF Dateien in {sentinel2_dir} gefunden")
        return None

    print(f"  [i] Lade {len(tif_files)} Sentinel-2 GeoTIFF Dateien...")

    # Lade erstes GeoTIFF mit Georeferenzierung
    tif_file = tif_files[0]

    try:
        with rasterio.open(tif_file) as src:
            # Lese RGB Bänder (Band 1, 2, 3)
            if src.count >= 3:
                rgb_data = np.dstack([src.read(i) for i in range(1, 4)])
            elif src.count == 1:
                # Grayscale zu RGB
                band = src.read(1)
                rgb_data = np.dstack([band, band, band])
            else:
                print(f"    [!] Unerwartete Band-Anzahl: {src.count}")
                return None

            # Extrahiere Metadaten
            transform = src.transform
            bounds = src.bounds
            bounds_utm = (bounds.left, bounds.bottom, bounds.right, bounds.top)

            print(f"    - {tif_file.name}: {src.width}×{src.height} ({src.crs})")
            print(
                f"      UTM Bounds: X=[{bounds.left:.0f}..{bounds.right:.0f}], Y=[{bounds.bottom:.0f}..{bounds.top:.0f}]"
            )
            print(f"      Breite: {bounds.right - bounds.left:.0f}m, Höhe: {bounds.top - bounds.bottom:.0f}m")

            # Normalisiere auf 0-255 falls nötig
            if rgb_data.max() > 255:
                rgb_data = (rgb_data / rgb_data.max() * 255).astype(np.uint8)
            else:
                rgb_data = rgb_data.astype(np.uint8)

            print(f"  [OK] Sentinel-2 geladen: {rgb_data.shape}")

            return rgb_data, bounds_utm, transform

    except Exception as e:
        print(f"    [!] Fehler beim Laden: {e}")
        return None


def generate_horizon_mesh(height_points, height_elevations, local_offset):
    """
    Generiert Mesh aus 200m Horizont-Grid (einfache Triangulation).

    Args:
        height_points: (N, 2) Grid-Punkte bereits in lokalen Koordinaten
        height_elevations: (N,) Höhenwerte in lokalen Koordinaten
        local_offset: (ox, oy, oz) Transformation – hier nur noch für Konsistenz/Logging genutzt

    Returns:
        Tuple (vertices, faces, nx, ny)
        vertices: (M, 3) Array mit XYZ-Koordinaten
        faces: List von [v0, v1, v2] Indizes
        nx, ny: Grid-Dimensionen (für Texturierung)
    """
    _ = local_offset  # behalten für Aufrufer-Signatur; Daten sind bereits lokal

    # Punkte und Höhen liegen bereits lokal vor
    local_points = height_points
    local_elevations = height_elevations

    # Erstelle reguläres Grid aus unregelmäßigen Punkten
    # Bestimme Grid-Dimensionen (200m Spacing)
    x_min, x_max = local_points[:, 0].min(), local_points[:, 0].max()
    y_min, y_max = local_points[:, 1].min(), local_points[:, 1].max()

    grid_spacing = 200.0
    x_coords = np.arange(x_min, x_max + grid_spacing * 0.5, grid_spacing)
    y_coords = np.arange(y_min, y_max + grid_spacing * 0.5, grid_spacing)

    nx = len(x_coords)
    ny = len(y_coords)

    print(f"  [i] Erstelle Horizont-Mesh: {nx}×{ny} Grid")

    # Erstelle Grid mit Nearest-Neighbor-Interpolation
    from scipy.spatial import cKDTree

    tree = cKDTree(local_points)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points_flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Finde nächste Punkte
    distances, indices = tree.query(grid_points_flat)
    grid_elevations = local_elevations[indices]

    # Erstelle 3D Vertices - Horizont 150m0m unter Z-Level für Kern-Mesh Separation
    vertices = np.column_stack([grid_points_flat, grid_elevations - 50.0])

    # Generiere Faces (Triangulation)
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            # Indizes der 4 Ecken einer Zelle
            v0 = y * nx + x
            v1 = y * nx + (x + 1)
            v2 = (y + 1) * nx + x
            v3 = (y + 1) * nx + (x + 1)

            # Zwei Dreiecke pro Zelle
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    print(f"  [OK] {len(faces)} Dreiecke generiert")

    return vertices, faces, nx, ny


def get_terrain_tile_bounds(items_json_path=None, beamng_dir=None):
    """
    Liest main.items.json und extrahiert die Bounds aller Terrain-Tiles aus Item-Namen.

    Format der Item-Namen: "terrain_X_Y" wobei X, Y die lokalen Koordinaten der Tile-Ursprünge sind.
    Jedes Terrain-Tile ist 2000×2000m (2×2 km).
    Beispiel: "terrain_-1000_-1000" → Bounds X=[-1000, 1000], Y=[-1000, 1000]

    Args:
        items_json_path: Pfad zu main.items.json (wird geprüft zuerst)
        beamng_dir: BeamNG Verzeichnis (Fallback für items_json_path)

    Returns:
        List von Tuples [(x_min, y_min, x_max, y_max), ...] in lokalen Koordinaten
        oder leere Liste, wenn Datei nicht existiert/lesbar
    """
    # Versuche items_json_path direkt
    if items_json_path is None and beamng_dir is not None:
        items_json_path = os.path.join(beamng_dir, "main.items.json")

    tile_bounds = []
    tile_size = 2000  # Jeder Terrain-Tile ist 2000×2000m (2×2 km)
    collected_tiles = set()

    # Lese main.items.json
    if items_json_path and os.path.exists(items_json_path):
        try:
            with open(items_json_path, "r", encoding="utf-8") as f:
                items = json.load(f)

            for item_name in items.keys():
                # Suche nach terrain_X_Y Einträgen (nicht terrain_tile_!)
                if not item_name.startswith("terrain_"):
                    continue

                # terrain_tile_... sind Einträge der Tile-Slices, nicht die 2x2km Tiles
                if "tile_" in item_name:
                    continue

                # Parse Koordinaten aus "terrain_X_Y"
                # Format: "terrain_-1000_-1000" oder "terrain_0_500"
                parts = item_name.split("_")
                if len(parts) < 3:
                    continue

                try:
                    # Lese X und Y aus den letzten zwei Teilen
                    coord_x = int(parts[-2])
                    coord_y = int(parts[-1])
                    collected_tiles.add((coord_x, coord_y))
                except ValueError:
                    continue

        except Exception as e:
            print(f"  [!] Fehler beim Lesen von main.items.json: {e}")

    # Konvertiere collected_tiles zu Bounds (Koordinaten sind X_min, Y_min des Tiles)
    # Jedes Terrain-Tile ist 2000×2000m
    for coord_x, coord_y in collected_tiles:
        x_min = coord_x
        x_max = coord_x + tile_size
        y_min = coord_y
        y_max = coord_y + tile_size
        tile_bounds.append((x_min, y_min, x_max, y_max))

    if tile_bounds:
        print(f"  [OK] {len(tile_bounds)} Terrain-Tile-Bereiche aus main.items.json gelesen")
        for idx, (x_min, y_min, x_max, y_max) in enumerate(tile_bounds):
            print(f"      Tile {idx}: X=[{x_min}..{x_max}], Y=[{y_min}..{y_max}]")
    else:
        print(f"  [i] Keine Tile-Bounds in main.items.json gefunden")

    return tile_bounds


def clip_horizon_mesh_to_tiles(vertices, faces, tile_bounds):
    """
    Entfernt alle Faces aus dem Horizont-Mesh, die vollständig in einem Tile-Bereich liegen.

    Ein Face wird entfernt, wenn **alle 3 Vertices** vollständig innerhalb mindestens eines Tile-Bounds liegen.

    Args:
        vertices: (M, 3) Array mit Vertex-Koordinaten (lokal)
        faces: List von [v0, v1, v2] Face-Indizes
        tile_bounds: List von (x_min, y_min, x_max, y_max) Tuples

    Returns:
        Tuple (filtered_vertices, filtered_faces, removed_count)
    """
    if not tile_bounds:
        print(f"  [i] Keine Tile-Bounds zum Clipping vorhanden")
        return vertices, faces, 0

    print(f"  [i] Clippe Horizont-Mesh gegen {len(tile_bounds)} Tile-Bereiche...")

    def vertex_in_tile(vertex, tile_bounds):
        """Prüft ob Vertex komplett in mindestens einem Tile liegt."""
        x, y = vertex[0], vertex[1]
        for x_min, y_min, x_max, y_max in tile_bounds:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    # Filtere Faces: Behalte nur Faces, bei denen mindestens ein Vertex NICHT in einem Tile liegt
    original_face_count = len(faces)
    filtered_faces = []

    for face in faces:
        v0, v1, v2 = face
        v0_in = vertex_in_tile(vertices[v0], tile_bounds)
        v1_in = vertex_in_tile(vertices[v1], tile_bounds)
        v2_in = vertex_in_tile(vertices[v2], tile_bounds)

        # Entferne Face nur wenn **alle 3 Vertices** in Tiles liegen
        if not (v0_in and v1_in and v2_in):
            filtered_faces.append(face)

    removed_count = original_face_count - len(filtered_faces)
    print(f"  [OK] {removed_count} Faces entfernt (aus {original_face_count}), {len(filtered_faces)} behalten")

    return vertices, filtered_faces, removed_count


def texture_horizon_mesh(vertices, horizon_image, nx, ny, bounds_utm, transform, global_offset):
    """
    Mappt Sentinel-2 RGB Textur auf Horizon-Mesh mit korrekter Georeferenzierung.

    Args:
        vertices: (M, 3) Mesh-Vertices in lokalen Koordinaten
        horizon_image: (H, W, 3) RGB-Array
        nx, ny: Grid-Dimensionen
        bounds_utm: (x_min, y_min, x_max, y_max) Texture Bounds in UTM
        transform: rasterio Affine Transform
        global_offset: (ox, oy, oz) für Rück-Konvertierung lokal → UTM

    Returns:
        Dict mit Textur-Informationen
    """
    if horizon_image is None:
        return {"texture_path": None, "uv_map": None}

    os.makedirs(config.BEAMNG_DIR_TEXTURES, exist_ok=True)

    # Konvertiere Mesh-Vertices zurück zu UTM für Koordinaten-Mapping
    ox, oy, oz = global_offset
    vertices_utm = vertices.copy()
    vertices_utm[:, 0] += ox
    vertices_utm[:, 1] += oy

    # Prüfe Übereinstimmung
    mesh_x_min, mesh_x_max = vertices_utm[:, 0].min(), vertices_utm[:, 0].max()
    mesh_y_min, mesh_y_max = vertices_utm[:, 1].min(), vertices_utm[:, 1].max()

    tex_x_min, tex_y_min, tex_x_max, tex_y_max = bounds_utm

    print(f"  [i] Koordinaten-Check:")
    print(f"      Mesh (UTM):    X=[{mesh_x_min:.0f}..{mesh_x_max:.0f}], Y=[{mesh_y_min:.0f}..{mesh_y_max:.0f}]")
    print(f"      Texture (UTM): X=[{tex_x_min:.0f}..{tex_x_max:.0f}], Y=[{tex_y_min:.0f}..{tex_y_max:.0f}]")

    # Berechne Überlappung
    overlap_x = (min(mesh_x_max, tex_x_max) - max(mesh_x_min, tex_x_min)) / (mesh_x_max - mesh_x_min) * 100
    overlap_y = (min(mesh_y_max, tex_y_max) - max(mesh_y_min, tex_y_min)) / (mesh_y_max - mesh_y_min) * 100

    print(f"      Überlappung: X={overlap_x:.1f}%, Y={overlap_y:.1f}%")

    # Speichere temporär als TIF für texconv
    import tempfile
    import subprocess

    temp_tif = os.path.join(tempfile.gettempdir(), "horizon_temp.tif")

    img_pil = Image.fromarray(horizon_image.astype("uint8"), "RGB")
    img_pil.save(temp_tif, "TIFF")

    # Konvertiere mit texconv.exe zu DDS (BC1, 8192x8192, Mipmaps)
    texconv_exe = "bin/texconv.exe"
    dds_output = os.path.join(config.BEAMNG_DIR_TEXTURES, "horizon_sentinel2.dds")

    if not os.path.exists(texconv_exe):
        raise FileNotFoundError(f"texconv.exe nicht gefunden: {texconv_exe}")

    # texconv Parameter:
    # -f BC1_UNORM: BC1 Kompression
    # -w 8192 -h 8192: Zielauflösung
    # -m 0: Volle Mipmap-Kette
    # -o: Output-Verzeichnis
    # -y: Überschreiben ohne Rückfrage
    cmd = [
        texconv_exe,
        "-f",
        "BC1_UNORM",
        "-w",
        "8192",
        "-h",
        "8192",
        "-m",
        "0",
        "-y",
        "-o",
        config.BEAMNG_DIR_TEXTURES,
        temp_tif,
    ]

    print(f"  [i] Konvertiere zu DDS (BC1, 8192x8192, Mipmaps)...")
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    # texconv benennt Output nach Input: horizon_temp.dds -> umbenennen
    texconv_output = os.path.join(config.BEAMNG_DIR_TEXTURES, "horizon_temp.dds")
    if os.path.exists(texconv_output):
        if os.path.exists(dds_output):
            os.remove(dds_output)
        os.rename(texconv_output, dds_output)

    # Aufräumen
    if os.path.exists(temp_tif):
        os.remove(temp_tif)

    print(f"  [OK] Horizont-Textur (DDS) gespeichert: {dds_output}")

    # Relative Pfade für materials.json
    relative_texture_path = config.RELATIVE_DIR_TEXTURES + "horizon_sentinel2.dds"

    return {
        "texture_path": relative_texture_path,
        "image_size": horizon_image.shape,
        "bounds_utm": bounds_utm,
        "mesh_coverage": (overlap_x, overlap_y),
    }


def export_horizon_dae(vertices, faces, texture_info, output_dir, level_name="default", global_offset=None):
    """
    Exportiert Horizon-Mesh als DAE (Collada) Datei mit UV-Mapping.

    Clippt automatisch Horizont-Mesh gegen Terrain-Tile-Bereiche aus main.items.json.

    Args:
        vertices: (M, 3) Mesh-Vertices in lokalen Koordinaten
        faces: List von [v0, v1, v2] Face-Indizes
        texture_info: Dict mit Textur-Informationen (bounds_utm, mesh_coverage)
        output_dir: Zielverzeichnis (BeamNG Level Verzeichnis)
        level_name: Name des Levels
        global_offset: (ox, oy, oz) für UTM-Konvertierung

    Returns:
        Pfad zur erzeugten DAE-Datei
    """
    # ===== Clipping: Entferne Horizont-Faces in Tile-Bereichen =====
    tile_bounds = get_terrain_tile_bounds(beamng_dir=output_dir)

    if tile_bounds:
        vertices, faces, removed = clip_horizon_mesh_to_tiles(vertices, faces, tile_bounds)
        if removed > 0:
            print(f"  [OK] Clipping abgeschlossen: {removed} Faces entfernt")
    else:
        print(f"  [i] Kein Clipping: Keine Tile-Bounds gefunden")

    dae_path = os.path.join(output_dir, "art", "shapes", "terrain_horizon.dae")
    os.makedirs(os.path.dirname(dae_path), exist_ok=True)

    # Berechne UV-Offsets basierend auf Koordinaten-Mismatch
    bounds_utm = texture_info.get("bounds_utm", None)

    if bounds_utm and global_offset:
        ox, oy, oz = global_offset
        tex_x_min, tex_y_min, tex_x_max, tex_y_max = bounds_utm

        # Mesh-Bounds in UTM
        mesh_x_min, mesh_x_max = vertices[:, 0].min() + ox, vertices[:, 0].max() + ox
        mesh_y_min, mesh_y_max = vertices[:, 1].min() + oy, vertices[:, 1].max() + oy

        # Offsets in UTM-Metern
        offset_x_m = mesh_x_min - tex_x_min
        offset_y_m = mesh_y_min - tex_y_min

        # Texture Größe: 100km × 100km für ±50km
        tex_width_m = tex_x_max - tex_x_min
        tex_height_m = tex_y_max - tex_y_min

        # UV-Offset (normalisiert auf 0..1)
        uv_offset_x = offset_x_m / tex_width_m
        uv_offset_y = offset_y_m / tex_height_m

        # Mesh-Größe in lokalen Koordinaten
        mesh_width_m = vertices[:, 0].max() - vertices[:, 0].min()
        mesh_height_m = vertices[:, 1].max() - vertices[:, 1].min()

        # UV-Skalierung (Mesh-Größe zu Texture-Größe)
        uv_scale_x = mesh_width_m / tex_width_m
        uv_scale_y = mesh_height_m / tex_height_m

        print(f"  [i] UV-Mapping mit Offset:")
        print(f"      UV-Offset: ({uv_offset_x:.4f}, {uv_offset_y:.4f})")
        print(f"      UV-Skalierung: ({uv_scale_x:.4f}, {uv_scale_y:.4f})")
    else:
        uv_offset_x, uv_offset_y = 0.0, 0.0
        uv_scale_x, uv_scale_y = 1.0, 1.0

    # Schreibe DAE direkt als formatiertes XML (nicht mit ElementTree um großen Text zu vermeiden)
    with open(dae_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<COLLADA version="1.4.1" xmlns="http://www.collada.org/2005/11/COLLADASchema">\n')

        # Asset
        f.write("  <asset>\n")
        f.write("    <created>2025-01-07T00:00:00</created>\n")
        f.write("    <modified>2025-01-07T00:00:00</modified>\n")
        f.write("  </asset>\n")

        # Library Geometries
        f.write("  <library_geometries>\n")
        f.write('    <geometry id="horizon_mesh" name="horizon">\n')
        f.write("      <mesh>\n")

        # === Vertices Source ===
        f.write('        <source id="horizon_vertices">\n')
        f.write(f'          <float_array id="horizon_vertices_array" count="{len(vertices) * 3}">')

        # Schreibe Vertices mit minimalem Overhead - jeden Float auf eigene Zeile
        for vertex in vertices:
            f.write(f"\n{vertex[0]:.2f} {vertex[1]:.2f} {vertex[2]:.2f}")

        f.write("\n          </float_array>\n")
        f.write("          <technique_common>\n")
        f.write(f'            <accessor source="#horizon_vertices_array" count="{len(vertices)}" stride="3">\n')
        f.write('              <param name="X" type="float"/>\n')
        f.write('              <param name="Y" type="float"/>\n')
        f.write('              <param name="Z" type="float"/>\n')
        f.write("            </accessor>\n")
        f.write("          </technique_common>\n")
        f.write("        </source>\n")

        # === UV Coordinates Source ===
        mesh_x_min, mesh_x_max = vertices[:, 0].min(), vertices[:, 0].max()
        mesh_y_min, mesh_y_max = vertices[:, 1].min(), vertices[:, 1].max()

        uv_coords = []
        for vertex in vertices:
            uv_x = (vertex[0] - mesh_x_min) / (mesh_x_max - mesh_x_min) if mesh_x_max > mesh_x_min else 0.0
            uv_y = (vertex[1] - mesh_y_min) / (mesh_y_max - mesh_y_min) if mesh_y_max > mesh_y_min else 0.0

            uv_x = uv_offset_x + uv_x * uv_scale_x
            uv_y = uv_offset_y + uv_y * uv_scale_y

            uv_coords.append((uv_x, uv_y))

        f.write('        <source id="horizon_uvs">\n')
        f.write(f'          <float_array id="horizon_uvs_array" count="{len(uv_coords) * 2}">')

        # Schreibe UV-Koordinaten - minimal one Paar pro Zeile
        for u, v in uv_coords:
            f.write(f"\n{u:.6f} {v:.6f}")

        f.write("\n          </float_array>\n")
        f.write("          <technique_common>\n")
        f.write(f'            <accessor source="#horizon_uvs_array" count="{len(uv_coords)}" stride="2">\n')
        f.write('              <param name="S" type="float"/>\n')
        f.write('              <param name="T" type="float"/>\n')
        f.write("            </accessor>\n")
        f.write("          </technique_common>\n")
        f.write("        </source>\n")
        f.write('        <vertices id="horizon_vertices_input">\n')
        f.write('          <input semantic="POSITION" source="#horizon_vertices_array"/>\n')
        f.write("        </vertices>\n")

        # === Triangles ===
        f.write(f'        <triangles material="horizon_terrain" count="{len(faces)}">\n')
        f.write('          <input semantic="VERTEX" source="#horizon_vertices_input" offset="0"/>\n')
        # TEXCOORD verweist auf den <source id="horizon_uvs"> (nicht auf das float_array)
        f.write('          <input semantic="TEXCOORD" source="#horizon_uvs" offset="1" set="0"/>\n')
        f.write("          <p>")

        # Schreibe Face-Indizes mit separaten Offsets: v0 uv0 v1 uv1 v2 uv2
        for face in faces:
            f.write(f"\n{face[0]} {face[0]} {face[1]} {face[1]} {face[2]} {face[2]}")

        f.write("\n          </p>\n")
        f.write("        </triangles>\n")

        # Close mesh, geometry
        f.write("      </mesh>\n")
        f.write("    </geometry>\n")
        f.write("  </library_geometries>\n")

        # === Library Visual Scenes ===
        f.write("  <library_visual_scenes>\n")
        f.write('    <visual_scene id="Scene" name="Scene">\n')
        f.write('      <node id="Horizon" name="Horizon" type="NODE">\n')
        f.write('        <instance_geometry url="#horizon_mesh">\n')
        f.write("          <bind_material>\n")
        f.write("            <technique_common>\n")
        f.write('              <instance_material symbol="horizon_terrain" target="#horizon_terrain_material"/>\n')
        f.write("            </technique_common>\n")
        f.write("          </bind_material>\n")
        f.write("        </instance_geometry>\n")
        f.write("      </node>\n")
        f.write("    </visual_scene>\n")
        f.write("  </library_visual_scenes>\n")

        # === Scene ===
        f.write("  <scene>\n")
        f.write('    <instance_visual_scene url="#Scene"/>\n')
        f.write("  </scene>\n")

        f.write("</COLLADA>\n")

    print(f"  [OK] DAE exportiert mit UV-Mapping: {os.path.basename(dae_path)}")

    # Überprüfe ob Datei existiert
    print(f"  [DEBUG] Prüfe Dateiexistenz: {dae_path}")
    if os.path.exists(dae_path):
        file_size = os.path.getsize(dae_path)
        print(f"      Dateigröße: {file_size:,} Bytes")
    else:
        print(f"      [WARNING] Datei existiert nicht!")

    return os.path.basename(dae_path)
