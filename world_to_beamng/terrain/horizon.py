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


def generate_horizon_mesh(height_points, height_elevations, local_offset, tile_bounds=None, vertex_manager=None):
    """
    Generiert Horizont-Mesh mit SEPARATEM VertexManager (saubere Architektur).

    ARCHITEKTUR:
    - IMMER separater VM (vertex_manager Parameter wird IGNORIERT für Sauberness)
    - Stitching arbeitet NUR mit dem Horizon-VM
    - UVs werden NACH dem Stitching generiert (in horizon_workflow.py)
    - Rückgabe: (mesh, nx, ny, horizon_vertex_indices, global_to_horizon_map)
      - horizon_vertex_indices: Alle Horizon-Vertices (im separaten VM)
      - global_to_horizon_map: IMMER None (nur separater VM)

    OPTIMIERUNGEN:
    - Vektorisierte Batch-VertexManager-Einfügung
    - Effiziente Grid-Indizierung
    - KEINE UV-Berechnung (kommt später!)
    - Keine redundanten Lookups
    - Quads über Terrain-Tiles filtern (optional)

    Args:
        height_points: (N, 2) Grid-Punkte bereits in lokalen Koordinaten
        height_elevations: (N,) Höhenwerte in lokalen Koordinaten
        local_offset: (ox, oy, oz) Transformation – hier nur noch für Konsistenz/Logging genutzt
        tile_bounds: Optional - Liste von (x_min, y_min, x_max, y_max) Tuples in lokalen Koordinaten
                     zum Überspringen von Quads die über Terrain liegen
        vertex_manager: Optional - Bestehender VertexManager (z.B. vom Terrain für Stitching)
                        Falls None, wird ein SEPARATER erstellt (EMPFOHLEN)

    Returns:
        Tuple (mesh, nx, ny, horizon_vertex_indices, global_to_horizon_map)
        mesh: Mesh-Objekt mit VertexManager + deduplizierten UVs
        nx, ny: Grid-Dimensionen (für Texturierung)
        horizon_vertex_indices: Vertex-Indizes im VertexManager
        global_to_horizon_map: Dict {global_index → local_index} (nur wenn shared VM gegeben)
    """
    from ..mesh.vertex_manager import VertexManager
    from ..mesh.mesh import Mesh

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
    if vertex_manager is None:
        print(f"      Mit SEPARATEM VertexManager (saubere Architektur)")
    else:
        print(f"      Mit GEMEINSAMEN VertexManager (für Boundary-Stitching)")

    # Erstelle Grid mit Nearest-Neighbor-Interpolation
    from scipy.spatial import cKDTree

    tree = cKDTree(local_points)

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points_flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Finde nächste Punkte
    distances, indices = tree.query(grid_points_flat)
    grid_elevations = local_elevations[indices]

    # Erstelle 3D Vertices - Horizont 50m unter Z-Level für Kern-Mesh Separation
    vertices = np.column_stack([grid_points_flat, grid_elevations])

    # === ARCHITEKTUR: IMMER Separater VertexManager ===
    # vertex_manager Parameter wird für Sauberness ignoriert
    # Boundary-Stitching arbeitet mit Horizon-VM + kopiert Terrain-Ring in Horizon-VM
    vm = VertexManager(tolerance=0.001)
    global_to_horizon_map = None  # Immer None (nur separater VM)

    mesh = Mesh(vm)

    # === OPTIMIERUNG: Batch Vertex-Einfügung (OHNE Hash-Lookup!) ===
    # Für reguläre Grids: Alle Vertices sind unterschiedlich, kein Dedup nötig!
    # Nutze add_vertices_direct_nohash() statt einzelner add_vertex() Calls (258k Aufrufe!)
    vertex_indices = np.array(vm.add_vertices_direct_nohash(vertices), dtype=int)

    # Reshape vertex_indices in (ny, nx) Grid für einfachen Zugriff
    vertex_grid = vertex_indices.reshape(ny, nx)

    # === OPTIMIERUNG 3: Vektorisierte Quad-Filterung über Tile-Bounds ===
    quads_mask = np.ones((ny - 1, nx - 1), dtype=bool)

    if tile_bounds:
        print(f"  [i] Filtere {len(tile_bounds)} Terrain-Tiles (2x2 km) mit vektorisiertem Lookup...")

        import time

        t0 = time.time()

        # Pre-compute alle 4 Vertex-Positionen für jedes Quad (vektorisiert)
        # Quad (y, x) hat Vertices bei:
        #   v0 = (x_coords[x], y_coords[y])       - unten links
        #   v1 = (x_coords[x+1], y_coords[y])     - unten rechts
        #   v2 = (x_coords[x], y_coords[y+1])     - oben links
        #   v3 = (x_coords[x+1], y_coords[y+1])   - oben rechts

        # Erstelle Meshgrids für alle 4 Ecken
        x_left = x_coords[:-1]
        x_right = x_coords[1:]
        y_bottom = y_coords[:-1]
        y_top = y_coords[1:]

        # Prüfe für jeden Tile ob mindestens ein Vertex darin liegt
        for tile_x_min, tile_y_min, tile_x_max, tile_y_max in tile_bounds:
            # Für jedes Quad: Prüfe ob IRGENDEIN Vertex im Tile liegt
            # Vertex liegt im Tile wenn: tile_x_min <= x < tile_x_max AND tile_y_min <= y < tile_y_max

            # Prüfe alle 4 Vertices (vektorisiert über alle Quads)
            # v0 (unten links): (x_left, y_bottom)
            v0_inside = ((x_left >= tile_x_min) & (x_left < tile_x_max))[:, None] & (
                (y_bottom >= tile_y_min) & (y_bottom < tile_y_max)
            )[None, :]

            # v1 (unten rechts): (x_right, y_bottom)
            v1_inside = ((x_right >= tile_x_min) & (x_right < tile_x_max))[:, None] & (
                (y_bottom >= tile_y_min) & (y_bottom < tile_y_max)
            )[None, :]

            # v2 (oben links): (x_left, y_top)
            v2_inside = ((x_left >= tile_x_min) & (x_left < tile_x_max))[:, None] & (
                (y_top >= tile_y_min) & (y_top < tile_y_max)
            )[None, :]

            # v3 (oben rechts): (x_right, y_top)
            v3_inside = ((x_right >= tile_x_min) & (x_right < tile_x_max))[:, None] & (
                (y_top >= tile_y_min) & (y_top < tile_y_max)
            )[None, :]

            # Quad entfernen wenn MINDESTENS EIN Vertex im Tile liegt
            tile_mask = v0_inside | v1_inside | v2_inside | v3_inside
            quads_mask &= ~tile_mask.T  # Transpose weil Meshgrid (x, y) statt (y, x)

        skipped_count = np.sum(~quads_mask)
        if skipped_count > 0:
            print(f"  [OK] {skipped_count} Quads über Terrain gefiltert ({time.time() - t0:.2f}s)")

    # === OPTIMIERUNG 4: Batch-Insert direkter Arrays (KEINE Deduplizierung nötig) ===
    # Speichere Faces & UVs direkt ohne add_face() Overhead

    valid_quads = np.argwhere(quads_mask)  # (N, 2) Array mit (y, x) Indizes

    if len(valid_quads) == 0:
        print("  [!] Keine Quads zu generieren (alle gefiltert)")
        return mesh, nx, ny

    num_quads = len(valid_quads)

    # Erstelle Face-Arrays vektorisiert
    y_indices = valid_quads[:, 0]
    x_indices = valid_quads[:, 1]

    # Vertex-Indizes für alle Quads auf einmal
    v0 = vertex_grid[y_indices, x_indices]
    v1 = vertex_grid[y_indices, x_indices + 1]
    v2 = vertex_grid[y_indices + 1, x_indices]
    v3 = vertex_grid[y_indices + 1, x_indices + 1]

    # Erstelle zwei Dreiecke pro Quad direkt
    faces_tri1 = np.column_stack([v0, v1, v2])
    faces_tri2 = np.column_stack([v1, v3, v2])

    # Kombiniere und speichere direkt (KEINE add_face Loops!)
    faces_array = np.vstack([faces_tri1, faces_tri2]).astype(int)
    mesh.faces = list(map(tuple, faces_array))

    # === UVs werden NICHT hier generiert! ===
    # Sie werden in horizon_workflow.py NACH dem Stitching generiert (für alle Vertices zusammen)
    mesh.uvs = []
    mesh.uv_indices = {}

    face_count = len(mesh.faces)

    print(f"  [OK] {face_count} Dreiecke generiert")
    print(f"  [OK] {len(mesh.uvs)} UVs (1 pro Vertex, ohne Deduplizierung)")

    # Gebe Mesh, Grid-Dimensionen, Horizon-Vertex-Indizes UND Mapping zurück
    # vertex_indices ist flaches Array - konvertiere zu Liste
    horizon_vertex_indices = vertex_indices.tolist()

    return mesh, nx, ny, horizon_vertex_indices, global_to_horizon_map


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


def export_horizon_dae(mesh, texture_info, output_dir, level_name="default", global_offset=None, tile_bounds=None):
    """
    Exportiert Horizon-Mesh als DAE (Collada) Datei mit deduplizierten UVs.

    Tile-Bounds werden bereits während Mesh-Generierung gefiltert (siehe generate_horizon_mesh).
    Diese Funktion ist rein für DAE-Export zuständig.

    Args:
        mesh: Mesh-Objekt mit zentralem VertexManager + deduplizierten UVs (bereits gefiltert)
        texture_info: Dict mit Textur-Informationen (bounds_utm, mesh_coverage)
        output_dir: Zielverzeichnis (BeamNG Level Verzeichnis)
        level_name: Name des Levels
        global_offset: (ox, oy, oz) für UTM-Konvertierung
        tile_bounds: (UNBENUTZT - nur für API-Kompatibilität, Filterung erfolgt in generate_horizon_mesh)

    Returns:
        Pfad zur erzeugten DAE-Datei
    """
    # tile_bounds wird hier nicht mehr benötigt - Filterung erfolgt bereits im Mesh-Generieren!
    _ = tile_bounds  # Unbenutzt - Filterung erfolgt in generate_horizon_mesh()

    dae_path = os.path.join(output_dir, "art", "shapes", "terrain_horizon.dae")
    os.makedirs(os.path.dirname(dae_path), exist_ok=True)

    # Berechne UV-Offsets basierend auf Koordinaten-Mismatch
    bounds_utm = texture_info.get("bounds_utm", None)
    vertices = mesh.vertex_manager.vertices
    faces = mesh.faces  # Extrahiere Faces aus dem Mesh

    print(f"  [i] DAE-Export: {len(vertices)} Vertices, {len(faces)} Faces")

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

    # Skaliere deduplizierte UVs VEKTORISIERT mit Offset und Skalierung
    uvs_array = np.array(mesh.uvs, dtype=np.float32)
    scaled_uvs_array = uvs_array.copy()
    scaled_uvs_array[:, 0] = uv_offset_x + uvs_array[:, 0] * uv_scale_x
    scaled_uvs_array[:, 1] = uv_offset_y + uvs_array[:, 1] * uv_scale_y

    # Konvertiere zu Liste für Kompatibilität
    scaled_uvs = [(u, v) for u, v in scaled_uvs_array]

    # Schreibe DAE mit StringIO Buffer (schneller als direkte File-I/O)
    from io import StringIO

    buffer = StringIO()

    # Schreibe alles in Buffer (viel schneller als direkt in Datei)
    f = buffer
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<COLLADA version="1.4.1" xmlns="http://www.collada.org/2005/11/COLLADASchema">\n')

    # Asset
    f.write("  <asset>\n")
    f.write("    <created>2025-01-07T00:00:00</created>\n")
    f.write("    <modified>2025-01-07T00:00:00</modified>\n")
    f.write("  </asset>\n")

    # === Library Materials ===
    f.write("  <library_materials>\n")
    f.write('    <material id="horizon_terrain" name="horizon_terrain">\n')
    f.write('      <instance_effect url="#horizon_terrain_effect"/>\n')
    f.write("    </material>\n")
    f.write("  </library_materials>\n")

    # === Library Effects ===
    f.write("  <library_effects>\n")
    f.write('    <effect id="horizon_terrain_effect">\n')
    f.write("      <profile_COMMON>\n")
    f.write('        <technique sid="common">\n')
    if texture_info and texture_info.get("texture_path"):
        # Mit Textur
        f.write("          <phong>\n")
        f.write("            <diffuse>\n")
        f.write("              <color>1.0 1.0 1.0 1.0</color>\n")
        f.write("            </diffuse>\n")
        f.write("            <shininess>\n")
        f.write("              <float>1.0</float>\n")
        f.write("            </shininess>\n")
        f.write("          </phong>\n")
    else:
        # Ohne Textur - Fallback-Farbe
        f.write("          <phong>\n")
        f.write("            <diffuse>\n")
        f.write("              <color>0.8 0.8 0.8 1.0</color>\n")
        f.write("            </diffuse>\n")
        f.write("            <shininess>\n")
        f.write("              <float>1.0</float>\n")
        f.write("            </shininess>\n")
        f.write("          </phong>\n")
    f.write("        </technique>\n")
    f.write("      </profile_COMMON>\n")
    f.write("    </effect>\n")
    f.write("  </library_effects>\n")

    # Library Geometries
    f.write("  <library_geometries>\n")
    f.write('    <geometry id="horizon_mesh" name="horizon">\n')
    f.write("      <mesh>\n")

    # === Vertices Source ===
    f.write('        <source id="horizon_vertices">\n')
    f.write(f'          <float_array id="horizon_vertices_array" count="{len(vertices) * 3}">')

    # Schreibe Vertices mit minimalem Overhead
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

    # === Normals Source (für BeamNG-Kompatibilität) ===
    # Berechne Smooth Normals VEKTORISIERT aus den Faces
    normals = np.zeros((len(vertices), 3), dtype=np.float32)

    # Konvertiere Faces zu Numpy Array für vektorisierte Verarbeitung
    faces_array = np.array(faces, dtype=np.int32)

    # Extrahiere alle Vertices für alle Faces auf einmal
    v0_all = vertices[faces_array[:, 0]]
    v1_all = vertices[faces_array[:, 1]]
    v2_all = vertices[faces_array[:, 2]]

    # Berechne Edges vektorisiert
    edge1_all = v1_all - v0_all
    edge2_all = v2_all - v0_all

    # Berechne Face-Normals vektorisiert
    face_normals = np.cross(edge1_all, edge2_all)
    face_normals_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(
        face_normals, face_normals_len, out=np.zeros_like(face_normals), where=face_normals_len > 0
    )

    # Akkumuliere Face-Normals zu Vertex-Normals
    for i, face in enumerate(faces_array):
        normals[face[0]] += face_normals[i]
        normals[face[1]] += face_normals[i]
        normals[face[2]] += face_normals[i]

    # Normalisiere Vertex-Normals vektorisiert
    normals_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, normals_len, out=np.tile([0.0, 0.0, 1.0], (len(normals), 1)), where=normals_len > 0)

    f.write('        <source id="horizon_normals">\n')
    f.write(f'          <float_array id="horizon_normals_array" count="{len(normals) * 3}">')
    # Schreibe Normals
    for normal in normals:
        f.write(f"\n{normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
    f.write("\n          </float_array>\n")
    f.write("          <technique_common>\n")
    f.write(f'            <accessor source="#horizon_normals_array" count="{len(vertices)}" stride="3">\n')
    f.write('              <param name="X" type="float"/>\n')
    f.write('              <param name="Y" type="float"/>\n')
    f.write('              <param name="Z" type="float"/>\n')
    f.write("            </accessor>\n")
    f.write("          </technique_common>\n")
    f.write("        </source>\n")

    # === UV Coordinates Source (dedupliziert aus mesh.uvs) ===
    f.write('        <source id="horizon_uvs">\n')
    f.write(f'          <float_array id="horizon_uvs_array" count="{len(scaled_uvs) * 2}">')

    # Schreibe deduplizierte UV-Koordinaten
    for u, v in scaled_uvs:
        f.write(f"\n{u:.6f} {v:.6f}")

    f.write("\n          </float_array>\n")
    f.write("          <technique_common>\n")
    f.write(f'            <accessor source="#horizon_uvs_array" count="{len(scaled_uvs)}" stride="2">\n')
    f.write('              <param name="S" type="float"/>\n')
    f.write('              <param name="T" type="float"/>\n')
    f.write("            </accessor>\n")
    f.write("          </technique_common>\n")
    f.write("        </source>\n")
    f.write('        <vertices id="horizon_vertices_input">\n')
    f.write('          <input semantic="POSITION" source="#horizon_vertices"/>\n')
    f.write("        </vertices>\n")

    # === Triangles (mit deduplizierten UV-Indizes und Normals) ===
    f.write(f'        <triangles material="horizon_terrain" count="{len(faces)}">\n')
    f.write('          <input semantic="VERTEX" source="#horizon_vertices_input" offset="0"/>\n')
    f.write('          <input semantic="NORMAL" source="#horizon_normals" offset="1"/>\n')
    f.write('          <input semantic="TEXCOORD" source="#horizon_uvs" offset="2" set="0"/>\n')
    f.write("          <p>")

    # Schreibe Face-Indizes mit Normals + deduplizierten UV-Indizes aus mesh.uv_indices
    # Format: v0 n0 uv0 v1 n1 uv1 v2 n2 uv2
    for face_idx, face in enumerate(faces):
        if face_idx in mesh.uv_indices:
            uv_indices = mesh.uv_indices[face_idx]
            f.write(
                f"\n{face[0]} {face[0]} {uv_indices[0]} {face[1]} {face[1]} {uv_indices[1]} {face[2]} {face[2]} {uv_indices[2]}"
            )
        else:
            # Fallback: Nutze Vertex-Indizes als UV-Indizes (sollte nicht vorkommen)
            f.write(f"\n{face[0]} {face[0]} {face[0]} {face[1]} {face[1]} {face[1]} {face[2]} {face[2]} {face[2]}")

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
    f.write('              <instance_material symbol="horizon_terrain" target="#horizon_terrain"/>\n')
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

    # Schreibe Buffer-Inhalt auf einmal in Datei (viel schneller)
    with open(dae_path, "w", encoding="utf-8") as file:
        file.write(buffer.getvalue())
    buffer.close()

    print(f"  [OK] DAE exportiert mit deduplizierten UVs: {os.path.basename(dae_path)}")
    print(f"  [OK] UV-Statistik: {len(mesh.uvs)} deduplizierte UVs, {len(mesh.faces)} Faces")

    # Überprüfe ob Datei existiert
    if os.path.exists(dae_path):
        file_size = os.path.getsize(dae_path)
        print(f"      Dateigröße: {file_size:,} Bytes")
    else:
        print(f"      [WARNING] Datei existiert nicht!")

    return os.path.basename(dae_path)
