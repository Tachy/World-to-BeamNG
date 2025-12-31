"""
Hoehendaten-Verwaltung (Laden, Caching, Interpolation).
"""

import os
import glob
import hashlib
import zipfile
import io
import json
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from .. import config


def get_height_data_hash():
    """Erstellt einen Hash basierend auf den Dateien im height-data Ordner."""
    xyz_files = sorted(glob.glob(os.path.join(config.HEIGHT_DATA_DIR, "*.xyz")))
    zip_files = sorted(glob.glob(os.path.join(config.HEIGHT_DATA_DIR, "*.zip")))
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
    """Lädt alle Hoehendaten aus .xyz oder .zip Dateien (mit Caching)."""
    print("\nLade Hoehendaten...")

    # Pruefe ob gecachte Rohdaten existieren
    height_hash = get_height_data_hash()
    cache_file = None

    if height_hash:
        cache_file = os.path.join(config.CACHE_DIR, f"height_raw_{height_hash}.npz")

        if os.path.exists(cache_file):
            print(f"  [OK] Cache gefunden: {os.path.basename(cache_file)}")
            data = np.load(cache_file)
            points = data["points"]
            elevations = data["elevations"]
            print(f"  [OK] {len(elevations)} Hoehenpunkte aus Cache geladen")
            return points, elevations
        else:
            print(f"  Cache nicht gefunden, lade aus Dateien...")

    # Lade aus Dateien
    xyz_files = glob.glob(os.path.join(config.HEIGHT_DATA_DIR, "*.xyz"))
    zip_files = glob.glob(os.path.join(config.HEIGHT_DATA_DIR, "*.zip"))

    if not xyz_files and not zip_files:
        raise FileNotFoundError(
            f"Keine .xyz oder .zip Dateien in {config.HEIGHT_DATA_DIR} gefunden!"
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

    print(f"  [OK] {len(elevations)} Hoehenpunkte geladen")

    # Cache die Rohdaten (immer wenn wir frisch geladen haben)
    if height_hash:
        cache_file_path = os.path.join(
            config.CACHE_DIR, f"height_raw_{height_hash}.npz"
        )
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_file_path, points=points, elevations=elevations)
        print(f"  [OK] Cache erstellt: {os.path.basename(cache_file_path)}")

    return points, elevations


def get_elevation_cache(bbox):
    """Lädt den Elevation-Cache fuer eine BBox (Koordinate -> Hoehe)."""
    from ..io.cache import get_cache_path

    cache_path = get_cache_path(bbox, "elevations")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                print(f"  [OK] Elevation-Cache geladen: {len(cache_data)} Koordinaten")
                return cache_data
        except:
            pass
    return {}


def save_elevation_cache(bbox, cache_data):
    """Speichert den Elevation-Cache."""
    from ..io.cache import get_cache_path

    cache_path = get_cache_path(bbox, "elevations")
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        print(f"  [OK] Elevation-Cache gespeichert: {len(cache_data)} Koordinaten")
    except Exception as e:
        print(f"  [!] Fehler beim Speichern des Elevation-Cache: {e}")


def get_elevations_for_points(pts, bbox, height_points, height_elevations):
    """Holt Hoehendaten fuer Koordinaten - aus Cache oder durch Interpolation aus lokalen Daten."""
    from ..geometry.coordinates import transformer_to_utm
    from scipy.interpolate import griddata

    # Lade bestehenden Cache
    elevation_cache = get_elevation_cache(bbox)

    # Finde fehlende Koordinaten
    missing_pts = []
    missing_indices = []

    for idx, pt in enumerate(pts):
        # Erstelle eindeutigen Key fuer Koordinate (gerundet auf 6 Dezimalstellen)
        coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
        if coord_key not in elevation_cache:
            missing_pts.append(pt)
            missing_indices.append(idx)

    # Berechne fehlende Hoehen durch Interpolation
    if missing_pts:
        print(f"  Interpoliere {len(missing_pts)} Hoehenwerte...")

        # Konvertiere WGS84 zu UTM und dann zu lokal
        from .. import config

        missing_pts_local = []
        for pt in missing_pts:
            x, y = transformer_to_utm.transform(pt[1], pt[0])  # lon, lat -> x, y
            # Transformiere zu lokalen Koordinaten
            if config.LOCAL_OFFSET is not None:
                ox, oy, oz = config.LOCAL_OFFSET
                x -= ox
                y -= oy
            missing_pts_local.append([x, y])

        missing_pts_local = np.array(missing_pts_local)

        # Interpoliere Hoehen (nearest neighbor fuer schnellere Berechnung)
        new_elevations = griddata(
            height_points, height_elevations, missing_pts_local, method="nearest"
        )

        # Fuege zum Cache hinzu
        for pt, elev in zip(missing_pts, new_elevations):
            coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
            elevation_cache[coord_key] = float(elev)

        # Speichere aktualisierten Cache
        save_elevation_cache(bbox, elevation_cache)

    # Erstelle Elevation-Array fuer alle Punkte
    elevations = []
    for pt in pts:
        coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
        elevations.append(elevation_cache.get(coord_key, 0))

    return elevations
