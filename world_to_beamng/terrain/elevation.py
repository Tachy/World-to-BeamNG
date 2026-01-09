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
    """Erstellt einen Hash basierend auf den Dateien im data/DGM1 Ordner.

    Falls height_data_hash.txt fehlt oder unterschiedlich ist, werden alle alten
    Cache-Dateien gelöscht (erzwingt Neugenerierung).
    """
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

    new_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

    # Prüfe ob height_data_hash.txt existiert und einen ANDEREN Hash enthält
    hash_file = os.path.join(config.CACHE_DIR, "height_data_hash.txt")
    old_hash = None

    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                old_hash = f.read().strip()
        except:
            pass

    # Wenn Hash sich geändert hat oder Datei fehlt: Cleanup
    if old_hash != new_hash:
        if old_hash is None:
            print(f"  [i] height_data_hash.txt fehlt - loesche alte Cache-Dateien...")
        else:
            print(f"  [i] Hoehendaten geaendert ({old_hash} -> {new_hash}) - loesche alte Cache-Dateien...")

        # Lösche alle alten Cache-Dateien (wenn old_hash bekannt ist)
        if old_hash:
            for pattern in [
                f"height_raw_{old_hash}.npz",
                f"grid_v3_{old_hash}_*.npz",
                f"osm_all_{old_hash}.json",
                f"elevations_{old_hash}.json",
            ]:
                glob_pattern = os.path.join(config.CACHE_DIR, pattern)
                for old_file in glob.glob(glob_pattern):
                    try:
                        os.remove(old_file)
                        print(f"    • Geloescht: {os.path.basename(old_file)}")
                    except Exception as e:
                        print(f"    [!] Fehler beim Loeschen von {os.path.basename(old_file)}: {e}")
        else:
            # Wenn old_hash leer/None: Lösche ALLE potentiellen alten Caches (Sicherheitsmaßnahme)
            print(f"    Loeschen aller _*.npz und _*.json Cache-Dateien...")
            for pattern in ["height_raw_*.npz", "grid_v3_*.npz", "osm_all_*.json", "elevations_*.json"]:
                glob_pattern = os.path.join(config.CACHE_DIR, pattern)
                for old_file in glob.glob(glob_pattern):
                    try:
                        os.remove(old_file)
                        print(f"    • Geloescht: {os.path.basename(old_file)}")
                    except Exception as e:
                        print(f"    [!] Fehler beim Loeschen von {os.path.basename(old_file)}: {e}")

        # Lösche auch die generierten DAE-Tiles im BeamNG-Verzeichnis
        print(f"    Loeschen von Terrain-Tiles im BeamNG-Verzeichnis...")
        beamng_shapes = config.BEAMNG_DIR_SHAPES
        if os.path.exists(beamng_shapes):
            for file in glob.glob(os.path.join(beamng_shapes, "*.dae")):
                try:
                    os.remove(file)
                    print(f"    • Geloescht: {os.path.basename(file)}")
                except Exception as e:
                    print(f"    [!] Fehler beim Loeschen von {os.path.basename(file)}: {e}")
            # Lösche auch DAE-Index-Datei falls vorhanden
            for meta_file in ["index.json", "manifest.json"]:
                meta_path = os.path.join(beamng_shapes, meta_file)
                if os.path.exists(meta_path):
                    try:
                        os.remove(meta_path)
                        print(f"    • Geloescht: {os.path.basename(meta_path)}")
                    except Exception as e:
                        print(f"    [!] Fehler beim Loeschen von {os.path.basename(meta_path)}: {e}")

        # Lösche auch Texture-Tiles
        print(f"    Loeschen von Texture-Tiles im BeamNG-Verzeichnis...")
        beamng_textures = config.BEAMNG_DIR_TEXTURES
        if os.path.exists(beamng_textures):
            for file in glob.glob(os.path.join(beamng_textures, "tile*")):
                try:
                    os.remove(file)
                    print(f"    • Geloescht: {os.path.basename(file)}")
                except Exception as e:
                    print(f"    [!] Fehler beim Loeschen von {os.path.basename(file)}: {e}")

        # Speichere neuen Hash
        try:
            os.makedirs(config.CACHE_DIR, exist_ok=True)
            with open(hash_file, "w") as f:
                f.write(new_hash)
        except:
            pass

    return new_hash


def load_height_data():
    """Lädt alle Hoehendaten aus .xyz oder .zip Dateien (mit Caching)."""

    # Pruefe ob gecachte Rohdaten existieren
    height_hash = get_height_data_hash()
    cache_file = None
    loaded_from_cache = False

    if height_hash:
        cache_file = os.path.join(config.CACHE_DIR, f"height_raw_{height_hash}.npz")

        if os.path.exists(cache_file):
            print(f"  [OK] Cache gefunden: {os.path.basename(cache_file)}")
            data = np.load(cache_file)
            points = data["points"]
            elevations = data["elevations"]
            print(f"  [OK] {len(elevations)} Hoehenpunkte aus Cache geladen")
            loaded_from_cache = True
            # Rückgabe: needs_aerial_processing=False (aus Cache geladen)
            return points, elevations, False
        else:
            print(f"  Cache nicht gefunden, lade aus Dateien...")

    # Lade aus Dateien
    xyz_files = glob.glob(os.path.join(config.HEIGHT_DATA_DIR, "*.xyz"))
    zip_files = glob.glob(os.path.join(config.HEIGHT_DATA_DIR, "*.zip"))

    if not xyz_files and not zip_files:
        raise FileNotFoundError(f"Keine .xyz oder .zip Dateien in {config.HEIGHT_DATA_DIR} gefunden!")

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
        cache_file_path = os.path.join(config.CACHE_DIR, f"height_raw_{height_hash}.npz")
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_file_path, points=points, elevations=elevations)
        print(f"  [OK] Cache erstellt: {os.path.basename(cache_file_path)}")

    # Rückgabe: (points, elevations, needs_aerial_processing)
    # needs_aerial_processing=True weil neu geladen (nicht aus Cache)
    return points, elevations, True


def get_elevation_cache(bbox, height_hash=None):
    """Lädt den Elevation-Cache fuer eine BBox (Koordinate -> Hoehe).

    Args:
        bbox: Bounding Box
        height_hash: Optional - tile_hash für Cache-Konsistenz
    """
    from ..io.cache import get_cache_path

    # Verwende übergebenes height_hash oder fallback auf config (wenn vorhanden)
    effective_hash = height_hash or (config.HEIGHT_HASH if hasattr(config, "HEIGHT_HASH") else None)

    if effective_hash:
        cache_path = get_cache_path(bbox, "elevations", effective_hash)
    else:
        cache_path = get_cache_path(bbox, "elevations")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                # Cache-Version prüfen (v2 = normalisierte Z-Werte)
                if cache_data.get("_cache_version") == 2:
                    print(f"  [OK] Elevation-Cache geladen: {len(cache_data)-1} Koordinaten")
                    return cache_data
                else:
                    print(f"  [i] Alter Cache-Format erkannt, wird ignoriert")
        except:
            pass
    return {"_cache_version": 2}


def save_elevation_cache(bbox, cache_data, height_hash=None):
    """Speichert den Elevation-Cache.

    Args:
        bbox: Bounding Box
        cache_data: Cache-Daten zu speichern
        height_hash: Optional - tile_hash für Cache-Konsistenz
    """
    from ..io.cache import get_cache_path

    # Verwende übergebenes height_hash oder fallback auf config (wenn vorhanden)
    effective_hash = height_hash or (config.HEIGHT_HASH if hasattr(config, "HEIGHT_HASH") else None)

    if effective_hash:
        cache_path = get_cache_path(bbox, "elevations", effective_hash)
    else:
        cache_path = get_cache_path(bbox, "elevations")

    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        coord_count = len(cache_data) - 1  # -1 für _cache_version
        print(f"  [OK] Elevation-Cache gespeichert: {coord_count} Koordinaten")
    except Exception as e:
        print(f"  [!] Fehler beim Speichern des Elevation-Cache: {e}")


def get_elevations_for_points(pts, bbox, height_points, height_elevations, global_offset, height_hash=None):
    """Holt Hoehendaten fuer Koordinaten - aus Cache oder durch Interpolation aus lokalen Daten.

    Args:
        pts: Koordinaten (lat, lon) in WGS84
        bbox: Bounding Box
        height_points: Höhendaten-Punkte (XY) - LOKAL, bereits normalisiert!
        height_elevations: Z-Werte - LOKAL, bereits normalisiert!
        global_offset: (origin_x, origin_y) für Transformation WGS84->UTM->Lokal
        height_hash: Optional - tile_hash für Cache-Konsistenz
    """
    from ..geometry.coordinates import transformer_to_utm
    from scipy.interpolate import griddata

    # Lade bestehenden Cache
    elevation_cache = get_elevation_cache(bbox, height_hash=height_hash)

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

        # Konvertiere WGS84 zu UTM und dann zu lokal mit global_offset
        ox, oy = global_offset

        missing_pts_local = []
        for pt in missing_pts:
            x_utm, y_utm = transformer_to_utm.transform(pt[1], pt[0])  # lon, lat -> x, y
            # Transformiere zu lokalen Koordinaten
            x = x_utm - ox
            y = y_utm - oy
            missing_pts_local.append([x, y])

        missing_pts_local = np.array(missing_pts_local)

        # Interpoliere Hoehen (nearest neighbor fuer schnellere Berechnung)
        # WICHTIG: height_elevations ist BEREITS normalisiert (lokal)!
        new_elevations = griddata(height_points, height_elevations, missing_pts_local, method="nearest")

        # Fuege zum Cache hinzu
        for pt, elev in zip(missing_pts, new_elevations):
            coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
            elevation_cache[coord_key] = float(elev)

        # Speichere aktualisierten Cache (mit Version)
        save_elevation_cache(bbox, elevation_cache, height_hash=height_hash)

    # Erstelle Elevation-Array fuer alle Punkte
    elevations = []
    for pt in pts:
        coord_key = f"{pt[0]:.6f},{pt[1]:.6f}"
        elevations.append(elevation_cache.get(coord_key, 0))

    return elevations
