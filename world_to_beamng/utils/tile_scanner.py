"""
Multi-Tile-Scanner für LGL DGM1-Dateien.

DGM1-Dateien folgen dem Namensschema: dgm1_<easting>_<northing>.xyz.zip
Beispiel: dgm1_4658000_5394000.xyz.zip

Diese Funktion scannet das data/DGM1-Verzeichnis und extrahiert die Koordinaten.
"""

import os
import re
import numpy as np
from pathlib import Path


def scan_lgl_tiles(dgm1_dir):
    """
    Scannt das DGM1-Verzeichnis nach Tile-Dateien.

    Args:
        dgm1_dir: Pfad zum data/DGM1 Verzeichnis

    Returns:
        List[Dict]: Sortierte Liste mit Tile-Metadaten
            [{
                'filename': 'dgm1_4658000_5394000.xyz.zip',
                'easting': 4658000,
                'northing': 5394000,
                'tile_x': 4658000,  # World-Koordinate (Easting in Metern)
                'tile_y': 5394000,  # World-Koordinate (Northing in Metern)
                'tile_size': 2000,  # Standard DGM1 Kachel-Größe
                'bbox_utm': (easting, easting+2000, northing, northing+2000),
            }, ...]
    """

    if not os.path.exists(dgm1_dir):
        print(f"[WARNUNG] DGM1-Verzeichnis nicht gefunden: {dgm1_dir}")
        return []

    tiles = []
    # Pattern für LGL DGM1: dgm1_32_<grid_x>_<grid_y>_2_bw.zip (nur .zip, nicht .zip_)
    # Die Gitter-Indizes werden in UTM-Koordinaten umgerechnet
    pattern = re.compile(r"dgm1_(\d+)_(\d+)_(\d+)_\d+_bw\.zip$")

    # Scanne alle .zip Dateien im Verzeichnis
    for filename in sorted(os.listdir(dgm1_dir)):
        match = pattern.match(filename)
        if match:
            zone = int(match.group(1))  # Zone (z.B. 32)
            grid_x = int(match.group(2))  # Gitter-X (z.B. 399)
            grid_y = int(match.group(3))  # Gitter-Y (z.B. 5296)

            # DGM1 LGL: Gitter-Indizes zu UTM konvertieren
            # Zone 32: Easting = 399 * 1000 + 160000 (Basis für Zone 32)
            # Northing = 5296 * 1000 + 5000000 (Basis für UTM)
            # Aber der einfachste Weg: jeder Gitter-Index ist 1000m x 1000m
            # ABER: Jedes ZIP enthält 4 Kacheln (2x2), also 2000m x 2000m!
            tile_size = 2000

            # Berechne UTM-Koordinaten (Index * 1000, da Grid in 1km-Schritten)
            # Aber: ZIP deckt 2×2 Kacheln ab, also Grid-Index ist die untere linke Ecke
            easting = grid_x * 1000
            northing = grid_y * 1000

            tiles.append(
                {
                    "filename": filename,
                    "easting": easting,
                    "northing": northing,
                    "tile_x": easting,  # World-Koordinate
                    "tile_y": northing,  # World-Koordinate
                    "tile_size": tile_size,
                    "bbox_utm": (easting, easting + tile_size, northing, northing + tile_size),
                    "filepath": os.path.join(dgm1_dir, filename),
                }
            )

    if not tiles:
        print(f"[WARNUNG] Keine DGM1-Dateien gefunden in: {dgm1_dir}")
    else:
        print(f"[INFO] {len(tiles)} DGM1-Kacheln gefunden")
        for tile in tiles:
            print(f"  - {tile['filename']} → Easting={tile['easting']}, Northing={tile['northing']}")

    return tiles


def compute_global_bbox(tiles):
    """
    Berechnet die globale Bounding Box über alle Tiles.

    Args:
        tiles: Ergebnis von scan_lgl_tiles()

    Returns:
        Tuple: (min_x, max_x, min_y, max_y) in UTM-Koordinaten
    """
    if not tiles:
        return None

    min_x = min(t["easting"] for t in tiles)
    max_x = max(t["easting"] + t["tile_size"] for t in tiles)
    min_y = min(t["northing"] for t in tiles)
    max_y = max(t["northing"] + t["tile_size"] for t in tiles)

    return (min_x, max_x, min_y, max_y)


def compute_global_center(tiles):
    """
    Berechnet den globalen Center-Punkt über alle Tiles.

    Args:
        tiles: Ergebnis von scan_lgl_tiles()

    Returns:
        Tuple: (center_x, center_y) in UTM-Koordinaten
    """
    bbox = compute_global_bbox(tiles)
    if bbox is None:
        return (0.0, 0.0)

    min_x, max_x, min_y, max_y = bbox
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    return (center_x, center_y)
