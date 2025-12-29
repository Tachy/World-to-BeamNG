"""
Polygon-Operationen und Straßen-Extraktion.
"""

import numpy as np
from shapely.geometry import Polygon

from ..terrain.elevation import get_elevations_for_points
from ..geometry.coordinates import transformer_to_utm
from .. import config


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


def get_road_centerline_robust(road_poly):
    """
    Berechne die Mittellinie eines Straßen-Polygons mittels PCA und Mittelwertsbildung.

    Args:
        road_poly: Shapely Polygon der Straße

    Returns:
        centerline: (N, 2) NumPy Array mit Mittellinie-Koordinaten
    """
    coords = np.array(road_poly.exterior.coords[:-1])  # ohne Wiederholung des Endpunkts

    if len(coords) < 4:
        return coords

    # Berechne Schwerpunkt
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    # PCA: Finde Hauptrichtung (längste Achse = Straßenrichtung)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Hauptrichtung (Eigenvector mit größtem Eigenwert)
    main_direction = eigvecs[:, -1]

    # Senkrechte Richtung
    perp_direction = np.array([-main_direction[1], main_direction[0]])

    # Projiziere alle Punkte auf beide Richtungen
    proj_along = centered @ main_direction  # Entlang der Straße
    proj_perp = centered @ perp_direction  # Quer zur Straße (linke/rechte Seite)

    # Teile Punkte in zwei Seiten: links und rechts des Mittels
    median_perp = np.median(proj_perp)

    left_mask = proj_perp <= median_perp
    right_mask = proj_perp > median_perp

    left_indices = np.where(left_mask)[0]
    right_indices = np.where(right_mask)[0]

    if len(left_indices) < 2 or len(right_indices) < 2:
        return coords  # Fallback wenn Teilung nicht funktioniert

    # Sortiere beide Seiten nach der Längsprojektion (entlang der Straße)
    left_indices = left_indices[np.argsort(proj_along[left_indices])]
    right_indices = right_indices[np.argsort(proj_along[right_indices])]

    # Interpoliere beide Seiten auf die gleiche Anzahl von Punkten
    # So können wir sie direkt miteinander vergleichen
    n_samples = max(len(left_indices), len(right_indices))

    # Interpoliere linke Seite
    left_coords = coords[left_indices]
    left_interp_x = np.interp(
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, len(left_indices)),
        left_coords[:, 0],
    )
    left_interp_y = np.interp(
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, len(left_indices)),
        left_coords[:, 1],
    )

    # Interpoliere rechte Seite
    right_coords = coords[right_indices]
    right_interp_x = np.interp(
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, len(right_indices)),
        right_coords[:, 0],
    )
    right_interp_y = np.interp(
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, len(right_indices)),
        right_coords[:, 1],
    )

    # Berechne Mittelpunkte zwischen linker und rechter Seite
    centerline_x = (left_interp_x + right_interp_x) / 2
    centerline_y = (left_interp_y + right_interp_y) / 2

    centerline = np.column_stack([centerline_x, centerline_y])
    return centerline
