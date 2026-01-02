"""
Grid-Conforming: Klassifiziert Terrain-Faces als "road" oder "terrain".

Einfache, schnelle Methode:
- Terrain-Grid wird normal generiert
- Jedes Face wird klassifiziert basierend auf Mittelpunkt
- Keine komplexen Schnitt-Berechnungen nötig!
"""

import numpy as np
from shapely.geometry import Point, Polygon


def adjust_grid_to_roads_simple(road_polygons_2d):
    """
    Dummy-Funktion: Gibt einfach die Input-Polygone zurück.
    Die echte Conforming-Magie passiert in classify_face_by_center()!

    Args:
        road_polygons_2d: Liste von Road-Polygonen

    Returns:
        road_polygons_2d (unverändert), empty dict
    """
    print(
        f"  [Conforming] SKIP: Nutze Face-Klassifizierung statt Grid-Anpassung (schneller!)"
    )
    return road_polygons_2d, {}


def classify_face_by_center(face_vertices, road_polygons_2d):
    """
    Klassifiziert ein Face basierend auf seinem Mittelpunkt.

    Args:
        face_vertices: Liste von Vertex-Koordinaten [(x,y), ...]
        road_polygons_2d: Liste von Road-Polygonen

    Returns:
        "road" oder "terrain"
    """
    # Berechne Face-Mittelpunkt
    face_center = np.mean(face_vertices, axis=0)

    # Prüfe ob Mittelpunkt in Road-Polygon liegt
    point = Point(face_center[0], face_center[1])

    for poly_coords in road_polygons_2d:
        try:
            poly = Polygon(poly_coords)
            if poly.contains(point) or poly.touches(point):
                return "road"
        except:
            continue

    return "terrain"
