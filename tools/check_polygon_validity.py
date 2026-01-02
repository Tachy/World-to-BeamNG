"""
Prüfe ob ein Road-Polygon gültig ist (für Shapely)

Simuliert die Polygon-Erstellung aus road_mesh.py und prüft Validität
"""

import sys
import json
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity

DEBUG_JUNCTIONS_PATH = "C:/Users/johan/AppData/Local/BeamNG.drive/0.36/levels/World_to_BeamNG/art/shapes/debug_junctions.json"


def create_road_polygon_simple(centerline, width=7.0):
    """Erstelle Road-Polygon wie in road_mesh.py (vereinfacht)."""

    half_width = width / 2.0

    # Erstelle linke und rechte Kante
    left_points = []
    right_points = []

    for i in range(len(centerline)):
        x, y = centerline[i][0], centerline[i][1]

        # Berechne Richtung (vereinfacht: zum nächsten Punkt)
        if i < len(centerline) - 1:
            dx = centerline[i + 1][0] - x
            dy = centerline[i + 1][1] - y
        else:
            dx = x - centerline[i - 1][0]
            dy = y - centerline[i - 1][1]

        length = np.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            continue

        # Normalisiere
        dx /= length
        dy /= length

        # Perpendicular (90° gedreht)
        perp_x = -dy
        perp_y = dx

        # Links und rechts
        left_x = x + perp_x * half_width
        left_y = y + perp_y * half_width
        right_x = x - perp_x * half_width
        right_y = y - perp_y * half_width

        left_points.append([left_x, left_y])
        right_points.append([right_x, right_y])

    # Kombiniere: links vorwärts, rechts rückwärts
    polygon_points = left_points + list(reversed(right_points))

    return polygon_points


def check_polygon_validity(road_id):
    """Prüfe Polygon-Validität für eine Road-ID."""

    # Lade Roads
    with open(DEBUG_JUNCTIONS_PATH, "r") as f:
        data = json.load(f)

    roads = data["roads"]

    # Finde Road
    target_road = None
    for road in roads:
        if road.get("id") == road_id:
            target_road = road
            break

    if not target_road:
        print(f"[!] Road-ID {road_id} nicht gefunden")
        return

    coords = target_road["coords"]
    centerline = [(c[0], c[1]) for c in coords]

    print(f"\n{'='*70}")
    print(f"Polygon-Validitäts-Check für Road-ID: {road_id}")
    print(f"{'='*70}")
    print(f"Centerline: {len(centerline)} Punkte")

    # Erstelle Polygon
    poly_points = create_road_polygon_simple(centerline, width=7.0)
    print(
        f"Polygon: {len(poly_points)} Punkte ({len(centerline)} links + {len(centerline)} rechts)"
    )

    # Prüfe mit Shapely
    try:
        poly = Polygon(poly_points)
        is_valid = poly.is_valid

        print(f"\nShapely-Validierung:")
        print(f"  is_valid: {is_valid}")

        if not is_valid:
            reason = explain_validity(poly)
            print(f"  Grund: {reason}")
            print(
                f"\n  [!] UNGÜLTIGES POLYGON - wird von classify_grid_vertices ignoriert!"
            )
        else:
            print(f"  [OK] Polygon ist gültig")

            # Prüfe ob Punkt in Polygon liegt
            from shapely.geometry import Point

            test_point = Point(2788.0, 3744.0)
            contains = poly.contains(test_point)
            print(f"\n  Enthält Punkt (2788.0, 3744.0): {contains}")

            if not contains:
                print(
                    f"  [!] Punkt liegt NICHT im Polygon - überprüfe Polygon-Konstruktion"
                )

        # Zusätzliche Checks
        print(f"\nPolygon-Eigenschaften:")
        print(f"  Fläche: {poly.area:.2f} m²")
        print(f"  Umfang: {poly.length:.2f} m")

        # Prüfe auf Self-Intersections
        if not poly.is_simple:
            print(f"  [!] Self-Intersections gefunden (is_simple=False)")

    except Exception as e:
        print(f"\n[!] Fehler beim Erstellen des Polygons: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: python check_polygon_validity.py <Road-ID>")
        print("Beispiel:   python check_polygon_validity.py 50822566002")
        sys.exit(1)

    road_id = int(sys.argv[1])
    check_polygon_validity(road_id)
