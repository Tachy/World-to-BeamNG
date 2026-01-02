"""
Prüfe welches Road-Polygon am nächsten zu einer Position liegt

Verwendung:
    python check_road_polygon_at_position.py <X> <Y>
"""

import sys
import numpy as np
import json
from shapely.geometry import Point, Polygon

# Pfade
DEBUG_JUNCTIONS_PATH = "C:/Users/johan/AppData/Local/BeamNG.drive/0.36/levels/World_to_BeamNG/art/shapes/debug_junctions.json"


def check_road_polygon_at_position(x, y):
    """Finde Road-Polygon an Position und prüfe Geometrie."""

    # Lade Road-Daten
    with open(DEBUG_JUNCTIONS_PATH, "r") as f:
        data = json.load(f)

    roads = data["roads"]

    print(f"\n{'='*70}")
    print(f"Suche Road-Polygon an Position ({x:.1f}, {y:.1f})")
    print(f"{'='*70}")
    print(f"Gesamt: {len(roads)} Roads geladen\n")

    point = Point(x, y)

    # Finde alle Roads, die den Punkt enthalten oder nahe sind
    containing_roads = []
    nearby_roads = []

    for idx, road in enumerate(roads):
        coords = road["coords"]
        if len(coords) < 3:
            continue

        # Erstelle Centerline
        centerline_points = [(c[0], c[1]) for c in coords]

        # Berechne minimale Distanz zur Centerline
        min_dist = float("inf")
        for i in range(len(centerline_points) - 1):
            p1 = np.array(centerline_points[i])
            p2 = np.array(centerline_points[i + 1])

            # Punkt-zu-Linie Distanz
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len < 0.001:
                continue

            point_vec = np.array([x, y]) - p1
            t = np.dot(point_vec, line_vec) / (line_len * line_len)
            t = max(0, min(1, t))

            closest = p1 + t * line_vec
            dist = np.linalg.norm(np.array([x, y]) - closest)
            min_dist = min(min_dist, dist)

        # Bei 7m Straßenbreite: Wenn Distanz < 3.5m, dann liegt Punkt auf Straße
        if min_dist < 3.5:
            containing_roads.append((idx, road, min_dist))
        elif min_dist < 20:
            nearby_roads.append((idx, road, min_dist))

    if not containing_roads and not nearby_roads:
        print(f"[!] Keine Roads in 20m Umkreis gefunden")
        return

    if containing_roads:
        print(f"Roads die Position enthalten sollten (Distanz < 3.5m):")
        for idx, road, dist in sorted(containing_roads, key=lambda x: x[2]):
            coords = road["coords"]
            print(f"\n  Road-ID: {road.get('id', idx)}")
            print(f"    Distanz zur Centerline: {dist:.2f}m")
            print(f"    Anzahl Koordinaten: {len(coords)}")

            # Zeige erste und letzte Koordinate
            if coords:
                print(
                    f"    Start: ({coords[0][0]:.2f}, {coords[0][1]:.2f}, {coords[0][2]:.2f})"
                )
                print(
                    f"    Ende:  ({coords[-1][0]:.2f}, {coords[-1][1]:.2f}, {coords[-1][2]:.2f})"
                )

            # Berechne Road-Polygon (vereinfacht: 3.5m Breite)
            centerline_2d = [(c[0], c[1]) for c in coords]

            # Prüfe ob sehr kurz
            total_length = 0
            for i in range(len(centerline_2d) - 1):
                p1 = np.array(centerline_2d[i])
                p2 = np.array(centerline_2d[i + 1])
                total_length += np.linalg.norm(p2 - p1)

            print(f"    Gesamtlänge: {total_length:.2f}m")

            if total_length < 2.0:
                print(
                    f"    [!] SEHR KURZE STRASSE (< 2m) - könnte zwischen Grid-Punkte fallen!"
                )

            # Berechne Bounding Box
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            print(
                f"    Bounding Box: X=[{min(xs):.1f}, {max(xs):.1f}], Y=[{min(ys):.1f}, {max(ys):.1f}]"
            )
            bbox_width = max(xs) - min(xs)
            bbox_height = max(ys) - min(ys)
            print(f"    Bbox Größe: {bbox_width:.1f}m × {bbox_height:.1f}m")

    if nearby_roads and not containing_roads:
        print(f"\nNahe Roads (3.5-20m Distanz):")
        for idx, road, dist in sorted(nearby_roads, key=lambda x: x[2])[:5]:
            coords = road["coords"]
            print(
                f"  Road-ID {road.get('id', idx)}: {dist:.2f}m entfernt, {len(coords)} Punkte"
            )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Verwendung: python check_road_polygon_at_position.py <X> <Y>")
        sys.exit(1)

    x = float(sys.argv[1])
    y = float(sys.argv[2])

    check_road_polygon_at_position(x, y)
