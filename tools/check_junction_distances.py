"""
Debug-Tool: Prüfe Abstände von Straßen zu einem Junction Point
"""

import json
import numpy as np
import os

# Finde die debug_junctions.json Datei
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
debug_path = os.path.join(base_dir, "cache", "debug_junctions.json")

# Lade debug_junctions.json
with open(debug_path, "r", encoding="utf-8") as f:
    data = json.load(f)

junctions = data["junctions"]
roads = data["roads"]

# Finde Junction 381
junction_id = 381
if junction_id < len(junctions):
    junction = junctions[junction_id]
    jp_pos = np.array(junction["position"])
    road_indices = junction["road_indices"]

    print(f"=== Junction Point {junction_id} ===")
    print(f"Position: {jp_pos}")
    print(f"Anzahl verbundener Straßen: {len(road_indices)}")
    print(f"Verbundene Road-Indices: {road_indices}")
    print()

    # Für jede verbundene Straße
    for road_idx in road_indices:
        if road_idx < len(roads):
            road = roads[road_idx]
            coords = np.array(road["coords"])

            print(f"--- Road Index {road_idx} (OSM ID: {road.get('id', 'N/A')}) ---")
            print(f"Anzahl Punkte: {len(coords)}")

            if len(coords) > 0:
                # Abstand vom ersten Punkt zum Junction
                first_point = coords[0]
                dist_first = np.linalg.norm(first_point[:2] - jp_pos[:2])

                # Abstand vom letzten Punkt zum Junction
                last_point = coords[-1]
                dist_last = np.linalg.norm(last_point[:2] - jp_pos[:2])

                print(
                    f"Erster Punkt: {first_point[:2]} → Abstand zum JP: {dist_first:.2f}m"
                )
                print(
                    f"Letzter Punkt: {last_point[:2]} → Abstand zum JP: {dist_last:.2f}m"
                )
                print(f"Nächster Punkt zum JP: {min(dist_first, dist_last):.2f}m")

                # Berechne für alle Punkte
                distances = [np.linalg.norm(c[:2] - jp_pos[:2]) for c in coords]
                min_dist = min(distances)
                min_idx = distances.index(min_dist)

                print(
                    f"Minimaler Abstand (aller Punkte): {min_dist:.2f}m bei Index {min_idx}"
                )

                # Bei 7m Straßenbreite (3.5m half_width):
                # - Linke Kante: Punkt + 3.5m senkrecht
                # - Rechte Kante: Punkt - 3.5m senkrecht
                # Für min_edge_distance = 3.5m: Beide Kanten >= 3.5m vom JP
                # Das bedeutet: Centerline >= 7m vom JP (bei gerader Annäherung)

                if min_dist < 3.5:
                    print(f"⚠️  WARNUNG: Centerline ist < 3.5m vom Junction!")
                if min_dist < 7.0:
                    print(
                        f"⚠️  WARNUNG: Bei 7m Straßenbreite können Kanten < 3.5m sein!"
                    )

            print()
else:
    print(f"Junction {junction_id} nicht gefunden!")
