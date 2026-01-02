"""
Analysiere getrimmte Straßen-Koordinaten aus debug_trimmed_roads.json

Überprüft, ob die Straßen die korrekten Mindestabstände von Junctions einhalten:
- 2-Straßen-Junction: mind. 3.5m
- 3+-Straßen-Junction: mind. 8.5m
"""

import json
import math


def calculate_distance(p1, p2):
    """Berechne euklidische Distanz zwischen zwei Punkten (2D oder 3D)."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def main():
    # Lade getrimmte Straßen-Daten
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    json_path = os.path.join(project_root, "cache", "debug_trimmed_roads.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    roads = data["roads"]
    junctions = data["junctions"]

    # Erstelle Junction-Lookup (ID -> Position + Anzahl Straßen)
    junction_lookup = {}
    for idx, j in enumerate(junctions):
        num_roads = len(j["road_indices"])
        junction_lookup[idx] = {
            "position": j["position"],
            "num_roads": num_roads,
            "min_distance": 3.5 if num_roads == 2 else 8.5,
        }

    print(f"=== Analyse von {len(roads)} getrimmten Straßen ===\n")

    # Analysiere jede Straße
    warnings = []

    for road in roads:
        road_id = road.get("road_id")
        coords = road["coords"]
        num_points = road["num_points"]
        start_junc_id = road.get("junction_start_id")
        end_junc_id = road.get("junction_end_id")

        # Überprüfe Start-Junction
        if start_junc_id is not None and start_junc_id in junction_lookup:
            junc = junction_lookup[start_junc_id]
            if num_points > 0:
                first_point = coords[0]
                dist = calculate_distance(first_point, junc["position"])
                min_dist = junc["min_distance"]

                if dist < min_dist - 0.1:  # 10cm Toleranz
                    warnings.append(
                        {
                            "road_id": road_id,
                            "junction_id": start_junc_id,
                            "position": "START",
                            "distance": dist,
                            "min_required": min_dist,
                            "num_roads": junc["num_roads"],
                            "diff": min_dist - dist,
                        }
                    )

        # Überprüfe End-Junction
        if end_junc_id is not None and end_junc_id in junction_lookup:
            junc = junction_lookup[end_junc_id]
            if num_points > 0:
                last_point = coords[-1]
                dist = calculate_distance(last_point, junc["position"])
                min_dist = junc["min_distance"]

                if dist < min_dist - 0.1:  # 10cm Toleranz
                    warnings.append(
                        {
                            "road_id": road_id,
                            "junction_id": end_junc_id,
                            "position": "END",
                            "distance": dist,
                            "min_required": min_dist,
                            "num_roads": junc["num_roads"],
                            "diff": min_dist - dist,
                        }
                    )

    # Ausgabe
    if warnings:
        print(f"⚠️  {len(warnings)} Warnungen gefunden:\n")
        for w in warnings:
            print(
                f"Road {w['road_id']} ({w['position']}) -> Junction {w['junction_id']}"
            )
            print(
                f"  {w['num_roads']}-Straßen-Junction (min: {w['min_required']:.2f}m)"
            )
            print(f"  Ist-Abstand: {w['distance']:.2f}m")
            print(f"  Fehlt: {w['diff']:.2f}m ⚠️\n")
    else:
        print("✅ Alle Straßen halten die Mindestabstände ein!")

    # Spezielle Analyse für Junction Point 381 (falls vorhanden)
    print("\n=== Spezial-Check: Junction Point 381 ===")
    if 381 in junction_lookup:
        junc_381 = junction_lookup[381]
        junc_data = junctions[381]
        print(
            f"Junction 381: {junc_381['num_roads']} Straßen, min: {junc_381['min_distance']}m"
        )
        print(f"Position: {junc_381['position']}")
        print(f"Road-Indices laut Junction: {junc_data['road_indices']}")

        # Suche nach junction_start_id / junction_end_id
        roads_at_381 = [
            r
            for r in roads
            if r.get("junction_start_id") == 381 or r.get("junction_end_id") == 381
        ]
        print(
            f"\nStraßen mit junction_start_id oder junction_end_id == 381: {len(roads_at_381)}"
        )

        # Suche nach original_idx
        roads_by_original_idx = [
            r for r in roads if r.get("original_idx") in junc_data["road_indices"]
        ]
        print(
            f"Straßen mit original_idx in {junc_data['road_indices']}: {len(roads_by_original_idx)}"
        )

        if roads_by_original_idx:
            print("\nGefundene Straßen über original_idx:")
            for road in roads_by_original_idx:
                road_id = road.get("road_id")
                original_idx = road.get("original_idx")
                coords = road["coords"]
                start_junc = road.get("junction_start_id")
                end_junc = road.get("junction_end_id")

                print(f"\n  Road {road_id} (original_idx: {original_idx})")
                print(
                    f"    junction_start_id: {start_junc}, junction_end_id: {end_junc}"
                )
                print(f"    {len(coords)} Punkte")

                if len(coords) > 0:
                    # Berechne Distanzen zum ersten und letzten Punkt
                    dist_start = calculate_distance(coords[0], junc_381["position"])
                    dist_end = calculate_distance(coords[-1], junc_381["position"])
                    print(f"    Erster Punkt: {dist_start:.2f}m von JP381")
                    print(f"    Letzter Punkt: {dist_end:.2f}m von JP381")

                    # Bestimme welches Ende näher ist
                    if dist_start < dist_end:
                        status = (
                            "✅"
                            if dist_start >= junc_381["min_distance"] - 0.1
                            else "⚠️"
                        )
                        print(f"    → START an JP381: {dist_start:.2f}m {status}")
                    else:
                        status = (
                            "✅" if dist_end >= junc_381["min_distance"] - 0.1 else "⚠️"
                        )
                        print(f"    → END an JP381: {dist_end:.2f}m {status}")

        if roads_at_381:
            print("\nStraßen mit expliziter junction_start_id/end_id:")
            for road in roads_at_381:
                road_id = road.get("road_id")
                coords = road["coords"]

                if road.get("junction_start_id") == 381 and len(coords) > 0:
                    dist = calculate_distance(coords[0], junc_381["position"])
                    status = "✅" if dist >= junc_381["min_distance"] - 0.1 else "⚠️"
                    print(f"  Road {road_id} (START): {dist:.2f}m {status}")

                if road.get("junction_end_id") == 381 and len(coords) > 0:
                    dist = calculate_distance(coords[-1], junc_381["position"])
                    status = "✅" if dist >= junc_381["min_distance"] - 0.1 else "⚠️"
                    print(f"  Road {road_id} (END): {dist:.2f}m {status}")
    else:
        print("Junction 381 nicht in Daten gefunden.")


if __name__ == "__main__":
    main()
