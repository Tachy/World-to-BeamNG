"""
Finde fehlende Straßen in debug_trimmed_roads.json

Vergleicht die road_indices aus Junctions mit den tatsächlich vorhandenen
original_idx Werten in den getrimmten Straßen.
"""

import json


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

    # Sammle alle road_indices aus Junctions
    expected_road_indices = set()
    for j in junctions:
        expected_road_indices.update(j["road_indices"])

    # Sammle alle original_idx aus getrimmten Straßen
    found_road_indices = set()
    roads_without_original_idx = []

    for road in roads:
        original_idx = road.get("original_idx")
        if original_idx is not None:
            found_road_indices.add(original_idx)
        else:
            roads_without_original_idx.append(road.get("road_id", "unknown"))

    # Finde fehlende Straßen
    missing_road_indices = expected_road_indices - found_road_indices

    print(f"=== Straßen-Analyse ===")
    print(f"Erwartete road_indices (aus Junctions): {len(expected_road_indices)}")
    print(f"Gefundene road_indices (in trimmed roads): {len(found_road_indices)}")
    print(f"Fehlende road_indices: {len(missing_road_indices)}\n")

    if missing_road_indices:
        print(
            f"⚠️  Fehlende Straßen: {sorted(list(missing_road_indices))[:20]}"
        )  # Erste 20
        if len(missing_road_indices) > 20:
            print(f"   ... und {len(missing_road_indices) - 20} weitere\n")

    if roads_without_original_idx:
        print(f"\n⚠️  {len(roads_without_original_idx)} Straßen OHNE original_idx:")
        print(f"   Road IDs: {roads_without_original_idx[:10]}")
        if len(roads_without_original_idx) > 10:
            print(f"   ... und {len(roads_without_original_idx) - 10} weitere")

    # Spezielle Analyse für Straßen 1846 und 1847 (Junction 381)
    print("\n=== Spezial-Check: Straßen 1846 & 1847 (Junction 381) ===")

    for idx in [1846, 1847]:
        if idx in found_road_indices:
            # Finde die Straße
            road = next((r for r in roads if r.get("original_idx") == idx), None)
            if road:
                print(f"✅ Straße {idx} gefunden:")
                print(f"   Road ID: {road.get('road_id')}")
                print(f"   Punkte: {road.get('num_points')}")
                print(f"   Junction Start: {road.get('junction_start_id')}")
                print(f"   Junction End: {road.get('junction_end_id')}")
        else:
            print(f"❌ Straße {idx} NICHT in getrimmten Daten gefunden")

    # Prüfe ob es Straßen gibt, die junction_start_id=381 oder junction_end_id=381 haben
    # aber andere original_idx
    print("\n=== Straßen mit Junction-Referenz auf 381 ===")
    roads_at_381 = [
        r
        for r in roads
        if r.get("junction_start_id") == 381 or r.get("junction_end_id") == 381
    ]

    if roads_at_381:
        print(f"Gefunden: {len(roads_at_381)} Straßen")
        for road in roads_at_381:
            print(
                f"  Road {road.get('road_id')} (original_idx: {road.get('original_idx')})"
            )
            print(f"    Punkte: {road.get('num_points')}")
            print(
                f"    Start: {road.get('junction_start_id')}, End: {road.get('junction_end_id')}"
            )
    else:
        print(
            "Keine Straßen gefunden, die junction_start_id=381 oder junction_end_id=381 haben"
        )

    # Zeige Statistik über Straßen-Längen
    print("\n=== Straßen-Längen-Statistik ===")
    num_points_dist = {}
    for road in roads:
        n = road.get("num_points", 0)
        num_points_dist[n] = num_points_dist.get(n, 0) + 1

    for num_pts in sorted(num_points_dist.keys()):
        count = num_points_dist[num_pts]
        print(f"  {num_pts} Punkte: {count} Straßen")


if __name__ == "__main__":
    main()
