#!/usr/bin/env python3
"""
Analysiere Junction Point Winkel und Buffer nach der is_end_junction Fix.
Nutze den gemeinsamen Dump cache/debug_network.json (Junctions + getrimmte Straßen).

Usage: python analyze_jp210_fix.py <junction_point_id>
"""
import json
import math
import numpy as np
import sys
from pathlib import Path

# Füge world_to_beamng zum Path hinzu
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_dumps():
    """Lade den konsistenten Debug-Dump (Junctions + getrimmte Straßen)."""
    cache_dir = project_root / "cache"
    network_path = cache_dir / "debug_network.json"

    if not network_path.exists():
        print(f"[ERROR] {network_path} existiert nicht")
        return None, None

    print(f"[INFO] Lade {network_path}")
    with open(network_path, encoding="utf-8") as f:
        data = json.load(f)

    junctions = data.get("junctions", []) if isinstance(data, dict) else []
    roads = data.get("roads", []) if isinstance(data, dict) else []
    return junctions, roads


def get_road_direction_at_junction(coords, is_end_junction=True):
    """
    Extrahiert den Richtungsvektor der Centerline am Junction Point.
    Die Richtung (hin/weg) ist egal für die Winkelberechnung -
    nur der geometrische Winkel zwischen den Linien zählt.
    """
    coords_2d = np.asarray(
        coords[:, :2] if isinstance(coords, np.ndarray) else [c[:2] for c in coords],
        dtype=np.float64,
    )

    if len(coords_2d) < 2:
        return np.array([1.0, 0.0])

    if is_end_junction:
        # Letzter Punkt ist am Junction - nimm Richtung zum vorletzten
        direction = coords_2d[-2] - coords_2d[-1]
    else:
        # Erster Punkt ist am Junction - nimm Richtung zum zweiten
        direction = coords_2d[1] - coords_2d[0]

    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0])

    return direction / norm


def get_angle_between_vectors(v1, v2):
    """
    Berechnet den geometrischen Winkel zwischen zwei 2D-Vektoren (0-90°).
    Die Richtung der Vektoren ist egal - nur der visuelle XY-Winkel zählt.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    v1_norm = v1 / n1
    v2_norm = v2 / n2

    # Dot product gibt Cosinus des Winkels
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    # Nimm immer den kleineren Winkel (0-90°), Richtung ist egal
    return min(angle_deg, 180.0 - angle_deg)


def analyze_junction_point(junctions_data, roads_data, jp_id):
    """Analysiere einen Junction Point."""
    # Finde JP (Index im Array)
    junctions_list = junctions_data or []
    roads_list = roads_data or []

    if len(junctions_list) <= jp_id:
        print(
            f"[ERROR] JP {jp_id} nicht vorhanden (nur {len(junctions_list)} Junctions)"
        )
        return

    junction = junctions_list[jp_id]

    print(f"\n[OK] JP {jp_id} gefunden (Index {jp_id})!")
    print(f"\n{'='*80}")
    print(f"VOLLSTÄNDIGE DATENSTRUKTUR VON JUNCTION {jp_id}:")
    print(f"{'='*80}")
    print(json.dumps(junction, indent=2, default=str))
    print(f"{'='*80}")

    print(f"\n  Position: {junction.get('position')}")
    print(f"  Straßen an dieser Junction: {junction.get('road_indices', [])}")

    # Sammle Straßen an JP
    road_indices = junction.get("road_indices", [])
    print(f"\n[INFO] Road-Indizes in junction.road_indices: {road_indices}")
    print(f"[INFO] Anzahl Segmente in road_indices: {len(road_indices)}")

    # Prüfe Straßen aus road_indices
    print(f"\n[DEBUG] Analysiere Straßen aus road_indices...")
    for idx in road_indices:
        if idx < len(roads_list):
            road = roads_list[idx]

            print(f"\n{'='*80}")
            print(f"VOLLSTÄNDIGE DATENSTRUKTUR VON ROAD {idx}:")
            print(f"{'='*80}")
            print(json.dumps(road, indent=2, default=str))
            print(f"{'='*80}")

            coords = np.array(road.get("coords", []))
            road_id = road.get("road_id", road.get("id", f"idx_{idx}"))
            junction_start = road.get("junction_start_id")
            junction_end = road.get("junction_end_id")

            if len(coords) >= 2:
                start_pos = coords[0][:2]
                end_pos = coords[-1][:2]
                jp_pos = np.array(junction.get("position", [0, 0, 0]))[:2]
                dist_start = np.linalg.norm(start_pos - jp_pos)
                dist_end = np.linalg.norm(end_pos - jp_pos)
                print(
                    f"  Idx {idx} (OSM {road_id}): {len(coords)} Punkte, dist_start={dist_start:.2f}m, dist_end={dist_end:.2f}m"
                )
                print(
                    f"    junction_indices: start={junction_start}, end={junction_end}"
                )
            else:
                print(f"  Idx {idx} (OSM {road_id}): KEINE COORDS!")

    # Suche ALLE Straßen die junction_indices auf diesen JP haben
    print(
        f"\n[DEBUG] Suche ALLE Straßen mit junction_indices.start oder .end == {jp_id}..."
    )
    roads_with_junction_ref = []
    jp_position = np.array(junction.get("position", [0, 0, 0]))[:2]

    for idx, road in enumerate(roads_list):
        has_start = road.get("junction_start_id") == jp_id
        has_end = road.get("junction_end_id") == jp_id

        if has_start or has_end:
            coords = np.array(road.get("coords", []))
            road_id = road.get("road_id", road.get("id", f"idx_{idx}"))

            if len(coords) >= 2:
                start_pos = coords[0][:2]
                end_pos = coords[-1][:2]
                dist_start = np.linalg.norm(start_pos - jp_position)
                dist_end = np.linalg.norm(end_pos - jp_position)

                junction_type = []
                if has_start:
                    junction_type.append(f"START")
                if has_end:
                    junction_type.append(f"END")

                roads_with_junction_ref.append(
                    {
                        "idx": idx,
                        "id": road_id,
                        "type": "/".join(junction_type),
                        "dist_start": dist_start,
                        "dist_end": dist_end,
                        "num_points": len(coords),
                        "in_road_indices": idx in road_indices,
                    }
                )

    print(
        f"  Gefunden: {len(roads_with_junction_ref)} Straße(n) mit junction_indices Referenz"
    )
    for r in roads_with_junction_ref:
        in_list = "✓" if r["in_road_indices"] else "✗"
        print(
            f"    [{in_list}] Idx {r['idx']} (OSM {r['id']}): {r['type']}, {r['num_points']} Punkte, dist_start={r['dist_start']:.2f}m, dist_end={r['dist_end']:.2f}m"
        )

    # Baue vollständige Liste: Alle Straßen mit junction_indices Referenz
    print(
        f"\n[INFO] Berechne Richtungen für alle {len(roads_with_junction_ref)} Straße(n)..."
    )
    roads_complete = []

    for r in roads_with_junction_ref:
        idx = r["idx"]
        road = roads_list[idx]
        coords = np.array(road.get("coords", []))

        if len(coords) < 2:
            print(f"  [!] Straße {idx} hat < 2 Punkte, überspringe")
            continue

        # Bestimme welches Ende am Junction ist
        has_start = road.get("junction_start_id") == jp_id
        has_end = road.get("junction_end_id") == jp_id

        # Wenn beide, nimm das näheste
        if has_start and has_end:
            is_end_junction = r["dist_end"] < r["dist_start"]
        elif has_end:
            is_end_junction = True
        else:
            is_end_junction = False

        connection_point = "END" if is_end_junction else "START"

        # Berechne Richtung
        direction = get_road_direction_at_junction(
            coords, is_end_junction=is_end_junction
        )

        roads_complete.append(
            {
                "idx": idx,
                "id": str(
                    r["id"]
                ),  # IDs ggf. numerisch -> als String für Vergleiche/Ausgabe
                "connection": connection_point,
                "dist_start": r["dist_start"],
                "dist_end": r["dist_end"],
                "direction": direction,
                "is_end_junction": is_end_junction,
                "in_road_indices": r["in_road_indices"],
            }
        )

    print(f"\n[INFO] Alle {len(roads_complete)} Straße(n) mit Richtungen:")
    for r in roads_complete:
        in_list = "✓" if r["in_road_indices"] else "✗"
        dir_str = f"[{r['direction'][0]:.3f}, {r['direction'][1]:.3f}]"
        print(
            f"    [{in_list}] Idx {r['idx']} (OSM {r['id']}): {r['connection']}, dist_start={r['dist_start']:.2f}m, dist_end={r['dist_end']:.2f}m, dir={dir_str}"
        )

    # Prüfe Winkel-Sweep um 360° (sortiert nach Richtung)
    if roads_complete:
        bearings = []
        for r in roads_complete:
            vx, vy = r["direction"]
            bearing = math.degrees(math.atan2(vy, vx))  # -180..180
            if bearing < 0:
                bearing += 360.0
            bearings.append((bearing, r))

        bearings.sort(key=lambda x: x[0])
        sweep_angles = []
        for i in range(len(bearings)):
            ang_current = bearings[i][0]
            ang_next = bearings[(i + 1) % len(bearings)][0]
            diff = ang_next - ang_current
            if diff < 0:
                diff += 360.0
            sweep_angles.append(diff)

        print(f"\n[INFO] Winkel-Sweep (Summe sollte 360° sein):")
        total_sweep = 0.0
        for (bearing, r), sector in zip(bearings, sweep_angles):
            total_sweep += sector
            print(
                f"    Richtung {bearing:6.2f}° -> Sektor {sector:6.2f}° | Idx {r['idx']} (OSM {r['id']}, {r['connection']})"
            )
        print(f"    Summe aller Sektoren: {total_sweep:.2f}°")

    # Gruppiere nach OSM-ID um tatsächliche Anzahl Centerlines zu ermitteln
    osm_ids_at_junction = set()
    for road in roads_complete:
        osm_id = road["id"]
        if not osm_id.startswith("idx_"):  # Nur echte OSM IDs
            osm_ids_at_junction.add(osm_id)

    print(
        f"\n[INFO] Tatsächliche Anzahl unterschiedlicher Centerlines (OSM-IDs): {len(osm_ids_at_junction)}"
    )
    if osm_ids_at_junction:
        print(f"[INFO] OSM-IDs: {sorted(osm_ids_at_junction)}")

    # Berechne Winkel zwischen allen Straßen
    print(f"\n[INFO] Berechne Winkel zwischen allen Centerline-Paaren...")

    # Berechne ALLE Winkel zwischen allen Paaren
    print(f"\n[INFO] Alle Winkel:")
    all_angles = []
    for i, road_1 in enumerate(roads_complete):
        for road_2 in roads_complete[i + 1 :]:
            angle = get_angle_between_vectors(road_1["direction"], road_2["direction"])
            all_angles.append(angle)
            print(
                f"    Idx {road_1['idx']} (OSM {road_1['id']}, {road_1['connection']}) <-> Idx {road_2['idx']} (OSM {road_2['id']}, {road_2['connection']}): {angle:.1f}°"
            )

    if all_angles:
        min_angle = min(all_angles)
    else:
        min_angle = 180.0

    print(f"\n[RESULT] Anzahl ankommende Centerlines: {len(roads_complete)}")
    print(f"[RESULT] Minimum Winkel an JP {jp_id}: {min_angle:.1f}°")

    if min_angle < 60.0:
        print(f"[✓] BUFFER SOLLTE 5m sein (Winkel < 60°)")
    else:
        print(f"[✓] BUFFER SOLLTE 0m sein (Winkel >= 60°)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_jp210_fix.py <junction_point_id>")
        print("Beispiel: python analyze_jp210_fix.py 485")
        sys.exit(1)

    try:
        jp_id = int(sys.argv[1])
    except ValueError:
        print(f"[ERROR] Junction Point ID muss eine Zahl sein: {sys.argv[1]}")
        sys.exit(1)

    junctions_data, roads_data = load_dumps()

    if junctions_data is None or roads_data is None:
        print("[ERROR] Dumps konnten nicht geladen werden")
        sys.exit(1)

    analyze_junction_point(junctions_data, roads_data, jp_id)
