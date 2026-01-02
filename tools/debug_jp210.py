"""Debug-Tool für Junction Point 210 - Winkelanalyse."""

import json
import sys
import numpy as np
from pathlib import Path

# Lade debug_junctions.json
debug_file = Path(__file__).parent.parent / "cache" / "debug_junctions.json"

if not debug_file.exists():
    print(f"[ERROR] {debug_file} nicht gefunden")
    sys.exit(1)

with open(debug_file) as f:
    junctions_data = json.load(f)

with open(debug_file) as f:
    junctions_data = json.load(f)

# Handle verschachtelte Struktur
if isinstance(junctions_data, dict) and "junctions" in junctions_data:
    jp_list = junctions_data["junctions"]
else:
    jp_list = junctions_data if isinstance(junctions_data, list) else []

# Finde JP 210
jp_210 = None
for jp in jp_list:
    if jp.get("id") == 210:
        jp_210 = jp
        break

if not jp_210:
    print("[ERROR] JP 210 nicht gefunden")
    print("\n[INFO] Verfügbare Junctions:")
    if isinstance(junctions_data, dict):
        for key in list(junctions_data.keys())[:10]:
            print(f"  {key}")
    elif isinstance(junctions_data, list):
        for i, jp in enumerate(junctions_data[:10]):
            if isinstance(jp, dict):
                print(f"  #{i}: {jp.get('id', 'NO ID')}")
    print("\n[INFO] Suche nach 'JP Nr. 210' oder ähnlich...")
    sys.exit(1)

print(f"[JP 210] Position: {jp_210['position']}")
print(f"[JP 210] Road IDs: {jp_210['road_indices']}")
print()

# Lade road_polygons.json für Straßen-Koordinaten
roads_file = Path(__file__).parent.parent / "cache" / "road_polygons.json"
if not roads_file.exists():
    print(f"[ERROR] {roads_file} nicht gefunden")
    sys.exit(1)

with open(roads_file) as f:
    roads_raw = json.load(f)

# Handle nested structure
if isinstance(roads_raw, dict) and "roads" in roads_raw:
    roads_data = roads_raw["roads"]
else:
    roads_data = roads_raw if isinstance(roads_raw, list) else []

# Erstelle Lookup-Dict
roads_by_id = {r["id"]: r for r in roads_data}


def get_angle_between_vectors(v1, v2):
    """Berechnet Winkel zwischen zwei 2D-Vektoren in Grad (0-180)."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    v1_norm = v1 / n1
    v2_norm = v2 / n2

    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return min(angle_deg, 180.0 - angle_deg)


def get_road_direction_at_junction(coords, is_end_junction=True):
    """Berechnet Straßen-Richtung am Junction."""
    coords_2d = np.asarray(coords[:2] for coord in coords)

    if len(coords_2d) < 2:
        return np.array([1.0, 0.0])

    if is_end_junction:
        direction = coords_2d[-1] - coords_2d[-2]
    else:
        direction = coords_2d[1] - coords_2d[0]

    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0])

    return direction / norm


# Analysiere jede Straße an JP 210
print("=" * 80)
print("WINKELANALYSE FÜR JP 210")
print("=" * 80)
print()

road_ids = jp_210["road_indices"]

for i, road_id in enumerate(road_ids):
    road = roads_by_id.get(road_id)
    if not road:
        print(f"[!] Road {road_id} nicht in road_polygons.json")
        continue

    coords = road["coords"]
    if len(coords) < 2:
        print(f"[!] Road {road_id} hat < 2 Punkte")
        continue

    # Bestimme ob Start oder End Junction
    first_point = np.array(coords[0][:2])
    last_point = np.array(coords[-1][:2])
    jp_pos = np.array(jp_210["position"][:2])

    dist_start = np.linalg.norm(first_point - jp_pos)
    dist_end = np.linalg.norm(last_point - jp_pos)

    is_end = dist_end < dist_start

    print(f"Road {road_id}:")
    print(f"  First point: {coords[0][:2]} (dist to JP: {dist_start:.2f}m)")
    print(f"  Last point:  {coords[-1][:2]} (dist to JP: {dist_end:.2f}m)")
    print(f"  Junction-Type: {'END' if is_end else 'START'}")

    # Berechne Richtung
    direction = get_road_direction_at_junction(coords, is_end_junction=is_end)
    print(f"  Direction: [{direction[0]:.4f}, {direction[1]:.4f}]")
    print()

# Berechne Winkel zwischen allen Paaren
print("=" * 80)
print("WINKEL ZWISCHEN STRASSENPAAREN")
print("=" * 80)
print()

for i in range(len(road_ids)):
    for j in range(i + 1, len(road_ids)):
        road_id_a = road_ids[i]
        road_id_b = road_ids[j]

        road_a = roads_by_id.get(road_id_a)
        road_b = roads_by_id.get(road_id_b)

        if (
            not road_a
            or not road_b
            or len(road_a["coords"]) < 2
            or len(road_b["coords"]) < 2
        ):
            continue

        coords_a = road_a["coords"]
        coords_b = road_b["coords"]

        # Bestimme Junction-Type für beide
        first_a = np.array(coords_a[0][:2])
        last_a = np.array(coords_a[-1][:2])
        first_b = np.array(coords_b[0][:2])
        last_b = np.array(coords_b[-1][:2])
        jp_pos = np.array(jp_210["position"][:2])

        is_end_a = np.linalg.norm(last_a - jp_pos) < np.linalg.norm(first_a - jp_pos)
        is_end_b = np.linalg.norm(last_b - jp_pos) < np.linalg.norm(first_b - jp_pos)

        dir_a = get_road_direction_at_junction(coords_a, is_end_junction=is_end_a)
        dir_b = get_road_direction_at_junction(coords_b, is_end_junction=is_end_b)

        angle = get_angle_between_vectors(dir_a, dir_b)
        buffer = "5.0m" if angle < 60.0 else "0m"

        print(
            f"Road {road_id_a} ({'END' if is_end_a else 'START'}) <-> Road {road_id_b} ({'END' if is_end_b else 'START'})"
        )
        print(f"  Angle: {angle:.1f}°  →  Buffer: {buffer}")
        print()
