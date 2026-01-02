"""
Grid-Klassifizierung an einer Position prüfen

Verwendung:
    python check_grid_at_position.py <X> <Y> [radius]

Beispiel:
    python check_grid_at_position.py 100.5 200.3 10

    Zeigt alle Grid-Punkte in 10m Radius um Position (100.5, 200.3)
    und deren Klassifizierung (road, slope, terrain).
"""

import sys
import numpy as np
import os

# Pfad zur Debug-Datei
DEBUG_GRID_PATH = "C:/Users/johan/AppData/Local/BeamNG.drive/0.36/levels/World_to_BeamNG/art/shapes/debug_grid_classification.npz"


def check_grid_at_position(x, y, radius=10.0):
    """Prüfe Grid-Klassifizierung an einer Position."""

    if not os.path.exists(DEBUG_GRID_PATH):
        print(f"[!] Debug-Datei nicht gefunden: {DEBUG_GRID_PATH}")
        print("    Führe erst world_to_beamng.py aus!")
        return

    # Lade Grid-Daten
    data = np.load(DEBUG_GRID_PATH)
    grid_points = data["grid_points"]
    vertex_types = data["vertex_types"]
    nx = int(data["nx"])
    ny = int(data["ny"])
    grid_spacing = float(data["grid_spacing"])

    print(f"\n{'='*60}")
    print(f"Grid-Klassifizierung an Position ({x:.1f}, {y:.1f})")
    print(f"{'='*60}")
    print(f"Grid: {nx}×{ny} Punkte, Spacing: {grid_spacing}m")
    print(f"Suchradius: {radius}m\n")

    # Finde Grid-Punkte in Radius
    distances = np.sqrt((grid_points[:, 0] - x) ** 2 + (grid_points[:, 1] - y) ** 2)
    in_radius = distances <= radius

    indices = np.where(in_radius)[0]

    if len(indices) == 0:
        print(f"[!] Keine Grid-Punkte in {radius}m Radius gefunden")
        return

    # Zähle Typen
    types_in_radius = vertex_types[indices]
    type_names = {0: "terrain", 1: "road", 2: "slope"}
    type_counts = {name: 0 for name in type_names.values()}

    for t in types_in_radius:
        type_name = type_names.get(t, f"unknown({t})")
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    print(f"Gefunden: {len(indices)} Grid-Punkte in {radius}m Radius")
    print(f"\nTyp-Verteilung:")
    for type_name, count in type_counts.items():
        if count > 0:
            percent = 100.0 * count / len(indices)
            print(f"  {type_name:8s}: {count:4d} ({percent:5.1f}%)")

    # Zeige nächste 5 Punkte
    print(f"\nNächste 5 Grid-Punkte:")
    closest_5 = np.argsort(distances[indices])[:5]

    for i, idx_in_subset in enumerate(closest_5):
        global_idx = indices[idx_in_subset]
        px, py = grid_points[global_idx]
        dist = distances[global_idx]
        t = vertex_types[global_idx]
        type_name = type_names.get(t, f"unknown({t})")

        print(
            f"  {i+1}. Position: ({px:7.2f}, {py:7.2f}), Distanz: {dist:5.2f}m, Typ: {type_name}"
        )

    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Verwendung: python check_grid_at_position.py <X> <Y> [radius]")
        print("Beispiel:   python check_grid_at_position.py 100.5 200.3 10")
        sys.exit(1)

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    check_grid_at_position(x, y, radius)
