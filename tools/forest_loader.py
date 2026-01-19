"""
Forest Layer Loader für dae_viewer

Lädt forest.json und erstellt Punkt-Instanzen für Bäume.
"""

import json
import numpy as np
import pyvista as pv
from pathlib import Path


def load_forest_layer(viewer, forest_json_path: Path):
    """
    Lade Forest-Daten aus forest.json und füge Baum-Punkt-Cloud zur Scene hinzu.

    Args:
        viewer: DAETileViewer Instanz
        forest_json_path: Pfad zur forest.json
    """

    if not forest_json_path.exists():
        print(f"  [!] Forest-JSON nicht gefunden: {forest_json_path}")
        return None

    try:
        with open(forest_json_path, "r", encoding="utf-8") as f:
            forest_data = json.load(f)
    except Exception as e:
        print(f"  [!] Fehler beim Laden der Forest-JSON: {e}")
        return None

    # Forest.json Format: {"formatVersion": 1, "trees": [...]}
    if not forest_data or "trees" not in forest_data:
        print(f"  [!] Keine trees in forest.json")
        return None

    tree_instances = forest_data["trees"]
    if not tree_instances:
        print(f"  [!] Baum-Instanzen sind leer")
        return None

    print(f"  [Forest] Lade {len(tree_instances)} Baum-Instanzen...")

    try:
        # Extrahiere Positionen
        positions = []
        for inst in tree_instances:
            pos = inst.get("pos")
            if pos and len(pos) >= 3:
                positions.append(pos[:3])

        if not positions:
            print(f"  [!] Keine gültigen Positionen gefunden")
            return None

        positions_array = np.array(positions, dtype=np.float32)

        # Erstelle Punkt-Cloud mit PyVista
        point_cloud = pv.PolyData(positions_array)

        # Rendere als Punkt-Cloud
        actor = viewer.plotter.add_mesh(
            point_cloud,
            point_size=3.0,
            color="lime",  # Helles Grün
            opacity=0.8,
            label="Trees",
            render_points_as_spheres=False,
        )

        # Speichere Actor-Referenz für Toggle
        viewer.forest_actors.append(actor)

        # Statistik
        tree_types = {}
        for inst in tree_instances:
            tree_type = inst.get("type", "unknown")
            tree_types[tree_type] = tree_types.get(tree_type, 0) + 1

        print(f"  [✓] {len(tree_instances)} Bäume geladen")
        print(f"      Typen: {tree_types}")

        return actor

    except Exception as e:
        print(f"  [!] Fehler beim Verarbeiten der Forest-Daten: {e}")
        import traceback

        traceback.print_exc()
        return None
