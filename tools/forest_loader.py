"""
Forest Layer Loader für dae_viewer

Lädt forest.json und erstellt Punkt-Instanzen für Bäume.
Alle Bäume werden in EINEM Actor gerendert (optimiert für 30k+ Instanzen).
"""

import json
import numpy as np
import pyvista as pv
from pathlib import Path


# Farb-Zuordnung für Baum-Typen
TREE_TYPE_COLORS = {
    "oak": (0.2, 0.6, 0.2),  # Dunkelgrün
    "birch": (0.8, 0.7, 0.5),  # Hellbraun
    "spruce": (0.1, 0.4, 0.1),  # Dunkelgrün
    "ash": (0.4, 0.6, 0.3),  # Mittleres Grün
    "pine": (0.15, 0.45, 0.15),  # Tannengrün
    "beech": (0.25, 0.55, 0.25),  # Dunkelgrün
    "default": (0.3, 0.8, 0.3),  # Helles Grün
}


def load_forest_layer(viewer, forest_json_path: Path):
    """
    Lade Forest-Daten aus forest.json und füge ALLE Bäume als einen einzigen Actor zur Scene hinzu.

    Optimiert für 30.000+ Instanzen: Alle Punkte in EINEM PolyData Mesh.

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

    print(f"  [Forest] Lade {len(tree_instances)} Baum-Instanzen in EINEM Actor...")

    try:
        # Extrahiere Positionen und Typen
        positions = []
        colors = []

        tree_types = {}
        for inst in tree_instances:
            pos = inst.get("pos")
            if pos and len(pos) >= 3:
                positions.append(pos[:3])

                # Bestimme Farbe basierend auf Baum-Typ
                tree_type = inst.get("type", "default")
                tree_types[tree_type] = tree_types.get(tree_type, 0) + 1
                color = TREE_TYPE_COLORS.get(tree_type.lower(), TREE_TYPE_COLORS["default"])
                colors.append(color)

        if not positions:
            print(f"  [!] Keine gültigen Positionen gefunden")
            return None

        positions_array = np.array(positions, dtype=np.float32)
        colors_array = np.array(colors, dtype=np.float32)

        # Erstelle EINEN einzigen PolyData mit ALLEN Bäumen
        point_cloud = pv.PolyData(positions_array)

        # Füge Farben zu den Punkten hinzu
        point_cloud["Colors"] = colors_array

        # Rendere alle Bäume in EINEM Actor!
        actor = viewer.plotter.add_mesh(
            point_cloud,
            scalars="Colors",
            point_size=5.0,
            opacity=0.85,
            label="Trees",
            render_points_as_spheres=True,
            rgb=True,  # Farben sind RGB
        )

        # Speichere Actor-Referenz für Toggle
        viewer.forest_actors.append(actor)

        print(f"  [✓] {len(tree_instances)} Bäume geladen in EINEM Actor")
        print(f"      Typen: {tree_types}")

        return actor

    except Exception as e:
        print(f"  [!] Fehler beim Verarbeiten der Forest-Daten: {e}")
        import traceback

        traceback.print_exc()
        return None
