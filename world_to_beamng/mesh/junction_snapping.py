"""
T-Junction Snapping mit VertexManager.
Verbindet Straßen an Kreuzungen durch automatische Vertex-Deduplication.
"""

import numpy as np
from .. import config


def snap_junctions_with_edges(
    vertex_manager, road_slope_polygons_2d, t_junctions, original_to_mesh_idx
):
    """
    T-Junctions werden automatisch durch VertexManager-Deduplication verbunden.

    Diese Funktion ist jetzt ein Placeholder - die eigentliche Arbeit macht der
    VertexManager durch seine automatische Deduplizierung innerhalb der Toleranz.

    Args:
        vertex_manager: Zentrale Vertex-Verwaltung (nutzt Deduplication)
        road_slope_polygons_2d: Liste von Road/Slope-Polygon-Daten
        t_junctions: Liste von T-Junction-Dictionaries
        original_to_mesh_idx: Mapping original road_idx → mesh_idx
    """
    if not t_junctions:
        return

    print(f"    → T-Junctions werden automatisch durch Vertex-Deduplication verbunden")
    print(f"    → {len(t_junctions)} Kreuzungen werden vom VertexManager verarbeitet")
    print(f"    → Toleranz: {vertex_manager.tolerance * 1000:.1f}mm")
