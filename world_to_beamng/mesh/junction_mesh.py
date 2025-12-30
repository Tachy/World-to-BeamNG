"""
Meshing von Junction-Polygonen.

Konvertiert die zentralen Junction-Polygone in Mesh-Faces.
"""

import numpy as np


def mesh_junction_polygons(junction_polys, vertex_manager):
    """
    Konvertiere Junction-Polygone zu Mesh-Faces.

    Nutzt Fan-Triangulation für beliebige Polygone (4, 5, 6+ Ecken).

    Args:
        junction_polys: Liste von Junction-Polygon-Dicts aus build_all_junction_polygons()
        vertex_manager: VertexManager zum Hinzufügen der Vertices

    Returns:
        Tuple: (all_faces, all_face_indices)
        - all_faces: List of triangles [v0, v1, v2]
        - all_face_indices: List of junction indices (welcher Junction jedes Face angehört)
    """
    if not junction_polys:
        return [], []

    all_junction_faces = []
    all_junction_indices = []

    # Statistik nach Typ
    type_counts = {}
    type_faces = {}

    for junction_idx, poly in enumerate(junction_polys):
        vertices_3d = poly["vertices_3d"]
        poly_type = poly.get("type", "unknown")

        if len(vertices_3d) < 3:
            # Mindestens 3 Vertices für Triangle
            continue

        # Füge alle Vertices hinzu
        vertex_indices = vertex_manager.add_vertices_batch_dedup_fast(vertices_3d)

        # Fan-Triangulation: v0 ist Zentrum, Triangles von v0 zu je 2 benachbarten Vertices
        # Für N Vertices: N-2 Triangles
        num_triangles = 0
        for i in range(1, len(vertex_indices) - 1):
            triangle = [vertex_indices[0], vertex_indices[i], vertex_indices[i + 1]]
            all_junction_faces.append(triangle)
            all_junction_indices.append(junction_idx)
            num_triangles += 1

        # Statistik
        type_counts[poly_type] = type_counts.get(poly_type, 0) + 1
        type_faces[poly_type] = type_faces.get(poly_type, 0) + num_triangles

    # Ausgabe der Statistik
    if all_junction_faces:
        print(
            f"      ✓ {len(all_junction_faces)} Triangles aus {len(junction_polys)} Junctions:"
        )
        for poly_type in sorted(type_counts.keys()):
            count = type_counts[poly_type]
            faces = type_faces[poly_type]
            avg = faces / count if count > 0 else 0
            print(
                f"        - {count} {poly_type:15s} = {faces:4d} Triangles (ø {avg:.1f} pro Junction)"
            )

        # Debug: Zeige Min/Max Z-Werte
        z_values = [v[2] for poly in junction_polys for v in poly["vertices_3d"]]
        if z_values:
            print(
                f"      ℹ Junction Z-Range: {min(z_values):.1f} to {max(z_values):.1f}m"
            )

    return all_junction_faces, all_junction_indices


def add_junction_polygons_to_mesh(vertex_manager, junction_polys):
    """
    Convenience-Funktion: Meshe Junction-Polygone und gib zurück.

    Args:
        vertex_manager: VertexManager
        junction_polys: Junction-Polygone

    Returns:
        (junction_faces, junction_face_indices)
    """
    return mesh_junction_polygons(junction_polys, vertex_manager)
