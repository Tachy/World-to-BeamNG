"""
Terrain-Grid Mesh Generierung mit zentraler Vertex-Verwaltung.
"""

import numpy as np

from ..geometry.coordinates import apply_local_offset


def generate_full_grid_mesh(
    grid_points, modified_heights, vertex_types, nx, ny, vertex_manager, dedup=True
):
    """
    Generiert vollständiges Grid-Mesh mit zentraler Vertex-Verwaltung.

    Args:
        grid_points: Grid-Punkte (x, y)
        modified_heights: Höhenwerte
        vertex_types: Vertex-Typen (0=Terrain, >0=Road/Slope)
        nx, ny: Grid-Dimensionen
        vertex_manager: Zentrale Vertex-Verwaltung

    Returns:
        terrain_faces: Liste von Face-Indizes (0-basiert)
    """
    print("  Füge Grid-Vertices zum VertexManager hinzu...")

    # Transformiere und füge alle Vertices hinzu
    x_local, y_local, z_local = apply_local_offset(
        grid_points[:, 0], grid_points[:, 1], modified_heights
    )

    # Füge alle Vertices zum Manager hinzu und speichere Indices (vektorisiert)
    coords = np.column_stack([x_local, y_local, z_local])

    if not dedup:
        # Endphase: kein Dedup mehr nötig → schneller Append ohne Hash/KDTree
        vertex_indices = vertex_manager.add_vertices_direct_nohash(coords)
    else:
        vertex_indices = vertex_manager.add_vertices_batch_dedup_fast(coords)

    print(
        f"  ✓ {len(vertex_indices)} Grid-Vertices hinzugefügt (gesamt: {vertex_manager.get_count()})"
    )

    print("  Generiere Grid-Faces (vektorisiert)...")

    # Erstelle Index-Grid (0-basiert mit VertexManager-Indices)
    idx_grid = np.array(vertex_indices).reshape(ny, nx)

    # Extrahiere alle Quad-Ecken auf einmal
    tl = idx_grid[:-1, :-1].ravel()  # top-left
    tr = idx_grid[:-1, 1:].ravel()  # top-right
    br = idx_grid[1:, 1:].ravel()  # bottom-right
    bl = idx_grid[1:, :-1].ravel()  # bottom-left

    # Material-Typ pro Quad (Maximum der 4 Ecken)
    vertex_types_2d = vertex_types.reshape(ny, nx)
    mat_tl = vertex_types_2d[:-1, :-1].ravel()
    mat_tr = vertex_types_2d[:-1, 1:].ravel()
    mat_br = vertex_types_2d[1:, 1:].ravel()
    mat_bl = vertex_types_2d[1:, :-1].ravel()
    quad_materials = np.maximum.reduce([mat_tl, mat_tr, mat_br, mat_bl])

    # Erzeuge Faces NUR für Quads ohne markierte Vertices (alles material == 0)
    valid_quads = quad_materials == 0
    num_valid = np.count_nonzero(valid_quads)

    # Erstelle nur notwendige Dreiecke (2 pro gültigem Quad)
    if num_valid > 0:
        terrain_faces = np.empty((num_valid * 2, 3), dtype=np.int32)
        terrain_faces[0::2] = np.column_stack(
            [tl[valid_quads], tr[valid_quads], br[valid_quads]]
        )
        terrain_faces[1::2] = np.column_stack(
            [tl[valid_quads], br[valid_quads], bl[valid_quads]]
        )
        terrain_faces = terrain_faces.tolist()
    else:
        terrain_faces = []

    print(f"  ✓ {len(terrain_faces)} Terrain-Faces (Straßen ausgeschnitten)")

    return terrain_faces, vertex_indices
