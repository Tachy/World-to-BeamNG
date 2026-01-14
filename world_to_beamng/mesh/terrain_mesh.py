"""
Terrain-Grid Mesh Generierung mit zentraler Vertex-Verwaltung.
"""

import numpy as np


def generate_full_grid_mesh(grid_points, modified_heights, vertex_types, nx, ny, vertex_manager, dedup=True):
    """
    Generiert vollständiges Grid-Mesh mit zentraler Vertex-Verwaltung.

    Args:
        grid_points: Grid-Punkte (x, y)
        modified_heights: Hoehenwerte
        vertex_types: Vertex-Typen (0=Terrain, >0=Road/Slope)
        nx, ny: Grid-Dimensionen
        vertex_manager: Zentrale Vertex-Verwaltung
        dedup: Ob Deduplication benutzt werden soll

    Returns:
        terrain_faces: Liste von Face-Indizes (0-basiert)
        vertex_indices: Liste aller Vertex-Indizes
        
    HINWEIS: UVs werden NICHT hier berechnet, sondern am Ende per mesh.compute_terrain_uvs_batch()
    für maximale Performance (vektorisiert über alle Terrain-Faces).
    """
    print("  Fuege Grid-Vertices zum VertexManager hinzu...")

    # Grid-Punkte sind bereits in lokalen Koordinaten - keine Transformation mehr noetig!
    # Fuege alle Vertices zum Manager hinzu und speichere Indices (vektorisiert)
    coords = np.column_stack([grid_points[:, 0], grid_points[:, 1], modified_heights])

    if not dedup:
        # Endphase: kein Dedup mehr noetig -> schneller Append ohne Hash/KDTree
        vertex_indices = vertex_manager.add_vertices_direct_nohash(coords)
    else:
        vertex_indices = vertex_manager.add_vertices_batch_dedup_fast(coords)

    print(f"  [OK] {len(vertex_indices)} Grid-Vertices hinzugefuegt (gesamt: {vertex_manager.get_count()})")

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

    # Erzeuge Faces NUR fuer Quads ohne markierte Vertices (alles material == 0)
    valid_quads = quad_materials == 0
    num_valid = np.count_nonzero(valid_quads)

    # Erstelle nur notwendige Dreiecke (2 pro gueltigem Quad)
    if num_valid > 0:
        terrain_faces = np.empty((num_valid * 2, 3), dtype=np.int32)
        terrain_faces[0::2] = np.column_stack([tl[valid_quads], tr[valid_quads], br[valid_quads]])
        terrain_faces[1::2] = np.column_stack([tl[valid_quads], br[valid_quads], bl[valid_quads]])
        terrain_faces = terrain_faces.tolist()
    else:
        terrain_faces = []

    print(f"  [OK] {len(terrain_faces)} Terrain-Faces (Strassen ausgeschnitten)")

    return terrain_faces, vertex_indices
