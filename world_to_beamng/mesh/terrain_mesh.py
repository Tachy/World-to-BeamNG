"""
Terrain-Grid Mesh Generierung.
"""

import numpy as np

from ..geometry.coordinates import apply_local_offset


def generate_full_grid_mesh(grid_points, modified_heights, vertex_types, nx, ny):
    """
    Generiert vollständiges Grid-Mesh OHNE Vereinfachung.
    Einfache Grid-basierte Triangulation.
    """
    print("  Transformiere Vertices...")
    x_local, y_local, z_local = apply_local_offset(
        grid_points[:, 0], grid_points[:, 1], modified_heights
    )
    vertices = np.column_stack([x_local, y_local, z_local])

    print("  Generiere Grid-Faces (vektorisiert)...")

    # VEKTORISIERTE Grid-Face-Generierung (100x schneller!)
    # Erstelle Index-Grid (1-basiert für OBJ)
    idx_grid = np.arange(1, len(vertices) + 1).reshape(ny, nx)

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

    # Erstelle alle Dreiecke (2 pro Quad)
    num_quads = len(tl)
    all_tris = np.empty((num_quads * 2, 3), dtype=np.int32)
    all_tris[0::2] = np.column_stack([tl, tr, br])  # Dreieck 1
    all_tris[1::2] = np.column_stack([tl, br, bl])  # Dreieck 2

    # Verdopple Material-Maske (2 Dreiecke pro Quad)
    tri_materials = np.repeat(quad_materials, 2)

    # Trenne nach Material - NUR TERRAIN (Straßen/Böschungen werden ausgeschnitten!)
    terrain_faces = all_tris[tri_materials == 0].tolist()

    road_faces = []
    slope_faces = []

    print(f"  ✓ {len(vertices)} Vertices")
    print(f"  ✓ {len(terrain_faces)} Terrain-Faces (Straßen ausgeschnitten)")
    print(f"  ✓ Straßen/Böschungen werden separat generiert")

    return vertices.tolist(), road_faces, slope_faces, terrain_faces
