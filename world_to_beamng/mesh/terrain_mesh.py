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
    grid_points_2d = grid_points[:, :2]

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

    # DEBUG: Zeige Material-Verteilung
    marked_quads = np.count_nonzero(quad_materials > 0)
    print(
        f"  DEBUG: {marked_quads:,} Quads mit mindestens einer markierten Ecke (Material > 0)"
    )
    print(
        f"  DEBUG: vertex_types shape: {vertex_types.shape}, vertex_types_2d shape: {vertex_types_2d.shape}"
    )
    marked_vertices = np.count_nonzero(vertex_types > 0)
    print(f"  DEBUG: {marked_vertices:,} Vertices markiert")

    # Erzeuge Faces NUR für Quads ohne IRGENDEINE markierte Vertices (alles material == 0)
    # WICHTIG: Wenn ein Quad ALLE 4 Ecken hat (Material = 0), wird es generiert
    #          Nur wenn mindestens EINE Ecke markiert ist (Material > 0), wird es NICHT generiert
    valid_quads = quad_materials == 0
    num_valid = np.count_nonzero(valid_quads)
    print(f"  DEBUG: {num_valid:,} Quads mit ALLEN Ecken = Terrain (Material = 0)")

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

    road_faces = []
    slope_faces = []

    print(f"  ✓ {len(vertices)} Vertices")
    print(f"  ✓ {len(terrain_faces)} Terrain-Faces (Straßen ausgeschnitten)")
    print(f"  ✓ Straßen/Böschungen werden separat generiert")

    return vertices.tolist(), road_faces, slope_faces, terrain_faces
