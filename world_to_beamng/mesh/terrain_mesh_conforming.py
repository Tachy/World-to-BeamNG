"""
Terrain-Grid Mesh Generierung (vereinfacht).

Die Road-Faces sind bereits separat mit material="road" in der Mesh!
Wir generieren nur das Terrain-Mesh mit material="terrain".
Keine komplexe Klassifizierung nötig → schnell!
"""

import numpy as np


def generate_full_grid_mesh_conforming(
    grid_points, modified_heights, nx, ny, vertex_manager, road_polygons_2d=None
):
    """
    Generiert Grid-Mesh OHNE Face-Klassifizierung (da Roads bereits separat sind).

    Args:
        grid_points: Grid-Punkte (x, y)
        modified_heights: Hoehenwerte
        nx, ny: Grid-Dimensionen
        vertex_manager: Zentrale Vertex-Verwaltung
        road_polygons_2d: (Ignoriert - nur für Kompatibilität)

    Returns:
        terrain_faces: Liste von Face-Indizes
        terrain_materials: Liste von Material-Strings (alle "terrain")
    """
    print("  Fuege Grid-Vertices zum VertexManager hinzu...")

    # Füge alle Vertices hinzu
    coords = np.column_stack([grid_points[:, 0], grid_points[:, 1], modified_heights])
    vertex_indices = vertex_manager.add_vertices_direct_nohash(coords)

    print(
        f"  [OK] {len(vertex_indices)} Grid-Vertices hinzugefuegt (gesamt: {vertex_manager.get_count()})"
    )

    print("  Generiere Grid-Faces...")

    # Erstelle Index-Grid
    idx_grid = np.array(vertex_indices).reshape(ny, nx)

    # Extrahiere Quad-Ecken
    tl = idx_grid[:-1, :-1].ravel()  # top-left
    tr = idx_grid[:-1, 1:].ravel()  # top-right
    br = idx_grid[1:, 1:].ravel()  # bottom-right
    bl = idx_grid[1:, :-1].ravel()  # bottom-left

    # Erzeuge zwei Dreiecke pro Quad
    num_quads = len(tl)
    terrain_faces = np.empty((num_quads * 2, 3), dtype=np.int32)
    terrain_faces[0::2] = np.column_stack([tl, tr, br])
    terrain_faces[1::2] = np.column_stack([tl, br, bl])

    # Alle Terrain (keine Klassifizierung nötig!)
    terrain_materials = ["terrain"] * len(terrain_faces)

    print(f"  [OK] {len(terrain_faces)} Terrain-Faces generiert")

    return terrain_faces.tolist(), terrain_materials
