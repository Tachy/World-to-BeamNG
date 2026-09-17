"""
Senkt das Terrain-Heightmap-Array entlang von Straßen ab, damit das
(unveränderte) Straßen-/Böschungs-Mesh sauber eingebettet liegt, statt zu
schweben oder das Terrain zu durchstechen (Spec Abschnitt 4).

Kern-Idee: für jede Rasterzelle nahe einer Straße wird die Höhe der
Straßen-/Böschungs-Mesh-Oberfläche an exakt dieser XY-Position abgefragt
(baryzentrische Interpolation im jeweiligen Dreieck) und das Terrain auf
diesen Wert minus Sicherheitsabstand abgesenkt - nie angehoben.
"""

from typing import Dict, List, Tuple

import numpy as np


def road_mesh_to_arrays(
    road_mesh_data: List[Dict], all_vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wandelt die strukturierte Road-Mesh-Ausgabe von RoadMeshBuilder.build() in
    ein einfaches (vertices, triangles)-Paar für embed_roads_into_heightmap() um.

    Args:
        road_mesh_data: Liste von {"vertices": [i0, i1, i2], ...} Dicts
                        (road_mesh[0] aus TerrainWorkflow.process_tile())
        all_vertices: (N, 3) Array aller Mesh-Vertex-Positionen
                      (vertex_manager.get_array())

    Returns:
        (all_vertices, triangles) - triangles ist ein (K, 3) int Array
    """
    if not road_mesh_data:
        return all_vertices, np.empty((0, 3), dtype=np.int64)
    triangles = np.array([face["vertices"] for face in road_mesh_data], dtype=np.int64)
    return all_vertices, triangles


def embed_roads_into_heightmap(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    road_vertices: np.ndarray,
    road_triangles: np.ndarray,
    margin: float,
) -> np.ndarray:
    """
    Senkt heights dort ab, wo das Straßenmesh liegt. Verändert heights NICHT
    in-place, gibt eine neue Kopie zurück.

    Args:
        heights: (size, size) float Array
        origin_x, origin_y: Welt-Koordinaten der Zelle [*, 0] bzw. [0, *]
        square_size: Meter pro Rasterzelle
        road_vertices: (M, 3) Array aller Straßen-Vertex-Positionen (x, y, z)
        road_triangles: (K, 3) Array von Vertex-Indizes (in road_vertices) pro Dreieck
        margin: Sicherheitsabstand in Metern (config.ROAD_EMBED_MARGIN)

    Returns:
        Neues (size, size) float Array
    """
    result = heights.copy()
    size_y, size_x = heights.shape

    for tri in road_triangles:
        p0 = road_vertices[tri[0]]
        p1 = road_vertices[tri[1]]
        p2 = road_vertices[tri[2]]
        _embed_triangle(result, origin_x, origin_y, square_size, p0, p1, p2, margin, size_x, size_y)

    return result


def _embed_triangle(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    margin: float,
    size_x: int,
    size_y: int,
) -> None:
    """Senkt alle Rasterzellen ab, die (in 2D-Draufsicht) innerhalb des Dreiecks liegen."""
    min_x = min(p0[0], p1[0], p2[0])
    max_x = max(p0[0], p1[0], p2[0])
    min_y = min(p0[1], p1[1], p2[1])
    max_y = max(p0[1], p1[1], p2[1])

    col_start = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row_start = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))

    if col_start > col_end or row_start > row_end:
        return

    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(denom) < 1e-9:
        return  # entartetes Dreieck (Fläche ~0)

    cols = np.arange(col_start, col_end + 1)
    rows = np.arange(row_start, row_end + 1)
    cell_x = origin_x + cols * square_size
    cell_y = origin_y + rows * square_size
    grid_x, grid_y = np.meshgrid(cell_x, cell_y)  # shape (len(rows), len(cols))

    w0 = ((p1[1] - p2[1]) * (grid_x - p2[0]) + (p2[0] - p1[0]) * (grid_y - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (grid_x - p2[0]) + (p0[0] - p2[0]) * (grid_y - p2[1])) / denom
    w2 = 1.0 - w0 - w1

    inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not np.any(inside):
        return

    interpolated_z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
    target = interpolated_z - margin

    sub = heights[row_start : row_end + 1, col_start : col_end + 1]
    np.minimum(sub, np.where(inside, target, sub), out=sub)
