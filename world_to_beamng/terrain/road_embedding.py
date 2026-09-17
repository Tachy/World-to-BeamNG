"""
Senkt das Terrain-Heightmap-Array entlang von Straßen ab, damit das
(unveränderte) Straßen-/Böschungs-Mesh sauber eingebettet liegt, statt zu
schweben oder das Terrain zu durchstechen (Spec Abschnitt 4).

Kern-Idee: für jede Rasterzelle nahe einer Straße wird die Höhe der
Straßen-/Böschungs-Mesh-Oberfläche an exakt dieser XY-Position abgefragt
(baryzentrische Interpolation im jeweiligen Dreieck) und das Terrain auf
diesen Wert minus Sicherheitsabstand abgesenkt - nie angehoben.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


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


def sample_heightmap_bilinear(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    points_xy: np.ndarray,
) -> np.ndarray:
    """
    Liest die Heightmap an beliebigen (nicht Grid-ausgerichteten) XY-Punkten
    per bilinearer Interpolation - für Straßenkanten-Punkte, die nicht exakt
    auf einer Rasterzelle liegen.

    Args:
        heights: (size, size) float Array
        origin_x, origin_y, square_size: wie in heightmap.build_heightmap()
        points_xy: (N, 2) Array von Weltkoordinaten (x, y)

    Returns:
        (N,) Array interpolierter Höhenwerte
    """
    size_y, size_x = heights.shape
    col_f = (points_xy[:, 0] - origin_x) / square_size
    row_f = (points_xy[:, 1] - origin_y) / square_size

    # Clip to the valid coordinate domain [0, size-1] (exact, no epsilon
    # fudge - an epsilon here would distort fc/fr and corrupt exact-grid-point
    # results). Out-of-bounds access is instead avoided below by clamping
    # col0/row0 so that col1/row1 stay in range.
    col_f = np.clip(col_f, 0.0, size_x - 1)
    row_f = np.clip(row_f, 0.0, size_y - 1)

    col0 = np.clip(np.floor(col_f).astype(int), 0, max(size_x - 2, 0))
    row0 = np.clip(np.floor(row_f).astype(int), 0, max(size_y - 2, 0))
    col1 = col0 + 1
    row1 = row0 + 1

    fc = col_f - col0
    fr = row_f - row0

    h00 = heights[row0, col0]
    h01 = heights[row0, col1]
    h10 = heights[row1, col0]
    h11 = heights[row1, col1]

    h0 = h00 * (1 - fc) + h01 * fc
    h1 = h10 * (1 - fc) + h11 * fc
    return h0 * (1 - fr) + h1 * fr


def build_road_embankment_profiles(
    road_slope_polygons_2d: list,
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    osm_mapper,
    slope_angle_deg: float,
    min_slope_width: float,
    max_slope_width: float = 30.0,
) -> list:
    """
    Baut pro Straße die Kanten-/Böschungs-Profildaten für apply_embankment_blend().

    Für jede Straße wird entlang der Centerline (bereits dicht abgetastet,
    ca. 1m Punktabstand) links und rechts der Kantenpunkt bei halber
    Straßenbreite berechnet, die natürliche Terrainhöhe dort aus dem noch
    unveränderten Heightmap abgetastet, und daraus eine Böschungsbreite über
    config.SLOPE_ANGLE abgeleitet (Breite = Höhendifferenz / tan(Winkel),
    mindestens min_slope_width, gedeckelt bei max_slope_width).

    Args:
        road_slope_polygons_2d: Liste von Dicts mit "trimmed_centerline"
            ((N,3) Koordinaten x,y,z) und "osm_tags" (Dict) - dieselbe
            Struktur, die TerrainWorkflow.process_tile() bereits für
            Material-Mapping verwendet
        heights: (size, size) float Array MIT der noch unveränderten,
            natürlichen Terrainhöhe (vor embed_roads_into_heightmap und vor
            apply_embankment_blend aufrufen)
        origin_x, origin_y, square_size: wie in heightmap.build_heightmap()
        osm_mapper: OSMMapper-Instanz (für get_road_properties()["width"])
        slope_angle_deg: config.SLOPE_ANGLE
        min_slope_width: config.MIN_SLOPE_WIDTH
        max_slope_width: Obergrenze der Böschungsbreite (Meter)

    Returns:
        Liste von Dicts, je Straße:
            {
                "left_edge_xyz": (N,3) float Array (x, y, road_z),
                "right_edge_xyz": (N,3) float Array (x, y, road_z),
                "left_slope_width": (N,) float Array,
                "right_slope_width": (N,) float Array,
                "left_natural_z": (N,) float Array,
                "right_natural_z": (N,) float Array,
            }
    """
    tan_angle = math.tan(math.radians(slope_angle_deg))
    roads = []

    for poly in road_slope_polygons_2d:
        centerline = np.asarray(poly.get("trimmed_centerline", []), dtype=float)
        if len(centerline) < 2:
            continue

        width = osm_mapper.get_road_properties(poly.get("osm_tags", {}))["width"]
        half_width = width / 2.0

        xy = centerline[:, :2]
        z = centerline[:, 2]

        directions = np.diff(xy, axis=0)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        directions = directions / norms

        point_dirs = np.empty_like(xy)
        point_dirs[0] = directions[0]
        point_dirs[-1] = directions[-1]
        for i in range(1, len(xy) - 1):
            avg = directions[i - 1] + directions[i]
            n = np.linalg.norm(avg)
            point_dirs[i] = avg / n if n > 1e-9 else directions[i - 1]

        perp = np.column_stack([-point_dirs[:, 1], point_dirs[:, 0]])

        # Hinweis: "links"/"rechts" ist hier eine reine Namenskonvention ohne
        # geometrische Bedeutung (perp zeigt in Fahrtrichtung nach links,
        # aber die Zuordnung + / - ist willkürlich) - apply_embankment_blend
        # behandelt beide Seiten symmetrisch, daher ist die Wahl unkritisch.
        left_xy = xy - perp * half_width
        right_xy = xy + perp * half_width

        left_natural_z = sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, left_xy)
        right_natural_z = sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, right_xy)

        left_diff = np.abs(left_natural_z - z)
        right_diff = np.abs(right_natural_z - z)

        left_slope_width = np.clip(np.maximum(min_slope_width, left_diff / tan_angle), None, max_slope_width)
        right_slope_width = np.clip(np.maximum(min_slope_width, right_diff / tan_angle), None, max_slope_width)

        roads.append(
            {
                "left_edge_xyz": np.column_stack([left_xy, z]),
                "right_edge_xyz": np.column_stack([right_xy, z]),
                "left_slope_width": left_slope_width,
                "right_slope_width": right_slope_width,
                "left_natural_z": left_natural_z,
                "right_natural_z": right_natural_z,
            }
        )

    return roads


def apply_embankment_blend(heights: np.ndarray, origin_x: float, origin_y: float, square_size: float, roads: list) -> np.ndarray:
    """
    Überblendet das Terrain zwischen Straßenkante und natürlicher Umgebung
    (Böschung) direkt im Heightmap-Raster - ersetzt die nie fertiggestellte
    Böschungs-Mesh-Geometrie aus mesh/road_mesh.py (config.GENERATE_SLOPES
    bleibt False).

    Für jede Rasterzelle im Böschungskorridor (zwischen Straßenkante und
    Kante+Böschungsbreite) wird linear zwischen der Straßenkanten-Höhe (an
    der Kante) und der ursprünglichen natürlichen Terrainhöhe (am
    Korridor-Rand) interpoliert. Funktioniert für Damm (Straße höher) und
    Einschnitt (Straße tiefer) gleichermaßen, weil einfach in Richtung
    "natürliche Höhe" interpoliert wird, ohne Vorzeichen-Annahme.

    Bekannte Einschränkung: überlappende Korridore mehrerer Straßen (z.B. an
    Kreuzungen) werden nicht speziell behandelt - die zuletzt verarbeitete
    Straße gewinnt. Dokumentiert in der Spec als akzeptierte Vereinfachung.

    Args:
        heights: (size, size) float Array MIT der noch unveränderten,
            natürlichen Terrainhöhe (vor embed_roads_into_heightmap
            aufrufen - diese Funktion muss VOR der Straßen-Einbettung
            laufen, damit "natürliche Höhe" wirklich natürlich ist)
        origin_x, origin_y, square_size: wie in heightmap.build_heightmap()
        roads: Rückgabe von build_road_embankment_profiles()

    Returns:
        Neues (size, size) float Array (Kopie, Original unverändert)
    """
    result = heights.copy()
    size_y, size_x = heights.shape

    for road in roads:
        _blend_one_side(result, origin_x, origin_y, square_size, size_x, size_y,
                         road["left_edge_xyz"], road["left_slope_width"], road["left_natural_z"])
        _blend_one_side(result, origin_x, origin_y, square_size, size_x, size_y,
                         road["right_edge_xyz"], road["right_slope_width"], road["right_natural_z"])

    return result


def _blend_one_side(heights, origin_x, origin_y, square_size, size_x, size_y, edge_xyz, slope_width, natural_z):
    """Überblendet eine Straßenseite (links oder rechts) in-place in heights."""
    if len(edge_xyz) == 0:
        return

    max_width = float(np.max(slope_width)) if len(slope_width) else 0.0
    if max_width <= 0:
        return

    min_x = float(np.min(edge_xyz[:, 0])) - max_width
    max_x = float(np.max(edge_xyz[:, 0])) + max_width
    min_y = float(np.min(edge_xyz[:, 1])) - max_width
    max_y = float(np.max(edge_xyz[:, 1])) + max_width

    col_start = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row_start = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))

    if col_start > col_end or row_start > row_end:
        return

    tree = cKDTree(edge_xyz[:, :2])

    cols = np.arange(col_start, col_end + 1)
    rows = np.arange(row_start, row_end + 1)
    cell_x = origin_x + cols * square_size
    cell_y = origin_y + rows * square_size
    grid_x, grid_y = np.meshgrid(cell_x, cell_y)
    query_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    dist, idx = tree.query(query_points)

    nearest_edge_z = edge_xyz[idx, 2]
    nearest_slope_width = slope_width[idx]
    nearest_natural_z = natural_z[idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(nearest_slope_width > 0, dist / nearest_slope_width, 1.0)
    t = np.clip(t, 0.0, 1.0)

    blended = nearest_edge_z + (nearest_natural_z - nearest_edge_z) * t

    in_corridor = (dist > 0) & (dist <= nearest_slope_width)

    sub_shape = (row_end - row_start + 1, col_end - col_start + 1)
    sub = heights[row_start : row_end + 1, col_start : col_end + 1].reshape(-1)
    write_mask = in_corridor
    sub[write_mask] = blended[write_mask]
    heights[row_start : row_end + 1, col_start : col_end + 1] = sub.reshape(sub_shape)
