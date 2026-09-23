"""
Setzt das Terrain-Heightmap-Array entlang von Straßen exakt auf die
Straßen-Centerline-Höhe.

Seit der Umstellung auf BeamNG `DecalRoad` - Straßen werden nicht mehr als eigenes Mesh
exportiert, sondern als Decal zur Laufzeit direkt auf die Terrain-Oberfläche
projiziert - gibt es keine zweite, separat kodierte Straßen-Oberfläche mehr,
die "getroffen" werden müsste - das Terrain IST die sichtbare Straße. Daher
kein Sicherheitsabstand/Gefälle-Kompensation mehr nötig (im Gegensatz zur
früheren Mesh-Einbettung): die Ziel-Höhe pro Rasterzelle ist exakt die
Centerline-Höhe an der nächstgelegenen Position.

Kern-Idee: für jede Straße wird pro Rasterzelle im (bereits vorhandenen)
2D-Straßenpolygon geprüft, ob sie darin liegt (Punkt-in-Polygon), und falls
ja die Höhe per Projektion der Zellmitte auf die nächstgelegene Position
entlang der Centerline bestimmt (lineare Interpolation zwischen den beiden
nächsten Centerline-Punkten) und direkt gesetzt.
"""

import math
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree
from shapely import intersects_xy
from shapely.geometry import Polygon

from .. import config


def embed_roads_into_heightmap(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    road_slope_polygons_2d: List[Dict],
) -> np.ndarray:
    """
    Setzt heights dort exakt auf Straßen-Centerline-Höhe, wo eine Straße
    liegt. Verändert heights NICHT in-place, gibt eine neue Kopie zurück.

    Args:
        heights: (size, size) float Array
        origin_x, origin_y: Welt-Koordinaten der Zelle [*, 0] bzw. [0, *]
        square_size: Meter pro Rasterzelle
        road_slope_polygons_2d: Liste von Dicts mit "road_polygon" ((M,2)
            2D-Straßenumriss, bereits um halbe Straßenbreite gebuffert) und
            "trimmed_centerline" ((N,3) x,y,z-Punkte) - dieselbe Struktur wie
            für build_road_embankment_profiles()

    Returns:
        Neues (size, size) float Array
    """
    result = heights.copy()
    size_y, size_x = heights.shape

    for road in road_slope_polygons_2d:
        _embed_road(result, origin_x, origin_y, square_size, road, size_x, size_y)

    return result


def _points_in_polygon_2d(qx: np.ndarray, qy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vektorisierter Punkt-in-Polygon-Test (Ray-Casting/Crossing-Number)."""
    polygon = np.asarray(polygon, dtype=np.float64)
    px = polygon[:, 0]
    py = polygon[:, 1]
    n = len(polygon)
    inside = np.zeros(qx.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = (xj - xi) * (qy - yi) / (yj - yi) + xi
        crosses = (yi > qy) != (yj > qy)
        inside ^= crosses & (qx < x_intersect)
        j = i
    return inside


def _cells_in_polygon(qx: np.ndarray, qy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Punkt-in-Polygon für ein Zellraster. Schnell über Shapely (C, vorbereitetes Polygon);
    nur bei ungültigen Polygonen (z.B. selbstüberschneidender Centerline-Fallback) greift
    der Ray-Casting-Test, dessen Even-Odd-Regel dort das bisherige Verhalten liefert.
    """
    shape = Polygon(polygon)
    if not shape.is_valid:
        return _points_in_polygon_2d(qx, qy, polygon)
    return intersects_xy(shape, qx, qy)


PROJECT_CHUNK = 96  # Zellen je Block in _project_onto_polyline


def _project_onto_polyline(
    qx: np.ndarray, qy: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray, poly_z: np.ndarray
) -> np.ndarray:
    """
    Projiziert Query-Punkte (qx, qy, beliebige gleiche Shape) auf die
    nächstgelegene Position entlang der durch (poly_x, poly_y, poly_z)
    definierten Polylinie und gibt die dort linear interpolierte Z-Höhe
    zurück (gleiche Shape wie qx/qy).

    Ergebnisgleich zur einfachen Schleife über alle Segmente (bei Gleichstand gewinnt das erste Segment), aber
    nur mit den Segmenten, die für einen Block räumlich benachbarter Zellen überhaupt in Frage kommen: der
    nächste Segment-Endpunkt liegt auf der Polylinie und begrenzt damit die Entfernung zum nächsten Segment von
    oben; Segmente, deren Bounding Box weiter als diese Schranke vom Block entfernt ist, können nicht gewinnen.
    """
    shape = np.shape(qx)
    points = np.column_stack([np.ravel(qx), np.ravel(qy)]).astype(np.float64)
    count = len(points)
    best_z = np.zeros(count)
    if count == 0 or len(poly_x) < 2:
        return best_z.reshape(shape)

    ax, ay, az = poly_x[:-1], poly_y[:-1], poly_z[:-1]
    bx, by, bz = poly_x[1:], poly_y[1:], poly_z[1:]
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    valid = np.flatnonzero(~(seg_len_sq < 1e-9))  # Nulllängen-Segmente entfallen
    if len(valid) == 0:
        return best_z.reshape(shape)
    ax, ay, az, bx, by, bz, dx, dy, seg_len_sq = (a[valid] for a in (ax, ay, az, bx, by, bz, dx, dy, seg_len_sq))
    segments = len(valid)

    seg_min_x, seg_max_x = np.minimum(ax, bx), np.maximum(ax, bx)
    seg_min_y, seg_max_y = np.minimum(ay, by), np.maximum(ay, by)

    endpoints = np.concatenate([np.column_stack([ax, ay]), np.column_stack([bx, by])])
    endpoint_dist, nearest_endpoint = cKDTree(endpoints).query(points)
    # Nach Lage entlang der Linie sortieren: aufeinanderfolgende Zellen bilden kompakte Blöcke
    order = np.argsort(nearest_endpoint % segments, kind="stable")

    for start in range(0, count, PROJECT_CHUNK):
        cells = order[start : start + PROJECT_CHUNK]
        cx, cy = points[cells, 0], points[cells, 1]
        bound = endpoint_dist[cells].max() + 1e-9
        candidates = np.flatnonzero(
            (seg_max_x >= cx.min() - bound)
            & (seg_min_x <= cx.max() + bound)
            & (seg_max_y >= cy.min() - bound)
            & (seg_min_y <= cy.max() + bound)
        )
        cax, cay, cdx, cdy, clen = ax[candidates], ay[candidates], dx[candidates], dy[candidates], seg_len_sq[candidates]
        col_x, col_y = cx[:, None], cy[:, None]
        t = np.clip(((col_x - cax) * cdx + (col_y - cay) * cdy) / clen, 0.0, 1.0)
        dist = np.hypot(col_x - (cax + t * cdx), col_y - (cay + t * cdy))
        nearest = np.argmin(dist, axis=1)  # erstes Minimum wie die Schleife mit "dist < best"
        t_best = t[np.arange(len(cells)), nearest]
        seg = candidates[nearest]
        best_z[cells] = az[seg] + t_best * (bz[seg] - az[seg])

    return best_z.reshape(shape)


def _embed_road(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    road: Dict,
    size_x: int,
    size_y: int,
) -> None:
    """Setzt alle Rasterzellen innerhalb des Straßenpolygons auf Centerline-Höhe."""
    polygon = np.asarray(road["road_polygon"], dtype=np.float64)
    centerline = np.asarray(road["trimmed_centerline"], dtype=np.float64)
    if len(polygon) < 3 or len(centerline) < 2:
        return

    min_x, min_y = polygon[:, 0].min(), polygon[:, 1].min()
    max_x, max_y = polygon[:, 0].max(), polygon[:, 1].max()

    col_start = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row_start = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))

    if col_start > col_end or row_start > row_end:
        return

    cols = np.arange(col_start, col_end + 1)
    rows = np.arange(row_start, row_end + 1)
    cell_x = origin_x + cols * square_size
    cell_y = origin_y + rows * square_size
    grid_x, grid_y = np.meshgrid(cell_x, cell_y)  # shape (len(rows), len(cols))

    inside = _cells_in_polygon(grid_x, grid_y, polygon)
    if not np.any(inside):
        return

    # Nur die Zellen im Polygon projizieren: bei langen, diagonalen Straßen ist die
    # Bounding-Box riesig, der eigentliche Streifen aber schmal (Faktor 100+ weniger Zellen).
    target_z = _project_onto_polyline(
        grid_x[inside], grid_y[inside], centerline[:, 0], centerline[:, 1], centerline[:, 2]
    )

    sub = heights[row_start : row_end + 1, col_start : col_end + 1]
    sub[inside] = target_z


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
    max_slope_width: float = config.MAX_SLOPE_WIDTH,
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

    Optionales Feld je Straßen-Dict: "slope_width_override" (Dict, Schlüssel "left"/"right", Wert = feste
    Böschungsbreite in Metern) - ersetzt die berechnete Böschungsbreite auf der jeweiligen Seite durch
    einen festen Wert statt sie aus der Höhendifferenz zum natürlichen Gelände abzuleiten (0.0 = gar keine
    Böschung, das Gelände bleibt dort auf natürlicher Höhe stehen). "left"/"right" folgen dabei der
    STANDARD-Konvention (wie offset_points()/resolve_open_side(): links = Centerline-Richtung um +90°
    gedreht) - NICHT der (rein internen, siehe Hinweis unten) links/rechts-Zuordnung dieser Funktion; die
    Übersetzung passiert intern.

    Für Galerien: die bergseitige Böschung braucht keinen künstlichen Winkel mehr, weil die (jetzt massive,
    siehe config.GALLERY_WALL_THICKNESS) Wand ohnehin bis in den Hang reicht (Override 0.0). Die talseitige
    Böschung bekommt stattdessen einen kurzen FESTEN Wert (statt der berechneten Breite): das DGM zeigt an
    einer Galerie nicht das ursprüngliche Gelände, sondern die reale Talseiten-Struktur (Brüstung/
    Dachüberstand) - die daraus abgeleitete Höhendifferenz/Böschungsbreite wäre entsprechend verrauscht und
    ergäbe eine sichtbar facettierte, spitze Böschung statt einer glatten Angleichung ans Gelände (siehe
    config.GALLERY_VALLEY_SLOPE_WIDTH). Siehe tunnels/gallery_mesh.py::resolve_open_side().

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

        # STANDARD-"links" (point + perp*half) ist oben "right_xy", STANDARD-"rechts" ist "left_xy" (siehe
        # Hinweis) - die slope_width_override-Zuordnung muss deshalb gespiegelt werden.
        override = poly.get("slope_width_override") or {}
        if "left" in override:
            right_slope_width = np.full_like(right_slope_width, override["left"])
        if "right" in override:
            left_slope_width = np.full_like(left_slope_width, override["right"])

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
    Böschungs-Mesh-Geometrie.

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


BLEND_BLOCK = 32  # Kantenlänge der Zellblöcke, die _blend_one_side auf Nähe zur Straße vorprüft


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

    # Der Korridor endet spätestens bei max_width. Bei langen, diagonalen Straßen ist die Bounding Box aber riesig
    # und fast leer: statt alle ihre Zellen abzufragen, werden nur Blöcke von BLEND_BLOCK x BLEND_BLOCK Zellen
    # berücksichtigt, deren Mittelpunkt nah genug an einem Kantenpunkt liegt (jede Zelle des Blocks ist höchstens
    # die halbe Blockdiagonale vom Mittelpunkt entfernt - weiter entfernte Blöcke enthalten garantiert keine
    # Korridorzelle). Das Ergebnis je Zelle ist unverändert.
    limit = max_width + 1e-9
    n_rows = row_end - row_start + 1
    n_cols = col_end - col_start + 1
    block_r0 = np.arange(0, n_rows, BLEND_BLOCK)
    block_c0 = np.arange(0, n_cols, BLEND_BLOCK)
    r0, c0 = np.meshgrid(block_r0, block_c0, indexing="ij")
    r0, c0 = r0.ravel(), c0.ravel()
    r1 = np.minimum(r0 + BLEND_BLOCK, n_rows) - 1
    c1 = np.minimum(c0 + BLEND_BLOCK, n_cols) - 1
    center_x = origin_x + (col_start + (c0 + c1) / 2.0) * square_size
    center_y = origin_y + (row_start + (r0 + r1) / 2.0) * square_size
    half_diagonal = 0.5 * np.hypot(c1 - c0, r1 - r0) * square_size
    block_dist, _ = tree.query(np.column_stack([center_x, center_y]), distance_upper_bound=limit + half_diagonal.max() + 1e-6)
    keep = np.isfinite(block_dist) & (block_dist <= limit + half_diagonal + 1e-6)
    if not keep.any():
        return

    row_parts, col_parts = [], []
    for br0, br1, bc0, bc1 in zip(r0[keep], r1[keep], c0[keep], c1[keep]):
        rr, cc = np.meshgrid(np.arange(br0, br1 + 1), np.arange(bc0, bc1 + 1), indexing="ij")
        row_parts.append(rr.ravel())
        col_parts.append(cc.ravel())
    rows = np.concatenate(row_parts) + row_start
    cols = np.concatenate(col_parts) + col_start
    query_points = np.column_stack([origin_x + cols * square_size, origin_y + rows * square_size])

    dist, idx = tree.query(query_points, distance_upper_bound=limit)
    near = np.isfinite(dist)
    dist, idx = dist[near], idx[near]

    nearest_edge_z = edge_xyz[idx, 2]
    nearest_slope_width = slope_width[idx]
    nearest_natural_z = natural_z[idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(nearest_slope_width > 0, dist / nearest_slope_width, 1.0)
    t = np.clip(t, 0.0, 1.0)

    blended = nearest_edge_z + (nearest_natural_z - nearest_edge_z) * t

    in_corridor = (dist > 0) & (dist <= nearest_slope_width)

    target_rows = rows[near][in_corridor]
    target_cols = cols[near][in_corridor]
    heights[target_rows, target_cols] = blended[in_corridor]
