"""
Strassen-Mesh und Boeschungs-Generierung.

Hinweis: Junction-Detection wurde nach Schritt 6a verschoben (detect_junctions_in_centerlines).
Die hier frueher stattfindende nachträgliche T-Junction-Erkennung ist nicht mehr erforderlich.
"""

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import ceil, atan2, degrees

from .. import config


def get_angle_between_vectors(v1, v2):
    """
    Berechnet den geometrischen Winkel zwischen zwei 2D-Vektoren (0-90°).
    Die Richtung der Vektoren ist egal - nur der visuelle XY-Winkel zählt.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0  # Undefined - treat as maximum

    v1_norm = v1 / n1
    v2_norm = v2 / n2

    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    # Gib immer den kleineren Winkel zurück (0-90°)
    # Richtung ist egal, nur der geometrische Winkel zählt
    return min(angle_deg, 180.0 - angle_deg)


def get_road_direction_at_junction(coords, junction_pos, is_end_junction=True):
    """
    Extrahiert den Richtungsvektor der Centerline am Junction Point.
    Die Richtung (hin/weg) ist egal für die Winkelberechnung -
    nur der geometrische Winkel zwischen den Linien zählt.

    Args:
        coords: Straßen-Centerline-Punkte
        junction_pos: Junction-Position (ungenutzt, nur für Kompatibilität)
        is_end_junction: True wenn Junction am Road-Ende ist, False am Anfang

    Returns:
        2D-Richtungsvektor (normalisiert)
    """
    coords_2d = np.asarray(coords[:, :2] if len(coords[0]) >= 2 else coords, dtype=np.float64)

    if len(coords_2d) < 2:
        return np.array([1.0, 0.0])  # Default-Richtung

    if is_end_junction:
        # Richtung vom vorletzten zum letzten Punkt (Road-Ende)
        direction = coords_2d[-1] - coords_2d[-2]
    else:
        # Richtung vom ersten zum zweiten Punkt (Road-Anfang)
        direction = coords_2d[1] - coords_2d[0]

    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0])  # Default bei praktisch Null-Länge

    return direction / norm


def calculate_junction_buffer(
    road_idx,
    junction_idx,
    coords,
    junction_centers,
    junctions_full,
    road_polygons,
    is_end_junction=True,
):
    """
    Berechnet den dynamischen junction_stop_buffer über benachbarte Winkel.

    - Straßen an der Junction werden über ihre junction_indices aus road_polygons
      ermittelt (kein Vertrauen in junctions_full.road_indices nötig).
    - Richtungen zeigen vom Junction weg (Ende → invertiert).
    - Wenn einer der beiden Nachbarsektoren < config.JUNCTION_STOP_ANGLE_THRESHOLD ist → Buffer 5m, sonst 0m.
    """

    def _bearing_from_direction(direction_vec):
        v = np.asarray(direction_vec, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        v = v / n
        ang = degrees(np.arctan2(v[1], v[0]))
        return ang + 360.0 if ang < 0.0 else ang

    # Sammle alle Straßen, die an dieser Junction hängen
    connected = []  # (idx, junction_indices)
    for idx, r in enumerate(road_polygons):
        ji = r.get("junction_indices", {}) or {}
        if ji.get("start") == junction_idx or ji.get("end") == junction_idx:
            connected.append((idx, ji))

    if len(connected) < 2:
        return 0.0

    # Bearings ermitteln (vom Junction weg)
    bearings = []
    for idx, ji in connected:
        coords_arr = np.asarray(road_polygons[idx].get("coords", []), dtype=float)
        if len(coords_arr) < 2:
            continue

        is_end = ji.get("end") == junction_idx and ji.get("start") != junction_idx
        is_start = ji.get("start") == junction_idx and ji.get("end") != junction_idx

        if ji.get("start") == junction_idx and ji.get("end") == junction_idx:
            if idx == road_idx:
                is_end = is_end_junction
                is_start = not is_end_junction
            else:
                is_end = True
                is_start = False

        if is_end:
            direction = coords_arr[-1, :2] - coords_arr[-2, :2]
            direction = -direction  # vom Junction weg
        else:
            direction = coords_arr[1, :2] - coords_arr[0, :2]

        ang = _bearing_from_direction(direction)
        if ang is not None:
            bearings.append((idx, ang))

    if len(bearings) < 2:
        return 0.0

    bearings.sort(key=lambda x: x[1])
    my_idx_in_list = next((i for i, b in enumerate(bearings) if b[0] == road_idx), None)
    if my_idx_in_list is None:
        return 0.0

    def _sector(a, b):
        diff = b - a
        return diff + 360.0 if diff < 0.0 else diff

    prev_idx = (my_idx_in_list - 1) % len(bearings)
    next_idx = (my_idx_in_list + 1) % len(bearings)
    ang_prev = _sector(bearings[prev_idx][1], bearings[my_idx_in_list][1])
    ang_next = _sector(bearings[my_idx_in_list][1], bearings[next_idx][1])

    angle_threshold = getattr(config, "JUNCTION_STOP_ANGLE_THRESHOLD", 60.0)
    return 5.0 if min(ang_prev, ang_next) < angle_threshold else 0.0


def calculate_stop_distance(centerline_coords, junction_center, road_width, min_edge_distance=1.0):
    """
    Berechnet, wo eine Straße stoppen soll, um die Junction zu erreichen.

    Constraint: Beide Kanten (Links/Rechts) müssen mindestens min_edge_distance
    vom Junction-Center entfernt bleiben.

    Args:
        centerline_coords: (N, 2) Array - Centerline-Punkte (x, y)
        junction_center: (2,) - Junction-Koordinaten (x, y)
        road_width: float - Straßenbreite (z.B. 7m)
        min_edge_distance: float - Minimum Abstand Kante zu Center (default 1m)

    Returns:
        stop_index: int - Index, wo Quad-Bauen stoppen soll (oder len(coords) wenn vor Junction)
    """
    if len(centerline_coords) < 2:
        return len(centerline_coords)

    half_width = road_width / 2.0
    coords_2d = np.asarray(centerline_coords, dtype=np.float64)
    center_2d = np.asarray(junction_center, dtype=np.float64)

    # Berechne Abstand jedes Punktes zur Junction
    dists = np.linalg.norm(coords_2d - center_2d[np.newaxis, :], axis=1)

    # Berechne Richtungen zwischen aufeinanderfolgenden Punkten
    diffs = np.diff(coords_2d, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    lengths = np.maximum(lengths, 0.001)
    directions = diffs / lengths[:, np.newaxis]

    # Erweitere auf alle Punkte (erste/letzte Punkt besonders behandelt)
    point_dirs = np.zeros((len(coords_2d), 2))
    point_dirs[0] = directions[0]
    point_dirs[-1] = directions[-1]
    for i in range(1, len(coords_2d) - 1):
        avg_dir = (directions[i - 1] + directions[i]) / 2.0
        norm = np.linalg.norm(avg_dir)
        point_dirs[i] = avg_dir / norm if norm > 1e-6 else directions[i]

    # Senkrechte zu Strassenrichtung
    perp = np.column_stack([-point_dirs[:, 1], point_dirs[:, 0]])

    # Berechne linke und rechte Kante-Punkte
    left_edges = coords_2d + perp * half_width
    right_edges = coords_2d - perp * half_width

    # Abstand von Kanten zur Junction
    dist_left_to_center = np.linalg.norm(left_edges - center_2d[np.newaxis, :], axis=1)
    dist_right_to_center = np.linalg.norm(right_edges - center_2d[np.newaxis, :], axis=1)

    # Abstand von Centerline-Punkten zur Junction
    dist_center_to_center = np.linalg.norm(coords_2d - center_2d[np.newaxis, :], axis=1)

    # Constraint:
    # 1. Beide Kanten müssen >= min_edge_distance vom Center entfernt sein
    # 2. Centerline-Punkt selbst muss auch >= min_edge_distance vom Center entfernt sein
    valid_indices = np.where(
        (dist_left_to_center >= min_edge_distance)
        & (dist_right_to_center >= min_edge_distance)
        & (dist_center_to_center >= min_edge_distance)
    )[0]

    if len(valid_indices) == 0:
        # Straße startet schon zu nah an Junction
        return 0

    # Finde den letzten gültigen Index
    stop_index = int(valid_indices[-1]) + 1

    return min(stop_index, len(coords_2d))


def clip_road_to_bounds(coords, bounds_local):
    """Clippt eine Strasse an den Grid-Bounds (lokale Koordinaten).

    Findet das längste zusammenhängende Segment innerhalb des Grids.
    """
    if not coords or bounds_local is None:
        return coords

    # bounds_local ist (min_x, max_x, min_y, max_y)
    min_x, max_x, min_y, max_y = bounds_local
    buffer = 0.0
    min_x -= buffer
    min_y -= buffer
    max_x += buffer
    max_y += buffer

    # Finde alle zusammenhängenden Segmente im Grid
    segments = []
    current_segment = []

    for x, y, z in coords:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            current_segment.append((x, y, z))
        else:
            # Punkt außerhalb - speichere aktuelles Segment wenn nicht leer
            if current_segment:
                segments.append(current_segment)
                current_segment = []

    # Letztes Segment speichern
    if current_segment:
        segments.append(current_segment)

    # Gib längstes Segment zurück
    if not segments:
        return []

    return max(segments, key=len)


def create_minimal_quad_for_junction(point1, point2, z_value, half_width=3.5):
    """
    Erstellt ein minimales 10cm langes Quad mit voller Straßenbreite zwischen zwei Punkten.
    Das Quad ist zentriert auf der Verbindungslinie zwischen point1 und point2.

    Args:
        point1: (x, y) - Startpunkt (z.B. Junction-A)
        point2: (x, y) - Endpunkt (z.B. Junction-B)
        z_value: float - Z-Höhe des Quads
        half_width: float - Halbe Straßenbreite (default 3.5m = 7m breit)

    Returns:
        Tuple: (quad_vertices, quad_faces, quad_poly_2d) oder ([], [], []) wenn zu kurz
    """
    if len(point1) < 2 or len(point2) < 2:
        return [], [], []

    p1 = np.array([point1[0], point1[1]], dtype=np.float64)
    p2 = np.array([point2[0], point2[1]], dtype=np.float64)

    # Berechne Richtung zwischen den Punkten
    direction = p2 - p1
    length = np.linalg.norm(direction)

    # Skip wenn praktisch auf dem gleichen Punkt
    if length < 0.01:
        return [], [], []

    # Normalisiere Richtung
    direction_normalized = direction / length

    # Berechne Mittelpunkt der Verbindungslinie
    mid_point = (p1 + p2) / 2.0

    # Positioniere 10cm langes Quad zentriert auf Mittelpunkt
    quad_start = mid_point - direction_normalized * 0.05
    quad_end = mid_point + direction_normalized * 0.05

    # Senkrechte Richtung für die volle 7m Straßenbreite
    perp = np.array([-direction_normalized[1], direction_normalized[0]], dtype=np.float64)

    # Vier Ecken des Quads mit voller 7m Breite
    qs_left = quad_start + perp * half_width
    qs_right = quad_start - perp * half_width
    qe_left = quad_end + perp * half_width
    qe_right = quad_end - perp * half_width

    quad_vertices = [
        (qs_left[0], qs_left[1], z_value),
        (qs_right[0], qs_right[1], z_value),
        (qe_left[0], qe_left[1], z_value),
        (qe_right[0], qe_right[1], z_value),
    ]

    # Zwei Triangles für das Quad: (0,1,2) und (1,3,2)
    quad_faces = [(0, 1, 2), (1, 3, 2)]

    # 2D-Polygon für Grid-Ausschneiden
    quad_poly_2d = [
        (qs_left[0], qs_left[1]),
        (qs_right[0], qs_right[1]),
        (qe_right[0], qe_right[1]),
        (qe_left[0], qe_left[1]),
    ]

    return quad_vertices, quad_faces, quad_poly_2d


def _process_road_batch(
    batch,
    half_width,
    height_points,
    height_elevations,
    local_offset,
    grid_bounds,
    lookup_mode,
    junction_centers=None,
    junctions_full=None,
    all_road_polygons=None,
):
    """Worker-Funktion fuer Strassen-Batch (multiprocessing-fähig)."""
    use_kdtree = lookup_mode == "kdtree"
    tree = cKDTree(height_points) if use_kdtree else None
    interpolator = NearestNDInterpolator(height_points, height_elevations) if not use_kdtree else None

    if local_offset is None:
        local_offset = (0.0, 0.0, 0.0)

    def interpolate_height(xy, height_points, height_elevations, tree, interpolator):
        """Interpoliere Höhe an Position xy."""
        if use_kdtree and tree is not None:
            dist, idx = tree.query(xy)
            return height_elevations[idx]
        elif interpolator is not None:
            return float(interpolator(xy[0], xy[1]))
        else:
            return 0.0

    def _apply_offset(x, y, z):
        # Koordinaten sind bereits in polygon.py transformiert (UTM -> Local)
        # Keine zweite Transformation mehr!
        return (x, y, z)

    batch_vertices = []
    batch_per_road = []
    clipped_local = 0

    for original_road_idx, road in batch:
        coords = road["coords"]
        road_id = road.get("id")

        junction_indices = road.get("junction_indices", {}) or {}

        junction_buffer_start = None
        junction_buffer_end = None

        # Trim am Endpunkt (Road-Ende) falls Junction vorhanden
        end_junction_idx = junction_indices.get("end")
        if end_junction_idx is not None and junction_centers is not None:
            if 0 <= end_junction_idx < len(junction_centers):
                # Berechne dynamischen junction_stop_buffer basierend auf Winkeln
                junction_buffer = calculate_junction_buffer(
                    original_road_idx,
                    end_junction_idx,
                    coords,
                    junction_centers,
                    junctions_full,
                    all_road_polygons,
                    is_end_junction=True,
                )
                junction_buffer_end = junction_buffer

                # Spezialfall: 2-Straßen-Junction (nur half_width, kein Extra-Buffer)
                num_roads_at_junction = 2  # Default
                if junctions_full and 0 <= end_junction_idx < len(junctions_full):
                    num_roads_at_junction = len(junctions_full[end_junction_idx].get("road_indices", []))

                # 2-Straßen-Junctions: min_edge_distance = half_width (3.5m - harte Grenze)
                # 3+-Straßen-Junctions: min_edge_distance = half_width + dynamischer_buffer
                if num_roads_at_junction == 2:
                    min_edge_distance = half_width  # Nur Straßenbreite, kein Buffer
                else:
                    min_edge_distance = half_width + junction_buffer

                coords_arr = np.asarray(coords)
                stop_idx = calculate_stop_distance(
                    coords_arr[:, :2],
                    np.asarray(junction_centers[end_junction_idx])[:2],
                    road_width=half_width * 2.0,
                    min_edge_distance=min_edge_distance,
                )
                coords = coords[:stop_idx] if stop_idx > 0 else []

        # Trim am Startpunkt (Road-Anfang) falls Junction vorhanden
        start_junction_idx = junction_indices.get("start")
        if start_junction_idx is not None and junction_centers is not None and len(coords) > 0:
            if 0 <= start_junction_idx < len(junction_centers):
                # Berechne dynamischen junction_stop_buffer basierend auf Winkeln
                junction_buffer = calculate_junction_buffer(
                    original_road_idx,
                    start_junction_idx,
                    coords,
                    junction_centers,
                    junctions_full,
                    all_road_polygons,
                    is_end_junction=False,
                )
                junction_buffer_start = junction_buffer

                # Spezialfall: 2-Straßen-Junction (nur half_width, kein Extra-Buffer)
                num_roads_at_junction = 2  # Default
                if junctions_full and 0 <= start_junction_idx < len(junctions_full):
                    num_roads_at_junction = len(junctions_full[start_junction_idx].get("road_indices", []))

                # 2-Straßen-Junctions: min_edge_distance = half_width (3.5m - harte Grenze)
                # 3+-Straßen-Junctions: min_edge_distance = half_width + dynamischer_buffer
                if num_roads_at_junction == 2:
                    min_edge_distance = half_width  # Nur Straßenbreite, kein Buffer
                else:
                    min_edge_distance = half_width + junction_buffer

                coords_rev = list(reversed(coords))
                coords_arr = np.asarray(coords_rev)
                stop_idx = calculate_stop_distance(
                    coords_arr[:, :2],
                    np.asarray(junction_centers[start_junction_idx])[:2],
                    road_width=half_width * 2.0,
                    min_edge_distance=min_edge_distance,
                )
                coords_rev = coords_rev[:stop_idx] if stop_idx > 0 else []
                coords = list(reversed(coords_rev))

        # Prüfe ZUERST ob Straße nach Junction-Trimming zu kurz ist
        # (VOR Grid-Clipping, damit wir wissen ob es wegen Junctions ist)
        coords_before_grid_clip = coords
        coords_too_short_from_junctions = len(coords) < 2

        coords = clip_road_to_bounds(coords, grid_bounds)

        # Prüfe ob Straße nach Clipping noch Punkte hat
        if len(coords) < 2:
            # NEUE LOGIK: Erstelle minimales Quad wenn Junction-Trimming die Straße zu kurz machte
            # Fall 1: Beide Junctions vorhanden → Quad zwischen Junctions
            # Fall 2: Nur eine Junction (Stichstraße) → Quad zwischen Junction und Straßenende
            minimal_quad_created = False

            if coords_too_short_from_junctions and junction_centers is not None:
                start_junction_idx = junction_indices.get("start")
                end_junction_idx = junction_indices.get("end")

                point1 = None
                point2 = None

                # Fall 1: Beide Junctions vorhanden (normale Straße zwischen 2 Junctions)
                if (
                    start_junction_idx is not None
                    and end_junction_idx is not None
                    and 0 <= start_junction_idx < len(junction_centers)
                    and 0 <= end_junction_idx < len(junction_centers)
                ):

                    point1 = junction_centers[start_junction_idx]
                    point2 = junction_centers[end_junction_idx]

                # Fall 2: Nur Start-Junction (Stichstraße endet ohne Junction)
                elif (
                    start_junction_idx is not None
                    and 0 <= start_junction_idx < len(junction_centers)
                    and len(coords_before_grid_clip) > 0
                ):

                    point1 = junction_centers[start_junction_idx]
                    # Nimm letzten Punkt der Centerline (nach Junction-Trimming)
                    point2 = coords_before_grid_clip[-1]

                # Fall 3: Nur End-Junction (Stichstraße beginnt ohne Junction)
                elif (
                    end_junction_idx is not None
                    and 0 <= end_junction_idx < len(junction_centers)
                    and len(coords_before_grid_clip) > 0
                ):

                    # Nimm ersten Punkt der Centerline (nach Junction-Trimming)
                    point1 = coords_before_grid_clip[0]
                    point2 = junction_centers[end_junction_idx]

                # Wenn zwei Punkte gefunden wurden, erstelle Minimalquad
                if point1 is not None and point2 is not None:
                    # Berechne Mittelpunkt und Höhe
                    mid_point = (np.array(point1[:2]) + np.array(point2[:2])) / 2.0
                    z_val = interpolate_height(mid_point, height_points, height_elevations, tree, interpolator)

                    # Erstelle minimales Quad zwischen den beiden Punkten
                    quad_verts, quad_faces, quad_poly_2d = create_minimal_quad_for_junction(
                        point1[:2], point2[:2], z_val
                    )

                    if quad_verts and quad_faces:
                        batch_vertices.extend(quad_verts)
                        left_start = len(batch_vertices) - 4

                        # Verwende EXAKT die gleiche Struktur wie normale Straßen
                        batch_per_road.append(
                            {
                                "road_id": road_id,
                                "original_idx": original_road_idx,
                                "trimmed_centerline": [
                                    (*point1[:2], z_val),
                                    (*point2[:2], z_val),
                                ],
                                "junction_start_id": start_junction_idx,
                                "junction_end_id": end_junction_idx,
                                "num_points": 2,
                                "left_start": left_start,
                                "right_start": left_start + 1,
                                "slope_left_start": -1,
                                "slope_right_start": -1,
                                "road_poly_2d": quad_poly_2d,
                                "slope_poly_2d": quad_poly_2d,
                                "road_left_2d": [quad_poly_2d[0], quad_poly_2d[3]],
                                "road_right_2d": [quad_poly_2d[1], quad_poly_2d[2]],
                                # Kopiere ALLE road-Metadaten für einheitliche Struktur
                                "source_road": road,  # Gesamtes Original-Road-Dict für Zugriff auf tags etc.
                            }
                        )
                        minimal_quad_created = True

            if not minimal_quad_created:
                clipped_local += 1
            continue

        cleaned_coords = [coords[0]]
        for i in range(1, len(coords)):
            dx = coords[i][0] - cleaned_coords[-1][0]
            dy = coords[i][1] - cleaned_coords[-1][1]
            if dx * dx + dy * dy > 0.01:
                cleaned_coords.append(coords[i])

        if len(cleaned_coords) < 2:
            clipped_local += 1
            continue

        coords = cleaned_coords
        coords_array = np.array(coords)
        num_points = len(coords_array)

        diffs = np.diff(coords_array[:, :2], axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        lengths = np.maximum(lengths, 0.001)
        directions = diffs / lengths[:, np.newaxis]

        point_dirs = np.zeros((num_points, 2))
        point_dirs[0] = directions[0]
        point_dirs[-1] = directions[-1]
        for i in range(1, num_points - 1):
            incoming_dir = directions[i - 1]
            outgoing_dir = directions[i]
            avg_dir = (incoming_dir + outgoing_dir) / 2.0
            length = np.linalg.norm(avg_dir)
            point_dirs[i] = avg_dir / length if length > 1e-6 else incoming_dir

        perp = np.column_stack([-point_dirs[:, 1], point_dirs[:, 0]])

        # === WICHTIG: Stelle sicher, dass perp-Richtung konsistent bleibt ===
        # Bestimme die "richtige" Perpendikular-Richtung basierend auf der Gesamtrichtung
        # Erste und letzte Perpendikular sollten in gleiche Richtung zeigen
        angle_first = np.arctan2(perp[0, 1], perp[0, 0])
        angle_last = np.arctan2(perp[-1, 1], perp[-1, 0])

        # Berechne die kleinste Drehung
        angle_diff = angle_last - angle_first
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Wenn sich perp von Anfang zu Ende um > 90° dreht, flippe alle Punkte ab der Hälfte
        # Das deutet darauf hin, dass die Richtung zwischendurch verkehrt wurde
        if abs(angle_diff) > np.pi / 2:
            # Finde den Punkt mit dem größten Richtungswechsel
            for i in range(1, num_points):
                angle_prev = np.arctan2(perp[i - 1, 1], perp[i - 1, 0])
                angle_curr = np.arctan2(perp[i, 1], perp[i, 0])

                angle_change = angle_curr - angle_prev
                while angle_change > np.pi:
                    angle_change -= 2 * np.pi
                while angle_change < -np.pi:
                    angle_change += 2 * np.pi

                # Wenn großer Sprung, flippe ab hier
                if abs(angle_change) > np.pi / 2:
                    perp[i:] = -perp[i:]
                    break

        left_xy = coords_array[:, :2] + perp * half_width
        right_xy = coords_array[:, :2] - perp * half_width
        z_vals = coords_array[:, 2]

        if use_kdtree:
            _, left_idx = tree.query(left_xy)
            _, right_idx = tree.query(right_xy)
            terrain_left_height = height_elevations[left_idx]
            terrain_right_height = height_elevations[right_idx]
        else:
            terrain_left_height = interpolator(left_xy)
            terrain_right_height = interpolator(right_xy)

        height_diff_left = terrain_left_height - z_vals
        height_diff_right = terrain_right_height - z_vals

        abs_left = np.abs(height_diff_left)
        abs_right = np.abs(height_diff_right)
        min_slope = getattr(config, "MIN_SLOPE_WIDTH", 0.2)
        slope_width_left = np.clip(np.maximum(min_slope, abs_left), None, 30.0)
        slope_width_right = np.clip(np.maximum(min_slope, abs_right), None, 30.0)

        # Keine Transformation mehr noetig - ALLE Koordinaten sind bereits lokal
        left_local = (left_xy[:, 0], left_xy[:, 1], z_vals)
        right_local = (right_xy[:, 0], right_xy[:, 1], z_vals)

        road_left_vertices = list(zip(left_local[0], left_local[1], left_local[2]))
        road_right_vertices = list(zip(right_local[0], right_local[1], right_local[2]))

        # Böschungs-Vertices nur erzeugen, wenn aktiviert
        if config.GENERATE_SLOPES:
            slope_left_outer_xy = left_xy + perp * slope_width_left[:, np.newaxis]
            slope_right_outer_xy = right_xy - perp * slope_width_right[:, np.newaxis]
            slope_left_outer_z = z_vals + height_diff_left
            slope_right_outer_z = z_vals + height_diff_right

            slope_left_outer_local = (
                slope_left_outer_xy[:, 0],
                slope_left_outer_xy[:, 1],
                slope_left_outer_z,
            )
            slope_right_outer_local = (
                slope_right_outer_xy[:, 0],
                slope_right_outer_xy[:, 1],
                slope_right_outer_z,
            )

            slope_left_outer_vertices = list(
                zip(
                    slope_left_outer_local[0],
                    slope_left_outer_local[1],
                    slope_left_outer_local[2],
                )
            )
            slope_right_outer_vertices = list(
                zip(
                    slope_right_outer_local[0],
                    slope_right_outer_local[1],
                    slope_right_outer_local[2],
                )
            )
        else:
            slope_left_outer_vertices = []
            slope_right_outer_vertices = []

        # Verwende LOKALE Koordinaten fuer 2D-Polygone (nach Offset-Transformation)
        road_left_2d = [(v[0], v[1]) for v in road_left_vertices]
        road_right_2d = [(v[0], v[1]) for v in road_right_vertices]
        road_poly_2d = road_left_2d + list(reversed(road_right_2d))

        # WICHTIG: Prüfe ob Polygon gültig ist (mindestens 3 Punkte für ein Dreieck)
        # Bei nur 2 Punkten links + 2 rechts = 4 Punkte → gültiges Viereck
        if len(road_poly_2d) < 3:
            clipped_local += 1
            continue

        # Slope-2D-Polygone nur erzeugen, wenn aktiviert
        if config.GENERATE_SLOPES:
            slope_left_2d = [(v[0], v[1]) for v in slope_left_outer_vertices]
            slope_right_2d = [(v[0], v[1]) for v in slope_right_outer_vertices]
            slope_poly_2d = slope_left_2d + list(reversed(slope_right_2d))
        else:
            slope_poly_2d = []

        left_start = len(batch_vertices)
        batch_vertices.extend(road_left_vertices)
        right_start = len(batch_vertices)
        batch_vertices.extend(road_right_vertices)

        # Böschungs-Vertices nur hinzufügen, wenn aktiviert
        if config.GENERATE_SLOPES:
            slope_left_start = len(batch_vertices)
            batch_vertices.extend(slope_left_outer_vertices)
            slope_right_start = len(batch_vertices)
            batch_vertices.extend(slope_right_outer_vertices)
        else:
            slope_left_start = -1
            slope_right_start = -1

        batch_per_road.append(
            {
                "road_id": road_id,
                "original_idx": original_road_idx,
                "trimmed_centerline": coords,
                "junction_start_id": start_junction_idx if junction_indices else None,
                "junction_end_id": end_junction_idx if junction_indices else None,
                "junction_buffer_start": junction_buffer_start,
                "junction_buffer_end": junction_buffer_end,
                "num_points": num_points,
                "left_start": left_start,
                "right_start": right_start,
                "slope_left_start": slope_left_start,
                "slope_right_start": slope_right_start,
                "road_poly_2d": road_poly_2d,
                "slope_poly_2d": slope_poly_2d,
                "road_left_2d": road_left_2d,
                "road_right_2d": road_right_2d,
            }
        )

    return (
        batch_per_road,
        batch_vertices,
        clipped_local,
    )


def generate_road_mesh_strips(road_polygons, height_points, height_elevations, vertex_manager, junctions=None):
    """
    Generiert Strassen als separate Mesh-Streifen mit zentraler Vertex-Verwaltung.

    Args:
        road_polygons: Liste von Strassen-Polygonen
        height_points: Terrain-Hoehenpunkte
        height_elevations: Terrain-Hoehen
        vertex_manager: Zentrale Vertex-Verwaltung (ERFORDERLICH)

    Returns:
        Tuple: (road_faces, slope_faces, road_slope_polygons_2d, original_to_mesh_idx)
    """
    half_width = config.ROAD_WIDTH / 2.0
    slope_gradient = np.tan(np.radians(config.SLOPE_ANGLE))

    junction_centers = None
    junctions_full = None
    if junctions:
        junction_centers = [np.asarray(j["position"]) for j in junctions if "position" in j]
        junctions_full = junctions  # Vollständige Junction-Objekte für Anzahl-Prüfung
    junction_stop_buffer = getattr(config, "JUNCTION_STOP_BUFFER", 5.0)

    all_road_faces = []
    all_road_face_to_idx = []
    all_slope_faces = []
    road_slope_polygons_2d = []

    # Mapping: original road_polygons index -> road_slope_polygons_2d index
    # (wegen Clipping koennen Indices unterschiedlich sein)
    original_to_mesh_idx = {}

    total_roads = len(road_polygons)

    use_mp = config.USE_MULTIPROCESSING
    num_workers = config.NUM_WORKERS or None
    local_offset = config.LOCAL_OFFSET
    grid_bounds = config.GRID_BOUNDS_LOCAL

    lookup_mode = (getattr(config, "HEIGHT_LOOKUP_MODE", "kdtree") or "kdtree").lower()

    worker_func = partial(
        _process_road_batch,
        half_width=half_width,
        height_points=height_points,
        height_elevations=height_elevations,
        local_offset=local_offset,
        grid_bounds=grid_bounds,
        lookup_mode=lookup_mode,
        junction_centers=junction_centers,
        junctions_full=junctions_full,
        all_road_polygons=road_polygons,
    )

    per_road_data = []
    all_vertices_concat = []
    clipped_roads = 0

    if use_mp and total_roads > 0:
        workers = num_workers or None
        max_roads_per_batch = getattr(config, "MAX_ROADS_PER_BATCH", 500) or 500
        chunk_size = ceil(total_roads / (workers or 1))
        chunk_size = max(1, min(chunk_size, max_roads_per_batch))
        batches = []
        for start in range(0, total_roads, chunk_size):
            end = min(start + chunk_size, total_roads)
            batch = list(zip(range(start, end), road_polygons[start:end]))
            batches.append(batch)

        with ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, result in enumerate(ex.map(worker_func, batches), 1):
                (
                    batch_per_road,
                    batch_vertices,
                    clipped_local,
                ) = result

                offset = len(all_vertices_concat)
                all_vertices_concat.extend(batch_vertices)

                for entry in batch_per_road:
                    entry["left_start"] += offset
                    entry["right_start"] += offset
                    entry["slope_left_start"] += offset
                    entry["slope_right_start"] += offset
                per_road_data.extend(batch_per_road)

                clipped_roads += clipped_local

                print(f"  Batch {idx}/{len(batches)} fertig (Vertices: {len(batch_vertices):,})")
    else:
        # Single-thread fallback - AUCH mit Batching für Performance!
        max_roads_per_batch = getattr(config, "MAX_ROADS_PER_BATCH", 500) or 500
        batches = []
        for start in range(0, total_roads, max_roads_per_batch):
            end = min(start + max_roads_per_batch, total_roads)
            batch = list(zip(range(start, end), road_polygons[start:end]))
            batches.append(batch)

        for idx, batch in enumerate(batches, 1):
            (
                batch_per_road,
                batch_vertices,
                clipped_local,
            ) = worker_func(batch)

            offset = len(all_vertices_concat)
            all_vertices_concat.extend(batch_vertices)

            for entry in batch_per_road:
                entry["left_start"] += offset
                entry["right_start"] += offset
                entry["slope_left_start"] += offset
                entry["slope_right_start"] += offset

            per_road_data.extend(batch_per_road)
            clipped_roads += clipped_local

            print(f"  Batch {idx}/{len(batches)} fertig (Vertices: {len(batch_vertices):,})")

    # === Globaler Insert in einem Rutsch ===
    print(f"  Fuege {len(all_vertices_concat):,} Strassen/Boeschungs-Vertices in einem Rutsch hinzu...")
    # WICHTIG: Keine globale Deduplizierung über Roads hinweg!
    # Jede Road hat ihre eigenen Vertices - sie dürfen nicht mit anderen Roads geteilt werden
    # Das würde zu falschen Verbindungen zwischen verschiedenen Roads führen
    # MEGA-OPTIMIZATION: Nutze add_vertices_bulk() statt Loop (100x schneller!)
    global_indices = vertex_manager.add_vertices_bulk(all_vertices_concat)

    print(f"  [OK] VertexManager: {vertex_manager.get_count():,} Vertices nach Strassen")

    junction_fans = {}
    if junction_centers:
        junction_fans = {idx: {"center_idx": None, "rim": []} for idx in range(len(junction_centers))}

    # === Faces und Polygone aufbauen mit globalen Indices ===
    for road_meta in per_road_data:
        n = road_meta["num_points"]
        if n < 2:
            continue

        ls = road_meta["left_start"]
        rs = road_meta["right_start"]
        sls = road_meta["slope_left_start"]
        srs = road_meta["slope_right_start"]

        road_vertex_indices_left = global_indices[ls : ls + n]
        road_vertex_indices_right = global_indices[rs : rs + n]
        slope_left_outer_indices = global_indices[sls : sls + n]
        slope_right_outer_indices = global_indices[srs : srs + n]

        start_jid = road_meta.get("junction_start_id")
        end_jid = road_meta.get("junction_end_id")

        def _add_rim(jid, left_idx, right_idx):
            if junction_fans is None or jid is None:
                return
            data = junction_fans.get(jid)
            if data is None:
                return
            if data["center_idx"] is None and junction_centers is not None:
                if 0 <= jid < len(junction_centers):
                    cx, cy, cz = junction_centers[jid]
                    data["center_idx"] = vertex_manager.add_vertex(cx, cy, cz)
            data["rim"].append((left_idx, right_idx))

        # Sammle Endkanten für Junction-Hub-Fans
        _add_rim(start_jid, road_vertex_indices_left[0], road_vertex_indices_right[0])
        _add_rim(end_jid, road_vertex_indices_left[-1], road_vertex_indices_right[-1])

        road_id = road_meta["road_id"]

        # Strassen-Faces
        for i in range(n - 1):
            left1 = road_vertex_indices_left[i]
            left2 = road_vertex_indices_left[i + 1]
            right1 = road_vertex_indices_right[i]
            right2 = road_vertex_indices_right[i + 1]

            # KORRIGIERTE Triangulation mit CCW Winding-Order (gegen Uhrzeigersinn)
            # Diese Reihenfolge erzeugt positive Z-Normals
            all_road_faces.append([left1, right2, right1])
            all_road_face_to_idx.append(road_id)
            all_road_faces.append([left1, left2, right2])
            all_road_face_to_idx.append(road_id)

        # Boeschungs-Faces nur erzeugen, wenn aktiviert
        if config.GENERATE_SLOPES:
            slope_road_left_indices = road_vertex_indices_left
            slope_road_right_indices = road_vertex_indices_right

            for i in range(n - 1):
                road_left1 = slope_road_left_indices[i]
                road_left2 = slope_road_left_indices[i + 1]
                slope_left1 = slope_left_outer_indices[i]
                slope_left2 = slope_left_outer_indices[i + 1]

                all_slope_faces.append([road_left1, slope_left1, slope_left2])
                all_slope_faces.append([road_left1, slope_left2, road_left2])

                road_right1 = slope_road_right_indices[i]
                road_right2 = slope_road_right_indices[i + 1]
                slope_right1 = slope_right_outer_indices[i]
                slope_right2 = slope_right_outer_indices[i + 1]

                all_slope_faces.append([road_right1, slope_right2, slope_right1])
                all_slope_faces.append([road_right1, road_right2, slope_right2])

        # 2D-Polygone + Mapping fuer Snapping/Klassifizierung
        poly_data = {
            "road_polygon": road_meta["road_poly_2d"],
            "slope_polygon": road_meta["slope_poly_2d"],
            "trimmed_centerline": road_meta["trimmed_centerline"],
            "road_id": road_meta.get("road_id"),
            "original_idx": road_meta.get("original_idx"),
            "junction_start_id": road_meta.get("junction_start_id"),
            "junction_end_id": road_meta.get("junction_end_id"),
            "junction_buffer_start": road_meta.get("junction_buffer_start"),
            "junction_buffer_end": road_meta.get("junction_buffer_end"),
            "road_vertex_indices": {
                "left": road_vertex_indices_left,
                "right": road_vertex_indices_right,
            },
        }

        # slope_outer_indices nur hinzufügen, wenn Slopes aktiviert sind
        if config.GENERATE_SLOPES:
            poly_data["slope_outer_indices"] = {
                "left": slope_left_outer_indices,
                "right": slope_right_outer_indices,
            }

        road_slope_polygons_2d.append(poly_data)

        original_idx = road_meta["original_idx"]
        original_to_mesh_idx[original_idx] = len(road_slope_polygons_2d) - 1

    # Junction-Hubs mit Fan-Triangulation schließen
    if junction_fans:
        for j_id, data in junction_fans.items():
            center_idx = data.get("center_idx")
            rim = data.get("rim") or []
            if center_idx is None or len(rim) < 1:  # Mindestens 1 Rim-Paar (2 Vertices)
                continue

            center_xy = vertex_manager.vertices[center_idx][:2]
            boundary_points = []
            for left_idx, right_idx in rim:
                for vid in (left_idx, right_idx):
                    vx, vy = vertex_manager.vertices[vid][:2]
                    boundary_points.append((atan2(vy - center_xy[1], vx - center_xy[0]), vid))

            boundary_points.sort(key=lambda x: x[0])
            # OPTIMIZATION: Für 2er-Junctions mit nur 2 Vertices auch Dreieck erzeugen
            # (Dupliziere einfach das erste Vertex, um ein Dreieck zu bilden)
            if len(boundary_points) < 2:
                continue

            # Wenn weniger als 3 Punkte: verdoppele den letzten Punkt
            while len(boundary_points) < 3:
                boundary_points.append(boundary_points[-1])

            for i in range(len(boundary_points)):
                a = boundary_points[i][1]
                b = boundary_points[(i + 1) % len(boundary_points)][1]
                all_road_faces.append([a, b, center_idx])
                all_road_face_to_idx.append(-1)  # Fan-Faces gehören zu keiner spezifischen Road

    print(f"  [OK] {len(all_road_faces)} Strassen-Faces")
    if config.GENERATE_SLOPES:
        print(f"  [OK] {len(all_slope_faces)} Boeschungs-Faces")
        print(f"  [OK] Boeschungen OK")
    else:
        print(f"  [i] Boeschungs-Generierung deaktiviert (config.GENERATE_SLOPES=False)")
    print(f"  [OK] {len(road_slope_polygons_2d)} Road/Slope-Polygone fuer Grid-Ausschneiden (2D)")
    if clipped_roads > 0:
        print(f"  [i] {clipped_roads} Strassen komplett ausserhalb Grid (ignoriert)")

    # Sammle alle Road-Polygone (2D) für Grid-Conforming
    all_road_polygons_2d = [poly_data["road_polygon"] for poly_data in road_slope_polygons_2d]
    all_slope_polygons_2d = [poly_data["slope_polygon"] for poly_data in road_slope_polygons_2d]

    # Füge Junction-Fan-Bereiche als Polygone hinzu (für Grid-Ausschneiden)
    junction_fan_polygons_added = 0
    if junction_fans and junction_centers:
        for j_id, data in junction_fans.items():
            center_idx = data.get("center_idx")
            rim = data.get("rim") or []
            if center_idx is None or len(rim) < 1:  # Auch 2er-Junctions mit 1 Rim-Paar
                continue

            # Sammle alle Rim-Vertices als Polygon
            rim_vertices_2d = []
            center_xy = vertex_manager.vertices[center_idx][:2]

            # Sortiere Rim-Punkte nach Winkel um Center
            boundary_points = []
            for left_idx, right_idx in rim:
                for vid in (left_idx, right_idx):
                    vx, vy = vertex_manager.vertices[vid][:2]
                    boundary_points.append((atan2(vy - center_xy[1], vx - center_xy[0]), [vx, vy]))

            boundary_points.sort(key=lambda x: x[0])
            rim_vertices_2d = [pt[1] for pt in boundary_points]

            if len(rim_vertices_2d) >= 2:
                # Füge als Road-Polygon hinzu (auch wenn nur 2 Punkte)
                # Die Polygone werden ohnehin als Punkte-Liste behandelt
                all_road_polygons_2d.append(np.array(rim_vertices_2d))
                junction_fan_polygons_added += 1

    return (
        all_road_faces,
        all_road_face_to_idx,
        all_slope_faces,
        road_slope_polygons_2d,
        original_to_mesh_idx,
        all_road_polygons_2d,  # ← NEU
        all_slope_polygons_2d,  # ← NEU
    )
