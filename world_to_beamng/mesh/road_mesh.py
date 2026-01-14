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
from ..config import OSM_MAPPER


def smooth_junction_centers_z(junction_fans, vertex_manager):
    """
    Glätte die Z-Werte der Rim-Vertices um Junction-Zentralpunkte.

    Der zentrale Punkt bleibt unverändert. Die umlaufenden Rim-Vertices werden
    mit Chaikin-Filter geglättet, um Höhen-Spikes zu reduzieren.

    Args:
        junction_fans: Dict aus generate_road_mesh_strips() mit Junction-Daten
        vertex_manager: VertexManager mit allen Vertices
    """
    if not junction_fans:
        return

    smoothed_count = 0
    for j_id, data in junction_fans.items():
        center_idx = data.get("center_idx")
        rim = data.get("rim", [])

        if center_idx is None or len(rim) < 2:
            continue

        # Sammle Rim-Vertex-Indices in Reihenfolge (zirkulär)
        rim_indices = []
        for rim_entry in rim:
            if isinstance(rim_entry, dict):
                left_idx = rim_entry.get("left_idx")
                right_idx = rim_entry.get("right_idx")
            else:
                left_idx, right_idx = rim_entry

            if left_idx is not None:
                rim_indices.append(left_idx)
            if right_idx is not None:
                rim_indices.append(right_idx)

        # Duplikate entfernen (Vertices können mehrfach vorkommen)
        rim_indices = list(dict.fromkeys(rim_indices))

        if len(rim_indices) < 2:
            continue

        # Hole Z-Werte
        rim_z_values = np.array([vertex_manager.vertices[vid][2] for vid in rim_indices])

        # Chaikin-Glättung (1 Iteration) auf Z-Werte, zirkulär
        smoothed_z = rim_z_values.copy()
        for i in range(len(smoothed_z)):
            prev_idx = (i - 1) % len(smoothed_z)
            next_idx = (i + 1) % len(smoothed_z)
            # Neuer Z-Wert: 50% aktuell + 25% links + 25% rechts
            smoothed_z[i] = 0.5 * rim_z_values[i] + 0.25 * rim_z_values[prev_idx] + 0.25 * rim_z_values[next_idx]

        # Schreibe geglättete Z-Werte zurück
        for vid, z in zip(rim_indices, smoothed_z):
            vertex_manager.vertices[vid, 2] = z

        smoothed_count += 1

    if smoothed_count > 0:
        print(f"    -> {smoothed_count} Junction-Rim-Vertex-Ringe Z-geglättet")


def compute_road_uv_coords(centerline_coords, tiling_distance=10.0):
    """
    Berechne UV-Koordinaten für eine Straße entlang des Centerline-Strips.

    U = Distance entlang Centerline (repetierend alle tiling_distance Meter)
    V = Querposition (0 = links, 1 = rechts)

    Args:
        centerline_coords: (N, 3) Array mit Centerline-Punkten (x, y, z)
        tiling_distance: Meter zwischen U-Wiederholung (z.B. 10m für Straßenmuster)

    Returns:
        List[float]: Kumulative Distanzen normalisiert auf [0, tiling_count]
                    Jede Position hat 1 Distanz-Wert (für Links und Rechts UV)
    """
    if len(centerline_coords) < 2:
        return [0.0] * len(centerline_coords)

    # Berechne kumulative Länge entlang des Centerline
    coords_2d = np.asarray(centerline_coords[:, :2], dtype=np.float64)
    diffs = np.diff(coords_2d, axis=0)  # Kanten zwischen Punkten
    segment_lengths = np.linalg.norm(diffs, axis=1)

    # Kumulative Länge (startend bei 0)
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    # Normalisiere auf Tiling-Entfernung
    # U in Einheiten von tiling_distance (z.B. alle 10m = 1.0 U-Einheit)
    # WICHTIG: Kein % 1.0 mehr - UVs sollen wachsen, nicht wiederholen!
    # Die Wiederholung passiert in der Textur selbst (mit wrap mode)
    u_coords = cumulative_length / tiling_distance

    return u_coords.tolist()


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
    road_id=None,
):
    """
    Berechnet den dynamischen, winkelabhängigen junction_stop_buffer.

    - Straßen an der Junction werden über ihre junction_indices aus road_polygons ermittelt.
    - Wenn der kleinste Nachbarsektor < JUNCTION_STOP_ANGLE_THRESHOLD (90°) ist:
      Buffer = half_width / sin(angle_min / 2) - half_width (asymmetrisch pro Straße)
    - Sonst: Buffer = 0 (Straßen nicht eng beieinander)
    - Der Buffer wird geclampet auf [0, half_width * 3].
    """

    def _bearing_from_direction(direction_vec):
        v = np.asarray(direction_vec, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        v = v / n
        ang = degrees(np.arctan2(v[1], v[0]))
        return ang + 360.0 if ang < 0.0 else ang

    def _direction_with_fallback(coords_arr, is_end, junction_center):
        """Liefere einen nicht-degenerierten Richtungsvektor.

        Primär nutzen wir den Strahl vom Junction-Center zum ersten Nicht-Junction-Punkt
        ("outward"), damit die Bearings einem Sweep um den Knoten entsprechen.
        Fallback: nächster nicht-degenerierter Segment-Vektor.
        """

        coords2 = coords_arr[:, :2]
        n = len(coords2)
        if n < 2:
            return np.array([0.0, 0.0]), "insufficient"

        if junction_center is not None:
            jc = junction_center[:2]
            if is_end:
                for k in range(n - 2, -1, -1):
                    direction = coords2[k] - jc
                    if np.linalg.norm(direction) > 1e-3:
                        return direction, "outward"
            else:
                for k in range(1, n):
                    direction = coords2[k] - jc
                    if np.linalg.norm(direction) > 1e-3:
                        return direction, "outward"

        # Fallback: Segment-basierte Richtung (früheres Verhalten)
        if is_end:
            anchor = coords2[-1]
            for k in range(n - 2, -1, -1):
                direction = anchor - coords2[k]
                if np.linalg.norm(direction) > 1e-3:
                    return direction, "segment"
        else:
            anchor = coords2[0]
            for k in range(1, n):
                direction = coords2[k] - anchor
                if np.linalg.norm(direction) > 1e-3:
                    return direction, "segment"

        if junction_center is not None:
            direction = coords2[-1] - junction_center[:2] if is_end else coords2[0] - junction_center[:2]
            return direction, "center"

        return np.array([0.0, 0.0]), "zero"

    # Sammle alle Straßen, die an dieser Junction hängen
    connected = []  # (road_id, junction_indices, osm_tags)
    for idx, r in enumerate(road_polygons):
        ji = r.get("junction_indices", {}) or {}
        if ji.get("start") == junction_idx or ji.get("end") == junction_idx:
            r_id = r.get("id")
            osm_tags = r.get("osm_tags", {})
            connected.append((r_id, ji, r, osm_tags))

    # Vorab Breiten erfassen, um spätere Buffer-Basis auf maximale Breite zu setzen
    width_map = {r_id: OSM_MAPPER.get_road_properties(osm_tags)["width"] for r_id, ji, r, osm_tags in connected}
    max_half_width = max(width_map.values()) / 2.0 if width_map else 0.0

    if len(connected) < 2:
        return 0.0, max_half_width

    # Bearings ermitteln: Tangente am Straßenende/-anfang (letztes bzw. erstes Segment)
    bearings = []
    for r_id, ji, r, osm_tags in connected:
        coords_arr = np.asarray(r.get("coords", []), dtype=float)

        if len(coords_arr) < 2:
            continue

        is_end = ji.get("end") == junction_idx and ji.get("start") != junction_idx
        is_start = ji.get("start") == junction_idx and ji.get("end") != junction_idx

        if ji.get("start") == junction_idx and ji.get("end") == junction_idx:
            if r_id == road_id:
                is_end = is_end_junction
                is_start = not is_end_junction
            else:
                is_end = True
                is_start = False

        junction_center = np.asarray(junction_centers[junction_idx]) if junction_centers is not None else None
        direction, dir_source = _direction_with_fallback(coords_arr, is_end, junction_center)

        ang = _bearing_from_direction(direction)

        if ang is not None:
            bearings.append((r_id, ang, osm_tags))

    if len(bearings) < 2:
        return 0.0, max_half_width

    bearings.sort(key=lambda x: x[1])
    my_idx_in_list = next((i for i, b in enumerate(bearings) if b[0] == road_id), None)

    if my_idx_in_list is None:
        return 0.0, max_half_width

    # Berechne tatsächliche Nachbar-Winkel (geometrische Winkel, nicht Sektoren)
    # bearing = Kompassrichtung (0-360°). Der Winkel zwischen zwei Kompassrichtungen ist:
    # angle = |bearing1 - bearing2|, dann min(angle, 360-angle) für den kürzeren Bogen
    def _cw_diff(b_from, b_to):
        """Gerichteter Winkel im Uhrzeigersinn von b_from -> b_to (0..360)."""
        return (b_to - b_from) % 360.0

    # Berechne Winkel zu Nachbarn
    prev_idx = (my_idx_in_list - 1) % len(bearings)
    next_idx = (my_idx_in_list + 1) % len(bearings)

    ang_prev = _cw_diff(bearings[prev_idx][1], bearings[my_idx_in_list][1])
    ang_next = _cw_diff(bearings[my_idx_in_list][1], bearings[next_idx][1])

    # Nimm den kleinsten Winkel (spitzeste Ecke)
    angle_min = min(ang_prev, ang_next)
    angle_threshold = config.JUNCTION_STOP_ANGLE_THRESHOLD

    # Wenn Winkel >= threshold: kein Buffer nötig
    if angle_min >= angle_threshold:
        return 0.0, max_half_width

    # Berechne Buffer asymmetrisch
    # Extrahiere Straßenbreite der aktuellen Straße aus osm_tags
    my_osm_tags = next((b[2] for b in bearings if b[0] == road_id), {})
    road_width = width_map.get(road_id, OSM_MAPPER.get_road_properties(my_osm_tags)["width"])
    my_half_width = road_width / 2.0

    # Konvertiere Winkel zu Radiant für cot-Berechnung
    tan_half_angle = np.tan(np.radians(angle_min) / 2.0)

    # Geometrisch exakte Formel: buffer = max_halfwidth * cot(alpha/2) - max_halfwidth
    # cot(x) = 1/tan(x)
    if tan_half_angle > 1e-6:
        buffer = (max_half_width / tan_half_angle) - max_half_width + 0.2
    else:
        buffer = 0.2

    # Clamp auf [0, half_width * 5] zur Sicherheit
    buffer = max(0.0, min(buffer, max_half_width * 5.0))

    return buffer, max_half_width


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
        Tuple: (stop_index, snapped_point or None)
               - stop_index: int - Index, wo Quad-Bauen stoppen soll
               - snapped_point: (x, y) oder None - gesnappter Punkt bei exaktem Zielabstand
    """
    if len(centerline_coords) < 2:
        return len(centerline_coords), None

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
        return 0, None

    # Finde den letzten gültigen Index
    last_valid_idx = int(valid_indices[-1])
    stop_index = last_valid_idx + 1
    snapped_point = None

    # === SNAPPING: Berechne exakten Stop-Punkt zwischen last_valid und nächst ungültig ===
    # Wenn stop_index < len(coords_2d), interpoliere linear zu exaktem Zielabstand
    if stop_index < len(coords_2d):
        p_valid = coords_2d[last_valid_idx]
        p_next = coords_2d[stop_index]
        dir_vec = p_next - p_valid
        dist_seg = np.linalg.norm(dir_vec)

        if dist_seg > 1e-6:
            # Iterativ: teste t ∈ [0, 1] wo beide Kanten + Center >= min_edge_distance
            best_t = 0.0
            for t in np.linspace(0.0, 1.0, 20):
                p_test = p_valid + t * dir_vec

                # Berechne Senkrechte interpoliert zwischen den beiden Punkten
                # Linear interpolieren zwischen perp[last_valid_idx] und perp[stop_index]
                if stop_index < len(perp):
                    perp_test = perp[last_valid_idx] * (1.0 - t) + perp[stop_index] * t
                    perp_norm = np.linalg.norm(perp_test)
                    if perp_norm > 1e-6:
                        perp_test = perp_test / perp_norm
                    else:
                        perp_test = perp[last_valid_idx]
                else:
                    perp_test = perp[last_valid_idx]

                left_test = p_test + perp_test * half_width
                right_test = p_test - perp_test * half_width
                dist_left = np.linalg.norm(left_test - center_2d)
                dist_right = np.linalg.norm(right_test - center_2d)
                dist_center = np.linalg.norm(p_test - center_2d)

                # Prüfe: beide Kanten + Center >= min_edge_distance?
                if (
                    dist_left >= min_edge_distance
                    and dist_right >= min_edge_distance
                    and dist_center >= min_edge_distance
                ):
                    best_t = t
                else:
                    break

            # Berechne gesnappten Punkt
            snapped_point = p_valid + best_t * dir_vec

    return min(stop_index, len(coords_2d)), snapped_point


def clip_road_to_bounds(coords, bounds_local):
    """Clippt eine Strasse an den Grid-Bounds (lokale Koordinaten).

    Findet das längste zusammenhängende Segment innerhalb des Grids.
    """
    if not coords or bounds_local is None:
        return coords

    # bounds_local ist (min_x, max_x, min_y, max_y)
    min_x, max_x, min_y, max_y = bounds_local
    # Erweiterung der Bounds falls ROAD_CLIP_MARGIN negativ ist ("rausziehen")
    buffer = 0.0
    try:
        margin = float(config.ROAD_CLIP_MARGIN)
        if margin < 0:
            buffer = -margin
    except Exception:
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
    qe_left = quad_end + perp * half_width
    qs_right = quad_start - perp * half_width
    qe_right = quad_end - perp * half_width

    # Vertex-Order so, dass die ersten beiden die linke Kante bilden, die zweiten beiden die rechte
    quad_vertices = [
        (qs_left[0], qs_left[1], z_value),  # 0: left start
        (qe_left[0], qe_left[1], z_value),  # 1: left end
        (qs_right[0], qs_right[1], z_value),  # 2: right start
        (qe_right[0], qe_right[1], z_value),  # 3: right end
    ]

    # Faces kompatibel mit späterer Streifen-Triangulation
    quad_faces = [
        (0, 2, 3),
        (0, 3, 1),
    ]

    # 2D-Polygon für Grid-Ausschneiden (links vor, rechts zurück)
    quad_poly_2d = [
        (qs_left[0], qs_left[1]),
        (qe_left[0], qe_left[1]),
        (qe_right[0], qe_right[1]),
        (qs_right[0], qs_right[1]),
    ]

    return quad_vertices, quad_faces, quad_poly_2d


def _process_road_batch(
    batch,
    height_points,
    height_elevations,
    local_offset,
    grid_bounds,
    lookup_mode,
    junction_centers=None,
    junctions_full=None,
    all_road_polygons=None,
    junction_road_counts=None,
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

    def count_roads_at_junction(jid):
        if junction_road_counts and jid in junction_road_counts:
            return junction_road_counts[jid]
        return None

    batch_vertices = []
    batch_per_road = []
    clipped_local = 0

    for original_road_idx, road in batch:
        coords = road["coords"]
        road_id = road.get("id")
        osm_tags = road.get("osm_tags", {})  # OSM-Tags extrahieren

        # Berechne Straßenbreite dynamisch aus OSM-Tags
        road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
        half_width = road_width / 2.0

        junction_indices = road.get("junction_indices", {}) or {}

        junction_buffer_start = None
        junction_buffer_end = None

        # WICHTIG: Buffer-Berechnung mit ORIGINAL coords aus all_road_polygons durchführen,
        # nicht mit den bereits geclippten/getrimmten coords!
        original_coords_for_buffer = road["coords"]  # Original aus dem Batch

        # Trim am Endpunkt (Road-Ende) falls Junction vorhanden
        end_junction_idx = junction_indices.get("end")
        if end_junction_idx is not None and junction_centers is not None:
            if 0 <= end_junction_idx < len(junction_centers):
                # Berechne dynamischen junction_stop_buffer basierend auf Winkeln
                # VERWENDE ORIGINAL COORDS, nicht die lokalen (bereits geclippten) coords!
                junction_buffer, max_half_width = calculate_junction_buffer(
                    original_road_idx,
                    end_junction_idx,
                    original_coords_for_buffer,  # Verwende original coords!
                    junction_centers,
                    junctions_full,
                    all_road_polygons,
                    is_end_junction=True,
                    road_id=road_id,
                )
                junction_buffer_end = junction_buffer

                # VEREINHEITLICHTE LOGIK: Nutze immer den dynamischen Buffer
                # (dieser kann auch 0 sein bei breiten Winkeln >= 90°)
                # Verwende max_half_width aller ankommenden Straßen als Basis
                min_edge_distance = max_half_width + junction_buffer

                coords_arr = np.asarray(coords)
                stop_idx, snapped_point = calculate_stop_distance(
                    coords_arr[:, :2],
                    np.asarray(junction_centers[end_junction_idx])[:2],
                    road_width=half_width * 2.0,
                    min_edge_distance=min_edge_distance,
                )
                # Wende Snapping an, wenn gesnappter Punkt vorhanden
                if snapped_point is not None and stop_idx > 0:
                    # Ersetze die Höhe des gesnappten Punktes durch lineare Interpolation
                    old_point_3d = coords[stop_idx - 1] if stop_idx < len(coords) else coords[-1]
                    z_interp = old_point_3d[2]  # Höhe vom letzten Punkt
                    coords[stop_idx - 1] = (*snapped_point, z_interp)
                coords = coords[:stop_idx] if stop_idx > 0 else []

        # Trim am Startpunkt (Road-Anfang) falls Junction vorhanden
        start_junction_idx = junction_indices.get("start")
        if start_junction_idx is not None and junction_centers is not None and len(coords) > 0:
            if 0 <= start_junction_idx < len(junction_centers):
                # Berechne dynamischen junction_stop_buffer basierend auf Winkeln
                # VERWENDE ORIGINAL COORDS, nicht die lokalen (bereits getrimmten) coords!
                junction_buffer, max_half_width = calculate_junction_buffer(
                    original_road_idx,
                    start_junction_idx,
                    original_coords_for_buffer,  # Verwende original coords!
                    junction_centers,
                    junctions_full,
                    all_road_polygons,
                    is_end_junction=False,
                    road_id=road_id,
                )
                junction_buffer_start = junction_buffer

                # VEREINHEITLICHTE LOGIK: Nutze immer den dynamischen Buffer
                # (dieser kann auch 0 sein bei breiten Winkeln >= 90°)
                # Verwende max_half_width aller ankommenden Straßen als Basis
                min_edge_distance = max_half_width + junction_buffer

                coords_rev = list(reversed(coords))
                coords_arr = np.asarray(coords_rev)
                stop_idx, snapped_point = calculate_stop_distance(
                    coords_arr[:, :2],
                    np.asarray(junction_centers[start_junction_idx])[:2],
                    road_width=half_width * 2.0,
                    min_edge_distance=min_edge_distance,
                )
                # Wende Snapping an, wenn gesnappter Punkt vorhanden
                if snapped_point is not None and stop_idx > 0:
                    # Ersetze die Höhe des gesnappten Punktes
                    old_point_3d = coords_rev[stop_idx - 1] if stop_idx < len(coords_rev) else coords_rev[-1]
                    z_interp = old_point_3d[2]
                    coords_rev[stop_idx - 1] = (*snapped_point, z_interp)
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

                    # Erstelle minimales Quad zwischen den beiden Punkten (mit korrekter Straßenbreite!)
                    quad_verts, quad_faces, quad_poly_2d = create_minimal_quad_for_junction(
                        point1[:2], point2[:2], z_val, half_width=half_width
                    )

                    if quad_verts and quad_faces:
                        batch_vertices.extend(quad_verts)
                        left_start = len(batch_vertices) - 4
                        # Rechte Seite startet zwei Indices nach links (0,1 = left; 2,3 = right)
                        right_start = left_start + 2

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
                                # Auch im Minimal-Quad-Fall konsistente Buffer-Werte mitschreiben
                                "junction_buffer_start": (
                                    junction_buffer_start if junction_buffer_start is not None else 0.0
                                ),
                                "junction_buffer_end": junction_buffer_end if junction_buffer_end is not None else 0.0,
                                "num_points": 2,
                                "left_start": left_start,
                                "right_start": right_start,
                                "slope_left_start": -1,
                                "slope_right_start": -1,
                                "road_poly_2d": quad_poly_2d,
                                "slope_poly_2d": quad_poly_2d,
                                "road_left_2d": [quad_poly_2d[0], quad_poly_2d[1]],
                                "road_right_2d": [quad_poly_2d[3], quad_poly_2d[2]],
                                # Kopiere ALLE road-Metadaten für einheitliche Struktur
                                "source_road": road,  # Gesamtes Original-Road-Dict für Zugriff auf tags etc.
                                "osm_tags": osm_tags,  # OSM-Tags bei Minimal-Quads ebenso übernehmen
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

        # WICHTIG: Straßen-Centerline Z-Werte sind bereits normalisiert (aus polygon.py)!
        # Nutze sie direkt, KEINE erneute Interpolation!
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
        min_slope = config.MIN_SLOPE_WIDTH
        slope_width_left = np.clip(np.maximum(min_slope, abs_left), None, 30.0)
        slope_width_right = np.clip(np.maximum(min_slope, abs_right), None, 30.0)

        # Keine Transformation mehr noetig - ALLE Koordinaten sind bereits lokal
        left_local = (left_xy[:, 0], left_xy[:, 1], z_vals)
        right_local = (right_xy[:, 0], right_xy[:, 1], z_vals)

        road_left_vertices = list(zip(left_local[0], left_local[1], left_local[2]))
        road_right_vertices = list(zip(right_local[0], right_local[1], right_local[2]))

        # Debug: Prüfe erste Straßen-Z-Werte
        if original_road_idx < 3:
            print(
                f"    [DEBUG road_mesh] Road {original_road_idx}: centerline_z_raw={z_vals[0]:.2f}, "
                f"left_terrain={terrain_left_height[0]:.2f}, right_terrain={terrain_right_height[0]:.2f}"
            )

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
                "osm_tags": osm_tags,  # OSM-Tags hinzufügen
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
    slope_gradient = np.tan(np.radians(config.SLOPE_ANGLE))

    junction_centers = None
    junctions_full = None
    if junctions:
        junction_centers = [np.asarray(j["position"]) for j in junctions if "position" in j]
        junctions_full = junctions  # Vollständige Junction-Objekte für Anzahl-Prüfung
    junction_stop_buffer = config.JUNCTION_STOP_BUFFER

    # NEU: Strukturiertes Array statt drei separate Arrays
    # Jeder Eintrag: {'vertices': [v0, v1, v2], 'road_id': road_id, 'uvs': {v0: (u,v), ...}}
    road_mesh_data = []
    road_slope_polygons_2d = []

    # Mapping: original road_polygons index -> road_slope_polygons_2d index
    # (wegen Clipping koennen Indices unterschiedlich sein)
    original_to_mesh_idx = {}

    total_roads = len(road_polygons)

    use_mp = config.USE_MULTIPROCESSING
    num_workers = config.NUM_WORKERS or None
    local_offset = config.LOCAL_OFFSET
    grid_bounds = config.GRID_BOUNDS_LOCAL

    lookup_mode = (config.HEIGHT_LOOKUP_MODE or "kdtree").lower()

    # Pre-compute junction road counts (global view before batching)
    junction_road_counts = {}
    if junction_centers is not None:
        for jid in range(len(junction_centers)):
            cnt = 0
            for r in road_polygons:
                ji = r.get("junction_indices", {}) or {}
                if ji.get("start") == jid or ji.get("end") == jid:
                    cnt += 1
            junction_road_counts[jid] = cnt

    worker_func = partial(
        _process_road_batch,
        height_points=height_points,
        height_elevations=height_elevations,
        local_offset=local_offset,
        grid_bounds=grid_bounds,
        lookup_mode=lookup_mode,
        junction_centers=junction_centers,
        junctions_full=junctions_full,
        all_road_polygons=road_polygons,
        junction_road_counts=junction_road_counts,
    )

    per_road_data = []
    all_vertices_concat = []
    clipped_roads = 0

    if use_mp and total_roads > 0:
        workers = num_workers or None
        max_roads_per_batch = config.MAX_ROADS_PER_BATCH or 500
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
        max_roads_per_batch = config.MAX_ROADS_PER_BATCH or 500
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
        junction_fans = {
            idx: {
                "center_idx": None,
                "rim": [],  # list of dicts {left_idx, right_idx, tangent}
                "connected_road_ids": [],
                "vertex_tangent": {},  # vertex_idx -> tangent vec (3,)
            }
            for idx in range(len(junction_centers))
        }

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
        road_id = road_meta["road_id"]  # Vor _add_rim definieren, damit es in der Closure verfügbar ist

        # Centerline als Array (für Tangenten und UV-Berechnung)
        road_centerline = np.array(road_meta["trimmed_centerline"])

        def _add_rim(jid, left_idx, right_idx, tangent_vec):
            if junction_fans is None or jid is None:
                return
            data = junction_fans.get(jid)
            if data is None:
                return
            if data["center_idx"] is None and junction_centers is not None:
                if 0 <= jid < len(junction_centers):
                    cx, cy, cz = junction_centers[jid]
                    data["center_idx"] = vertex_manager.add_vertex(cx, cy, cz)
            data["rim"].append({"left_idx": left_idx, "right_idx": right_idx, "tangent": tangent_vec})
            # Speichere auch die angrenzende Straßen-ID für Material-Selection
            if "connected_road_ids" not in data:
                data["connected_road_ids"] = []
            data["connected_road_ids"].append(road_id)

            # Speichere Tangente pro Vertex für spätere UV-Ableitung
            if tangent_vec is not None:
                data.setdefault("vertex_tangent", {})[left_idx] = tangent_vec
                data.setdefault("vertex_tangent", {})[right_idx] = tangent_vec

        # Sammle Endkanten für Junction-Hub-Fans
        # Tangenten an Road-Anfang/Ende (Centerline)
        start_tangent = None
        end_tangent = None
        if road_centerline.shape[0] >= 2:
            start_vec = road_centerline[1] - road_centerline[0]
            sv_norm = np.linalg.norm(start_vec)
            if sv_norm > 1e-6:
                start_tangent = (start_vec / sv_norm).tolist()
        if road_centerline.shape[0] >= 2:
            end_vec = road_centerline[-1] - road_centerline[-2]
            ev_norm = np.linalg.norm(end_vec)
            if ev_norm > 1e-6:
                end_tangent = (end_vec / ev_norm).tolist()

        _add_rim(start_jid, road_vertex_indices_left[0], road_vertex_indices_right[0], start_tangent)
        _add_rim(end_jid, road_vertex_indices_left[-1], road_vertex_indices_right[-1], end_tangent)

        # Berechne UV-Koordinaten für diesen Road-Strip
        # U = Distance entlang Centerline, V = Querposition (0=links, 1=rechts)
        road_centerline = np.array(road_meta["trimmed_centerline"])
        u_coords = compute_road_uv_coords(road_centerline, tiling_distance=10.0)

        # Strassen-Faces
        for i in range(n - 1):
            left1 = road_vertex_indices_left[i]
            left2 = road_vertex_indices_left[i + 1]
            right1 = road_vertex_indices_right[i]
            right2 = road_vertex_indices_right[i + 1]

            # UV-Koordinaten für dieses Quad
            # GEDREHT 90°: left = (0, u), right = (1, u)
            # Dies ermöglicht, dass die Straßentextur in Fahrtrichtung korrekt verläuft
            u_curr = u_coords[i]
            u_next = u_coords[i + 1]

            uv_left1 = (0.0, u_curr)
            uv_left2 = (0.0, u_next)
            uv_right1 = (1.0, u_curr)
            uv_right2 = (1.0, u_next)

            # Quad: left1-right1-right2-left2 → zwei Dreiecke
            # Winding-Order wird automatisch in Mesh.add_face() korrigiert

            # Dreieck 1: left1-right1-right2
            road_mesh_data.append(
                {
                    "vertices": [left1, right1, right2],
                    "road_id": road_id,
                    "uvs": {left1: uv_left1, right1: uv_right1, right2: uv_right2},
                }
            )

            # Dreieck 2: left1-right2-left2
            road_mesh_data.append(
                {
                    "vertices": [left1, right2, left2],
                    "road_id": road_id,
                    "uvs": {left1: uv_left1, right2: uv_right2, left2: uv_left2},
                }
            )

        # Böschungen sind deaktiviert (config.GENERATE_SLOPES=False)

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
            "osm_tags": road_meta.get("osm_tags", {}),  # OSM-Tags hinzufügen
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
            vertex_tangent = data.get("vertex_tangent", {})
            for rim_entry in rim:
                if isinstance(rim_entry, dict):
                    left_idx = rim_entry.get("left_idx")
                    right_idx = rim_entry.get("right_idx")
                else:
                    left_idx, right_idx = rim_entry
                for vid in (left_idx, right_idx):
                    vx, vy = vertex_manager.vertices[vid][:2]
                    ang = atan2(vy - center_xy[1], vx - center_xy[0])
                    boundary_points.append((ang, vid))

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

                # UV-Achsen aus Tangenten ableiten (Option 1: entlang Road-Tangent)
                t_a = np.array(vertex_tangent.get(a), dtype=float) if a in vertex_tangent else None
                t_b = np.array(vertex_tangent.get(b), dtype=float) if b in vertex_tangent else None

                # Mittel-Tangent bestimmen
                if t_a is not None and t_b is not None:
                    t_axis = t_a + t_b
                elif t_a is not None:
                    t_axis = t_a
                elif t_b is not None:
                    t_axis = t_b
                else:
                    # Fallback: radial Richtung zu a
                    pa = vertex_manager.vertices[a]
                    pc = vertex_manager.vertices[center_idx]
                    t_axis = np.array(pa - pc, dtype=float)

                t_norm = np.linalg.norm(t_axis)
                if t_norm > 1e-12:
                    u_axis = t_axis / t_norm
                else:
                    u_axis = np.array([1.0, 0.0, 0.0])

                # V-Achse als 2D-Perp im XY-Plane
                v_axis = np.array([-u_axis[1], u_axis[0], 0.0])
                v_norm = np.linalg.norm(v_axis)
                if v_norm > 1e-12:
                    v_axis /= v_norm
                else:
                    v_axis = np.array([0.0, 1.0, 0.0])

                # Punkte
                pa = vertex_manager.vertices[a]
                pb = vertex_manager.vertices[b]
                pc = vertex_manager.vertices[center_idx]

                # Projektion auf Achsen, skaliert wie Straßen (10m Tiling)
                scale = 10.0

                def proj_uv(p):
                    d = p - pc
                    u = np.dot(d, u_axis) / scale
                    v = np.dot(d, v_axis) / scale
                    return (u, v)

                uv_a = proj_uv(pa)
                uv_b = proj_uv(pb)
                uv_c = proj_uv(pc)

                road_mesh_data.append(
                    {
                        "vertices": [a, b, center_idx],
                        "road_id": -(j_id + 1),
                        "uvs": {a: uv_a, b: uv_b, center_idx: uv_c},
                    }
                )

    print(f"  [OK] {len(road_mesh_data)} Strassen-Faces")
    if config.GENERATE_SLOPES:
        print(f"  [OK] {len(all_slope_faces)} Boeschungs-Faces")
        print(f"  [OK] Boeschungen OK")
    else:
        print(f"  [i] Boeschungs-Generierung deaktiviert (config.GENERATE_SLOPES=False)")

    # === Junction Z-Glättung (nach Fan-Triangulation) ===
    if junction_fans:
        print(f"  Glaette Junction-Zentralpunkte in Z-Richtung...")
        smooth_junction_centers_z(junction_fans, vertex_manager)

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
            for rim_entry in rim:
                left_idx = rim_entry.get("left_idx") if isinstance(rim_entry, dict) else rim_entry[0]
                right_idx = rim_entry.get("right_idx") if isinstance(rim_entry, dict) else rim_entry[1]
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
        road_mesh_data,  # Strukturiert: [{'vertices': [...], 'road_id': ..., 'uvs': {...}}, ...]
        road_slope_polygons_2d,
        original_to_mesh_idx,
        all_road_polygons_2d,
        junction_fans,  # Für Material-Selection bei Junctions
    )
