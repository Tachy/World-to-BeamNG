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
from math import ceil

from .. import config


def calculate_stop_distance(
    centerline_coords, junction_center, road_width, min_edge_distance=1.0
):
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
    dist_right_to_center = np.linalg.norm(
        right_edges - center_2d[np.newaxis, :], axis=1
    )

    # Constraint: Beide Kanten müssen >= min_edge_distance vom Center entfernt sein
    valid_indices = np.where(
        (dist_left_to_center >= min_edge_distance)
        & (dist_right_to_center >= min_edge_distance)
    )[0]

    if len(valid_indices) == 0:
        # Straße startet schon zu nah an Junction
        return 0

    # Finde den letzten gültigen Index
    stop_index = int(valid_indices[-1]) + 1

    return min(stop_index, len(coords_2d))


def clip_road_to_bounds(coords, bounds_local):
    """Clippt eine Strasse an den Grid-Bounds (lokale Koordinaten)."""
    if not coords or bounds_local is None:
        return coords

    # bounds_local ist (min_x, max_x, min_y, max_y)
    min_x, max_x, min_y, max_y = bounds_local
    buffer = 0.0
    min_x -= buffer
    min_y -= buffer
    max_x += buffer
    max_y += buffer

    clipped = []

    for x, y, z in coords:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            clipped.append((x, y, z))
        elif clipped:
            break

    return clipped


def _process_road_batch(
    batch,
    half_width,
    height_points,
    height_elevations,
    local_offset,
    grid_bounds,
    lookup_mode,
):
    """Worker-Funktion fuer Strassen-Batch (multiprocessing-fähig)."""
    use_kdtree = lookup_mode == "kdtree"
    tree = cKDTree(height_points) if use_kdtree else None
    interpolator = (
        NearestNDInterpolator(height_points, height_elevations)
        if not use_kdtree
        else None
    )

    if local_offset is None:
        local_offset = (0.0, 0.0, 0.0)

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

        if len(coords) < 2:
            continue

        coords = clip_road_to_bounds(coords, grid_bounds)

        if len(coords) < 2:
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
                "original_coords": coords,
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


def generate_road_mesh_strips(
    road_polygons, height_points, height_elevations, vertex_manager
):
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

                print(
                    f"  Batch {idx}/{len(batches)} fertig (Vertices: {len(batch_vertices):,})"
                )
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

            print(
                f"  Batch {idx}/{len(batches)} fertig (Vertices: {len(batch_vertices):,})"
            )

    # === Globaler Insert in einem Rutsch ===
    print(
        f"  Fuege {len(all_vertices_concat):,} Strassen/Boeschungs-Vertices in einem Rutsch hinzu..."
    )
    # WICHTIG: Keine globale Deduplizierung über Roads hinweg!
    # Jede Road hat ihre eigenen Vertices - sie dürfen nicht mit anderen Roads geteilt werden
    # Das würde zu falschen Verbindungen zwischen verschiedenen Roads führen
    global_indices = []
    for i, vertex in enumerate(all_vertices_concat):
        idx = vertex_manager.add_vertex(vertex[0], vertex[1], vertex[2])
        global_indices.append(idx)

    print(
        f"  [OK] VertexManager: {vertex_manager.get_count():,} Vertices nach Strassen"
    )

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
            "original_coords": road_meta["original_coords"],
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

    print(f"  [OK] {len(all_road_faces)} Strassen-Faces")
    if config.GENERATE_SLOPES:
        print(f"  [OK] {len(all_slope_faces)} Boeschungs-Faces")
        print(f"  [OK] Boeschungen OK")
    else:
        print(
            f"  [i] Boeschungs-Generierung deaktiviert (config.GENERATE_SLOPES=False)"
        )
    print(
        f"  [OK] {len(road_slope_polygons_2d)} Road/Slope-Polygone fuer Grid-Ausschneiden (2D)"
    )
    if clipped_roads > 0:
        print(f"  [i] {clipped_roads} Strassen komplett ausserhalb Grid (ignoriert)")

    # T-Junction Snapping: JETZT DEAKTIVIERT - wird direkt in Schritt 6a erkannt
    # Die Junctions werden bereits in der Pipeline erkannt und in generate_road_mesh_strips verarbeitet
    if config.ENABLE_ROAD_EDGE_SNAPPING:
        print(
            f"  [i] Junction-Handling: wird direkt in Centerlines erkannt (nicht mehr nachträglich)"
        )

    # Sammle alle Road-Polygone (2D) für Grid-Conforming
    all_road_polygons_2d = [
        poly_data["road_polygon"] for poly_data in road_slope_polygons_2d
    ]
    all_slope_polygons_2d = [
        poly_data["slope_polygon"] for poly_data in road_slope_polygons_2d
    ]

    return (
        all_road_faces,
        all_road_face_to_idx,
        all_slope_faces,
        road_slope_polygons_2d,
        original_to_mesh_idx,
        all_road_polygons_2d,  # ← NEU
        all_slope_polygons_2d,  # ← NEU
    )
