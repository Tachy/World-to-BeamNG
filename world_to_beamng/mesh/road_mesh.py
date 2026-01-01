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

        slope_left_outer_xy = left_xy + perp * slope_width_left[:, np.newaxis]
        slope_right_outer_xy = right_xy - perp * slope_width_right[:, np.newaxis]
        slope_left_outer_z = z_vals + height_diff_left
        slope_right_outer_z = z_vals + height_diff_right

        # Keine Transformation mehr noetig - ALLE Koordinaten sind bereits lokal
        left_local = (left_xy[:, 0], left_xy[:, 1], z_vals)
        right_local = (right_xy[:, 0], right_xy[:, 1], z_vals)

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

        road_left_vertices = list(zip(left_local[0], left_local[1], left_local[2]))
        road_right_vertices = list(zip(right_local[0], right_local[1], right_local[2]))
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

        # Verwende LOKALE Koordinaten fuer 2D-Polygone (nach Offset-Transformation)
        road_left_2d = [(v[0], v[1]) for v in road_left_vertices]
        road_right_2d = [(v[0], v[1]) for v in road_right_vertices]
        road_poly_2d = road_left_2d + list(reversed(road_right_2d))

        slope_left_2d = [(v[0], v[1]) for v in slope_left_outer_vertices]
        slope_right_2d = [(v[0], v[1]) for v in slope_right_outer_vertices]
        # Korrigierte Reihenfolge: Aussenkontur gegen Uhrzeigersinn
        slope_poly_2d = slope_left_2d + list(reversed(slope_right_2d))

        left_start = len(batch_vertices)
        batch_vertices.extend(road_left_vertices)
        right_start = len(batch_vertices)
        batch_vertices.extend(road_right_vertices)
        slope_left_start = len(batch_vertices)
        batch_vertices.extend(slope_left_outer_vertices)
        slope_right_start = len(batch_vertices)
        batch_vertices.extend(slope_right_outer_vertices)

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
        # Single-thread fallback
        for original_road_idx, road in enumerate(road_polygons):
            (
                batch_per_road,
                batch_vertices,
                clipped_local,
            ) = worker_func([(original_road_idx, road)])

            offset = len(all_vertices_concat)
            all_vertices_concat.extend(batch_vertices)
            for entry in batch_per_road:
                entry["left_start"] += offset
                entry["right_start"] += offset
                entry["slope_left_start"] += offset
                entry["slope_right_start"] += offset

            per_road_data.extend(batch_per_road)
            clipped_roads += clipped_local

            processed = original_road_idx + 1
            if processed % 50 == 0:
                print(f"  {processed}/{total_roads} Strassen (Geometrie)...")

    # === Globaler Insert in einem Rutsch ===
    print(
        f"  Fuege {len(all_vertices_concat):,} Strassen/Boeschungs-Vertices in einem Rutsch hinzu..."
    )
    global_indices = vertex_manager.add_vertices_batch_dedup_fast(all_vertices_concat)
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

            all_road_faces.append([left1, right1, right2])
            all_road_face_to_idx.append(road_id)
            all_road_faces.append([left1, right2, left2])
            all_road_face_to_idx.append(road_id)

        # Boeschungs-Faces
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
        road_slope_polygons_2d.append(
            {
                "road_polygon": road_meta["road_poly_2d"],
                "slope_polygon": road_meta["slope_poly_2d"],
                "original_coords": road_meta["original_coords"],
                "road_vertex_indices": {
                    "left": road_vertex_indices_left,
                    "right": road_vertex_indices_right,
                },
                "slope_outer_indices": {
                    "left": slope_left_outer_indices,
                    "right": slope_right_outer_indices,
                },
            }
        )

        original_idx = road_meta["original_idx"]
        original_to_mesh_idx[original_idx] = len(road_slope_polygons_2d) - 1

    print(f"  [OK] {len(all_road_faces)} Strassen-Faces")
    print(f"  [OK] {len(all_slope_faces)} Boeschungs-Faces")

    print(f"  [OK] Boeschungen OK")
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

    return (
        all_road_faces,
        all_road_face_to_idx,
        all_slope_faces,
        road_slope_polygons_2d,
        original_to_mesh_idx,
    )
