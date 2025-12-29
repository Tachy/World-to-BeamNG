"""
Face-zu-Face Überlappungsprüfung (STRtree-basiert).
"""

import numpy as np
import time as time_module
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.prepared import prep
from shapely.strtree import STRtree

from .. import config


def _process_road_chunk(args):
    """
    Worker-Funktion für Multiprocessing: Prüft einen Chunk von Roads gegen Terrain-Faces.
    Muss top-level Funktion sein (für pickle).
    """
    (
        road_chunk,
        grid_points_2d,
        terrain_faces_array,
        terrain_triangles_indices,
        chunk_id,
    ) = args

    # Rekonstruiere terrain_triangles aus Indizes
    terrain_triangles = []
    for idx in terrain_triangles_indices:
        v1, v2, v3 = terrain_faces_array[idx]
        tri_coords = grid_points_2d[[v1, v2, v3]]
        triangle = ShapelyPolygon(tri_coords)
        terrain_triangles.append(triangle)

    # Baue STRtree für diesen Worker
    spatial_index = STRtree(terrain_triangles)

    # Lokale face_types für diesen Chunk
    local_face_types = np.zeros(len(terrain_triangles), dtype=int)
    faces_deleted = 0

    for road_info in road_chunk:
        road_coords = road_info["road_polygon"]
        slope_coords = road_info["slope_polygon"]

        if isinstance(road_coords, list):
            road_geom = ShapelyPolygon(road_coords)
        else:
            road_geom = road_coords

        if isinstance(slope_coords, list):
            slope_geom = ShapelyPolygon(slope_coords)
        else:
            slope_geom = slope_coords

        road_prepared = prep(road_geom)
        slope_prepared = prep(slope_geom)

        road_candidates = spatial_index.query(road_geom)
        slope_candidates = spatial_index.query(slope_geom)
        all_candidates = set(road_candidates) | set(slope_candidates)

        for local_idx in all_candidates:
            if local_face_types[local_idx] == 0:
                triangle_obj = terrain_triangles[local_idx]

                if road_prepared.intersects(triangle_obj):
                    local_face_types[local_idx] = 1
                    faces_deleted += 1
                elif slope_prepared.intersects(triangle_obj):
                    local_face_types[local_idx] = 1
                    faces_deleted += 1

    return (chunk_id, local_face_types, faces_deleted)


def check_face_overlaps(grid_points, terrain_faces, road_slope_polygons_2d):
    """
    Prüfe Terrain-Faces auf Überlappung mit Road/Slope Polygonen.
    ULTRA-OPTIMIERT: Nutzt R-Tree Spatial Index + optionales Multiprocessing.
    """
    # Initialisiere face_types
    face_types = np.zeros(len(terrain_faces), dtype=int)

    # Extrahiere nur X-Y Koordinaten
    if grid_points.shape[1] >= 3:
        grid_points_2d = grid_points[:, :2]
    else:
        grid_points_2d = grid_points

    # Baue Terrain-Face Arrays
    print("  Baue Terrain-Face Polygone...")
    terrain_faces_array = np.array(terrain_faces, dtype=np.int32) - 1  # 0-basiert

    step_start = time_module.time()

    # MULTIPROCESSING oder SINGLE-THREAD?
    if config.USE_MULTIPROCESSING and len(road_slope_polygons_2d) > 100:
        # MULTIPROCESSING-Modus
        import multiprocessing as mp

        if config.NUM_WORKERS is None:
            num_workers = mp.cpu_count()
        else:
            num_workers = min(config.NUM_WORKERS, mp.cpu_count())

        print(f"  Multiprocessing: {num_workers} Worker-Prozesse")

        chunk_size = max(1, len(road_slope_polygons_2d) // num_workers)
        road_chunks = [
            road_slope_polygons_2d[i : i + chunk_size]
            for i in range(0, len(road_slope_polygons_2d), chunk_size)
        ]

        terrain_triangles_indices = list(range(len(terrain_faces_array)))

        worker_args = [
            (chunk, grid_points_2d, terrain_faces_array, terrain_triangles_indices, i)
            for i, chunk in enumerate(road_chunks)
        ]

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_process_road_chunk, worker_args)

        faces_deleted = 0
        for chunk_id, chunk_face_types, chunk_deleted in results:
            face_types = np.maximum(face_types, chunk_face_types)
            faces_deleted += chunk_deleted

        print(f"  ✓ Multiprocessing abgeschlossen")

    else:
        # SINGLE-THREAD-Modus (Fallback oder kleine Datenmengen)
        if not config.USE_MULTIPROCESSING:
            print("  Single-Thread-Modus (USE_MULTIPROCESSING=False)")
        else:
            print("  Single-Thread-Modus (zu wenige Roads für Multiprocessing)")

        # Baue Terrain-Face Polygone
        terrain_triangles = []
        for face_idx in range(len(terrain_faces_array)):
            v1, v2, v3 = terrain_faces_array[face_idx]
            tri_coords = grid_points_2d[[v1, v2, v3]]
            triangle = ShapelyPolygon(tri_coords)
            terrain_triangles.append(triangle)

        # STRtree Spatial Index
        print("  Baue STRtree Spatial Index...")
        spatial_index = STRtree(terrain_triangles)

        faces_deleted = 0

        # Prüfe jeden Road/Slope Bereich
        for road_num, road_info in enumerate(road_slope_polygons_2d):
            road_coords = road_info["road_polygon"]
            slope_coords = road_info["slope_polygon"]

            if isinstance(road_coords, list):
                road_geom = ShapelyPolygon(road_coords)
            else:
                road_geom = road_coords

            if isinstance(slope_coords, list):
                slope_geom = ShapelyPolygon(slope_coords)
            else:
                slope_geom = slope_coords

            road_prepared = prep(road_geom)
            slope_prepared = prep(slope_geom)

            road_candidates_indices = spatial_index.query(road_geom)
            slope_candidates_indices = spatial_index.query(slope_geom)
            all_candidates = set(road_candidates_indices) | set(
                slope_candidates_indices
            )

            for face_idx in all_candidates:
                if face_types[face_idx] == 0:
                    triangle_obj = terrain_triangles[face_idx]

                    if road_prepared.intersects(triangle_obj):
                        face_types[face_idx] = 1
                        faces_deleted += 1
                    elif slope_prepared.intersects(triangle_obj):
                        face_types[face_idx] = 1
                        faces_deleted += 1

            if (road_num + 1) % 100 == 0:
                elapsed = time_module.time() - step_start
                rate = (road_num + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"    {road_num + 1}/{len(road_slope_polygons_2d)} Roads ({rate:.0f}/s, {faces_deleted} Faces markiert)"
                )

    elapsed = time_module.time() - step_start
    print(
        f"  ✓ Face-Überlappungsprüfung abgeschlossen ({elapsed:.1f}s, {faces_deleted} Faces zum Löschen markiert)"
    )

    return face_types
