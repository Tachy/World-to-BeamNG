"""
Face-zu-Face Überlappungspruefung (KDTree-basiert, ULTRA-OPTIMIERT).
"""

import numpy as np
import time as time_module
from shapely.geometry import Polygon as ShapelyPolygon, LineString
from shapely.prepared import prep
from scipy.spatial import cKDTree

from ..geometry.polygon import get_road_centerline_robust
from .. import config


def _process_road_chunk(args):
    """
    Worker-Funktion fuer Multiprocessing: Prueft einen Chunk von Roads gegen Terrain-Faces.
    Muss top-level Funktion sein (fuer pickle).
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

    # Baue STRtree fuer diesen Worker
    spatial_index = STRtree(terrain_triangles)

    # Lokale face_types fuer diesen Chunk
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


def check_face_overlaps(
    grid_points, terrain_faces, road_slope_polygons_2d, vertex_types=None
):
    """
    Pruefe Terrain-Faces auf Überlappung mit Road/Slope Polygonen.
    ULTRA-OPTIMIERT: Nutzt KDTree + Centerline-basierte Vorfilterung.
    Prueft nur Faces, deren Vertices in 7m-Radius der Centerline liegen!

    Args:
        vertex_types: Optional array marking vertices as road (2), slope (1), or terrain (0)
                      If provided, uses this for vertex marking validation instead of KDTree
    """
    # Initialisiere face_types
    face_types = np.zeros(len(terrain_faces), dtype=int)

    # Extrahiere nur X-Y Koordinaten
    if grid_points.shape[1] >= 3:
        grid_points_2d = grid_points[:, :2]
    else:
        grid_points_2d = grid_points

    # Baue Terrain-Face Arrays
    print("  Baue Terrain-Face Arrays...")
    terrain_faces_array = np.array(terrain_faces, dtype=np.int32) - 1  # 0-basiert

    # Baue inversen Index: Vertex -> Liste von Face-Indizes
    print("  Baue Vertex-zu-Face Mapping...")
    vertex_to_faces = {}
    for face_idx, (v1, v2, v3) in enumerate(terrain_faces_array):
        if v1 not in vertex_to_faces:
            vertex_to_faces[v1] = []
        if v2 not in vertex_to_faces:
            vertex_to_faces[v2] = []
        if v3 not in vertex_to_faces:
            vertex_to_faces[v3] = []
        vertex_to_faces[v1].append(face_idx)
        vertex_to_faces[v2].append(face_idx)
        vertex_to_faces[v3].append(face_idx)

    print(
        f"  Vertex-Mapping: {len(vertex_to_faces)} Vertices -> {len(terrain_faces_array)} Faces"
    )

    # Baue KDTree ueber Grid-Punkte (EINMAL!)
    print("  Baue KDTree fuer Grid-Punkte...")
    kdtree = cKDTree(grid_points_2d)

    step_start = time_module.time()
    print("  Single-Thread-Modus mit KDTree-Vorfilterung")

    faces_deleted = 0
    search_radius = 8.0  # Erhoeht auf 8.0 fuer Grenzfall-Abdeckung

    # Pruefe jeden Road/Slope Bereich
    for road_num, road_info in enumerate(road_slope_polygons_2d):
        road_start = time_module.time()

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

        # OPTIMIERUNG: Nutze EXAKT GLEICHE Centerline-Berechnung wie in Vertex-Klassifizierung!
        try:
            # Verwende get_road_centerline_robust() fuer Konsistenz
            centerline_coords = get_road_centerline_robust(road_geom)
            if len(centerline_coords) < 2:
                continue

            centerline = LineString(centerline_coords)

            # Sample Punkte entlang der Centerline (alle 10m)
            total_length = centerline.length
            if total_length < 1.0:
                continue

            sample_distances = np.arange(0, total_length + 5, 10.0)
            centerline_samples = np.array(
                [
                    np.array(centerline.interpolate(dist).coords[0])
                    for dist in sample_distances
                ]
            )

        except Exception:
            continue

        # Finde alle Grid-Punkte im Radius um Centerline
        candidate_vertex_indices = set()
        for sample_pt in centerline_samples:
            nearby = kdtree.query_ball_point(sample_pt, r=search_radius)
            candidate_vertex_indices.update(nearby)

        if len(candidate_vertex_indices) == 0:
            continue

        # Finde alle Faces, die mindestens einen Candidate-Vertex verwenden
        # OPTIMIERT: Nutze inversen Index statt Schleife ueber alle Faces!
        candidate_face_indices = set()
        for vertex_idx in candidate_vertex_indices:
            if vertex_idx in vertex_to_faces:
                candidate_face_indices.update(vertex_to_faces[vertex_idx])

        if len(candidate_face_indices) == 0:
            continue

        # Jetzt nur diese Faces gegen Road/Slope-Polygone testen
        road_prepared = prep(road_geom)
        slope_prepared = prep(slope_geom)

        problematic_faces_this_road = 0
        for face_idx in candidate_face_indices:
            if face_types[face_idx] == 0:  # Noch nicht markiert
                v1, v2, v3 = terrain_faces_array[face_idx]
                tri_coords = grid_points_2d[[v1, v2, v3]]
                triangle_obj = ShapelyPolygon(tri_coords)

                # KRITISCH: Nur markieren wenn Triangle ÜBERSCHNEIDET und mindestens ein Vertex NICHT markiert!
                # Das bedeutet: Geometrische Verletzung (Luecke zwischen Grid und Strasse)
                triangle_intersects_road = road_prepared.intersects(triangle_obj)
                triangle_intersects_slope = slope_prepared.intersects(triangle_obj)

                # Pruefe ob alle 3 Vertices bereits als Strasse/Boeschung markiert sind
                if vertex_types is not None:
                    # BESSER: Nutze tatsächliche Markierung aus Vertex-Klassifizierung
                    # (2=Strasse, 1=Boeschung, 0=Gelände)
                    v1_marked = vertex_types[v1] > 0
                    v2_marked = vertex_types[v2] > 0
                    v3_marked = vertex_types[v3] > 0
                else:
                    # FALLBACK: Nutze KDTree Kandidaten
                    v1_marked = v1 in candidate_vertex_indices
                    v2_marked = v2 in candidate_vertex_indices
                    v3_marked = v3 in candidate_vertex_indices

                all_vertices_marked = v1_marked and v2_marked and v3_marked

                # Nur loeschen wenn ueberschneidend UND nicht alle Vertices markiert
                # = echte geometrische Verletzung
                if (
                    triangle_intersects_road or triangle_intersects_slope
                ) and not all_vertices_marked:
                    face_types[face_idx] = 1
                    faces_deleted += 1
                    problematic_faces_this_road += 1

        road_elapsed = time_module.time() - road_start
        total_elapsed = time_module.time() - step_start
        rate = (road_num + 1) / total_elapsed if total_elapsed > 0 else 0
        eta = (len(road_slope_polygons_2d) - road_num - 1) / rate if rate > 0 else 0

        # Zeige nur Roads mit Problemen
        if problematic_faces_this_road > 0:
            print(
                f"    Road {road_num + 1}/{len(road_slope_polygons_2d)}: {road_elapsed:.3f}s | "
                f"{len(candidate_face_indices)} Faces geprueft, {problematic_faces_this_road} PROBLEMATISCH!"
            )

    elapsed = time_module.time() - step_start
    print(
        f"  [OK] Face-Überlappungspruefung abgeschlossen ({elapsed:.1f}s, {faces_deleted} Faces zum Loeschen markiert)"
    )

    return face_types
