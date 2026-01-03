"""
Grid-Vertex Klassifizierung und Polygon-Tests.
"""

import numpy as np
import time as time_module
from shapely.geometry import Polygon, LineString, MultiPolygon
from scipy.spatial import cKDTree
from matplotlib.path import Path

from ..geometry.polygon import get_road_centerline_robust
from .. import config


def classify_grid_vertices(grid_points, grid_elevations, road_slope_polygons_2d):
    """
    Markiert Grid-Vertices unter Straßen.
    SUPER-OPTIMIERT: Nutzt KDTree + geometrische Tests!
    """
    print("  Markiere Strassen-Bereiche im Grid (KDTree + Face-Überlappung)...")

    vertex_types = np.zeros(len(grid_points), dtype=int)
    modified_heights = grid_elevations.copy()

    # Grid-Punkte sind bereits in lokalen Koordinaten (wie Polygone)
    grid_points_2d = grid_points[:, :2]

    # Konvertiere zu Shapely-Polygone
    print("  Konvertiere Polygone zu Shapely...")
    road_data = []

    for idx, poly_data in enumerate(road_slope_polygons_2d):
        road_poly_xy = poly_data["road_polygon"]
        trimmed_centerline = poly_data.get("trimmed_centerline", [])

        # Road-Polygon muss vorhanden sein
        if len(road_poly_xy) < 3:
            continue

        try:
            road_poly = Polygon(road_poly_xy)

            # Validiere Polygon
            if not road_poly.is_valid:
                road_poly = road_poly.buffer(0)
                # Nach buffer(0) kann MultiPolygon entstehen - nimm groesstes Teil
                if isinstance(road_poly, MultiPolygon):
                    road_poly = max(road_poly.geoms, key=lambda p: p.area)

            # Nochmal pruefen nach Fix
            if not isinstance(road_poly, Polygon):
                continue

            # Verwende die GETRIMTE Centerline (nach Junction-Trimming)!
            # (bereits in lokalen Koordinaten durch get_road_polygons)
            if trimmed_centerline and len(trimmed_centerline) >= 2:
                centerline_coords = np.array([(x, y) for x, y, z in trimmed_centerline])
                has_real_centerline = True
            else:
                # Fallback für Junction-Fans: Nutze Polygon-Center als "Centerline"
                # (Fans werden später per Polygon-Bounds statt Centerline-Sampling verarbeitet)
                centerline_coords = np.array([road_poly.centroid.coords[0]])
                has_real_centerline = False

            road_data.append(
                {
                    "road_geom": road_poly,
                    "centerline_points": centerline_coords,
                    "has_real_centerline": has_real_centerline,
                    "buffer_bounds": road_poly.bounds,
                }
            )
        except Exception:
            continue

    if not road_data:
        print("  [!] Keine gueltigen Polygone gefunden!")
        return vertex_types, modified_heights

    print(f"  Loaded {len(road_data)} valid roads for processing")

    process_start = time_module.time()

    # MEGA-OPTIMIERUNG: Baue KDTree ueber alle Grid-Punkte (EINMAL, dann reuse!)
    print(f"  Baue KDTree fuer {len(grid_points_2d)} Grid-Punkte...")
    kdtree = cKDTree(grid_points_2d)

    print(f"  Teste {len(road_data)} Roads gegen Grid-Punkte...")

    for road_num, road_info in enumerate(road_data):
        centerline_points = road_info["centerline_points"]
        has_real_centerline = road_info.get("has_real_centerline", True)

        # Zwei Modi:
        # 1. Regular Roads (mit Centerline): Sample entlang Centerline + KDTree
        # 2. Junction Fans (ohne Centerline): Nutze Polygon-Bounds für KDTree

        if has_real_centerline and len(centerline_points) >= 2:
            # Modus 1: Sample entlang Centerline
            centerline_linestring = LineString(centerline_points)

            if not centerline_linestring.is_valid or centerline_linestring.length == 0:
                continue

            total_length = centerline_linestring.length
            sample_spacing = config.CENTERLINE_SAMPLE_SPACING

            # Stelle sicher, dass auch sehr kurze Segmente mindestens mit Start+Ende abgetastet werden
            num_samples = max(2, int(np.ceil(total_length / sample_spacing)) + 1)
            sample_distances = np.linspace(0, total_length, num_samples)

            try:
                centerline = np.array(
                    [np.array(centerline_linestring.interpolate(dist).coords[0]) for dist in sample_distances]
                )
            except Exception:
                continue

            if len(centerline) == 0:
                continue

            buffer_indices_set = set()
            search_radius = config.CENTERLINE_SEARCH_RADIUS

            # Centerline-Samples
            for centerline_pt in centerline:
                nearby = kdtree.query_ball_point(centerline_pt, r=search_radius)
                buffer_indices_set.update(nearby)

            buffer_indices = list(buffer_indices_set)
        else:
            # Modus 2: Junction Fan - nutze Polygon-Bounds
            minx, miny, maxx, maxy = road_info["buffer_bounds"]

            # Finde alle Grid-Punkte in Bounding Box
            in_bbox = (
                (grid_points_2d[:, 0] >= minx)
                & (grid_points_2d[:, 0] <= maxx)
                & (grid_points_2d[:, 1] >= miny)
                & (grid_points_2d[:, 1] <= maxy)
            )
            buffer_indices = np.where(in_bbox)[0].tolist()

        if len(buffer_indices) == 0:
            continue

        # VEKTORISIERT: Markiere ALLE Vertices die der KDTree findet!
        # Bei 2m Grid und 7m Strassenbreite koennen Vertices ausserhalb des Polygons liegen
        # Der KDTree-Radius (10m) erfasst alle relevanten Vertices

        candidate_points = grid_points_2d[buffer_indices]

        # Test gegen Road-Polygon (7m Breite)
        # WICHTIG: road_geom enthält Regular Roads (getrimmt) UND Junction-Fans (aus main())
        road_coords = np.array(road_info["road_geom"].exterior.coords)
        road_path = Path(road_coords)
        inside_road = road_path.contains_points(candidate_points)

        # Markiere alle Vertices innerhalb des Road-Polygons
        for inside_idx in range(len(buffer_indices)):
            pt_idx = buffer_indices[inside_idx]
            if inside_road[inside_idx]:
                vertex_types[pt_idx] = 1  # Flag zum Ausblenden

    elapsed_total = time_module.time() - process_start
    marked_count = np.count_nonzero(vertex_types)

    print(f"  [OK] Klassifizierung abgeschlossen ({elapsed_total:.1f}s, {marked_count} Punkte markiert)")

    return vertex_types, modified_heights
