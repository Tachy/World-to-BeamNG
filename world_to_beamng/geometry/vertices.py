"""
Grid-Vertex Klassifizierung und Polygon-Tests.
"""

import numpy as np
import time as time_module
from shapely.geometry import Polygon, LineString
from scipy.spatial import cKDTree
from matplotlib.path import Path

from ..geometry.polygon import get_road_centerline_robust
from .. import config


def classify_grid_vertices(grid_points, grid_elevations, road_slope_polygons_2d):
    """
    Markiert Grid-Vertices unter Straßen/Böschungen.
    SUPER-OPTIMIERT: Nutzt KDTree + geometrische Tests!
    """
    print(
        "\nMarkiere Straßen-/Böschungsbereiche im Grid (KDTree + Face-Überlappung)..."
    )

    vertex_types = np.zeros(len(grid_points), dtype=int)
    modified_heights = grid_elevations.copy()

    # Extrahiere nur X-Y Koordinaten
    grid_points_2d = grid_points[:, :2]

    # Konvertiere zu Shapely-Polygone
    print("  Konvertiere Polygone zu Shapely...")
    road_data = []

    for poly_data in road_slope_polygons_2d:
        road_poly_xy = poly_data["road_polygon"]
        slope_poly_xy = poly_data["slope_polygon"]

        if len(slope_poly_xy) >= 3 and len(road_poly_xy) >= 3:
            try:
                road_poly = Polygon(road_poly_xy)
                slope_poly = Polygon(slope_poly_xy)

                centerline_coords = get_road_centerline_robust(road_poly)
                centerline = LineString(centerline_coords)
                buffer_zone = centerline.buffer(7.0)

                road_data.append(
                    {
                        "road_geom": road_poly,
                        "slope_geom": slope_poly,
                        "centerline_points": centerline_coords,
                        "buffer_zone": buffer_zone,
                        "buffer_bounds": buffer_zone.bounds,
                    }
                )
            except Exception:
                continue

    if not road_data:
        print("  ⚠ Keine gültigen Polygone gefunden!")
        return vertex_types, modified_heights

    print(f"  Loaded {len(road_data)} valid roads for processing")

    process_start = time_module.time()

    # MEGA-OPTIMIERUNG: Baue KDTree über alle Grid-Punkte (EINMAL, dann reuse!)
    print(f"  Baue KDTree für {len(grid_points_2d)} Grid-Punkte...")
    kdtree = cKDTree(grid_points_2d)

    print(f"  Teste {len(road_data)} Roads gegen Grid-Punkte...")

    for road_num, road_info in enumerate(road_data):
        centerline_points = road_info["centerline_points"]

        if centerline_points is None or len(centerline_points) < 2:
            continue

        centerline_linestring = LineString(centerline_points)

        if not centerline_linestring.is_valid or centerline_linestring.length == 0:
            continue

        total_length = centerline_linestring.length
        sample_spacing = 10.0

        sample_distances = np.arange(
            0, total_length + sample_spacing / 2, sample_spacing
        )

        if len(sample_distances) < 2:
            continue

        try:
            centerline = np.array(
                [
                    np.array(centerline_linestring.interpolate(dist).coords[0])
                    for dist in sample_distances
                ]
            )
        except Exception:
            continue

        if len(centerline) == 0:
            continue

        buffer_indices_set = set()
        search_radius = 7.0

        for centerline_pt in centerline:
            nearby = kdtree.query_ball_point(centerline_pt, r=search_radius)
            buffer_indices_set.update(nearby)

        buffer_indices = list(buffer_indices_set)

        if len(buffer_indices) == 0:
            continue

        # Test 1: Straßen-Bereich
        road_coords = np.array(road_info["road_geom"].exterior.coords)
        road_path = Path(road_coords)
        candidate_points = grid_points_2d[buffer_indices]
        inside_road = road_path.contains_points(candidate_points)

        # Test 2: Böschungs-Bereich
        slope_coords = np.array(road_info["slope_geom"].exterior.coords)
        slope_path = Path(slope_coords)
        inside_slope = slope_path.contains_points(candidate_points)

        for inside_idx in range(len(buffer_indices)):
            pt_idx = buffer_indices[inside_idx]
            if inside_road[inside_idx]:
                vertex_types[pt_idx] = 2
            elif inside_slope[inside_idx] and vertex_types[pt_idx] != 2:
                vertex_types[pt_idx] = 1

        if (road_num + 1) % 100 == 0:
            elapsed = time_module.time() - process_start
            rate = (road_num + 1) / elapsed if elapsed > 0 else 0
            eta = (len(road_data) - road_num - 1) / rate if rate > 0 else 0
            print(
                f"  {road_num + 1}/{len(road_data)} Roads ({rate:.0f}/s, ETA: {eta:.0f}s)"
            )

    elapsed_total = time_module.time() - process_start
    marked_count = np.count_nonzero(vertex_types)

    print(
        f"  ✓ Klassifizierung abgeschlossen ({elapsed_total:.1f}s, {marked_count} Punkte markiert)"
    )

    return vertex_types, modified_heights
