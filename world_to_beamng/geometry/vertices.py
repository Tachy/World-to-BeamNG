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

    roads_with_zero_points = []
    roads_processed = 0

    centerline_point_counts = []
    road_lengths = []
    sample_point_counts = []
    grid_points_per_road = []
    sample_gap_max = []

    for road_num, road_info in enumerate(road_data):
        centerline_points = road_info["centerline_points"]

        if centerline_points is None:
            roads_with_zero_points.append((road_num, "Centerline ist None"))
            continue

        if len(centerline_points) == 0:
            roads_with_zero_points.append((road_num, "Centerline ist leer"))
            continue

        if len(centerline_points) < 2:
            roads_with_zero_points.append(
                (road_num, f"Centerline zu kurz ({len(centerline_points)} Punkte)")
            )
            continue

        centerline_linestring = LineString(centerline_points)

        if not centerline_linestring.is_valid:
            roads_with_zero_points.append(
                (road_num, "Centerline ist ungültig (is_valid=False)")
            )
            continue

        if centerline_linestring.length == 0:
            roads_with_zero_points.append((road_num, "Centerline hat Länge 0"))
            continue

        total_length = centerline_linestring.length
        sample_spacing = 10.0

        sample_distances = np.arange(
            0, total_length + sample_spacing / 2, sample_spacing
        )

        if len(sample_distances) < 2:
            roads_with_zero_points.append(
                (road_num, f"Zu wenig Sample-Punkte ({len(sample_distances)})")
            )
            continue

        if len(sample_distances) > 1:
            diffs = np.diff(sample_distances)
            max_gap = np.max(diffs)
            if max_gap > 10.5:
                roads_with_zero_points.append(
                    (
                        road_num,
                        f"Lücke > 10m in Sample-Distances (max: {max_gap:.1f}m, {len(sample_distances)} Samples)",
                    )
                )
                continue

        try:
            centerline = np.array(
                [
                    np.array(centerline_linestring.interpolate(dist).coords[0])
                    for dist in sample_distances
                ]
            )
        except Exception as e:
            roads_with_zero_points.append((road_num, f"Interpolationsfehler: {str(e)}"))
            continue

        if len(centerline) == 0:
            roads_with_zero_points.append((road_num, "Keine interpolierten Punkte"))
            continue

        buffer_indices_set = set()
        search_radius = 7.0

        for centerline_pt in centerline:
            nearby = kdtree.query_ball_point(centerline_pt, r=search_radius)
            buffer_indices_set.update(nearby)

        buffer_indices = list(buffer_indices_set)

        if len(buffer_indices) == 0:
            roads_with_zero_points.append(
                (
                    road_num,
                    f"Keine Punkte im 7m-Radius (Centerline: {len(centerline)} Punkte, Länge: {total_length:.1f}m, Samples: {len(sample_distances)})",
                )
            )
            continue

        roads_processed += 1

        centerline_point_counts.append(len(centerline_points))
        road_lengths.append(total_length)
        sample_point_counts.append(len(sample_distances))
        grid_points_per_road.append(len(buffer_indices))
        if len(sample_distances) > 1:
            sample_gap_max.append(np.max(np.diff(sample_distances)))

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

    print(f"\n  DEBUG-INFO:")
    print(f"    • Verarbeitete Straßen: {roads_processed}/{len(road_data)}")
    print(f"    • Straßen ohne Prüfung: {len(roads_with_zero_points)}")
    if roads_with_zero_points:
        print(f"    • Gründe für fehlende Prüfung:")
        for road_idx, reason in roads_with_zero_points[:10]:
            print(f"      - Road {road_idx}: {reason}")
        if len(roads_with_zero_points) > 10:
            print(f"      ... und {len(roads_with_zero_points) - 10} weitere")

    if roads_processed > 0:
        print(f"\n  STATISTIKEN (für {roads_processed} verarbeitete Straßen):")
        print(f"    Centerline-Punkte:")
        print(
            f"      - Min: {np.min(centerline_point_counts)}, Max: {np.max(centerline_point_counts)}, Mittel: {np.mean(centerline_point_counts):.1f}"
        )
        print(f"    Straßenlängen:")
        print(
            f"      - Min: {np.min(road_lengths):.1f}m, Max: {np.max(road_lengths):.1f}m, Mittel: {np.mean(road_lengths):.1f}m"
        )
        print(f"    Sample-Punkte pro Straße:")
        print(
            f"      - Min: {np.min(sample_point_counts)}, Max: {np.max(sample_point_counts)}, Mittel: {np.mean(sample_point_counts):.1f}"
        )
        print(f"    Grid-Punkte pro Straße (im 7m-Radius):")
        print(
            f"      - Min: {np.min(grid_points_per_road)}, Max: {np.max(grid_points_per_road)}, Mittel: {np.mean(grid_points_per_road):.1f}"
        )
        if sample_gap_max:
            print(f"    Max. Abstände zwischen Sample-Punkten:")
            print(
                f"      - Min: {np.min(sample_gap_max):.2f}m, Max: {np.max(sample_gap_max):.2f}m, Mittel: {np.mean(sample_gap_max):.2f}m"
            )

    print(
        f"  ✓ Klassifizierung abgeschlossen ({elapsed_total:.1f}s, {marked_count} Punkte markiert)"
    )

    return vertex_types, modified_heights
