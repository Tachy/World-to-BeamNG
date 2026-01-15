"""
Grid-Vertex Klassifizierung und Polygon-Tests.
"""

import numpy as np
import time as time_module
from scipy.spatial import cKDTree

from ..config import OSM_MAPPER
from .. import config


def classify_grid_vertices(
    grid_points,
    grid_elevations,
    road_slope_polygons_2d,
):
    """
    Markiert Grid-Vertices unter Straßen.
    ULTRA-VEREINFACHT:
    - Suchradius = road_width/2 + 1
    - Schrittweite = Suchradius
    - Markiere alle Punkte im Suchradius
    """
    print("  Markiere Strassen-Bereiche im Grid (vereinfacht)...")

    vertex_types = np.zeros(len(grid_points), dtype=int)
    modified_heights = grid_elevations.copy()

    # Grid-Punkte sind bereits in lokalen Koordinaten (wie Polygone)
    grid_points_2d = grid_points[:, :2]

    # Baue KDTree über alle Grid-Punkte (EINMAL)
    print(f"  Baue KDTree für {len(grid_points_2d)} Grid-Punkte...")
    kdtree = cKDTree(grid_points_2d)

    print(f"  Markiere {len(road_slope_polygons_2d)} Roads...")

    process_start = time_module.time()

    for road_num, poly_data in enumerate(road_slope_polygons_2d):
        trimmed_centerline = poly_data.get("trimmed_centerline", [])
        osm_tags = poly_data.get("osm_tags", {})

        # Skip wenn keine Centerline
        if trimmed_centerline is None or len(trimmed_centerline) < 2:
            continue

        # Berechne Suchradius: road_width/2 + 1
        road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
        search_radius = road_width / 2.0 + 1.0
        step_size = search_radius  # Schrittweite = Suchradius

        # Konvertiere Centerline zu Array
        centerline_3d = np.array(trimmed_centerline, dtype=float)
        centerline_2d = centerline_3d[:, :2]

        # Berechne Distanzen entlang Centerline für äquidistante Samples
        diffs = centerline_2d[1:] - centerline_2d[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = cumulative_lengths[-1]

        if total_length <= 0:
            continue

        # Sample entlang Centerline mit step_size
        sample_distances = np.arange(0, total_length + step_size / 2, step_size)

        for sample_dist in sample_distances:
            if sample_dist > total_length:
                sample_dist = total_length

            # Interpoliere Position entlang Centerline
            seg_idx = np.searchsorted(cumulative_lengths, sample_dist, side="right") - 1
            seg_idx = max(0, min(seg_idx, len(segment_lengths) - 1))

            if segment_lengths[seg_idx] > 0:
                t = (sample_dist - cumulative_lengths[seg_idx]) / segment_lengths[seg_idx]
                t = np.clip(t, 0, 1)
                sample_pt = centerline_2d[seg_idx] + t * diffs[seg_idx]
            else:
                sample_pt = centerline_2d[seg_idx]

            # Markiere alle Vertices im Suchradius
            nearby_indices = kdtree.query_ball_point(sample_pt, r=search_radius)

            for idx in nearby_indices:
                vertex_types[idx] = 1  # Markiere als Straße

    # Dedupliziere (mehrere Samples können gleiche Vertices markieren)
    actual_marked = np.count_nonzero(vertex_types)

    elapsed_total = time_module.time() - process_start
    print(f"  [OK] Klassifizierung abgeschlossen ({elapsed_total:.1f}s, {actual_marked} Punkte markiert)")

    return vertex_types, modified_heights
