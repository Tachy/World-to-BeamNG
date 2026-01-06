"""
Grid-Vertex Klassifizierung und Polygon-Tests.
"""

import numpy as np
import time as time_module
from shapely.geometry import Polygon, LineString, MultiPolygon
from scipy.spatial import cKDTree
from matplotlib.path import Path

from ..geometry.polygon import get_road_centerline_robust
from ..config import OSM_MAPPER
from .. import config


def classify_grid_vertices(
    grid_points,
    grid_elevations,
    road_slope_polygons_2d,
):
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
        osm_tags = poly_data.get("osm_tags", {})

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

            # Berechne dynamischen Search-Radius basierend auf Straßenbreite
            road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
            # Formel: road_width + GRID_SPACING*2.5 (bei 2m Grid: road_width + 5m)
            dynamic_search_radius = road_width + config.GRID_SPACING * 2.5

            # Verwende die GETRIMTE Centerline (nach Junction-Trimming)!
            # (bereits in lokalen Koordinaten durch get_road_polygons)
            if trimmed_centerline and len(trimmed_centerline) >= 2:
                centerline_3d = np.array(trimmed_centerline, dtype=float)
                centerline_coords = centerline_3d[:, :2]
                has_real_centerline = True
            else:
                # Fallback für Junction-Fans: Nutze Polygon-Center als "Centerline"
                # (Fans werden später per Polygon-Bounds statt Centerline-Sampling verarbeitet)
                centerline_coords = np.array([road_poly.centroid.coords[0]])
                centerline_3d = None
                has_real_centerline = False

            road_data.append(
                {
                    "road_geom": road_poly,
                    "centerline_points": centerline_coords,
                    "centerline_3d": centerline_3d,
                    "has_real_centerline": has_real_centerline,
                    "buffer_bounds": road_poly.bounds,
                    "search_radius": dynamic_search_radius,  # Dynamischer Radius pro Straße
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

            centerline_3d = road_info.get("centerline_3d")

            # Vorberechnung für Z-Interpolation entlang der 3D-Centerline
            seg_len = None
            seg_cum = None
            if centerline_3d is not None and len(centerline_3d) >= 2:
                seg_vec = centerline_3d[1:, :2] - centerline_3d[:-1, :2]
                seg_len = np.linalg.norm(seg_vec, axis=1)
                seg_cum = np.concatenate(([0.0], np.cumsum(seg_len)))

            total_length = centerline_linestring.length
            # Dynamischer Sample-Spacing = dynamischer Suchradius (damit Kreise nahtlos anschließen)
            osm_tags = road_info.get("osm_tags", {})
            road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
            sample_spacing = road_width + config.GRID_SPACING * 2.5

            # Stelle sicher, dass auch sehr kurze Segmente mindestens mit Start+Ende abgetastet werden
            num_samples = max(2, int(np.ceil(total_length / sample_spacing)) + 1)
            sample_distances = np.linspace(0, total_length, num_samples)

            def _interp_z(distance):
                if centerline_3d is None or seg_cum is None or seg_len is None:
                    return 0.0
                if distance <= 0:
                    return float(centerline_3d[0, 2])
                if distance >= seg_cum[-1]:
                    return float(centerline_3d[-1, 2])

                idx = int(np.searchsorted(seg_cum, distance, side="right") - 1)
                idx = max(0, min(idx, len(seg_len) - 1))
                seg_length = seg_len[idx]

                if seg_length < 1e-9:
                    return float(centerline_3d[idx, 2])

                t = (distance - seg_cum[idx]) / seg_length
                return float(centerline_3d[idx, 2] + t * (centerline_3d[idx + 1, 2] - centerline_3d[idx, 2]))

            try:
                centerline = []
                for dist in sample_distances:
                    pt = centerline_linestring.interpolate(dist)
                    z_val = _interp_z(dist)
                    centerline.append([pt.x, pt.y, z_val])
                centerline = np.array(centerline)
            except Exception:
                continue

            if len(centerline) == 0:
                continue

            buffer_indices_set = set()
            # Dynamischer Search-Radius (wird in road_info gespeichert)
            search_radius = road_info["search_radius"]

            # Centerline-Samples
            for centerline_pt in centerline:
                nearby = kdtree.query_ball_point(centerline_pt[:2], r=search_radius)
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
