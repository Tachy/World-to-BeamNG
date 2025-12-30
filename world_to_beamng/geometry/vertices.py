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
    Markiert Grid-Vertices unter Straßen/Böschungen.
    SUPER-OPTIMIERT: Nutzt KDTree + geometrische Tests!
    """
    print(
        "\nMarkiere Straßen-/Böschungsbereiche im Grid (KDTree + Face-Überlappung)..."
    )

    vertex_types = np.zeros(len(grid_points), dtype=int)
    modified_heights = grid_elevations.copy()

    # Grid-Punkte sind bereits in lokalen Koordinaten (wie Polygone)
    grid_points_2d = grid_points[:, :2]

    if len(grid_points_2d) > 0:
        print(f"    DEBUG Grid: sample point 0 = {grid_points_2d[0]}")
        print(f"    DEBUG Grid: sample point 1000000 = {grid_points_2d[1000000]}")
        print(
            f"    DEBUG Grid: min = {grid_points_2d.min(axis=0)}, max = {grid_points_2d.max(axis=0)}"
        )

    # Konvertiere zu Shapely-Polygone
    print("  Konvertiere Polygone zu Shapely...")
    road_data = []

    print(f"    DEBUG: {len(road_slope_polygons_2d)} Polygone zu verarbeiten")

    for idx, poly_data in enumerate(road_slope_polygons_2d):
        road_poly_xy = poly_data["road_polygon"]
        slope_poly_xy = poly_data["slope_polygon"]
        original_coords = poly_data.get("original_coords", [])

        if idx == 0:
            print(f"    DEBUG Polygon 0: road_polygon hat {len(road_poly_xy)} Punkte")
            print(f"    DEBUG Polygon 0: slope_polygon hat {len(slope_poly_xy)} Punkte")
            if len(road_poly_xy) > 0:
                print(f"    DEBUG Polygon 0: road_polygon[0] = {road_poly_xy[0]}")
            if len(slope_poly_xy) > 0:
                print(f"    DEBUG Polygon 0: slope_polygon[0] = {slope_poly_xy[0]}")

        if len(slope_poly_xy) >= 3 and len(road_poly_xy) >= 3:
            try:
                road_poly = Polygon(road_poly_xy)
                slope_poly = Polygon(slope_poly_xy)

                # Validiere Polygone
                if not road_poly.is_valid:
                    road_poly = road_poly.buffer(0)
                    # Nach buffer(0) kann MultiPolygon entstehen - nimm größtes Teil
                    if isinstance(road_poly, MultiPolygon):
                        road_poly = max(road_poly.geoms, key=lambda p: p.area)

                if not slope_poly.is_valid:
                    slope_poly = slope_poly.buffer(0)
                    # Nach buffer(0) kann MultiPolygon entstehen - nimm größtes Teil
                    if isinstance(slope_poly, MultiPolygon):
                        slope_poly = max(slope_poly.geoms, key=lambda p: p.area)

                # Nochmal prüfen nach Fix
                if not isinstance(road_poly, Polygon) or not isinstance(
                    slope_poly, Polygon
                ):
                    continue

                # Verwende die ORIGINALE OSM-Straßengeometrie für die Centerline!
                # (bereits in lokalen Koordinaten durch get_road_polygons)
                if original_coords and len(original_coords) >= 2:
                    centerline_coords = np.array(
                        [(x, y) for x, y, z in original_coords]
                    )
                else:
                    # Fallback: Berechne aus Polygon (bereits in lokalen Coords)
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
        sample_spacing = (
            config.CENTERLINE_SAMPLE_SPACING
        )  # Abstand zwischen Sample-Points entlang der Centerline

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
        search_radius = config.CENTERLINE_SEARCH_RADIUS

        for centerline_pt in centerline:
            nearby = kdtree.query_ball_point(centerline_pt, r=search_radius)
            buffer_indices_set.update(nearby)

        buffer_indices = list(buffer_indices_set)

        if len(buffer_indices) == 0:
            continue

        # VEKTORISIERT: Markiere ALLE Vertices die der KDTree findet!
        # Bei 3m Grid und 7m Straßenbreite können Vertices außerhalb des Polygons liegen
        # Der KDTree-Radius (10m) erfasst alle relevanten Vertices

        candidate_points = grid_points_2d[buffer_indices]

        # Test gegen Böschungs-Bereich (enthält Straße+Böschung komplett)
        slope_coords = np.array(road_info["slope_geom"].exterior.coords)
        slope_path = Path(slope_coords)
        inside_slope = slope_path.contains_points(candidate_points)

        # Markiere alle Vertices innerhalb von Straße+Böschung
        for inside_idx in range(len(buffer_indices)):
            pt_idx = buffer_indices[inside_idx]
            if inside_slope[inside_idx]:
                vertex_types[pt_idx] = 1  # Flag zum Ausblenden

    elapsed_total = time_module.time() - process_start
    marked_count = np.count_nonzero(vertex_types)

    print(
        f"  ✓ Klassifizierung abgeschlossen ({elapsed_total:.1f}s, {marked_count} Punkte markiert)"
    )

    return vertex_types, modified_heights
