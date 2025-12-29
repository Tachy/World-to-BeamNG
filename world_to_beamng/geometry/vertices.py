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
        original_coords = poly_data.get("original_coords", [])

        if len(slope_poly_xy) >= 3 and len(road_poly_xy) >= 3:
            try:
                road_poly = Polygon(road_poly_xy)
                slope_poly = Polygon(slope_poly_xy)

                # Verwende die ORIGINALE OSM-Straßengeometrie für die Centerline!
                if original_coords and len(original_coords) >= 2:
                    centerline_coords = np.array(
                        [(x, y) for x, y, z in original_coords]
                    )
                else:
                    # Fallback: Berechne aus Polygon (sollte nicht passieren)
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

    # DEBUG: Sammle Centerlines und Sample-Points für Visualisierung
    debug_centerlines = []
    debug_sample_points = []
    debug_search_radii = []

    for road_num, road_info in enumerate(road_data):
        centerline_points = road_info["centerline_points"]

        if centerline_points is None or len(centerline_points) < 2:
            continue

        centerline_linestring = LineString(centerline_points)

        if not centerline_linestring.is_valid or centerline_linestring.length == 0:
            continue

        total_length = centerline_linestring.length
        sample_spacing = 10.0  # Abstand zwischen Sample-Points entlang der Centerline

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

        # DEBUG: Speichere Centerline für Visualisierung
        debug_centerlines.append(centerline.tolist())

        buffer_indices_set = set()
        # KDTree-Radius: 7m = halbe Straßenbreite (3.5m) + Puffer für Böschungen (3.5m)
        search_radius = 7.0

        for centerline_pt in centerline:
            nearby = kdtree.query_ball_point(centerline_pt, r=search_radius)
            buffer_indices_set.update(nearby)

            # DEBUG: Speichere Sample-Point und Radius
            debug_sample_points.append(centerline_pt.tolist())
            debug_search_radii.append(search_radius)

        buffer_indices = list(buffer_indices_set)

        if len(buffer_indices) == 0:
            continue

        # VEKTORISIERT: Markiere ALLE Vertices die der KDTree findet!
        # Bei 3m Grid und 7m Straßenbreite können Vertices außerhalb des Polygons liegen
        # Der KDTree-Radius (10m) erfasst alle relevanten Vertices

        candidate_points = grid_points_2d[buffer_indices]

        # Test 1: Straßen-Bereich (vektorisiert)
        road_coords = np.array(road_info["road_geom"].exterior.coords)
        road_path = Path(road_coords)
        inside_road = road_path.contains_points(candidate_points)

        # Test 2: Böschungs-Bereich (vektorisiert)
        slope_coords = np.array(road_info["slope_geom"].exterior.coords)
        slope_path = Path(slope_coords)
        inside_slope = slope_path.contains_points(candidate_points)

        # Markiere NUR Vertices die innerhalb von Straßen oder Böschungen liegen!
        for inside_idx in range(len(buffer_indices)):
            pt_idx = buffer_indices[inside_idx]
            if inside_road[inside_idx]:
                vertex_types[pt_idx] = 2  # Straße
            elif inside_slope[inside_idx]:
                vertex_types[pt_idx] = 1  # Böschung
            # else: Vertex ist im KDTree-Buffer, aber außerhalb → NICHT markieren!

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

    # DEBUG: Exportiere Centerlines und Suchradien als OBJ
    print("  DEBUG: Exportiere Centerlines und Suchradien zu debug_centerlines.obj...")
    _export_debug_visualization(
        debug_centerlines, debug_sample_points, debug_search_radii
    )

    return vertex_types, modified_heights


def _export_debug_visualization(centerlines, sample_points, search_radii):
    """Exportiert Centerlines und Suchradien als OBJ zur Visualisierung."""
    from ..geometry.coordinates import apply_local_offset

    # Erstelle MTL-Datei
    with open("debug_centerlines.mtl", "w") as f:
        f.write("# Debug Materialien\n\n")
        f.write("newmtl centerline\n")
        f.write("Ka 0.0 1.0 0.0\n")  # Ambient grün
        f.write("Kd 0.0 1.0 0.0\n")  # Diffuse grün
        f.write("Ks 0.0 0.0 0.0\n")  # Specular schwarz
        f.write("d 1.0\n\n")

        f.write("newmtl searchradius\n")
        f.write("Ka 1.0 0.0 0.0\n")  # Ambient rot
        f.write("Kd 1.0 0.0 0.0\n")  # Diffuse rot
        f.write("Ks 0.0 0.0 0.0\n")  # Specular schwarz
        f.write("d 0.5\n")  # Halb-transparent

    with open("debug_centerlines.obj", "w") as f:
        f.write("# Debug: Centerlines und Suchradien\n")
        f.write("mtllib debug_centerlines.mtl\n")
        f.write("# Grün = Centerlines, Rot = Sample Points mit Radius\n\n")

        vertex_count = 1

        # Exportiere Centerlines als Linien
        f.write("usemtl centerline\n")
        for centerline in centerlines:
            # Schreibe Vertices
            for x, y in centerline:
                x_local, y_local, z_local = apply_local_offset(x, y, 0.0)
                f.write(f"v {x_local} {y_local} {z_local}\n")

            # Schreibe Linien (l v1 v2 v3 ...)
            num_verts = len(centerline)
            line_indices = " ".join(str(vertex_count + i) for i in range(num_verts))
            f.write(f"l {line_indices}\n")
            vertex_count += num_verts

        f.write("\n# Sample Points mit Suchradius (Kreise)\n")
        f.write("usemtl searchradius\n")

        # Exportiere Sample Points als Kreise (gefüllte Polygone)
        for (x, y), radius in zip(sample_points, search_radii):
            # Erstelle Kreis mit 16 Segmenten als gefülltes Polygon
            num_segments = 16

            # Zentrum des Kreises
            x_local, y_local, z_local = apply_local_offset(x, y, 0.0)
            f.write(f"v {x_local} {y_local} {z_local}\n")
            center_idx = vertex_count
            vertex_count += 1

            # Kreis-Vertices
            circle_verts = []
            for i in range(num_segments):
                angle = 2 * np.pi * i / num_segments
                cx = x + radius * np.cos(angle)
                cy = y + radius * np.sin(angle)
                x_local, y_local, z_local = apply_local_offset(cx, cy, 0.0)
                f.write(f"v {x_local} {y_local} {z_local}\n")
                circle_verts.append(vertex_count)
                vertex_count += 1

            # Schreibe Kreis als Dreiecke (Fächer vom Zentrum)
            for i in range(num_segments):
                v1 = center_idx
                v2 = circle_verts[i]
                v3 = circle_verts[(i + 1) % num_segments]
                f.write(f"f {v1} {v2} {v3}\n")

    print("    ✓ debug_centerlines.obj + debug_centerlines.mtl erstellt")
