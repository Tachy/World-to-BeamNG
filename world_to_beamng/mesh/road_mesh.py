"""
Straßen-Mesh und Böschungs-Generierung.
"""

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from ..geometry.coordinates import apply_local_offset
from .. import config


def snap_road_edge_vertices(all_road_vertices, road_polygons, road_slope_polygons_2d):
    """
    Snappt die Rand-Vertices (links/rechts) von Straßen, deren Centerlines
    sich nahe kommen (z.B. an Kreuzungen).

    Args:
        all_road_vertices: Liste aller Road-Vertices (local coordinates)
        road_polygons: Original Road-Polygons mit 'coords' (centerline)
        road_slope_polygons_2d: Generierte Road/Slope-Polygone mit Metadaten

    Returns:
        Modifizierte all_road_vertices Liste
    """
    from scipy.spatial import cKDTree

    snap_distance = 5.0  # Gleicher Wert wie bei Centerline-Snap
    half_width = config.ROAD_WIDTH / 2.0

    # Sammle Info über Start/End-Vertices jeder Straße
    # Format: (vertex_idx_left, vertex_idx_right, centerline_point_xy, road_idx, is_start, normal_xy)
    endpoints_info = []
    vertex_offset = 0

    for road_idx, poly_data in enumerate(road_slope_polygons_2d):
        original_coords = poly_data["original_coords"]
        if len(original_coords) < 2:
            continue

        num_points = len(original_coords)

        # Start-Vertices (Index 0)
        start_left_idx = vertex_offset + 0
        start_right_idx = vertex_offset + num_points + 0
        start_center_xy = (original_coords[0][0], original_coords[0][1])
        start_tangent = np.array(original_coords[1][:2]) - np.array(
            original_coords[0][:2]
        )
        if np.linalg.norm(start_tangent) < 1e-9:
            start_tangent = np.array([1.0, 0.0])
        start_norm = start_tangent[[1, 0]] * np.array([-1, 1])  # (-dy, dx)

        endpoints_info.append(
            (
                start_left_idx,
                start_right_idx,
                start_center_xy,
                road_idx,
                True,
                start_norm,
            )
        )

        # End-Vertices (letzter Index)
        end_left_idx = vertex_offset + (num_points - 1)
        end_right_idx = vertex_offset + num_points + (num_points - 1)
        end_center_xy = (original_coords[-1][0], original_coords[-1][1])
        end_tangent = np.array(original_coords[-1][:2]) - np.array(
            original_coords[-2][:2]
        )
        if np.linalg.norm(end_tangent) < 1e-9:
            end_tangent = np.array([1.0, 0.0])
        end_norm = end_tangent[[1, 0]] * np.array([-1, 1])  # (-dy, dx)

        endpoints_info.append(
            (end_left_idx, end_right_idx, end_center_xy, road_idx, False, end_norm)
        )

        vertex_offset += num_points * 2  # links + rechts

    if len(endpoints_info) < 2:
        return all_road_vertices

    # Baue KDTree mit Centerline-Endpunkten
    centerline_endpoints = np.array([info[2] for info in endpoints_info])
    kdtree = cKDTree(centerline_endpoints)

    # Finde Endpunkte, die nahe beieinander liegen
    pairs = kdtree.query_pairs(snap_distance)

    if len(pairs) == 0:
        print(f"    → Keine nahen Straßenenden gefunden")
        return all_road_vertices

    # Gruppiere zu Clustern (Union-Find)
    from collections import defaultdict

    parent = list(range(len(endpoints_info)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in pairs:
        union(i, j)

    # Sammle Cluster
    clusters = defaultdict(list)
    for i in range(len(endpoints_info)):
        clusters[find(i)].append(i)

    # Für jeden Cluster: Snappe die Rand-Vertices zusammen
    all_road_vertices = list(all_road_vertices)  # Konvertiere zu modifizierbarer Liste
    snapped_clusters = 0

    for cluster_indices in clusters.values():
        if len(cluster_indices) <= 1:
            continue  # Keine Nachbarn

        # Sammle alle linken und rechten Vertices in diesem Cluster
        left_vertices = []
        right_vertices = []

        for idx in cluster_indices:
            left_idx, right_idx, _, _, _, norm_vec = endpoints_info[idx]

            ref_norm = endpoints_info[cluster_indices[0]][5]
            # Wenn die Normale entgegengesetzt zeigt, tausche links/rechts, um Verdrehungen zu vermeiden
            if np.dot(norm_vec, ref_norm) < 0:
                left_idx, right_idx = right_idx, left_idx

            left_vertices.append(all_road_vertices[left_idx])
            right_vertices.append(all_road_vertices[right_idx])

        # Berechne Schwerpunkte
        left_vertices_array = np.array(left_vertices)
        right_vertices_array = np.array(right_vertices)

        left_centroid = left_vertices_array.mean(axis=0)
        right_centroid = right_vertices_array.mean(axis=0)

        # Snap alle Vertices im Cluster zum jeweiligen Schwerpunkt
        ref_norm = endpoints_info[cluster_indices[0]][5]

        for idx in cluster_indices:
            left_idx, right_idx, _, _, _, norm_vec = endpoints_info[idx]

            if np.dot(norm_vec, ref_norm) < 0:
                left_idx, right_idx = right_idx, left_idx

            all_road_vertices[left_idx] = tuple(left_centroid)
            all_road_vertices[right_idx] = tuple(right_centroid)

        snapped_clusters += 1

    print(f"    → {snapped_clusters} Kreuzungen mit gesnappten Rändern")

    return all_road_vertices


def clip_road_to_bounds(coords, bounds_utm):
    """Clippt eine Straße an den Grid-Bounds (UTM-Koordinaten)."""
    if not coords or bounds_utm is None:
        return coords

    min_x, min_y, max_x, max_y = bounds_utm
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


def generate_road_mesh_strips(road_polygons, height_points, height_elevations):
    """
    Generiert Straßen als separate Mesh-Streifen mit perfekt parallelen Kanten.
    """
    half_width = config.ROAD_WIDTH / 2.0
    slope_gradient = np.tan(np.radians(config.SLOPE_ANGLE))

    terrain_interpolator = NearestNDInterpolator(height_points, height_elevations)

    all_road_vertices = []
    all_road_faces = []
    all_road_face_to_idx = []
    all_slope_vertices = []
    all_slope_faces = []
    road_slope_polygons_2d = []

    total_roads = len(road_polygons)
    processed = 0
    clipped_roads = 0

    # DEBUG: Statistiken für Böschungen
    slopes_with_left = 0
    slopes_with_right = 0
    slopes_with_both = 0
    min_height_diff = float("inf")
    max_height_diff = float("-inf")

    for road in road_polygons:
        coords = road["coords"]
        road_id = road.get("id")

        if len(coords) < 2:
            continue

        coords = clip_road_to_bounds(coords, config.GRID_BOUNDS_UTM)

        if len(coords) < 2:
            clipped_roads += 1
            continue

        # Entferne doppelte aufeinanderfolgende Punkte (Sicherheit)
        cleaned_coords = [coords[0]]
        for i in range(1, len(coords)):
            dx = coords[i][0] - cleaned_coords[-1][0]
            dy = coords[i][1] - cleaned_coords[-1][1]
            if dx * dx + dy * dy > 0.01:  # > 10cm Abstand
                cleaned_coords.append(coords[i])

        if len(cleaned_coords) < 2:
            clipped_roads += 1
            continue

        coords = cleaned_coords

        road_left_vertices = []
        road_right_vertices = []
        road_left_abs = []
        road_right_abs = []
        slope_left_outer_vertices = []
        slope_right_outer_vertices = []

        coords_array = np.array(coords)
        num_points = len(coords_array)

        # Berechne GEGLÄTTETE Tangenten für radiale Ausrichtung
        # Statt segment-basierte Richtungen, mitteln wir die Richtungen
        # für sanftere Übergänge an Kurven
        diffs = np.diff(coords_array[:, :2], axis=0)
        lengths = np.linalg.norm(diffs, axis=1)

        # Verhindere Division durch 0 bei identischen Punkten
        lengths = np.maximum(lengths, 0.001)  # Min 1mm
        directions = diffs / lengths[:, np.newaxis]

        # NEUE METHODE: Geglättete Tangenten für bessere Kurven
        point_dirs = np.zeros((num_points, 2))

        # Erster Punkt: Richtung zum nächsten
        point_dirs[0] = directions[0]

        # Letzter Punkt: Richtung vom vorherigen
        point_dirs[-1] = directions[-1]

        # Mittlere Punkte: Mittle die Richtungen für glatte Tangenten
        for i in range(1, num_points - 1):
            # Mittle zwischen eingehender und ausgehender Richtung
            incoming_dir = directions[i - 1]
            outgoing_dir = directions[i]

            # Gewichteter Durchschnitt
            avg_dir = (incoming_dir + outgoing_dir) / 2.0

            # Normalisiere
            length = np.linalg.norm(avg_dir)
            if length > 1e-6:
                point_dirs[i] = avg_dir / length
            else:
                # Fallback wenn Richtungen entgegengesetzt sind
                point_dirs[i] = incoming_dir

        for i, (x, y, z) in enumerate(coords):
            dir_x = point_dirs[i, 0]
            dir_y = point_dirs[i, 1]

            perp_x = -dir_y
            perp_y = dir_x

            road_left_x = x + perp_x * half_width
            road_left_y = y + perp_y * half_width
            road_right_x = x - perp_x * half_width
            road_right_y = y - perp_y * half_width

            road_left_abs.append((road_left_x, road_left_y, z))
            road_right_abs.append((road_right_x, road_right_y, z))

            p_left_local = apply_local_offset(road_left_x, road_left_y, z)
            p_right_local = apply_local_offset(road_right_x, road_right_y, z)

            road_left_vertices.append(p_left_local)
            road_right_vertices.append(p_right_local)

            terrain_left_height = terrain_interpolator([[road_left_x, road_left_y]])[0]
            terrain_right_height = terrain_interpolator([[road_right_x, road_right_y]])[
                0
            ]

            height_diff_left = terrain_left_height - z
            height_diff_right = terrain_right_height - z

            # DEBUG: Statistiken
            if abs(height_diff_left) > 0.01:
                min_height_diff = min(min_height_diff, abs(height_diff_left))
                max_height_diff = max(max_height_diff, abs(height_diff_left))
            if abs(height_diff_right) > 0.01:
                min_height_diff = min(min_height_diff, abs(height_diff_right))
                max_height_diff = max(max_height_diff, abs(height_diff_right))

            # Böschungs-Logik:
            # - Mindestbreite 0.2m (20cm)
            # - Solange |height_diff| < 0.2m: Breite = 0.2m
            # - Ab |height_diff| >= 0.2m: Breite = |height_diff| (1:1 = 45°)
            abs_left = abs(height_diff_left)
            abs_right = abs(height_diff_right)

            slope_width_left = max(0.2, abs_left)
            slope_width_right = max(0.2, abs_right)

            MAX_SLOPE_WIDTH = 30.0
            slope_width_left = min(slope_width_left, MAX_SLOPE_WIDTH)
            slope_width_right = min(slope_width_right, MAX_SLOPE_WIDTH)

            if slope_width_left < 0.01:  # Praktisch 0
                slope_left_outer_x = road_left_x
                slope_left_outer_y = road_left_y
                slope_left_outer_height = z
            else:
                slopes_with_left += 1
                slope_left_outer_x = road_left_x + perp_x * slope_width_left
                slope_left_outer_y = road_left_y + perp_y * slope_width_left
                slope_left_outer_height = (
                    z + height_diff_left
                )  # Use terrain height, not z

            if slope_width_right < 0.01:  # Praktisch 0
                slope_right_outer_x = road_right_x
                slope_right_outer_y = road_right_y
                slope_right_outer_height = z
            else:
                slopes_with_right += 1
                slope_right_outer_x = road_right_x - perp_x * slope_width_right
                slope_right_outer_y = road_right_y - perp_y * slope_width_right
                slope_right_outer_height = (
                    z + height_diff_right
                )  # Use terrain height, not z

            if slope_width_left >= 0.01 and slope_width_right >= 0.01:
                slopes_with_both += 1

            slope_left_outer_local = apply_local_offset(
                slope_left_outer_x, slope_left_outer_y, slope_left_outer_height
            )
            slope_right_outer_local = apply_local_offset(
                slope_right_outer_x, slope_right_outer_y, slope_right_outer_height
            )

            slope_left_outer_vertices.append(slope_left_outer_local)
            slope_right_outer_vertices.append(slope_right_outer_local)

        num_points = len(road_left_abs)

        road_start_idx = len(all_road_vertices)

        for i, (x_abs, y_abs, z_abs) in enumerate(road_left_abs):
            transformed = apply_local_offset(x_abs, y_abs, z_abs)
            all_road_vertices.append(transformed)
        for x_abs, y_abs, z_abs in road_right_abs:
            all_road_vertices.append(apply_local_offset(x_abs, y_abs, z_abs))

        for i in range(num_points - 1):
            left1 = road_start_idx + i
            left2 = road_start_idx + i + 1
            right1 = road_start_idx + num_points + i
            right2 = road_start_idx + num_points + i + 1

            # Erstelle zwei Dreiecke für ein Quad zwischen left und right
            # Face 1: left1 → right1 → right2
            all_road_faces.append([left1, right1, right2])
            all_road_face_to_idx.append(road_id)
            # Face 2: left1 → right2 → left2
            all_road_faces.append([left1, right2, left2])
            all_road_face_to_idx.append(road_id)

        # DEBUG: Prüfe ob alle Vertices verbunden sind
        if num_points < 2:
            print(f"    ⚠ Straße mit nur {num_points} Punkten - zu kurz!")

        slope_start_for_this_road = len(all_slope_vertices)

        for i, (x_abs, y_abs, z_abs) in enumerate(road_left_abs):
            transformed = apply_local_offset(x_abs, y_abs, z_abs)
            all_slope_vertices.append(transformed)
        for x_abs, y_abs, z_abs in road_right_abs:
            all_slope_vertices.append(apply_local_offset(x_abs, y_abs, z_abs))

        all_slope_vertices.extend(slope_left_outer_vertices)
        all_slope_vertices.extend(slope_right_outer_vertices)

        for i in range(num_points - 1):
            road_left1 = slope_start_for_this_road + i
            road_left2 = slope_start_for_this_road + i + 1
            slope_left1 = slope_start_for_this_road + 2 * num_points + i
            slope_left2 = slope_start_for_this_road + 2 * num_points + i + 1

            all_slope_faces.append([road_left1, slope_left1, slope_left2])
            all_slope_faces.append([road_left1, slope_left2, road_left2])

            road_right1 = slope_start_for_this_road + num_points + i
            road_right2 = slope_start_for_this_road + num_points + i + 1
            slope_right1 = slope_start_for_this_road + 3 * num_points + i
            slope_right2 = slope_start_for_this_road + 3 * num_points + i + 1

            all_slope_faces.append([road_right1, slope_right2, slope_right1])
            all_slope_faces.append([road_right1, road_right2, slope_right2])

        # WICHTIG: 2D-Polygone OHNE LOCAL_OFFSET für Vertex-Klassifizierung!
        # Die grid_points sind in globalen UTM-Koordinaten
        # LOCAL_OFFSET wird nur für 3D-Mesh-Vertices angewendet
        road_left_2d = [(x, y) for x, y, z in road_left_abs]
        road_right_2d = [(x, y) for x, y, z in road_right_abs]

        road_poly_2d = road_left_2d + list(reversed(road_right_2d))

        # Böschungs-Polygone OHNE LOCAL_OFFSET (für Vertex-Klassifizierung)
        # Polygon-Reihenfolge: road_left → slope_left → slope_right → road_right → zurück zu road_left
        slope_left_2d = [(v[0], v[1]) for v in slope_left_outer_vertices]
        slope_right_2d = [(v[0], v[1]) for v in slope_right_outer_vertices]

        # Korrekte Polygon-Reihenfolge (gegen Uhrzeigersinn)
        slope_poly_2d = (
            road_left_2d  # Innere linke Straßenkante
            + slope_left_2d  # Äußere linke Böschungskante
            + list(reversed(slope_right_2d))  # Äußere rechte Böschungskante (rückwärts)
            + list(reversed(road_right_2d))  # Innere rechte Straßenkante (rückwärts)
        )

        road_slope_polygons_2d.append(
            {
                "road_polygon": road_poly_2d,
                "slope_polygon": slope_poly_2d,
                "original_coords": coords,  # Original OSM-Straßenkoordinaten
            }
        )

        processed += 1
        if processed % 100 == 0:
            print(f"  {processed}/{total_roads} Straßen...")

    print(f"  ✓ {len(all_road_vertices)} Straßen-Vertices")
    print(f"  ✓ {len(all_road_faces)} Straßen-Faces")
    print(f"  ✓ {len(all_slope_vertices)} Böschungs-Vertices")
    print(f"  ✓ {len(all_slope_faces)} Böschungs-Faces")

    # DEBUG: Böschungsstatistiken
    print(f"\n  DEBUG: Böschungen-Statistik:")
    print(f"    • Straßen mit linker Böschung: {slopes_with_left}")
    print(f"    • Straßen mit rechter Böschung: {slopes_with_right}")
    print(f"    • Straßen mit BEIDEN Böschungen: {slopes_with_both}")
    if min_height_diff != float("inf"):
        print(
            f"    • Höhendifferenzen: {min_height_diff:.2f}m - {max_height_diff:.2f}m"
        )
    else:
        print(f"    • Keine Höhendifferenzen gefunden!")

    # DEBUG: Prüfe ob alle Straßen auch Böschungen haben
    if len(all_slope_vertices) == 0:
        print("    ⚠ WARNUNG: KEINE Böschungs-Vertices generiert!")
    elif len(all_slope_vertices) < len(all_road_vertices) * 2:
        print(
            f"    ⚠ WARNUNG: Weniger Böschungs- als Straßen-Vertices! ({len(all_slope_vertices)} vs {len(all_road_vertices)})"
        )
    else:
        print(f"    ✓ Böschungen OK")
    print(
        f"  ✓ {len(road_slope_polygons_2d)} Road/Slope-Polygone für Grid-Ausschneiden (2D)"
    )
    if clipped_roads > 0:
        print(f"  ℹ {clipped_roads} Straßen komplett außerhalb Grid (ignoriert)")

    # Snappe Straßenränder an Kreuzungen (wo Centerlines zusammentreffen)
    if config.ENABLE_ROAD_EDGE_SNAPPING:
        print(f"  Snappe Straßenrand-Vertices an Kreuzungen...")
        all_road_vertices = snap_road_edge_vertices(
            all_road_vertices, road_polygons, road_slope_polygons_2d
        )
    else:
        print(f"  Snappe Straßenränder SKIP (config.ENABLE_ROAD_EDGE_SNAPPING=False)")

    return (
        all_road_vertices,
        all_road_faces,
        all_road_face_to_idx,
        all_slope_vertices,
        all_slope_faces,
        road_slope_polygons_2d,
    )
