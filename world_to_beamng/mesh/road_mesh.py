"""
Straßen-Mesh und Böschungs-Generierung.
"""

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from ..geometry.coordinates import apply_local_offset
from .. import config


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
    all_slope_vertices = []
    all_slope_faces = []
    road_slope_polygons_2d = []

    total_roads = len(road_polygons)
    processed = 0
    clipped_roads = 0

    for road in road_polygons:
        coords = road["coords"]

        if len(coords) < 2:
            continue

        coords = clip_road_to_bounds(coords, config.GRID_BOUNDS_UTM)

        if len(coords) < 2:
            clipped_roads += 1
            continue

        road_left_vertices = []
        road_right_vertices = []
        road_left_abs = []
        road_right_abs = []
        slope_left_outer_vertices = []
        slope_right_outer_vertices = []

        coords_array = np.array(coords)
        num_points = len(coords_array)

        diffs = np.diff(coords_array[:, :2], axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        directions = diffs / lengths[:, np.newaxis]

        avg_dirs = np.zeros((num_points, 2))
        avg_dirs[0] = directions[0]
        avg_dirs[1:-1] = (directions[:-1] + directions[1:]) / 2
        avg_dirs[-1] = directions[-1]

        avg_lengths = np.linalg.norm(avg_dirs, axis=1)
        avg_dirs = avg_dirs / avg_lengths[:, np.newaxis]

        for i, (x, y, z) in enumerate(coords):
            dir_x = avg_dirs[i, 0]
            dir_y = avg_dirs[i, 1]

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

            slope_width_left = abs(height_diff_left) / slope_gradient
            slope_width_right = abs(height_diff_right) / slope_gradient

            MAX_SLOPE_WIDTH = 30.0
            slope_width_left = min(slope_width_left, MAX_SLOPE_WIDTH)
            slope_width_right = min(slope_width_right, MAX_SLOPE_WIDTH)

            if abs(height_diff_left) < 0.1:
                slope_width_left = 0.0
            if abs(height_diff_right) < 0.1:
                slope_width_right = 0.0

            if slope_width_left < 0.1:
                slope_left_outer_x = road_left_x
                slope_left_outer_y = road_left_y
                slope_left_outer_height = z
            else:
                slope_left_outer_x = road_left_x + perp_x * slope_width_left
                slope_left_outer_y = road_left_y + perp_y * slope_width_left
                slope_left_outer_height = terrain_left_height

            if slope_width_right < 0.1:
                slope_right_outer_x = road_right_x
                slope_right_outer_y = road_right_y
                slope_right_outer_height = z
            else:
                slope_right_outer_x = road_right_x - perp_x * slope_width_right
                slope_right_outer_y = road_right_y - perp_y * slope_width_right
                slope_right_outer_height = terrain_right_height

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

            all_road_faces.append([left1, right1, right2])
            all_road_faces.append([left1, right2, left2])

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

        road_poly_2d = [(x, y) for x, y, z in road_left_abs] + [
            (x, y) for x, y, z in reversed(road_right_abs)
        ]

        if config.LOCAL_OFFSET is not None:
            slope_left_2d = [
                (v[0] + config.LOCAL_OFFSET[0], v[1] + config.LOCAL_OFFSET[1])
                for v in slope_left_outer_vertices
            ]
            slope_right_2d = [
                (v[0] + config.LOCAL_OFFSET[0], v[1] + config.LOCAL_OFFSET[1])
                for v in slope_right_outer_vertices
            ]
        else:
            slope_left_2d = [(v[0], v[1]) for v in slope_left_outer_vertices]
            slope_right_2d = [(v[0], v[1]) for v in slope_right_outer_vertices]

        slope_poly_2d = (
            slope_left_2d
            + [(x, y) for x, y, z in reversed(road_left_abs)]
            + [(x, y) for x, y, z in road_right_abs]
            + list(reversed(slope_right_2d))
        )

        road_slope_polygons_2d.append(
            {"road_polygon": road_poly_2d, "slope_polygon": slope_poly_2d}
        )

        processed += 1
        if processed % 100 == 0:
            print(f"  {processed}/{total_roads} Straßen...")

    print(f"  ✓ {len(all_road_vertices)} Straßen-Vertices")
    print(f"  ✓ {len(all_road_faces)} Straßen-Faces")
    print(f"  ✓ {len(all_slope_vertices)} Böschungs-Vertices")
    print(f"  ✓ {len(all_slope_faces)} Böschungs-Faces")
    print(
        f"  ✓ {len(road_slope_polygons_2d)} Road/Slope-Polygone für Grid-Ausschneiden (2D)"
    )
    if clipped_roads > 0:
        print(f"  ℹ {clipped_roads} Straßen komplett außerhalb Grid (ignoriert)")

    return (
        all_road_vertices,
        all_road_faces,
        all_slope_vertices,
        all_slope_faces,
        road_slope_polygons_2d,
    )
