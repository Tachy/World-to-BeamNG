"""
Detection and handling of road junctions directly in centerlines.

This module detects crossings and T-junctions directly from the centerline coordinates
(where road points are connected), in order to mesh them cleanly during mesh generation.
"""

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection, box
from shapely.strtree import STRtree

from .. import config
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def _nearest_vertex_to_line(coords, line):
    """The vertex from `coords` closest to `line` (the first on a tie), as a Point - or None."""
    if len(coords) == 0:
        return None
    xy = np.array([(c[0], c[1]) for c in coords], dtype=float)
    shapely.prepare(line)  # prepared geometry: point-line distances via index instead of segment by segment
    dists = shapely.distance(line, shapely.points(xy))
    best = int(np.argmin(dists))
    if not np.isfinite(dists[best]):
        return None
    return Point(xy[best, 0], xy[best, 1])


class _JunctionIndex:
    """
    Spatial index of the junction positions for "is there already a junction within the tolerance?".

    Replaces the linear search over all junctions (with one np.array/np.sum per comparison, the main cost point
    with thousands of junctions and candidates). Like the linear search, it returns the FIRST junction in list
    order that lies within the tolerance (positions do not change after creation).
    """

    def __init__(self, junctions, tolerance):
        self.junctions = junctions
        self.tolerance = tolerance
        self.tolerance_sq = tolerance * tolerance
        self.cells = {}
        self.indexed = 0

    def _cell(self, x, y):
        return int(np.floor(x / self.tolerance)), int(np.floor(y / self.tolerance))

    def _sync(self):
        """Picks up junctions appended since the last call."""
        while self.indexed < len(self.junctions):
            position = self.junctions[self.indexed]["position"]
            self.cells.setdefault(self._cell(position[0], position[1]), []).append(self.indexed)
            self.indexed += 1

    def find(self, x, y):
        """First junction (list order) with distance <= tolerance to (x, y), or None."""
        self._sync()
        cx, cy = self._cell(x, y)
        best = None
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for index in self.cells.get((gx, gy), ()):
                    if best is not None and index >= best:
                        continue
                    position = self.junctions[index]["position"]
                    dx, dy = position[0] - x, position[1] - y
                    if dx * dx + dy * dy <= self.tolerance_sq:
                        best = index
        return None if best is None else self.junctions[best]


def detect_junctions_in_centerlines(road_polygons, height_points=None, height_elevations=None):
    """
    Detects junctions (crossings/T-junctions) directly in the centerlines.

    A junction is a point where several roads meet.
    OSM data has road ends as points that are connected -> natural junctions.

    Args:
        road_polygons: List of road dicts with 'coords', 'id', 'name'

    Returns:
        List of junctions with:
        {
            'position': (x, y, z),         # 3D position of the junction
            'road_indices': [i, j, ...],   # indices of the roads involved
            'connection_types': [...]      # 'end' or 'start' per road
        }
    """
    if not road_polygons:
        return []

    # Prepare DEM interpolator if provided
    dem_interpolator = None
    if height_points is not None and height_elevations is not None:
        dem_interpolator = cKDTree(height_points[:, :2])
        height_elevations_array = np.asarray(height_elevations)

    # Collect all road ends (start and end) and precompute segment data
    endpoints = []  # (x, y, z, road_idx, is_start)
    road_cache = []  # per road: precomputed segment data for direction/interpolation

    for road_idx, road in enumerate(road_polygons):
        coords = road["coords"]

        if len(coords) >= 2:
            # Start point
            start = coords[0]
            endpoints.append((start[0], start[1], start[2], road_idx, True))

            # End point
            end = coords[-1]
            endpoints.append((end[0], end[1], end[2], road_idx, False))

            coords_xy = coords[:, :2] if isinstance(coords, np.ndarray) else np.array(coords)[:, :2]
            coords_z = coords[:, 2] if isinstance(coords, np.ndarray) else np.array(coords)[:, 2]

            p1 = coords_xy[:-1]
            p2 = coords_xy[1:]
            seg = p2 - p1
            seg_len_sq = np.sum(seg * seg, axis=1)
            valid = seg_len_sq > 1e-12

            seg_valid = seg[valid]
            seg_len_sq_valid = seg_len_sq[valid]
            p1_valid = p1[valid]

            # Prepared start/end directions (normalized) for _direction_at_endpoint
            start_dir = seg_valid[0] if len(seg_valid) else np.array([1.0, 0.0])
            end_dir = seg_valid[-1] if len(seg_valid) else np.array([1.0, 0.0])
            sd_norm = np.sqrt(start_dir.dot(start_dir))
            ed_norm = np.sqrt(end_dir.dot(end_dir))
            if sd_norm > 0.01:
                start_dir = start_dir / sd_norm
            else:
                start_dir = np.array([1.0, 0.0])
            if ed_norm > 0.01:
                end_dir = end_dir / ed_norm
            else:
                end_dir = np.array([1.0, 0.0])

            # For Z interpolation: segment midpoints and mean Z per segment
            seg_mids = (p1 + p2) / 2.0
            z_mid = (coords_z[:-1] + coords_z[1:]) / 2.0

            road_cache.append(
                {
                    "coords": coords,
                    "coords_xy": coords_xy,
                    "coords_z": coords_z,
                    "p1_valid": p1_valid,
                    "seg_valid": seg_valid,
                    "seg_len_sq_valid": seg_len_sq_valid,
                    "seg_mids": seg_mids,
                    "z_mid": z_mid,
                    "start_dir": start_dir,
                    "end_dir": end_dir,
                }
            )
        else:
            road_cache.append(
                {
                    "coords": coords,
                    "coords_xy": (np.array(coords)[:, :2] if len(coords) else np.empty((0, 2))),
                    "coords_z": (np.array(coords)[:, 2] if len(coords) else np.empty((0,))),
                    "p1_valid": np.empty((0, 2)),
                    "seg_valid": np.empty((0, 2)),
                    "seg_len_sq_valid": np.empty((0,)),
                    "seg_mids": np.empty((0, 2)),
                    "z_mid": np.empty((0,)),
                    "start_dir": np.array([1.0, 0.0]),
                    "end_dir": np.array([1.0, 0.0]),
                }
            )

    if len(endpoints) < 2:
        return []

    # Build a KDTree for fast neighbor search (XY coordinates only)
    endpoints_xy = np.array([(p[0], p[1]) for p in endpoints])
    kdtree = cKDTree(endpoints_xy)

    # Find all points that are close together (tolerance 1 m)
    endpoint_merge_tol = 1.0
    junction_pairs = kdtree.query_pairs(r=endpoint_merge_tol)

    if not junction_pairs:
        return []

    # Group endpoints into junctions (union-find)
    from collections import defaultdict

    parent = list(range(len(endpoints)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in junction_pairs:
        union(i, j)

    # Collect clusters
    clusters = defaultdict(list)
    for i, endpoint in enumerate(endpoints):
        root = find(i)
        clusters[root].append(i)

    # Build junctions from clusters
    junctions = []

    def _direction_at_endpoint(road_idx, is_start):
        cache = road_cache[road_idx]
        return cache["start_dir"] if is_start else cache["end_dir"]

    def _get_direction_at_point(road_idx, proj_xy):
        """Computes the road direction at an arbitrary point (vectorized with reduce)."""
        cache = road_cache[road_idx]
        seg_valid = cache["seg_valid"]
        seg_len_sq_valid = cache["seg_len_sq_valid"]
        p1_valid = cache["p1_valid"]

        if seg_valid.size == 0:
            return np.array([1.0, 0.0])

        # OPTIMIZATION: use dot() instead of sum() for vector products
        vec_to_point = proj_xy - p1_valid  # Broadcasting: (n_valid, 2)
        # Dot product without reshape: sum(vec * seg) = sum along axis 1
        t = np.einsum("ij,ij->i", vec_to_point, seg_valid) / seg_len_sq_valid
        t = np.clip(t, 0, 1)

        # Nearest points on segments
        proj_pts = p1_valid + t[:, None] * seg_valid  # (n_valid, 2)

        # OPTIMIZATION: use einsum for the dot product (faster than sum+*+*)
        diff = proj_pts - proj_xy
        dists_sq = np.einsum("ij,ij->i", diff, diff)
        best_idx = int(np.argmin(dists_sq))

        # Direction of the best segment
        best_seg = seg_valid[best_idx]
        seg_norm = np.sqrt(seg_len_sq_valid[best_idx])
        return best_seg / seg_norm if seg_norm > 0.01 else np.array([1.0, 0.0])

    def _get_z_at_point(road_idx, xy_point):
        """Interpolates the Z coordinate at an XY point from the DEM (or from already normalized coords)."""
        if dem_interpolator is not None:
            # Use the DEM for normalized Z values
            dist, idx = dem_interpolator.query(xy_point)
            return float(height_elevations_array[idx])
        else:
            # Fallback: coords are already normalized (from polygon.py),
            # so we can interpolate directly without another DEM lookup
            cache = road_cache[road_idx]
            coords = cache["coords"]
            if not coords or len(coords) < 2:
                return 0.0 if not coords else coords[0][2]
            seg_mids = cache["seg_mids"]
            z_mid = cache["z_mid"]
            if seg_mids.size == 0:
                return coords[0][2]
            diff = seg_mids - np.array(xy_point)
            dists_sq = np.einsum("ij,ij->i", diff, diff)
            best_idx = int(np.argmin(dists_sq))
            return z_mid[best_idx]

    def _add_junction(position_xyz, cluster_indices, extra_connections=None):
        cluster_points = [endpoints[i] for i in cluster_indices]
        avg_x, avg_y, avg_z = position_xyz

        junction_roads = {}
        direction_vectors = {}

        for idx in cluster_indices:
            ep = endpoints[idx]
            road_idx = ep[3]
            is_start = ep[4]
            connection_type = "start" if is_start else "end"

            junction_roads.setdefault(road_idx, []).append(connection_type)
            direction_vectors[road_idx] = _direction_at_endpoint(road_idx, is_start)

        if extra_connections:
            for road_idx, conn_type, direction in extra_connections:
                junction_roads.setdefault(road_idx, []).append(conn_type)
                if direction is not None:
                    direction_vectors[road_idx] = direction

        road_list = sorted(list(junction_roads.keys()))
        if len(road_list) >= 2:
            junctions.append(
                {
                    "position": (avg_x, avg_y, avg_z),
                    "road_indices": road_list,
                    "connection_types": {r: junction_roads[r] for r in road_list},
                    "direction_vectors": direction_vectors,
                    "num_connections": len(cluster_indices) + (len(extra_connections) if extra_connections else 0),
                }
            )

    for cluster_indices in clusters.values():
        if len(cluster_indices) < 2:
            continue  # Not a real junction

        # Vectorized mean computation (use XY only!)
        cluster_points = np.array([endpoints[i][:3] for i in cluster_indices], dtype=np.float64)
        avg_xy = np.mean(cluster_points[:, :2], axis=0)

        # Interpolate the Z coordinate from the DEM (not from raw OSM data!)
        if dem_interpolator is not None:
            dist, idx = dem_interpolator.query(avg_xy)
            avg_z = float(height_elevations_array[idx])
        else:
            # Fallback: mean of the OSM Z values
            avg_z = float(np.mean(cluster_points[:, 2]))

        avg_point = (avg_xy[0], avg_xy[1], avg_z)
        _add_junction(avg_point, cluster_indices)

    # ---- Additional detection: endpoint on a continuous centerline (T-junction) ----
    # Detects points where an endpoint ends on the centerline of another road
    # (typical for T-junctions with a continuous road).

    # Search radius based on grid spacing only (OSM nodes usually lie exactly)
    t_search_radius = config.GRID_SPACING * 2.5
    t_line_tol = 1.0  # meters - tolerance for point-to-line distance (1.0 m)
    t_line_tol_sq = t_line_tol * t_line_tol
    merge_tol = 1.0  # merge tolerance to existing junctions (1.0 m)
    junction_index = _JunctionIndex(junctions, merge_tol)

    # Collect all line points with their road index and build LineStrings/indexes
    all_line_points = []  # [(x, y, road_idx, point_idx)]
    line_points_xy = []
    line_strings = []  # index-aligned with road_polygons (None for roads that are too short)
    indexed_geoms = []  # geometries that go into the STRtree
    geom_to_idx = {}  # STRtree geometry → road_idx

    for road_idx, road in enumerate(road_polygons):
        coords = road.get("coords", [])

        # Collect points (for the KDTree from T-junction detection)
        for pt_idx, coord in enumerate(coords):
            all_line_points.append((coord[0], coord[1], road_idx, pt_idx))
            line_points_xy.append([coord[0], coord[1]])

        # Build the LineString once
        if len(coords) >= 2:
            ls = LineString([(c[0], c[1]) for c in coords])
            line_strings.append(ls)
            indexed_geoms.append(ls)
            geom_to_idx[ls] = road_idx
        else:
            line_strings.append(None)

    if not line_points_xy:
        pass
    else:
        line_kdtree = cKDTree(np.array(line_points_xy))


        # STAGE 1: KDTree pre-selection - line points in the neighborhood, for all endpoints in ONE query
        endpoint_xy = np.array([(ep[0], ep[1]) for ep in endpoints], dtype=float)
        all_dists, all_indices = line_kdtree.query(endpoint_xy, k=50, distance_upper_bound=t_search_radius)
        n_line_points = len(all_line_points)

        for ep_number, (ep_x, ep_y, ep_z, road_idx, is_start) in enumerate(endpoints):
            nearby_indices = [
                i for i, d in zip(all_indices[ep_number].tolist(), all_dists[ep_number].tolist()) if d <= t_search_radius
            ]
            # For each endpoint, every other road is needed only once: the projection depends only on the endpoint
            # and the road, a repetition would hit the same junction and change nothing.
            seen_roads = set()

            for line_pt_idx in nearby_indices:
                if line_pt_idx >= n_line_points:
                    continue

                lx, ly, other_road_idx, pt_idx_on_line = all_line_points[line_pt_idx]

                if other_road_idx == road_idx:
                    continue  # Same road
                if other_road_idx in seen_roads:
                    continue
                seen_roads.add(other_road_idx)

                # STAGE 2: Project onto the entire line and check the point-to-line distance
                other_line = line_strings[other_road_idx]
                if other_line is None:
                    continue

                proj_dist = other_line.project(Point(ep_x, ep_y))
                proj_pt = other_line.interpolate(proj_dist)
                proj_xy = (proj_pt.x, proj_pt.y)

                # Compute the true point-to-line distance
                # STAGE 3: Only accept if the point is really close to the line (10 cm)
                dist_vec = np.array([ep_x - proj_pt.x, ep_y - proj_pt.y])
                point_to_line_dist_sq = np.sum(dist_vec * dist_vec)
                if point_to_line_dist_sq > t_line_tol_sq:
                    continue  # Point is too far from the line

                # Interpolate the Z coordinate from the DEM (not from raw OSM data)
                proj_z = _get_z_at_point(other_road_idx, proj_xy)

                # Check whether a junction already exists at THIS position (position-based!)
                # Allows several junctions between the same roads at different positions
                existing_junction = junction_index.find(proj_xy[0], proj_xy[1])

                # If a junction exists, try to add roads instead of skipping
                if existing_junction is not None:
                    # Try to add roads to this existing junction
                    through_dir = _get_direction_at_point(other_road_idx, proj_xy)

                    # Make sure connection_types is initialized
                    if "connection_types" not in existing_junction:
                        existing_junction["connection_types"] = {}
                    if "direction_vectors" not in existing_junction:
                        existing_junction["direction_vectors"] = {}

                    # Add the first road or update connection_type
                    if road_idx not in existing_junction["road_indices"]:
                        existing_junction["road_indices"].append(road_idx)

                    conn_type = "start" if is_start else "end"
                    if conn_type not in existing_junction["connection_types"].get(road_idx, []):
                        if road_idx not in existing_junction["connection_types"]:
                            existing_junction["connection_types"][road_idx] = []
                        existing_junction["connection_types"][road_idx].append(conn_type)
                    existing_junction["direction_vectors"][road_idx] = _direction_at_endpoint(road_idx, is_start)

                    # Add the second road or update connection_type
                    if other_road_idx not in existing_junction["road_indices"]:
                        existing_junction["road_indices"].append(other_road_idx)

                    if "mid" not in existing_junction["connection_types"].get(other_road_idx, []):
                        if other_road_idx not in existing_junction["connection_types"]:
                            existing_junction["connection_types"][other_road_idx] = []
                        existing_junction["connection_types"][other_road_idx].append("mid")
                    existing_junction["direction_vectors"][other_road_idx] = through_dir

                    continue  # Done with this check

                # Direction on the line at the projection point
                through_dir = _get_direction_at_point(other_road_idx, proj_xy)
                new_junc_pos = (proj_xy[0], proj_xy[1], proj_z)

                # Try to merge with an existing junction
                merged = False
                j = junction_index.find(new_junc_pos[0], new_junc_pos[1])
                if j is not None:
                    if road_idx not in j["road_indices"]:
                        j["road_indices"].append(road_idx)
                        j["connection_types"][road_idx] = ["start" if is_start else "end"]
                        j["direction_vectors"][road_idx] = _direction_at_endpoint(road_idx, is_start)
                    if other_road_idx not in j["road_indices"]:
                        j["road_indices"].append(other_road_idx)
                        j["connection_types"][other_road_idx] = ["mid"]
                        j["direction_vectors"][other_road_idx] = through_dir
                    merged = True

                if not merged:
                    extra_conns = [
                        (
                            road_idx,
                            "start" if is_start else "end",
                            _direction_at_endpoint(road_idx, is_start),
                        ),
                        (other_road_idx, "mid", through_dir),
                    ]
                    _add_junction(new_junc_pos, [], extra_connections=extra_conns)


    # ---- Third detection: line-on-line crossings (X-junctions without an endpoint match) ----
    # Detects crossings where two roads cross but the endpoints do not meet exactly

    ll_search_radius = config.GRID_SPACING * 2.5
    ll_line_tol = 1.0  # meters - tolerance for line-to-line distance (1 m)

    # Use the already created KDTree of the line points for performance
    if line_points_xy and indexed_geoms:
        tree = STRtree(indexed_geoms)

        for road_idx, road in enumerate(road_polygons):
            road_line = line_strings[road_idx]
            if road_line is None:
                continue

            road_bounds = road_line.bounds  # (minx, miny, maxx, maxy)
            pad = ll_search_radius + ll_line_tol
            query_geom = box(
                road_bounds[0] - pad,
                road_bounds[1] - pad,
                road_bounds[2] + pad,
                road_bounds[3] + pad,
            )

            # Only candidates with a larger road number, and check their distance to the line ONCE, vectorized
            # (candidate order stays that of the STRtree, so the junction order stays the same).
            candidates = [c for c in tree.query(query_geom).tolist() if c > road_idx]
            if not candidates:
                continue
            # dwithin via the STRtree (index + prepared geometry) instead of distance() per pair
            near = set(tree.query(road_line, predicate="dwithin", distance=ll_line_tol).tolist())

            for other_idx in candidates:
                other_road_idx = other_idx  # tree.query() returns indices in newer Shapely

                other_line = line_strings[other_road_idx]

                if other_idx not in near:
                    continue  # distance > ll_line_tol

                intersection = road_line.intersection(other_line)

                if intersection.is_empty:
                    nearest_pt1 = _nearest_vertex_to_line(road.get("coords", []), other_line)
                    nearest_pt2 = _nearest_vertex_to_line(road_polygons[other_road_idx].get("coords", []), road_line)

                    if nearest_pt1 and nearest_pt2:
                        cross_x = (nearest_pt1.x + nearest_pt2.x) / 2
                        cross_y = (nearest_pt1.y + nearest_pt2.y) / 2
                    else:
                        continue
                else:
                    if isinstance(intersection, Point):
                        cross_x, cross_y = intersection.x, intersection.y
                    elif isinstance(intersection, (MultiPoint, GeometryCollection)):
                        geoms = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
                        if geoms and isinstance(geoms[0], Point):
                            cross_x, cross_y = geoms[0].x, geoms[0].y
                        else:
                            continue
                    else:
                        continue

                if junction_index.find(cross_x, cross_y) is not None:
                    continue  # there is already a junction at this location

                best_z1 = _get_z_at_point(road_idx, (cross_x, cross_y))
                best_z2 = _get_z_at_point(other_road_idx, (cross_x, cross_y))
                cross_z = (best_z1 + best_z2) / 2

                dir1 = _get_direction_at_point(road_idx, (cross_x, cross_y))
                dir2 = _get_direction_at_point(other_road_idx, (cross_x, cross_y))

                new_junc_pos = (cross_x, cross_y, cross_z)

                merged = False
                j = junction_index.find(new_junc_pos[0], new_junc_pos[1])
                if j is not None:
                    if road_idx not in j["road_indices"]:
                        j["road_indices"].append(road_idx)
                        j["connection_types"][road_idx] = ["mid"]
                        j["direction_vectors"][road_idx] = dir1
                    if other_road_idx not in j["road_indices"]:
                        j["road_indices"].append(other_road_idx)
                        j["connection_types"][other_road_idx] = ["mid"]
                        j["direction_vectors"][other_road_idx] = dir2
                    merged = True

                if not merged:
                    extra_conns = [
                        (road_idx, "mid", dir1),
                        (other_road_idx, "mid", dir2),
                    ]
                    _add_junction(new_junc_pos, [], extra_connections=extra_conns)


    return junctions


def mark_junction_endpoints(road_polygons, junctions):
    """
    Marks road endpoints that belong to junctions.

    This is used later during mesh generation to identify these points
    when stitching.

    Args:
        road_polygons: List of roads (is modified)
        junctions: List of junctions from detect_junctions_in_centerlines()

    Returns:
        Modified road_polygons with a 'junction_indices' attribute
    """
    # Mark every endpoint
    for road_idx, road in enumerate(road_polygons):
        road["junction_indices"] = {"start": None, "end": None}

    for junction_idx, junction in enumerate(junctions):
        for road_idx in junction["road_indices"]:
            if road_idx < len(road_polygons):
                conn_types = junction["connection_types"].get(road_idx, [])
                for conn_type in conn_types:
                    if road_idx < len(road_polygons):
                        road_polygons[road_idx]["junction_indices"][conn_type] = junction_idx

    return road_polygons


def _same_xy(a, b) -> bool:
    """
    np.allclose(a[:2], b[:2], atol=1e-6) for exactly two coordinates - bit-identical formula (including the relative
    default tolerance rtol=1e-5, i.e. about 1 cm for coordinates around 1000 m), but without the numpy overhead:
    allclose costs ~50 µs per call and was called tens of thousands of times during the split.
    """
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    return abs(ax - bx) <= 1e-6 + 1e-5 * abs(bx) and abs(ay - by) <= 1e-6 + 1e-5 * abs(by)


def split_roads_at_mid_junctions(road_polygons, junctions, merge_tol=0.5):
    """
    Splits roads at junction points that were detected as "mid".

    Result: new road list in which each section is its own road with a
    start/end junction. All properties of the original road are
    copied, the IDs are extended by a part suffix.
    """

    if not junctions or not road_polygons:
        return road_polygons, junctions

    junction_positions = [np.asarray(j["position"], dtype=float) for j in junctions]

    def _dir_start(coords_arr):
        if len(coords_arr) < 2:
            return np.array([1.0, 0.0])
        v = coords_arr[1, :2] - coords_arr[0, :2]
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def _dir_end(coords_arr):
        if len(coords_arr) < 2:
            return np.array([1.0, 0.0])
        v = coords_arr[-1, :2] - coords_arr[-2, :2]
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([1.0, 0.0])

    def _new_id(base_id, part_idx):
        if isinstance(base_id, int):
            return base_id * 1000 + part_idx
        return f"{base_id}_p{part_idx}"

    # Road -> [(junction index, connection types)] in junction order - built once instead of iterating over all
    # junctions for each road (roads x junctions)
    connections_by_road = {}
    for j_idx, j in enumerate(junctions):
        for r_idx, conn in j.get("connection_types", {}).items():
            connections_by_road.setdefault(r_idx, []).append((j_idx, conn))

    new_roads = []
    old_to_new_map = {}  # old road idx -> list of new road indices

    for road_idx, road in enumerate(road_polygons):
        coords = np.asarray(road.get("coords", []), dtype=float)
        if len(coords) < 2:
            # Take over unchanged - with an explicit osm_tags copy
            new_road = {
                "id": road.get("id"),
                "coords": road.get("coords", []),
                "name": road.get("name", ""),
                "osm_tags": dict(road.get("osm_tags", {})),  # Deep copy
                "junction_indices": {"start": None, "end": None},
                "start_junction_id": None,
                "end_junction_id": None,
            }
            new_roads.append(new_road)
            # Set the mapping so that junctions keep their connections
            old_to_new_map[road_idx] = [len(new_roads) - 1]
            continue

        # Collect known start/end junctions from the original connection_types
        start_junc_id = None
        end_junc_id = None
        road_connections = connections_by_road.get(road_idx, [])
        for j_idx, conn in road_connections:
            if "start" in conn:
                start_junc_id = j_idx
            if "end" in conn:
                end_junc_id = j_idx

        # Collect all mid junctions for this road
        cut_marks = []
        for j_idx, conn in road_connections:
            if "mid" not in conn:
                continue
            pos = junction_positions[j_idx]
            p = coords[:, :2]
            seg = p[1:] - p[:-1]
            seg_len = np.linalg.norm(seg, axis=1)
            valid = seg_len > 1e-6
            if not np.any(valid):
                continue
            seg_len_sq = np.maximum(seg_len * seg_len, 1e-12)
            vec = pos[:2] - p[:-1]
            t = np.sum(vec * seg, axis=1) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj = p[:-1] + seg * t[:, None]
            dist_sq = np.sum((proj - pos[:2]) ** 2, axis=1)
            best = int(np.argmin(dist_sq))
            best_t = float(t[best])
            best_proj = proj[best]
            best_z = coords[best, 2] + best_t * (coords[best + 1, 2] - coords[best, 2])
            # Arc position as segment index + t
            cut_marks.append((best + best_t, best_proj[0], best_proj[1], best_z, j_idx))

        if not cut_marks:
            # No cuts - with an explicit osm_tags copy
            new_road = {
                "id": road.get("id"),
                "coords": road.get("coords", []),
                "name": road.get("name", ""),
                "osm_tags": dict(road.get("osm_tags", {})),  # Deep copy
                "junction_indices": {"start": None, "end": None},
                "start_junction_id": None,
                "end_junction_id": None,
            }
            new_roads.append(new_road)
            # Set the mapping so that unchanged roads remain in junctions
            old_to_new_map[road_idx] = [len(new_roads) - 1]
            continue

        # Merge duplicate cuts (close together)
        cut_marks.sort(key=lambda x: x[0])
        merged_cuts = []
        for c in cut_marks:
            if not merged_cuts:
                merged_cuts.append(c)
                continue
            last = merged_cuts[-1]
            if abs(c[0] - last[0]) <= 1e-4:
                merged_cuts[-1] = c  # replace with the last one (same segment)
            else:
                merged_cuts.append(c)

        # Map segment -> cuts
        cuts_by_seg = {}
        for c in merged_cuts:
            s_pos = c[0]
            seg_idx = int(np.floor(s_pos))
            t_seg = s_pos - seg_idx
            cuts_by_seg.setdefault(seg_idx, []).append((t_seg, c[1], c[2], c[3], c[4]))

        current_coords = [coords[0]]
        current_start_j = start_junc_id
        parts = []

        for seg_idx in range(len(coords) - 1):
            if seg_idx in cuts_by_seg:
                seg_cuts = sorted(cuts_by_seg[seg_idx], key=lambda x: x[0])
                for t_seg, x_cut, y_cut, z_cut, j_idx in seg_cuts:
                    cut_pt = np.array([x_cut, y_cut, z_cut])
                    # Only append the cut point if it is not already the last point (t_seg can fall exactly on the
                    # predecessor end point due to the floor() segment assignment in split_roads_at_mid_junctions)
                    # - otherwise a zero-length segment would arise (see test_junction_split.py). The split
                    # (new part) still happens, just without a duplicate point.
                    if not _same_xy(current_coords[-1], cut_pt):
                        current_coords.append(cut_pt)
                    parts.append((current_coords, current_start_j, j_idx))
                    current_coords = [cut_pt]
                    current_start_j = j_idx
            # add the end of the segment if no cut ends there (same zero-length guard as above: a
            # cut whose projection falls on the segment end point has already set current_coords there)
            next_pt = coords[seg_idx + 1]
            if not _same_xy(current_coords[-1], next_pt):
                current_coords.append(next_pt)

        # last part
        parts.append((current_coords, current_start_j, end_junc_id))

        # Build new roads from the parts
        base_id = road.get("id", f"road{road_idx}")
        new_ids_for_this = []
        for idx, (coords_part, start_j, end_j) in enumerate(parts, 1):
            coords_arr = np.asarray(coords_part)
            if len(coords_arr) < 2:
                continue
            # FIX: explicitly copy all fields including osm_tags
            new_road = {
                "id": _new_id(base_id, idx),
                "coords": coords_arr.tolist(),
                "name": road.get("name", ""),
                "osm_tags": dict(road.get("osm_tags", {})),  # Deep copy of osm_tags
                "start_junction_id": start_j,
                "end_junction_id": end_j,
                "junction_indices": {"start": start_j, "end": end_j},
            }
            new_roads.append(new_road)
            new_ids_for_this.append(len(new_roads) - 1)

        if new_ids_for_this:
            old_to_new_map[road_idx] = new_ids_for_this
        else:
            old_to_new_map[road_idx] = []

    # Rebuild the junction list based on the new roads, preserve the junction count
    new_junctions = []
    for j_idx, j in enumerate(junctions):
        pos = junction_positions[j_idx]
        roads_here = []
        conn_types = {}
        dir_vectors = {}

        for old_ridx, conn_list in j.get("connection_types", {}).items():
            mapped = old_to_new_map.get(old_ridx, [])
            for new_ridx in mapped:
                road = new_roads[new_ridx]
                coords_arr = np.asarray(road.get("coords", []), dtype=float)

                # Set missing start/end IDs if necessary
                if "start" in conn_list and road.get("start_junction_id") is None:
                    road["start_junction_id"] = j_idx
                    road["junction_indices"]["start"] = j_idx
                if "end" in conn_list and road.get("end_junction_id") is None:
                    road["end_junction_id"] = j_idx
                    road["junction_indices"]["end"] = j_idx

                # Only count if the part actually starts/ends at this junction
                if road.get("start_junction_id") == j_idx:
                    roads_here.append(new_ridx)
                    conn_types.setdefault(new_ridx, []).append("start")
                    dir_vectors[new_ridx] = _dir_start(coords_arr)
                if road.get("end_junction_id") == j_idx:
                    roads_here.append(new_ridx)
                    conn_types.setdefault(new_ridx, []).append("end")
                    dir_vectors[new_ridx] = _dir_end(coords_arr)

        roads_unique = sorted(set(roads_here))
        if len(roads_unique) >= 2:
            new_junctions.append(
                {
                    "position": tuple(pos.tolist()),
                    "road_indices": roads_unique,
                    "connection_types": {r: conn_types.get(r, []) for r in roads_unique},
                    "direction_vectors": {r: dir_vectors.get(r, np.array([1.0, 0.0])) for r in roads_unique},
                    "num_connections": sum(len(conn_types.get(r, [])) for r in roads_unique),
                }
            )

    return new_roads, new_junctions



OSM_NODE_JUNCTION_TOL = 1.0  # how close a tunnel junction must be to a shared OSM node, in meters


def _shared_osm_node_points(road_polygons):
    """
    Position (x, y) of all OSM nodes shared by at least two DIFFERENT ways - or None if nodes are not known
    for all roads (then nothing can be excluded).
    """
    ways_of_node = {}
    position = {}
    for road in road_polygons:
        nodes = road.get("osm_nodes")
        if not nodes:
            return None
        for node_id, x, y in nodes:
            ways_of_node.setdefault(node_id, set()).add(road.get("osm_way_id", road.get("id")))
            position[node_id] = (x, y)
    return np.array([position[n] for n, ways in ways_of_node.items() if len(ways) >= 2], dtype=float).reshape(-1, 2)


def _junction_network(road_polygons, allowed_points=None):
    """detect -> split -> mark for a self-contained road network. With `allowed_points` ((N, 2) array),
    only junctions that lie at most OSM_NODE_JUNCTION_TOL from one of these points are kept."""
    junctions = detect_junctions_in_centerlines(road_polygons)
    if allowed_points is not None:
        if len(allowed_points) == 0:
            junctions = []
        else:
            tree = cKDTree(allowed_points)
            junctions = [j for j in junctions if tree.query(j["position"][:2])[0] <= OSM_NODE_JUNCTION_TOL]
    road_polygons, junctions = split_roads_at_mid_junctions(road_polygons, junctions)  # Split FIRST
    road_polygons = mark_junction_endpoints(road_polygons, junctions)  # THEN mark
    return road_polygons, junctions


def build_junction_network(road_polygons):
    """
    Junction detection, split at mid junctions and endpoint marking (detect -> split -> mark) - separately for
    tunnels and all other roads.

    Tunnels lie on a different level and never form crossings with surface roads: a path that leads over a tunnel
    in 2D is not a junction - otherwise both would be split there, and the tunnel pieces would get portals in the
    middle of the mountain. Among themselves, however, tunnels do form real junctions (branches inside the tunnel),
    so they get their own detection - but only where the ways share a node in OSM: two tunnels that cross in 2D at
    different depths (e.g. fortress tunnels above the Gotthard road tunnel) have no shared node and stay
    unsplit. Both networks are merged afterwards (first the other roads, then the tunnels); road and junction
    indices of the tunnel network are shifted for this.

    Returns:
        (road_polygons, junctions)
    """
    from .road_structures import classify_structure

    is_tunnel = [classify_structure(r.get("osm_tags", {})) == "tunnel" for r in road_polygons]
    others, junctions = _junction_network([r for r, t in zip(road_polygons, is_tunnel) if not t])
    tunnel_roads = [r for r, t in zip(road_polygons, is_tunnel) if t]
    tunnels, tunnel_junctions = _junction_network(tunnel_roads, allowed_points=_shared_osm_node_points(tunnel_roads))

    road_offset, junction_offset = len(others), len(junctions)
    for junction in tunnel_junctions:
        junction["road_indices"] = [i + road_offset for i in junction["road_indices"]]
        for key in ("connection_types", "direction_vectors"):
            if key in junction:
                junction[key] = {i + road_offset: v for i, v in junction[key].items()}
    for tunnel in tunnels:
        for field in ("start_junction_id", "end_junction_id"):
            if tunnel.get(field) is not None:
                tunnel[field] += junction_offset
        tunnel["junction_indices"] = {
            end: (None if index is None else index + junction_offset) for end, index in tunnel["junction_indices"].items()
        }
    return others + tunnels, junctions + tunnel_junctions
