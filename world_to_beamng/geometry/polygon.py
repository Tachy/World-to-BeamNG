"""
Polygon operations and road extraction.
"""

import numpy as np
from shapely.geometry import Polygon

from ..terrain.elevation import get_elevations_for_points
from ..geometry.coordinates import transformer_to_utm
from ..config import OSM_MAPPER
from .. import config
from .road_structures import classify_structure
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()


def clip_road_polygons(road_polygons, grid_bounds_local, margin=3.0):
    """
    Clips road polygons at the grid edge with a margin.

    Args:
        road_polygons: List of road dictionaries with 'coords'
        grid_bounds_local: (min_x, max_x, min_y, max_y) in local coordinates
        margin: Distance from the grid edge in meters (default 3.0)
                Positive = roads are clipped BEFORE the edge
                Negative = roads are extended BEYOND the edge

    Returns:
        Clipped road_polygons (roads that lie completely outside are removed)
    """
    if not config.ENABLE_ROAD_CLIPPING:
        return road_polygons

    if not grid_bounds_local:
        return road_polygons

    min_x, max_x, min_y, max_y = grid_bounds_local

    # IMPORTANT: margin semantics:
    # - Positive (e.g. +10): clip box gets SMALLER → roads end 10 m BEFORE the grid edge
    # - Negative (e.g. -20): clip box gets LARGER → roads end 20 m BEYOND the grid edge
    # Formula: clip_min = min + margin (with margin=-20 → -1000 + (-20) = -1020 ✓)
    clip_min_x = min_x + margin
    clip_max_x = max_x - margin
    clip_min_y = min_y + margin
    clip_max_y = max_y - margin

    clipped_roads = []
    removed_count = 0
    segment_count = 0
    split_count = 0

    for road in road_polygons:
        coords = road["coords"]

        # IMPORTANT: Points removed by clipping must NOT simply be skipped -
        # otherwise the remaining points, which are actually far apart (e.g.
        # from a road that makes a wide arc far outside the tile and reaches
        # back into the tile at two completely different places), are joined
        # into a single artificial "teleport" straight line. That produces a
        # wrong centerline with wrong Z values, which then ends up as a huge,
        # unnatural cliff in the embankment blend. Instead: start a new,
        # independent road section at every gap (as in the old mesh workflow,
        # where roads actually ended at the edge).
        runs = []
        current_run = []
        for x, y, z in coords:
            if clip_min_x <= x <= clip_max_x and clip_min_y <= y <= clip_max_y:
                current_run.append((x, y, z))
            elif current_run:
                runs.append(current_run)
                current_run = []
        if current_run:
            runs.append(current_run)

        osm_tags = road.get("osm_tags", {})
        road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
        max_seg = config.GRID_SPACING

        for run_idx, new_coords in enumerate(runs):
            if len(new_coords) < 2:
                removed_count += 1
                continue

            # Subdivide long segments after clipping (to fill large gaps within
            # a contiguous section, e.g. for coarsely sampled OSM ways)
            final_coords = []
            for i, coord in enumerate(new_coords):
                final_coords.append(coord)

                # If not the last segment
                if i < len(new_coords) - 1:
                    next_coord = new_coords[i + 1]
                    # Compute distance to the next point
                    dist = np.sqrt(
                        (next_coord[0] - coord[0]) ** 2
                        + (next_coord[1] - coord[1]) ** 2
                        + (next_coord[2] - coord[2]) ** 2
                    )

                    # If the segment is longer than max_seg, interpolate intermediate points
                    if dist > max_seg:
                        num_intermediate = int(np.ceil(dist / max_seg)) - 1
                        for j in range(1, num_intermediate + 1):
                            t = j / (num_intermediate + 1)
                            inter_point = (
                                coord[0] + t * (next_coord[0] - coord[0]),
                                coord[1] + t * (next_coord[1] - coord[1]),
                                coord[2] + t * (next_coord[2] - coord[2]),
                            )
                            final_coords.append(inter_point)

            road_id = road["id"]
            if len(runs) > 1:
                # Several separate sections from the same road -> unique IDs
                road_id = f"{road_id}_c{run_idx}" if isinstance(road_id, str) else road_id * 1000 + run_idx
                split_count += 1

            clipped_roads.append(
                {
                    "id": road_id,
                    "coords": final_coords,
                    "name": road["name"],
                    "osm_tags": osm_tags,  # pass OSM tags through
                    "osm_way_id": road.get("osm_way_id"),
                    "osm_nodes": road.get("osm_nodes"),
                }
            )
            segment_count += len(coords) - len(final_coords)

    if removed_count > 0 or segment_count > 0 or split_count > 0:
        logger.info(
            f"  Clipping: {removed_count} road (sections) removed, "
            f"{segment_count} points outside the grid removed, "
            f"{split_count} roads split into separate sections at the edge"
        )

    return clipped_roads


def drop_close_nodes(nodes, min_dist):
    """Removes nodes that are closer (in XY) than min_dist to the previously kept node.

    Start and end nodes are kept exactly (junction connection to
    neighboring roads). If the last segment is too short, the
    second-to-last node is removed instead.

    Args:
        nodes: List of nodes [x, y, z, ...] (further entries such as the width
            remain unchanged)
        min_dist: Minimum distance in meters

    Returns:
        Filtered node list, or [] if the road is unusably short after filtering
        (fewer than 2 nodes or start-end distance < min_dist).
    """
    if len(nodes) < 2:
        return []

    def dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    kept = [nodes[0]]
    for node in nodes[1:-1]:
        if dist(node, kept[-1]) >= min_dist:
            kept.append(node)

    last = nodes[-1]
    if len(kept) > 1 and dist(last, kept[-1]) < min_dist:
        kept.pop()
    kept.append(last)

    if dist(kept[0], kept[-1]) < min_dist:
        return []
    return kept


def resample_road_xy_only(xy_coords, target_spacing):
    """Resamples the centerline in the XY plane with a fixed step size.

    Args:
        xy_coords: List of (x, y) coordinates
        target_spacing: Target spacing between points in meters

    Returns:
        List of resampled (x, y) coordinates
    """
    if len(xy_coords) < 2:
        return xy_coords

    coords_arr = np.array(xy_coords)

    # Compute cumulative distance
    diffs = np.diff(coords_arr, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum[-1]

    if total_len < 1e-6:
        return xy_coords

    # Compute sample positions
    num_samples = max(2, int(np.ceil(total_len / target_spacing)) + 1)
    t = np.linspace(0.0, total_len, num_samples)

    # Interpolate x, y
    x = np.interp(t, cum, coords_arr[:, 0])
    y = np.interp(t, cum, coords_arr[:, 1])

    # Make sure start/end stay exact (important for junctions!)
    x[0], y[0] = coords_arr[0, 0], coords_arr[0, 1]
    x[-1], y[-1] = coords_arr[-1, 0], coords_arr[-1, 1]

    return list(zip(x, y))


def _linear_elevation_profile(coords):
    """Replaces the Z values with linear interpolation between start and end point (arc-length weighted);
    start/end are kept exactly (the regular road connects there)."""
    arr = np.array(coords, dtype=float)
    xy = arr[:, :2]
    diffs = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-9:
        return coords
    t = cum / total
    z = arr[0, 2] + t * (arr[-1, 2] - arr[0, 2])
    return [(float(x), float(y), float(zz)) for (x, y), zz in zip(xy, z)]


def _endpoint_matches(pt_a, pt_b, tol=1e-4):
    return abs(pt_a[0] - pt_b[0]) < tol and abs(pt_a[1] - pt_b[1]) < tol and abs(pt_a[2] - pt_b[2]) < tol


def _find_unique_touching_road(road_polygons, point, exclude_id, predicate=None):
    """Finds the one road in road_polygons whose start or end point matches `point`
    (except `exclude_id`, optionally filtered via `predicate(osm_tags)`); if ambiguous (0 or 2+
    hits, e.g. at a real multi-way crossing), None is returned - there the sharp edge
    must be preserved for the junction logic.

    Returns:
        (road, touching_at_start) or None
    """
    found = []
    for road in road_polygons:
        if road["id"] == exclude_id:
            continue
        if predicate and not predicate(road.get("osm_tags", {})):
            continue
        coords = road["coords"]
        if len(coords) < 2:
            continue
        if _endpoint_matches(coords[0], point):
            found.append((road, True))
        elif _endpoint_matches(coords[-1], point):
            found.append((road, False))
    return found[0] if len(found) == 1 else None


def _walk_bridge_approach(ordered, slope_threshold, max_extension):
    """Walks along `ordered` from index 0 (contact point with the bridge) as long as the slope of the
    next segment stays >= slope_threshold (still part of the hillside that the too-short bridge does not
    cover) and the cumulative distance does not exceed max_extension; if the neighbor itself is shorter,
    the walk stops at its own end (no third road is included).

    Returns:
        (extension_points, remaining_ordered) - extension_points are the new bridge points in the
        direction away from the contact point (excluding the contact point itself); remaining_ordered is the
        remainder staying with the neighbor (starting with the new, shared boundary point).
    """
    idx = 0
    cum = 0.0
    n = len(ordered)
    while idx + 1 < n:
        a, b = ordered[idx], ordered[idx + 1]
        seg_dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if seg_dist < 1e-9:
            idx += 1
            continue
        slope = abs(b[2] - a[2]) / seg_dist
        if slope < slope_threshold:
            break
        if cum + seg_dist > max_extension + 1e-9:
            remaining = max_extension - cum
            if remaining > 1e-9:
                t = remaining / seg_dist
                point = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))
                ordered = ordered[: idx + 1] + [point] + ordered[idx + 1 :]
                idx += 1
            break
        cum += seg_dist
        idx += 1

    return ordered[1 : idx + 1], ordered[idx:]


def extend_short_bridges_to_natural_grade(road_polygons, slope_threshold=None, max_extension=None):
    """Extends too-short tagged bridges into their adjacent surface road until normal slope prevails
    there again: some OSM bridges already begin in the middle of the hillside instead of at road
    level, so the subsequent linear elevation interpolation (apply_structure_elevation_profiles)
    yields an unrealistically steep ramp. Runs BEFORE apply_structure_elevation_profiles, so that it works on
    the already extended course.

    Extension only happens if exactly ONE surface road adjoins a bridge end (unambiguous
    connection); tunnels/galleries as "neighbor" are ignored (their elevation profile is not real terrain).
    Config: BRIDGE_APPROACH_SLOPE_THRESHOLD, BRIDGE_APPROACH_MAX_EXTENSION.
    """
    if slope_threshold is None:
        slope_threshold = config.BRIDGE_APPROACH_SLOPE_THRESHOLD
    if max_extension is None:
        max_extension = config.BRIDGE_APPROACH_MAX_EXTENSION

    is_surface = lambda tags: classify_structure(tags) == "surface"

    for bridge in [r for r in road_polygons if classify_structure(r.get("osm_tags", {})) == "bridge"]:
        coords = bridge["coords"]
        if len(coords) < 2:
            continue

        for at_start in (True, False):
            touch_point = coords[0] if at_start else coords[-1]
            found = _find_unique_touching_road(road_polygons, touch_point, bridge["id"], predicate=is_surface)
            if not found:
                continue
            neighbor, touching_at_start = found
            ordered = neighbor["coords"] if touching_at_start else list(reversed(neighbor["coords"]))

            extension_points, remaining_ordered = _walk_bridge_approach(ordered, slope_threshold, max_extension)
            if not extension_points:
                continue

            coords = (list(reversed(extension_points)) + coords) if at_start else (coords + extension_points)
            neighbor["coords"] = remaining_ordered if touching_at_start else list(reversed(remaining_ordered))

        bridge["coords"] = coords

    return road_polygons


def _stable_grade_start(ordered, slope_threshold, stable_length, max_distance):
    """Walks `ordered` outward from index 0 (portal) and looks for the first point from which the slope over
    the next `stable_length` meters stays consistently below `slope_threshold`.

    Returns:
        (index, cum) - index and arc length of this point plus the cumulative arc lengths of all points,
        or None if no stable section begins within `max_distance` (or the road ends).
    """
    arr = np.asarray(ordered, dtype=float)
    seg_len = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(seg_len > 1e-9, np.abs(np.diff(arr[:, 2])) / seg_len, 0.0)

    for idx in range(len(arr) - 1):
        if cum[idx] > max_distance:
            return None
        window_end = cum[idx] + stable_length
        if window_end > cum[-1]:
            return None  # road too short to establish a stable slope
        in_window = (cum[:-1] >= cum[idx]) & (cum[:-1] < window_end)
        if np.all(slope[in_window] < slope_threshold):
            return idx, cum
    return None


def settle_tunnel_portals_to_approach_grade(road_polygons, slope_threshold=None, stable_length=None, max_distance=None):
    """Brings tunnel portals and gallery ends to the height of the approach road (at a gallery, the DGM shows the
    roof including the earth cover, not the carriageway - the same problem as at the tunnel portal).

    The OSM tunnel start often already lies in the hillside: the DGM there shows the portal embankment or the
    hillside above the portal instead of the road level, the approach climbs steeply over the last few meters and
    the tunnel begins (linear profile from this point, see apply_structure_elevation_profiles) several meters too
    high.

    For each tunnel end with exactly one adjoining surface road, the approach is sampled away from the portal
    until the slope stays stably below `slope_threshold` over `stable_length` meters. The slope measured
    there is extended to the portal: the portal point and the approach points before it then lie on this
    straight line. Runs BEFORE apply_structure_elevation_profiles, so that the tunnel profile already starts from
    the new portal height. Config: TUNNEL_APPROACH_SLOPE_THRESHOLD/_STABLE_LENGTH/_MAX_DISTANCE.
    """
    if slope_threshold is None:
        slope_threshold = config.TUNNEL_APPROACH_SLOPE_THRESHOLD
    if stable_length is None:
        stable_length = config.TUNNEL_APPROACH_STABLE_LENGTH
    if max_distance is None:
        max_distance = config.TUNNEL_APPROACH_MAX_DISTANCE

    is_surface = lambda tags: classify_structure(tags) == "surface"

    for tunnel in [r for r in road_polygons if classify_structure(r.get("osm_tags", {})) in ("tunnel", "gallery")]:
        for at_start in (True, False):
            coords = tunnel["coords"]
            if len(coords) < 2:
                break
            portal = coords[0] if at_start else coords[-1]
            # The approach must be unambiguous - at a real crossing directly at the portal everything stays as it is.
            if _find_unique_touching_road(road_polygons, portal, tunnel["id"]) is None:
                continue
            found = _find_unique_touching_road(road_polygons, portal, tunnel["id"], predicate=is_surface)
            if not found:
                continue
            neighbor, touching_at_start = found
            ordered = list(neighbor["coords"]) if touching_at_start else list(reversed(neighbor["coords"]))

            stable = _stable_grade_start(ordered, slope_threshold, stable_length, max_distance)
            if stable is None:
                continue
            idx, cum = stable
            if idx == 0:
                continue  # approach is already stable up to the portal

            arr = np.asarray(ordered, dtype=float)
            z_stable = arr[idx, 2]
            z_window_end = float(np.interp(cum[idx] + stable_length, cum, arr[:, 2]))
            grade = (z_window_end - z_stable) / stable_length  # slope outward (away from the portal)
            for j in range(idx):
                x, y, _ = ordered[j]
                ordered[j] = (x, y, float(z_stable - grade * (cum[idx] - cum[j])))

            neighbor["coords"] = ordered if touching_at_start else list(reversed(ordered))
            new_portal = ordered[0]
            tunnel["coords"] = [new_portal] + list(coords[1:]) if at_start else list(coords[:-1]) + [new_portal]
            logger.debug(
                f"  Tunnel {tunnel['id']}: portal at the {'start' if at_start else 'end'} from {portal[2]:.2f} to "
                f"{new_portal[2]:.2f} m (approach {neighbor['id']} stable from {cum[idx]:.1f} m)"
            )

    return road_polygons


def _xy_match(a, b, tol=1e-3) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def structure_chains(road_polygons):
    """
    Chains of tunnel/gallery ways that abut end to end UNAMBIGUOUSLY (exactly one structure partner and
    no other road at the joint). Returns: list of chains, each chain [(road, reversed), ...] in travel direction.
    """
    members = [
        r for r in road_polygons
        if classify_structure(r.get("osm_tags", {})) in ("tunnel", "gallery") and len(r["coords"]) >= 2
    ]

    def touches(road, point):
        return _xy_match(road["coords"][0], point) or _xy_match(road["coords"][-1], point)

    def partner(road, point):
        found = [o for o in members if o is not road and touches(o, point)]
        others = [o for o in road_polygons if o is not road and len(o["coords"]) >= 2 and touches(o, point)]
        if len(found) != 1 or len(others) != 1:
            return None
        return found[0], _xy_match(found[0]["coords"][0], point)

    chains, seen = [], set()
    for road in members:
        if id(road) in seen:
            continue
        # walk back to the start of the chain (ring guard via the member count)
        cur, rev = road, False
        for _ in range(len(members)):
            found = partner(cur, cur["coords"][-1] if rev else cur["coords"][0])
            if found is None or found[0] is road:
                break
            cur, rev = found[0], found[1]  # predecessor ends at the joint: if its start matches, walk it backwards
        chain = []
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            chain.append((cur, rev))
            found = partner(cur, cur["coords"][0] if rev else cur["coords"][-1])
            cur, rev = (found[0], not found[1]) if found else (None, False)
        chains.append(chain)
    return chains


def outside_map(point, bounds, edge_margin) -> bool:
    """Point lies outside the map or at most edge_margin inside its edge (bounds: min_x, max_x, min_y, max_y)."""
    min_x, max_x, min_y, max_y = bounds
    x, y = point[0], point[1]
    return not (min_x + edge_margin < x < max_x - edge_margin and min_y + edge_margin < y < max_y - edge_margin)


def _chain_profile(chain, roof_offset, min_end_distance, min_spacing, window, bounds=None, edge_margin=0.0, entrances=()):
    """Elevation profile of a chain, in place: piecewise linear by arc length through the two outer ends and the
    gallery roof sample points (model height - roof_offset; at least min_end_distance from both chain ends and
    min_spacing from each other, median over +-window). Joints between two structures are never sample points.
    If exactly one end of the chain extends past the map boundary (bounds) and the other end is an entrance (lies
    on an endpoint from `entrances`, i.e. a surface road connects), the whole chain lies at entrance height
    (slope 0) - the entrance is blocked there, see tunnels/roadblock.py."""
    points, is_roof_sample, owners = [], [], []  # owners: (road, index in road["coords"], index in points)
    for k, (road, rev) in enumerate(chain):
        coords = road["coords"]
        order = range(len(coords) - 1, -1, -1) if rev else range(len(coords))
        gallery = classify_structure(road.get("osm_tags", {})) == "gallery"
        for j, idx in enumerate(order):
            if k > 0 and j == 0:  # joint: same point as the end of the predecessor
                is_roof_sample[-1] = False
                owners.append((road, idx, len(points) - 1))
                continue
            points.append(coords[idx])
            is_roof_sample.append(gallery)  # joints are excluded above at the successor, chain ends via min_end_distance
            owners.append((road, idx, len(points) - 1))
    arr = np.asarray(points, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1])))])
    total = cum[-1]
    if total < 1e-9:
        return
    roof = np.asarray(is_roof_sample)

    if bounds is not None:
        start_out, end_out = outside_map(arr[0], bounds, edge_margin), outside_map(arr[-1], bounds, edge_margin)
        inner = arr[-1] if start_out else arr[0]
        if start_out != end_out and any(_xy_match(inner, e) for e in entrances):
            flat = float(inner[2])
            for road, idx, _ in owners:
                x, y = road["coords"][idx][0], road["coords"][idx][1]
                road["coords"][idx] = (float(x), float(y), flat)
            return

    stations, heights = [0.0], [float(arr[0, 2])]
    last = -np.inf
    for i in np.nonzero(roof)[0]:
        station = cum[i]
        if station < min_end_distance or station > total - min_end_distance or station - last < min_spacing:
            continue
        near = roof & (np.abs(cum - station) <= window)
        stations.append(float(station))
        heights.append(float(np.median(arr[near, 2])) - roof_offset)
        last = station
    stations.append(float(total))
    heights.append(float(arr[-1, 2]))

    z = np.interp(cum, stations, heights)
    for road, idx, point_idx in owners:
        x, y = road["coords"][idx][0], road["coords"][idx][1]
        road["coords"][idx] = (float(x), float(y), float(z[point_idx]))


def apply_structure_elevation_profiles(
    road_polygons, roof_offset=None, min_end_distance=None, min_spacing=None, window=None, bounds=None, edge_margin=None
):
    """
    Elevation profiles for structures instead of the raw DGM height at every point (e.g.
    the 16.9 km long Gotthard road tunnel would otherwise get the height of the mountain ridge above it):

    - Bridges: linear between their end points.
    - Tunnels and galleries: per CHAIN of abutting structures (structure_chains) one profile through the two
      outer ends (where the road connects, see settle_tunnel_portals_to_approach_grade) and through gallery roof
      sample points: at a gallery the DGM shows the roof, the carriageway lies roof_offset below it.
      Sample points at least min_end_distance from the chain ends and min_spacing from each other, median over
      +-window. Joints tunnel <-> gallery never read the terrain (there lies mountain or roof).
    - If a tunnel/gallery chain extends past the map boundary with exactly one end (`bounds` = min_x, max_x, min_y,
      max_y; at most edge_margin inside the edge already counts as outside) and a surface road connects at the
      other end (entrance), the whole chain lies at entrance height.
    """
    if roof_offset is None:
        roof_offset = config.GALLERY_HEIGHT + config.GALLERY_ROOF_THICKNESS
    if min_end_distance is None:
        min_end_distance = config.GALLERY_ROOF_SAMPLE_END_DISTANCE
    if min_spacing is None:
        min_spacing = config.GALLERY_ROOF_SAMPLE_SPACING
    if window is None:
        window = config.GALLERY_ROOF_SAMPLE_WINDOW
    if edge_margin is None:
        edge_margin = config.MAP_EDGE_TUNNEL_MARGIN

    for road in road_polygons:
        if classify_structure(road.get("osm_tags", {})) == "bridge" and len(road["coords"]) >= 2:
            road["coords"] = _linear_elevation_profile(road["coords"])
    entrances = [
        point
        for road in road_polygons
        if classify_structure(road.get("osm_tags", {})) == "surface" and len(road["coords"]) >= 2
        for point in (road["coords"][0], road["coords"][-1])
    ]
    for chain in structure_chains(road_polygons):
        _chain_profile(chain, roof_offset, min_end_distance, min_spacing, window, bounds, edge_margin, entrances)
    return road_polygons


def get_road_polygons(roads, bbox, height_points, height_elevations, global_offset, tile_hash=None):
    """Extracts road polygons with their coordinates and heights (NEW PIPELINE).

    Pipeline:
    1. OSM → local XY (without Z)
    2. Resampling in the XY plane (fixed step)
    3. Elevation sampling on the densified points
    4. Optional: mild XY smoothing

    Args:
        roads: OSM road data
        bbox: (lat_min, lon_min, lat_max, lon_max) BBox
        height_points: Elevation data points (LOCAL, already normalized!)
        height_elevations: Z values (LOCAL, already normalized!)
        global_offset: (origin_x, origin_y) for coordinate transformation
        tile_hash: Optional - tile_hash for cache consistency
    """
    road_polygons = []

    # Collect all coordinates for batch processing
    all_coords = []
    road_indices = []

    for way in roads:
        if "geometry" not in way:
            continue

        pts = [[p["lat"], p["lon"]] for p in way["geometry"]]
        if len(pts) < 2:
            continue

        road_indices.append((len(all_coords), len(all_coords) + len(pts), way))
        all_coords.extend(pts)

    if not all_coords:
        return road_polygons

    # Batch UTM transformation (vectorized) - XY ONLY
    lats = np.array([c[0] for c in all_coords])
    lons = np.array([c[1] for c in all_coords])
    xs_utm, ys_utm = transformer_to_utm.transform(lons, lats)

    # Transform to local coordinates with global_offset
    ox, oy = global_offset
    xs = xs_utm - ox
    ys = ys_utm - oy

    # Create temporary road_polygons with XY (without Z)
    temp_roads_xy = []
    for start_idx, end_idx, way in road_indices:
        xy_coords = [(xs[i], ys[i]) for i in range(start_idx, end_idx)]
        osm_tags = way.get("tags", {})
        # Keep the OSM nodes (ID + local position) before resampling: only a shared node of two ways is
        # a real connection (see geometry/junctions.py::build_junction_network())
        node_ids = way.get("nodes") or []
        osm_nodes = (
            [(int(n), float(x), float(y)) for n, (x, y) in zip(node_ids, xy_coords)] if len(node_ids) == len(xy_coords) else None
        )
        temp_roads_xy.append(
            {
                "id": way["id"],
                "xy_coords": xy_coords,
                "name": osm_tags.get("name", f"road_{way['id']}"),
                "osm_tags": osm_tags,
                "osm_nodes": osm_nodes,
            }
        )

    # STEP 2: XY resampling (densify centerlines BEFORE elevation sampling)
    logger.info(f"  Resampling centerlines in the XY plane...")
    points_before_resampling = sum(len(r["xy_coords"]) for r in temp_roads_xy)

    for road in temp_roads_xy:
        osm_tags = road.get("osm_tags", {})
        road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
        target_spacing = road_width * config.SAMPLE_SPACING_FACTOR

        resampled_xy = resample_road_xy_only(road["xy_coords"], target_spacing)
        road["xy_coords"] = resampled_xy

    points_after_resampling = sum(len(r["xy_coords"]) for r in temp_roads_xy)
    logger.info(
        f"    -> {points_before_resampling} points → {points_after_resampling} points ({points_after_resampling - points_before_resampling:+d})"
    )

    # STEP 3: Batch elevation lookup on the resampled XY points
    logger.info(f"  Loading elevations for {points_after_resampling} resampled points...")

    # Collect all XY points for the batch lookup
    all_xy_flat = []
    road_xy_indices = []
    for road in temp_roads_xy:
        start = len(all_xy_flat)
        all_xy_flat.extend(road["xy_coords"])
        end = len(all_xy_flat)
        road_xy_indices.append((start, end, road))

    # Convert XY back to lat/lon for the elevation lookup
    xs_flat = np.array([xy[0] for xy in all_xy_flat])
    ys_flat = np.array([xy[1] for xy in all_xy_flat])

    # Transform to UTM and then to lat/lon
    xs_utm_flat = xs_flat + ox
    ys_utm_flat = ys_flat + oy
    lons_flat, lats_flat = transformer_to_utm.transform(xs_utm_flat, ys_utm_flat, direction="INVERSE")

    latlon_coords = [[lats_flat[i], lons_flat[i]] for i in range(len(lats_flat))]
    all_elevations = get_elevations_for_points(
        latlon_coords, bbox, height_points, height_elevations, global_offset, height_hash=tile_hash
    )

    # Create final road_polygons with XYZ
    for start_idx, end_idx, road in road_xy_indices:
        xyz_coords = [(all_xy_flat[i][0], all_xy_flat[i][1], all_elevations[i]) for i in range(start_idx, end_idx)]
        road_polygons.append(
            {
                "id": road["id"],
                "coords": xyz_coords,
                "name": road["name"],
                "osm_tags": road["osm_tags"],
                "osm_way_id": road["id"],
                "osm_nodes": road["osm_nodes"],
            }
        )

    # STEP 3a: too-short tagged bridges are extended into their neighboring road until normal
    # slope prevails there again - BEFORE the linear elevation profile, so that it works on the
    # already extended course (see extend_short_bridges_to_natural_grade).
    road_polygons = extend_short_bridges_to_natural_grade(road_polygons)

    # STEP 3a': bring tunnel portals whose OSM end point already lies in the hillside to the stable slope of
    # the approach - likewise BEFORE the linear profile, which then starts from the corrected portal height.
    road_polygons = settle_tunnel_portals_to_approach_grade(road_polygons)

    # STEP 3b: bridges/tunnels/galleries get a linear elevation profile instead of the raw DGM sampling
    # - BEFORE smoothing, so that it works on the already correct profile.
    map_bounds = (
        float(np.min(height_points[:, 0])), float(np.max(height_points[:, 0])),
        float(np.min(height_points[:, 1])), float(np.max(height_points[:, 1])),
    )
    road_polygons = apply_structure_elevation_profiles(road_polygons, bounds=map_bounds)

    # STEP 4: Optional - mild XY smoothing (Z is kept or only lightly smoothed)
    if config.ENABLE_ROAD_SMOOTHING:
        logger.info(f"  Mild XY smoothing...")
        road_polygons = smooth_roads_xy_only(road_polygons)

        # STEP 4b: at unambiguous bridge/tunnel/gallery transitions, the kink left by the independent
        # per-road smoothing is additionally smoothed away (see smooth_structure_transitions).
        road_polygons = smooth_structure_transitions(road_polygons)
    else:
        logger.info(f"  Smoothing skipped (config.ENABLE_ROAD_SMOOTHING=False)")

    return road_polygons


def smooth_roads_xy_only(road_polygons):
    """Mild XY+Z smoothing with configurable strength.

    Smooths XY and Z with a Chaikin filter.
    Iterations and weighting are controllable via config:
    - ROAD_SMOOTH_ITERATIONS: 1-3 (higher = smoother)
    - ROAD_SMOOTH_WEIGHT: 0.5-0.9 (higher = less smoothing)

    Returns:
        Modified road_polygons with smoothed coordinates
    """
    total_points = sum(len(road["coords"]) for road in road_polygons)

    # Config parameters
    iterations = max(1, config.ROAD_SMOOTH_ITERATIONS)
    weight_center = config.ROAD_SMOOTH_WEIGHT  # e.g. 0.75
    weight_neighbor = (1.0 - weight_center) / 2.0  # e.g. 0.125

    for road in road_polygons:
        coords = road["coords"]
        if len(coords) < 3:
            continue

        coords_arr = np.array(coords)
        smoothed_arr = coords_arr.copy()

        # Chaikin smoothing for XYZ
        for iteration in range(iterations):
            temp = smoothed_arr.copy()
            for i in range(1, len(smoothed_arr) - 1):
                # Smooth XYZ with configurable weight
                temp[i, 0] = (
                    weight_center * smoothed_arr[i, 0]
                    + weight_neighbor * smoothed_arr[i - 1, 0]
                    + weight_neighbor * smoothed_arr[i + 1, 0]
                )
                temp[i, 1] = (
                    weight_center * smoothed_arr[i, 1]
                    + weight_neighbor * smoothed_arr[i - 1, 1]
                    + weight_neighbor * smoothed_arr[i + 1, 1]
                )
                temp[i, 2] = (
                    weight_center * smoothed_arr[i, 2]
                    + weight_neighbor * smoothed_arr[i - 1, 2]
                    + weight_neighbor * smoothed_arr[i + 1, 2]
                )
            smoothed_arr = temp

        # Keep start/end exactly (important for junctions!)
        smoothed_arr[0] = coords_arr[0]
        smoothed_arr[-1] = coords_arr[-1]

        road["coords"] = [(p[0], p[1], p[2]) for p in smoothed_arr]

    logger.info(f"    -> {total_points} points smoothed (XY+Z, {iterations} iter., weight={weight_center:.2f})")
    return road_polygons


def _blend_structure_boundary(road_a, at_start_a, road_b, at_start_b, window, iterations, weight_center):
    """Smooths the up to `window` points on each side of the shared boundary point of road_a/road_b
    TOGETHER (one Chaikin pass over the combined window sequence), so that the boundary point stays
    identical in both roads; the far window ends serve as fixed anchors."""
    weight_neighbor = (1.0 - weight_center) / 2.0
    coords_a = road_a["coords"]
    coords_b = road_b["coords"]
    wa = min(window, len(coords_a) - 1)
    wb = min(window, len(coords_b) - 1)

    a_slice = list(reversed(coords_a[: wa + 1])) if at_start_a else list(coords_a[-(wa + 1) :])
    b_slice = list(coords_b[: wb + 1]) if at_start_b else list(reversed(coords_b[-(wb + 1) :]))

    # a_slice ends with the boundary point, b_slice starts with the boundary point - merge it once
    window_seq = a_slice[:-1] + b_slice
    n = len(window_seq)
    if n < 3:
        return

    arr = np.array(window_seq, dtype=float)
    smoothed = arr.copy()
    for _ in range(max(1, iterations)):
        temp = smoothed.copy()
        for i in range(1, n - 1):
            temp[i] = weight_center * smoothed[i] + weight_neighbor * smoothed[i - 1] + weight_neighbor * smoothed[i + 1]
        smoothed = temp
    smoothed[0] = arr[0]  # far window ends stay fixed (anchors)
    smoothed[-1] = arr[-1]

    new_a_slice = [tuple(float(v) for v in p) for p in smoothed[: wa + 1]]
    new_b_slice = [tuple(float(v) for v in p) for p in smoothed[wa:]]

    if at_start_a:
        coords_a[: wa + 1] = list(reversed(new_a_slice))
    else:
        coords_a[-(wa + 1) :] = new_a_slice

    if at_start_b:
        coords_b[: wb + 1] = new_b_slice
    else:
        coords_b[-(wb + 1) :] = list(reversed(new_b_slice))


def smooth_structure_transitions(road_polygons, window=3, iterations=None, weight_center=None):
    """Softens the kink at unambiguous bridge/tunnel/gallery transitions.

    smooth_roads_xy_only smooths each road independently and keeps its end points exactly fixed, so a
    visible kink can arise at the shared transition to a structure - among other reasons because the structure
    gets a linear instead of the natural DGM elevation profile (apply_structure_elevation_profiles). This
    step runs AFTER smooth_roads_xy_only and smooths the few points on both sides of an unambiguous
    structure transition together (XYZ), so that the boundary point stays identical in both roads (no
    gap) and the road looks "of one piece".

    Only unambiguous 2-way transitions (structure <-> one other road, whether surface or again a
    structure) are smoothed; real multi-way crossings (3+ roads at one point) stay untouched,
    since the sharp edge is needed there for the junction logic (see _find_unique_touching_road).
    """
    iterations = config.ROAD_SMOOTH_ITERATIONS if iterations is None else iterations
    weight_center = config.ROAD_SMOOTH_WEIGHT if weight_center is None else weight_center

    processed = set()
    for road in road_polygons:
        if classify_structure(road.get("osm_tags", {})) == "surface":
            continue
        coords = road["coords"]
        if len(coords) < 2:
            continue

        for at_start in (True, False):
            key = (road["id"], at_start)
            if key in processed:
                continue
            touch_point = coords[0] if at_start else coords[-1]
            found = _find_unique_touching_road(road_polygons, touch_point, road["id"])
            if not found:
                continue
            neighbor, neighbor_at_start = found

            _blend_structure_boundary(road, at_start, neighbor, neighbor_at_start, window, iterations, weight_center)
            processed.add(key)
            processed.add((neighbor["id"], neighbor_at_start))

    return road_polygons
