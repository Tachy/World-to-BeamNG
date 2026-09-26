"""
Sets the terrain heightmap array exactly to the road centerline
height along roads.

Since the switch to BeamNG `DecalRoad` - roads are no longer exported as their own mesh,
but projected as a decal directly onto the terrain surface at runtime -
there is no second, separately encoded road surface anymore
that would have to be "hit" - the terrain IS the visible road. Therefore
no safety margin/gradient compensation is needed anymore (unlike the
earlier mesh embedding): the target height per raster cell is exactly the
centerline height at the nearest position.

Core idea: for each road, every raster cell is checked against the (already existing)
2D road polygon to see whether it lies inside (point-in-polygon), and if so
the height is determined by projecting the cell center onto the nearest position
along the centerline (linear interpolation between the two nearest centerline
points) and set directly.
"""

import math
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree
from shapely import intersects_xy
from shapely.geometry import Polygon

from .. import config


def embed_roads_into_heightmap(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    road_slope_polygons_2d: List[Dict],
    clamp_to_max: bool = False,
) -> np.ndarray:
    """
    Sets heights exactly to the road centerline height wherever a road
    lies. Does NOT modify heights in place, returns a new copy.

    Args:
        heights: (size, size) float array
        origin_x, origin_y: world coordinates of cell [*, 0] and [0, *] respectively
        square_size: meters per raster cell
        road_slope_polygons_2d: list of dicts with "road_polygon" ((M,2)
            2D road outline, already buffered by half the road width) and
            "trimmed_centerline" ((N,3) x,y,z points) - the same structure as
            for build_road_embankment_profiles()
        clamp_to_max: False (default) unconditionally sets the height to centerline level (normal roads/
            galleries). True only lowers cells that lie HIGHER than the centerline height (np.minimum),
            leaves lower cells unchanged - for bridge abutments: the terrain across the carriageway is
            not necessarily flat there and could otherwise poke through the (flat) deck in places, but the
            valley floor that the bridge spans must stay visible unchanged (see
            TerrainWorkflow.process_tile()).

    Returns:
        New (size, size) float array
    """
    result = heights.copy()
    size_y, size_x = heights.shape

    for road in road_slope_polygons_2d:
        _embed_road(result, origin_x, origin_y, square_size, road, size_x, size_y, clamp_to_max=clamp_to_max)

    return result


def near_deck_mask(
    heights: np.ndarray, origin_x: float, origin_y: float, square_size: float, bridges: List[Dict], clearance: float
) -> np.ndarray:
    """
    (size_y, size_x) bool raster: cells inside a bridge footprint ("road_polygon") whose terrain lies less than
    `clearance` below the deck (centerline height projected like in _embed_road()). There ground cover would grow
    through the deck of a hillside bridge; under a bridge spanning a valley the terrain lies deeper and keeps its grass.
    """
    mask = np.zeros(heights.shape, dtype=bool)
    size_y, size_x = heights.shape
    for bridge in bridges:
        polygon = np.asarray(bridge["road_polygon"], dtype=np.float64)
        centerline = np.asarray(bridge["trimmed_centerline"], dtype=np.float64)
        if len(polygon) < 3 or len(centerline) < 2:
            continue
        col0 = max(0, int(np.floor((polygon[:, 0].min() - origin_x) / square_size)))
        col1 = min(size_x - 1, int(np.ceil((polygon[:, 0].max() - origin_x) / square_size)))
        row0 = max(0, int(np.floor((polygon[:, 1].min() - origin_y) / square_size)))
        row1 = min(size_y - 1, int(np.ceil((polygon[:, 1].max() - origin_y) / square_size)))
        if col0 > col1 or row0 > row1:
            continue
        grid_x, grid_y = np.meshgrid(origin_x + np.arange(col0, col1 + 1) * square_size, origin_y + np.arange(row0, row1 + 1) * square_size)
        inside = _cells_in_polygon(grid_x, grid_y, polygon)
        if not np.any(inside):
            continue
        deck_z = _project_onto_polyline(grid_x[inside], grid_y[inside], centerline[:, 0], centerline[:, 1], centerline[:, 2])
        sub = mask[row0 : row1 + 1, col0 : col1 + 1]
        near = heights[row0 : row1 + 1, col0 : col1 + 1][inside] > deck_z - clearance
        sub[inside] |= near
    return mask


def _points_in_polygon_2d(qx: np.ndarray, qy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized point-in-polygon test (ray casting/crossing number)."""
    polygon = np.asarray(polygon, dtype=np.float64)
    px = polygon[:, 0]
    py = polygon[:, 1]
    n = len(polygon)
    inside = np.zeros(qx.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = (xj - xi) * (qy - yi) / (yj - yi) + xi
        crosses = (yi > qy) != (yj > qy)
        inside ^= crosses & (qx < x_intersect)
        j = i
    return inside


def _cells_in_polygon(qx: np.ndarray, qy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Point-in-polygon for a cell grid. Fast via Shapely (C, prepared polygon);
    only for invalid polygons (e.g. self-intersecting centerline fallback) does
    the ray-casting test apply, whose even-odd rule yields the previous behavior there.
    """
    shape = Polygon(polygon)
    if not shape.is_valid:
        return _points_in_polygon_2d(qx, qy, polygon)
    return intersects_xy(shape, qx, qy)


PROJECT_CHUNK = 96  # cells per block in _project_onto_polyline


def _project_onto_polyline(
    qx: np.ndarray, qy: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray, poly_z: np.ndarray
) -> np.ndarray:
    """
    Projects query points (qx, qy, any identical shape) onto the
    nearest position along the polyline defined by (poly_x, poly_y, poly_z)
    and returns the linearly interpolated Z height there
    (same shape as qx/qy).

    Gives the same result as the simple loop over all segments (on a tie the first segment wins), but
    only with the segments that are at all candidates for a block of spatially adjacent cells: the
    nearest segment endpoint lies on the polyline and thereby bounds the distance to the nearest segment
    from above; segments whose bounding box is farther than this bound from the block cannot win.
    """
    shape = np.shape(qx)
    points = np.column_stack([np.ravel(qx), np.ravel(qy)]).astype(np.float64)
    count = len(points)
    best_z = np.zeros(count)
    if count == 0 or len(poly_x) < 2:
        return best_z.reshape(shape)

    ax, ay, az = poly_x[:-1], poly_y[:-1], poly_z[:-1]
    bx, by, bz = poly_x[1:], poly_y[1:], poly_z[1:]
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    valid = np.flatnonzero(~(seg_len_sq < 1e-9))  # zero-length segments are dropped
    if len(valid) == 0:
        return best_z.reshape(shape)
    ax, ay, az, bx, by, bz, dx, dy, seg_len_sq = (a[valid] for a in (ax, ay, az, bx, by, bz, dx, dy, seg_len_sq))
    segments = len(valid)

    seg_min_x, seg_max_x = np.minimum(ax, bx), np.maximum(ax, bx)
    seg_min_y, seg_max_y = np.minimum(ay, by), np.maximum(ay, by)

    endpoints = np.concatenate([np.column_stack([ax, ay]), np.column_stack([bx, by])])
    endpoint_dist, nearest_endpoint = cKDTree(endpoints).query(points)
    # Sort by position along the line: consecutive cells form compact blocks
    order = np.argsort(nearest_endpoint % segments, kind="stable")

    for start in range(0, count, PROJECT_CHUNK):
        cells = order[start : start + PROJECT_CHUNK]
        cx, cy = points[cells, 0], points[cells, 1]
        bound = endpoint_dist[cells].max() + 1e-9
        candidates = np.flatnonzero(
            (seg_max_x >= cx.min() - bound)
            & (seg_min_x <= cx.max() + bound)
            & (seg_max_y >= cy.min() - bound)
            & (seg_min_y <= cy.max() + bound)
        )
        cax, cay, cdx, cdy, clen = ax[candidates], ay[candidates], dx[candidates], dy[candidates], seg_len_sq[candidates]
        col_x, col_y = cx[:, None], cy[:, None]
        t = np.clip(((col_x - cax) * cdx + (col_y - cay) * cdy) / clen, 0.0, 1.0)
        dist = np.hypot(col_x - (cax + t * cdx), col_y - (cay + t * cdy))
        nearest = np.argmin(dist, axis=1)  # first minimum, like the loop with "dist < best"
        t_best = t[np.arange(len(cells)), nearest]
        seg = candidates[nearest]
        best_z[cells] = az[seg] + t_best * (bz[seg] - az[seg])

    return best_z.reshape(shape)


def _embed_road(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    road: Dict,
    size_x: int,
    size_y: int,
    clamp_to_max: bool = False,
) -> None:
    """Sets all raster cells inside the road polygon to centerline height (or only caps them from
    above, see embed_roads_into_heightmap()'s clamp_to_max)."""
    polygon = np.asarray(road["road_polygon"], dtype=np.float64)
    centerline = np.asarray(road["trimmed_centerline"], dtype=np.float64)
    if len(polygon) < 3 or len(centerline) < 2:
        return

    min_x, min_y = polygon[:, 0].min(), polygon[:, 1].min()
    max_x, max_y = polygon[:, 0].max(), polygon[:, 1].max()

    col_start = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row_start = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))

    if col_start > col_end or row_start > row_end:
        return

    cols = np.arange(col_start, col_end + 1)
    rows = np.arange(row_start, row_end + 1)
    cell_x = origin_x + cols * square_size
    cell_y = origin_y + rows * square_size
    grid_x, grid_y = np.meshgrid(cell_x, cell_y)  # shape (len(rows), len(cols))

    inside = _cells_in_polygon(grid_x, grid_y, polygon)
    if not np.any(inside):
        return

    # Only project the cells inside the polygon: for long, diagonal roads the
    # bounding box is huge, but the actual strip is narrow (factor 100+ fewer cells).
    target_z = _project_onto_polyline(
        grid_x[inside], grid_y[inside], centerline[:, 0], centerline[:, 1], centerline[:, 2]
    )

    sub = heights[row_start : row_end + 1, col_start : col_end + 1]
    sub[inside] = np.minimum(sub[inside], target_z) if clamp_to_max else target_z


def sample_heightmap_bilinear(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    points_xy: np.ndarray,
) -> np.ndarray:
    """
    Reads the heightmap at arbitrary (not grid-aligned) XY points
    via bilinear interpolation - for road edge points that do not lie exactly
    on a raster cell.

    Args:
        heights: (size, size) float array
        origin_x, origin_y, square_size: as in heightmap.build_heightmap()
        points_xy: (N, 2) array of world coordinates (x, y)

    Returns:
        (N,) array of interpolated height values
    """
    size_y, size_x = heights.shape
    col_f = (points_xy[:, 0] - origin_x) / square_size
    row_f = (points_xy[:, 1] - origin_y) / square_size

    # Clip to the valid coordinate domain [0, size-1] (exact, no epsilon
    # fudge - an epsilon here would distort fc/fr and corrupt exact-grid-point
    # results). Out-of-bounds access is instead avoided below by clamping
    # col0/row0 so that col1/row1 stay in range.
    col_f = np.clip(col_f, 0.0, size_x - 1)
    row_f = np.clip(row_f, 0.0, size_y - 1)

    col0 = np.clip(np.floor(col_f).astype(int), 0, max(size_x - 2, 0))
    row0 = np.clip(np.floor(row_f).astype(int), 0, max(size_y - 2, 0))
    col1 = col0 + 1
    row1 = row0 + 1

    fc = col_f - col0
    fr = row_f - row0

    h00 = heights[row0, col0]
    h01 = heights[row0, col1]
    h10 = heights[row1, col0]
    h11 = heights[row1, col1]

    h0 = h00 * (1 - fc) + h01 * fc
    h1 = h10 * (1 - fc) + h11 * fc
    return h0 * (1 - fr) + h1 * fr


def build_road_embankment_profiles(
    road_slope_polygons_2d: list,
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    osm_mapper,
    slope_angle_deg: float,
    min_slope_width: float,
    max_slope_width: float = config.MAX_SLOPE_WIDTH,
) -> list:
    """
    Builds the per-road edge/embankment profile data for apply_embankment_blend().

    For each road, along the centerline (already densely sampled,
    approx. 1 m point spacing), the edge point at half the road width is computed
    on the left and right, the natural terrain height there is sampled from the still
    unmodified heightmap, and an embankment width is derived from it via
    config.SLOPE_ANGLE (width = height difference / tan(angle),
    at least min_slope_width, capped at max_slope_width).

    Args:
        road_slope_polygons_2d: list of dicts with "trimmed_centerline"
            ((N,3) coordinates x,y,z) and "osm_tags" (dict) - the same
            structure that TerrainWorkflow.process_tile() already uses for
            material mapping
        heights: (size, size) float array WITH the still unmodified,
            natural terrain height (call before embed_roads_into_heightmap and before
            apply_embankment_blend)
        origin_x, origin_y, square_size: as in heightmap.build_heightmap()
        osm_mapper: OSMMapper instance (for get_road_properties()["width"])
        slope_angle_deg: config.SLOPE_ANGLE
        min_slope_width: config.MIN_SLOPE_WIDTH
        max_slope_width: upper limit of the embankment width (meters)

    Optional field per road dict: "embankment_cuts" - [(point_xy, outward_normal_xy), ...]: the embankment ends flush
    at these lines (cells on the outward side are not touched). Without a cut the corridor of the first/last edge
    point reaches around the road end like a round cap - at a gallery end that cap overwrote the approach road's
    embankment with the gallery's valley-side slope (pit beside the road, see terrain_workflow._gallery_embankment_cuts()).

    Optional field per road dict: "slope_width_override" (dict, keys "left"/"right", value = fixed
    embankment width in meters, as a number or as an array per centerline point) - replaces the computed
    embankment width on the respective side with a fixed value instead of deriving it from the height
    difference to the natural terrain (0.0 = no embankment at all, the terrain stays at natural height
    there). For a value > 0 there are two modes for this side, controlled via the optional field
    "flat_shoulder_sides" (set/list with "left"/"right", STANDARD convention as below):
    - NOT in flat_shoulder_sides (default): natural_z is NOT sampled at the carriageway edge,
      but at the FAR end of the overridden corridor (edge + override width) - real downward/
      upward interpolation to the terrain there. See gallery valley side below for the reason.
    - IN flat_shoulder_sides: natural_z is set to the edge height itself - the whole corridor stays
      FLAT at carriageway height (no interpolation to the terrain). See gallery mountain side below.

    "left"/"right" follow the STANDARD convention (as in offset_points()/resolve_open_side(): left =
    centerline direction rotated by +90°) - NOT the (purely internal, see note below)
    left/right assignment of this function; the translation happens internally.

    For galleries:
    - Mountain side (flat_shoulder_sides): config.GALLERY_MOUNTAIN_EMBED_MARGIN as a narrow FLAT border at
      carriageway height directly at the inner edge of the (massive, see config.GALLERY_WALL_THICKNESS) wall - for
      a clean wall-to-ground transition, not an artificial embankment angle. Beyond that, the
      terrain stays unchanged (the wall reaches into the hillside anyway).
    - Valley side (NOT in flat_shoulder_sides): config.GALLERY_VALLEY_SLOPE_WIDTH as a short FIXED value
      (instead of the computed width) - at a gallery the DGM does not show the original terrain,
      but the real valley-side structure (parapet/roof overhang) - directly at the carriageway edge it
      therefore does NOT show the transition into the valley, but the structure surface itself (height jump
      of several meters typically already 2 m behind the edge, measured empirically at real galleries in the DGM).
      If natural_z were still sampled at the edge, the embankment would "smooth" to this raised value that lies
      barely below road level - visible as a remaining terrain spike instead of a downhill
      gradient toward the valley. Therefore natural_z is sampled here at the far end of the corridor
      (edge + GALLERY_VALLEY_SLOPE_WIDTH), where the DGM shows real terrain again.
      See tunnels/gallery_mesh.py::resolve_open_side().

    Returns:
        List of dicts, one per road:
            {
                "left_edge_xyz": (N,3) float array (x, y, road_z),
                "right_edge_xyz": (N,3) float array (x, y, road_z),
                "left_slope_width": (N,) float array,
                "right_slope_width": (N,) float array,
                "left_natural_z": (N,) float array,
                "right_natural_z": (N,) float array,
            }
    """
    tan_angle = math.tan(math.radians(slope_angle_deg))
    roads = []

    for poly in road_slope_polygons_2d:
        centerline = np.asarray(poly.get("trimmed_centerline", []), dtype=float)
        if len(centerline) < 2:
            continue

        width = osm_mapper.get_road_properties(poly.get("osm_tags", {}))["width"]
        half_width = width / 2.0

        xy = centerline[:, :2]
        z = centerline[:, 2]

        directions = np.diff(xy, axis=0)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        directions = directions / norms

        point_dirs = np.empty_like(xy)
        point_dirs[0] = directions[0]
        point_dirs[-1] = directions[-1]
        for i in range(1, len(xy) - 1):
            avg = directions[i - 1] + directions[i]
            n = np.linalg.norm(avg)
            point_dirs[i] = avg / n if n > 1e-9 else directions[i - 1]

        perp = np.column_stack([-point_dirs[:, 1], point_dirs[:, 0]])

        # NOTE: "left"/"right" is a pure naming convention here without
        # geometric meaning (perp points to the left in driving direction,
        # but the assignment + / - is arbitrary) - apply_embankment_blend
        # treats both sides symmetrically, so the choice is uncritical.
        left_xy = xy - perp * half_width
        right_xy = xy + perp * half_width

        left_natural_z = sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, left_xy)
        right_natural_z = sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, right_xy)

        left_diff = np.abs(left_natural_z - z)
        right_diff = np.abs(right_natural_z - z)

        left_slope_width = np.clip(np.maximum(min_slope_width, left_diff / tan_angle), None, max_slope_width)
        right_slope_width = np.clip(np.maximum(min_slope_width, right_diff / tan_angle), None, max_slope_width)

        # STANDARD "left" (point + perp*half) is "right_xy" above, STANDARD "right" is "left_xy" (see
        # note) - the slope_width_override assignment therefore has to be mirrored.
        override = poly.get("slope_width_override") or {}
        flat_sides = poly.get("flat_shoulder_sides") or ()
        def overridden(key, sign, slope_width, natural_z):
            """Embankment width/natural_z of an overridden side; key in STANDARD convention, sign = direction."""
            widths = np.broadcast_to(np.asarray(override[key], dtype=float), slope_width.shape).copy()
            if not np.any(widths > 0):
                return widths, natural_z
            if key in flat_sides:
                # Flat border at carriageway height (gallery mountain side): natural_z = edge height itself,
                # no interpolation to the terrain (see docstring).
                return widths, z.copy()
            # Sample natural_z NOT at the carriageway edge (there the DGM shows the structure itself at a gallery,
            # see docstring), but at the FAR end of the overridden corridor - only there does
            # the DGM show real terrain again. Width 0 thus samples at the edge by itself.
            far = xy + sign * perp * (half_width + widths)[:, None]
            return widths, sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, far)

        if "left" in override:
            right_slope_width, right_natural_z = overridden("left", 1.0, right_slope_width, right_natural_z)
        if "right" in override:
            left_slope_width, left_natural_z = overridden("right", -1.0, left_slope_width, left_natural_z)

        if poly.get("daylight_slopes"):
            # Road under a bridge: the terrain model shows a wall beside it (the deck level), often starting a few meters
            # from the edge - the height AT the edge says nothing about it. Both sides: slope up to where the slope line meets
            # the terrain.
            left_slope_width, left_natural_z = _daylight_slope(
                heights, origin_x, origin_y, square_size, left_xy, -perp, z, tan_angle, min_slope_width, max_slope_width
            )
            right_slope_width, right_natural_z = _daylight_slope(
                heights, origin_x, origin_y, square_size, right_xy, perp, z, tan_angle, min_slope_width, max_slope_width
            )

        outward = {}
        if poly.get("daylight_slopes"):
            outward = {"left_outward": -perp, "right_outward": perp}

        roads.append(
            {
                **outward,
                "left_edge_xyz": np.column_stack([left_xy, z]),
                "right_edge_xyz": np.column_stack([right_xy, z]),
                "left_slope_width": left_slope_width,
                "right_slope_width": right_slope_width,
                "left_natural_z": left_natural_z,
                "right_natural_z": right_natural_z,
                "cuts": list(poly.get("embankment_cuts") or ()),
            }
        )

    return roads


def _daylight_slope(heights, origin_x, origin_y, square_size, edge_xy, outward, edge_z, tan_angle, min_width, max_width):
    """
    (slope width, natural height at the far end) per edge point for a slope of the given angle that runs from the road edge
    outward until it meets the terrain ("daylighting"): the largest distance d where the terrain is still farther from the
    road height than the slope line (|terrain(d) - edge_z| > d * tan) - beyond it the slope line is inside/above the terrain
    again. Without such a distance the minimum width applies. The natural height sampled at the returned width makes the
    blend of apply_embankment_blend() exactly the slope angle.
    """
    distances = np.arange(square_size, max_width + 1e-9, square_size)
    width = np.full(len(edge_xy), float(min_width))
    for distance in distances:
        terrain = sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, edge_xy + outward * distance)
        exceeds = np.abs(terrain - edge_z) > distance * tan_angle + 0.05
        width = np.where(exceeds, np.maximum(width, distance + square_size), width)
    width = np.clip(width, min_width, max_width)
    far = edge_xy + outward * width[:, None]
    return width, sample_heightmap_bilinear(heights, origin_x, origin_y, square_size, far)


def apply_embankment_blend(heights: np.ndarray, origin_x: float, origin_y: float, square_size: float, roads: list) -> np.ndarray:
    """
    Blends the terrain between road edge and natural surroundings
    (embankment) directly in the heightmap raster - replaces the never completed
    embankment mesh geometry.

    For each raster cell in the embankment corridor (between road edge and
    edge + embankment width), it interpolates linearly between the road edge height (at
    the edge) and the original natural terrain height (at the
    corridor border). Works equally for fill (road higher) and
    cut (road lower), because it simply interpolates toward the
    "natural height", without a sign assumption.

    Known limitation: overlapping corridors of several roads (e.g. at
    junctions) are not handled specially - the last processed
    road wins. Documented in the spec as an accepted simplification.

    Args:
        heights: (size, size) float array WITH the still unmodified,
            natural terrain height (call before embed_roads_into_heightmap
            - this function must run BEFORE the road embedding,
            so that "natural height" is really natural)
        origin_x, origin_y, square_size: as in heightmap.build_heightmap()
        roads: return value of build_road_embankment_profiles()

    Returns:
        New (size, size) float array (copy, original unchanged)
    """
    result = heights.copy()
    size_y, size_x = heights.shape
    nearest = np.full(heights.shape, np.inf)  # distance to the nearest edge that wrote a cell (sides with `outward`)

    for road in roads:
        _blend_one_side(result, origin_x, origin_y, square_size, size_x, size_y,
                         road["left_edge_xyz"], road["left_slope_width"], road["left_natural_z"], road.get("cuts"),
                         road.get("left_outward"), nearest)
        _blend_one_side(result, origin_x, origin_y, square_size, size_x, size_y,
                         road["right_edge_xyz"], road["right_slope_width"], road["right_natural_z"], road.get("cuts"),
                         road.get("right_outward"), nearest)

    return result


BLEND_BLOCK = 8  # Edge length of the cell blocks _blend_one_side pre-checks for road proximity (embankments are
# only 2-8.5 m wide: 32-cell blocks queried 7 million cells, 8-cell blocks only 3 million - smaller ones gain nothing)


def _blend_one_side(heights, origin_x, origin_y, square_size, size_x, size_y, edge_xyz, slope_width, natural_z, cuts=None, outward=None, nearest=None):
    """Blends one road side (left or right) in place into heights. `cuts`: [(point_xy, outward_normal_xy), ...] - cells
    on the outward side of a cut line stay untouched (see build_road_embankment_profiles()). `outward` (per edge point,
    unit vector away from the road): only cells on that side of the nearest edge point are blended - the corridor of a
    narrow road's one side otherwise reaches across the road and overwrites the other side's terrain. With `nearest`
    (per cell: distance of the edge that wrote it last) such a side only writes cells that are nearer to its edge than to
    the edge that wrote them before - wide slopes of neighbouring pieces meeting at a bridge otherwise overwrite each
    other in processing order."""
    if len(edge_xyz) == 0:
        return

    max_width = float(np.max(slope_width)) if len(slope_width) else 0.0
    if max_width <= 0:
        return

    min_x = float(np.min(edge_xyz[:, 0])) - max_width
    max_x = float(np.max(edge_xyz[:, 0])) + max_width
    min_y = float(np.min(edge_xyz[:, 1])) - max_width
    max_y = float(np.max(edge_xyz[:, 1])) + max_width

    col_start = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row_start = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))

    if col_start > col_end or row_start > row_end:
        return

    tree = cKDTree(edge_xyz[:, :2])

    # The corridor ends at max_width at the latest. For long, diagonal roads, however, the bounding box is huge
    # and almost empty: instead of querying all its cells, only blocks of BLEND_BLOCK x BLEND_BLOCK cells are
    # considered whose center is close enough to an edge point (every cell of the block is at most
    # half the block diagonal away from the center - blocks farther away are guaranteed to contain no
    # corridor cell). The result per cell is unchanged.
    limit = max_width + 1e-9
    n_rows = row_end - row_start + 1
    n_cols = col_end - col_start + 1
    block_r0 = np.arange(0, n_rows, BLEND_BLOCK)
    block_c0 = np.arange(0, n_cols, BLEND_BLOCK)
    r0, c0 = np.meshgrid(block_r0, block_c0, indexing="ij")
    r0, c0 = r0.ravel(), c0.ravel()
    r1 = np.minimum(r0 + BLEND_BLOCK, n_rows) - 1
    c1 = np.minimum(c0 + BLEND_BLOCK, n_cols) - 1
    center_x = origin_x + (col_start + (c0 + c1) / 2.0) * square_size
    center_y = origin_y + (row_start + (r0 + r1) / 2.0) * square_size
    half_diagonal = 0.5 * np.hypot(c1 - c0, r1 - r0) * square_size
    block_dist, _ = tree.query(np.column_stack([center_x, center_y]), distance_upper_bound=limit + half_diagonal.max() + 1e-6)
    keep = np.isfinite(block_dist) & (block_dist <= limit + half_diagonal + 1e-6)
    if not keep.any():
        return

    # Cells of all kept blocks at once: full blocks via broadcasting, edge blocks (shorter at the end of the bounding
    # box) trimmed to n_rows/n_cols via the mask - the same cells as block by block
    offset_r, offset_c = np.meshgrid(np.arange(BLEND_BLOCK), np.arange(BLEND_BLOCK), indexing="ij")
    rows = (r0[keep][:, None] + offset_r.ravel()[None, :]).ravel()
    cols = (c0[keep][:, None] + offset_c.ravel()[None, :]).ravel()
    inside = (rows < n_rows) & (cols < n_cols)
    rows = rows[inside] + row_start
    cols = cols[inside] + col_start
    query_points = np.column_stack([origin_x + cols * square_size, origin_y + rows * square_size])

    dist, idx = tree.query(query_points, distance_upper_bound=limit)
    near = np.isfinite(dist)
    dist, idx = dist[near], idx[near]

    nearest_edge_z = edge_xyz[idx, 2]
    nearest_slope_width = slope_width[idx]
    nearest_natural_z = natural_z[idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(nearest_slope_width > 0, dist / nearest_slope_width, 1.0)
    t = np.clip(t, 0.0, 1.0)

    blended = nearest_edge_z + (nearest_natural_z - nearest_edge_z) * t

    in_corridor = (dist > 0) & (dist <= nearest_slope_width)
    if outward is not None:
        away = query_points[near] - edge_xyz[idx, :2]
        # half a cell of tolerance: the cells right at the edge keep the edge height instead of a leftover wall step
        in_corridor &= np.einsum("ij,ij->i", away, np.asarray(outward)[idx]) > -0.5 * square_size
    for point, normal in cuts or ():
        cell_xy = query_points[near]
        in_corridor &= (cell_xy[:, 0] - point[0]) * normal[0] + (cell_xy[:, 1] - point[1]) * normal[1] <= 0.0

    target_rows = rows[near][in_corridor]
    target_cols = cols[near][in_corridor]
    values = blended[in_corridor]
    if outward is not None and nearest is not None:
        target_dist = dist[in_corridor]
        closer = target_dist < nearest[target_rows, target_cols]
        target_rows, target_cols, values = target_rows[closer], target_cols[closer], values[closer]
        nearest[target_rows, target_cols] = target_dist[closer]
    heights[target_rows, target_cols] = values
