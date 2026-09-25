"""
Vineyard Generator: vine rows for vineyard areas (landuse=vineyard).

The vines are forest items (grape_vine: a row segment whose X axis is the
row direction; see io/vineyard_assets.py). For each vineyard polygon,
straight, parallel rows are generated. By default they run along the
fall line (slope gradient); each segment also follows the slope inclination in
row direction and stands upright.

Format per instance (BeamNG .forest4.json schema, as in ForestInstanceGenerator):
    {"type": "grape_vine", "pos": [x, y, z], "rotationMatrix": [9 values, row by row], "scale": 1.0}
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import shapely
from shapely import contains_xy
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..terrain.terrain_materials import get_landuse_category

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]

GRADIENT_SAMPLE_STEP = 4.0  # spacing of the samples for the fall line in meters
GRADIENT_DIFF_STEP = 1.0  # step size of the central difference in meters
MIN_ROW_FILL = 0.95  # shorter row pieces (in segment lengths) stay empty


def make_height_sampler(heights: np.ndarray, origin_x: float, origin_y: float, square_size: float) -> HeightAt:
    """
    Bilinear height query on the exported terrain heightmap.

    heights[i, j] belongs to the world position (origin_x + j * square_size, origin_y + i * square_size);
    outside the grid the query is clamped to the border.
    """
    rows, cols = heights.shape

    def height_at(x, y):
        fx = np.clip((np.asarray(x, dtype=float) - origin_x) / square_size, 0.0, cols - 1.001)
        fy = np.clip((np.asarray(y, dtype=float) - origin_y) / square_size, 0.0, rows - 1.001)
        j0 = fx.astype(int)
        i0 = fy.astype(int)
        tj = fx - j0
        ti = fy - i0
        return (
            heights[i0, j0] * (1 - ti) * (1 - tj)
            + heights[i0 + 1, j0] * ti * (1 - tj)
            + heights[i0, j0 + 1] * (1 - ti) * tj
            + heights[i0 + 1, j0 + 1] * ti * tj
        )

    return height_at


def _canonical(vec: np.ndarray) -> np.ndarray:
    """Axes have no sign: unique direction with ux > 0 (or uy > 0 if ux ≈ 0)."""
    vec = vec / np.linalg.norm(vec)
    if vec[0] < -1e-9 or (abs(vec[0]) <= 1e-9 and vec[1] < 0):
        vec = -vec
    return vec


def _sample_points(polygon: BaseGeometry, step: float) -> np.ndarray:
    min_x, min_y, max_x, max_y = polygon.bounds
    xs = np.arange(min_x + step / 2, max_x, step)
    ys = np.arange(min_y + step / 2, max_y, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    inside = contains_xy(polygon, grid_x, grid_y)
    return np.column_stack([grid_x[inside], grid_y[inside]])


def _gradient_samples(polygon: BaseGeometry, height_at: HeightAt):
    """(gx, gy) of the terrain gradient (m/m) at sample points in the polygon, or None."""
    points = _sample_points(polygon, GRADIENT_SAMPLE_STEP)
    if len(points) < 3:
        return None
    x, y = points[:, 0], points[:, 1]
    h = GRADIENT_DIFF_STEP
    gx = (height_at(x + h, y) - height_at(x - h, y)) / (2 * h)
    gy = (height_at(x, y + h) - height_at(x, y - h)) / (2 * h)
    return gx, gy


def _fall_line(polygon: BaseGeometry, height_at: HeightAt):
    """(fall-line axis as a unit vector, mean gradient in m/m) or None."""
    samples = _gradient_samples(polygon, height_at)
    if samples is None:
        return None
    gx, gy = samples
    # Structure tensor instead of the mean of the gradients: downslope to the left and to the right (crest)
    # cancels out in the mean but belongs to the same axis.
    tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    axis = eigenvectors[:, np.argmax(eigenvalues)]
    return _canonical(axis), float(np.mean(np.hypot(gx, gy)))


def _direction_agreement(polygon: BaseGeometry, height_at: HeightAt, min_slope_percent: float) -> float:
    """
    How uniform is the fall line in the polygon? 1.0 = same axis everywhere, 0 = wildly
    mixed (mean resultant of the doubled angles, weighted by gradient).
    On almost flat terrain the fall line is meaningless -> 1.0 (no splitting needed).
    """
    samples = _gradient_samples(polygon, height_at)
    if samples is None:
        return 1.0
    gx, gy = samples
    weight = np.hypot(gx, gy)
    if weight.mean() * 100.0 < min_slope_percent:
        return 1.0
    doubled = 2.0 * np.arctan2(gy, gx)
    resultant = np.hypot(np.sum(weight * np.cos(doubled)), np.sum(weight * np.sin(doubled)))
    return float(resultant / np.sum(weight))


def _long_axis(polygon: BaseGeometry) -> np.ndarray:
    """Long axis of the smallest enclosing rectangle."""
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        min_x, min_y, max_x, max_y = polygon.bounds
        return _canonical(np.array([1.0, 0.0]) if max_x - min_x >= max_y - min_y else np.array([0.0, 1.0]))
    coords = np.array(rectangle.exterior.coords)
    edge_a, edge_b = coords[1] - coords[0], coords[2] - coords[1]
    return _canonical(edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b)


def compute_row_direction(
    polygon: BaseGeometry, height_at: HeightAt, orientation: str = "gradient", min_slope_percent: float = 2.0
) -> np.ndarray:
    """
    Row direction (2D unit vector) for a vineyard polygon.

    orientation="gradient": rows along the fall line (slope gradient);
    orientation="contour": rows along the contour lines. On almost flat terrain
    (gradient < min_slope_percent) the fall line is unreliable - then the
    row runs along the long axis of the area.
    """
    fall = _fall_line(polygon, height_at)
    if fall is None or fall[1] * 100.0 < min_slope_percent:
        return _long_axis(polygon)
    axis = fall[0]
    if orientation == "contour":
        return _canonical(np.array([-axis[1], axis[0]]))
    return axis


def _polygon_parts(geometry: BaseGeometry) -> List[BaseGeometry]:
    return [g for g in getattr(geometry, "geoms", [geometry]) if g.geom_type == "Polygon" and g.area > 1e-6]


def _halve(polygon: BaseGeometry) -> List[BaseGeometry]:
    """Splits a polygon into two parts with a cut across the long axis in the middle."""
    a = _long_axis(polygon)
    b = np.array([-a[1], a[0]])
    coords = np.array(polygon.exterior.coords) if polygon.geom_type == "Polygon" else np.array(polygon.envelope.exterior.coords)
    t = coords @ a
    t_lo, t_hi, t_mid = float(t.min()) - 1.0, float(t.max()) + 1.0, float((t.min() + t.max()) / 2.0)
    big = 1e6
    first = Polygon([t_lo * a - big * b, t_mid * a - big * b, t_mid * a + big * b, t_lo * a + big * b])
    second = Polygon([t_mid * a - big * b, t_hi * a - big * b, t_hi * a + big * b, t_mid * a + big * b])
    return _polygon_parts(polygon.intersection(first)) + _polygon_parts(polygon.intersection(second))


def split_by_direction(
    polygon: BaseGeometry,
    height_at: HeightAt,
    max_spread_deg: float,
    min_area: float,
    min_slope_percent: float = 2.0,
    max_depth: int = 6,
) -> List[BaseGeometry]:
    """
    Splits a large polygon into blocks as long as the fall line within it varies too much.

    Straight vine rows can only follow ONE direction; on a curved slope
    this deviates from the local gradient. A block is halved if the fall line deviates on average
    by more than max_spread_deg from the main axis and both halves remain at least
    min_area in size. Each block then gets its own row direction.

    Returns:
        List of polygons that cover the input polygon without overlap.
    """
    threshold = float(np.cos(np.radians(2.0 * max_spread_deg)))
    result: List[BaseGeometry] = []
    stack = [(polygon, 0)]
    while stack:
        block, depth = stack.pop()
        if (
            depth >= max_depth
            or block.area < 2.0 * min_area
            or _direction_agreement(block, height_at, min_slope_percent) >= threshold
        ):
            result.append(block)
            continue
        halves = _halve(block)
        if len(halves) < 2:
            result.append(block)
            continue
        stack.extend((half, depth + 1) for half in halves)
    return result


def build_exclusion_geometry(shapes: Sequence[BaseGeometry], margin: float) -> Optional[BaseGeometry]:
    """Union of the areas (paths, buildings) buffered by `margin` meters, or None."""
    # Repair invalid polygons (self-intersection): the union would fail on them, the per-polygon buffering
    # has so far silently cleaned them up.
    parts = [shape if shape.is_valid else shape.buffer(0) for shape in shapes if shape is not None and not shape.is_empty]
    # Union first, then buffer once (Minkowski sum: same result, but much faster)
    return unary_union(parts).buffer(margin) if parts else None


def _line_parts(geometry: BaseGeometry) -> List[LineString]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        parts = []
        for part in geometry.geoms:
            parts.extend(_line_parts(part))
        return parts
    return []


def _rotation_matrices(forward: np.ndarray) -> np.ndarray:
    """
    Row-major 3x3 rotation matrices as an (N, 9) array for N normalized row directions
    (N, 3): ROWS = model X (row direction, follows the slope inclination), Y, Z (upright,
    perpendicular to X). BeamNG reads the axes as rows; verified on BeamNG's own
    vineyards (italy): there row 0 follows the terrain gradient to 98.5 %, column 0 is
    negatively correlated. Columns would invert the inclination (vines dive into the
    slope) and mirror the direction (across the slope).
    """
    up = np.array([0.0, 0.0, 1.0]) - forward[:, 2:3] * forward
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    side = np.cross(up, forward)
    return np.stack([forward, side, up], axis=1).reshape(-1, 9)


def _polygon_instances(polygon: BaseGeometry, height_at: HeightAt, rows: Dict, exclusion, rng) -> List[Dict]:
    orientation = rows.get("orientation", "gradient")
    spacing = float(rows["row_spacing"])
    segment = float(rows["segment_length"])
    scale_min, scale_max = rows.get("scale_range", [1.0, 1.0])

    edge_margin = float(rows.get("edge_margin", 0.0))
    area = polygon.buffer(-edge_margin) if edge_margin else (polygon if polygon.is_valid else polygon.buffer(0))
    if exclusion is not None and not area.is_empty:
        # The exclusion zone covers the whole road network (hundreds of thousands of vertices): first clip to the
        # bounding box of this block (linear, no topology operation), only then form the difference.
        local_exclusion = shapely.clip_by_rect(exclusion, *area.bounds)
        if not local_exclusion.is_empty:
            area = area.difference(local_exclusion)
    if area.is_empty:
        return []

    u = compute_row_direction(polygon, height_at, orientation, float(rows.get("min_slope_percent", 2.0)))
    v = np.array([-u[1], u[0]])

    corners = np.array(list(area.envelope.exterior.coords))
    t_all, v_all = corners @ u, corners @ v
    t_min, t_max = float(t_all.min()) - 1.0, float(t_all.max()) + 1.0
    v_min, v_max = float(v_all.min()), float(v_all.max())

    centers = []
    row_count = int(np.floor((v_max - v_min) / spacing)) + 1
    v_start = v_min + ((v_max - v_min) - (row_count - 1) * spacing) / 2.0
    for row in range(row_count):
        v_coord = v_start + row * spacing
        line = LineString([t_min * u + v_coord * v, t_max * u + v_coord * v])
        for part in _line_parts(line.intersection(area)):
            start, end = np.array(part.coords[0]), np.array(part.coords[-1])
            t0, t1 = sorted((float(start @ u), float(end @ u)))
            # The row extends exactly to the polygon edge or exclusion zone (no remainder at the ends): the segments
            # are distributed over the whole length, count = nearest integer. Rounded down (remainders > 0.5
            # segment) they move apart evenly, rounded up the end segments sit flush at the edge
            # and the inner ones overlap slightly - so no segment ever sticks out past the edge. Remainders below
            # MIN_ROW_FILL segment lengths stay empty.
            length = t1 - t0
            if length < segment * MIN_ROW_FILL:
                continue
            fill = length / segment
            count = max(1, int(np.floor(fill + 0.5)))
            if count == 1:
                t = np.array([(t0 + t1) / 2.0])
            elif fill >= count:
                t = t0 + (np.arange(count) + 0.5) * (length / count)
            else:
                t = np.linspace(t0 + segment / 2.0, t1 - segment / 2.0, count)
            centers.append(t[:, None] * u + v_coord * v)
    if not centers:
        return []

    # All segments of a block at once: heights of the segment ends, inclination, matrix.
    center = np.vstack(centers)
    end_a, end_b = center - u * segment / 2.0, center + u * segment / 2.0
    z_a, z_b = height_at(end_a[:, 0], end_a[:, 1]), height_at(end_b[:, 0], end_b[:, 1])
    forward = np.column_stack([np.full(len(center), segment * u[0]), np.full(len(center), segment * u[1]), z_b - z_a])
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    z_center = height_at(center[:, 0], center[:, 1])
    matrices = _rotation_matrices(forward).tolist()
    scales = rng.uniform(scale_min, scale_max, size=len(center))  # same random sequence as when drawn individually

    return [
        {"type": rows["item"], "pos": [x, y, z], "rotationMatrix": matrix, "scale": scale}
        for (x, y), z, matrix, scale in zip(center.tolist(), z_center.tolist(), matrices, scales.tolist())
    ]


def generate_vineyard_instances(
    polygon: BaseGeometry, height_at: HeightAt, rows: Dict, exclusion: Optional[BaseGeometry] = None
) -> List[Dict]:
    """
    Creates the vine row instances for a vineyard (multi-)polygon.

    Args:
        polygon: vineyard area in local coordinates (MultiPolygon: each part
            gets its own row direction)
        height_at: height query (see make_height_sampler())
        rows: row settings from landuse_mappings["vineyard"]["rows"]
            (item, orientation, row_spacing, segment_length, edge_margin,
            min_slope_percent, scale_range)
        exclusion: areas without vines (e.g. paths, buildings)

    Returns:
        List of forest instances; deterministic for identical inputs.
    """
    # Split MultiPolygon/GeometryCollection (e.g. after clipping with the terrain rectangle)
    # into individual polygons; line/point remnants are not areas.
    parts = _polygon_parts(polygon)
    spread = rows.get("max_direction_spread_deg")
    instances = []
    for part in parts:
        blocks = (
            split_by_direction(
                part,
                height_at,
                float(spread),
                float(rows.get("min_block_area", 2500.0)),
                float(rows.get("min_slope_percent", 2.0)),
            )
            if spread is not None
            else [part]
        )
        for block in blocks:
            centroid = block.centroid
            rng = np.random.RandomState(int(abs(centroid.x * 1000.0 + centroid.y)) % (2**31))
            instances.extend(_polygon_instances(block, height_at, rows, exclusion, rng))
    return instances


def generate_vineyards(
    landuse_polygons: Sequence[Dict],
    landuse_mappings: Dict,
    height_at: HeightAt,
    exclusion: Optional[BaseGeometry] = None,
    bounds: Optional[BaseGeometry] = None,
) -> List[Dict]:
    """
    Creates vine rows for all vineyard polygons (category with "rows" settings).

    Args:
        landuse_polygons: result of osm.landuse_polygons.build_landuse_polygons()
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]
        height_at: height query
        exclusion: areas without vines
        bounds: extent of the terrain. The OSM query reaches beyond it, and
            outside there is no elevation data (the heightmap clamps at the border) - there
            vines would float in the air.
    """
    instances = []
    for polygon in landuse_polygons:
        category = get_landuse_category(polygon["osm_tags"], landuse_mappings)
        rows = landuse_mappings.get(category, {}).get("rows") if category else None
        if not rows:
            continue
        geometry = polygon["geometry"]
        if bounds is not None:
            geometry = geometry.intersection(bounds)
            if geometry.is_empty:
                continue
        instances.extend(generate_vineyard_instances(geometry, height_at, rows, exclusion))
    return instances
