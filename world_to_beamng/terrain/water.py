"""
Real water for streams and water bodies.

In BeamNG the water surface, waves, underwater fog and buoyancy are separate objects - the aerial photo
alone is not enough:

- Streams become `River` splines: nodes [x, y, z, width, depth, nx, ny, nz] along the OSM line.
- Ponds/lakes become `WaterBlock` boxes (surface = position.z, depth downward). A block is
  always a rectangle, so the polygon is tiled with small blocks that cover it including a margin.
  The water level is the mean of the three lowest edge points; where the terrain is higher, the water is
  hidden, where it is lower (the whole hole), it is visible. Streams end at the bank (cut_line_by_area).

The DGM1 stays unchanged except for the pond hollows (carve_pond_basins: inside the OSM polygon 50 cm
lower, embankment 45 degrees inward); the water heights are derived from the terrain. Streams lie
just above the channel floor (water only visible in the channel, where the terrain behind it is higher);
the water only falls downstream and disappears under fills/culverts in the terrain.
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]
ToLocal = Callable[[Sequence[Dict]], List[Tuple[float, float]]]
Bounds = Tuple[float, float, float, float]

UNDERGROUND_TAGS = ("tunnel", "culvert")
NORMAL_UP = [0.0, 0.0, 1.0]


def _parse_width(value: Optional[str]) -> Optional[float]:
    """OSM `width` like "4", "3.5 m" or "2,5" -> meters or None."""
    if not value:
        return None
    match = re.match(r"\s*(\d+(?:[.,]\d+)?)", str(value))
    return float(match.group(1).replace(",", ".")) if match else None


def select_waterways(osm_data: Sequence[Dict], to_local: ToLocal, widths: Dict[str, float]) -> List[Dict]:
    """
    Visible streams/rivers from OSM ways: only kinds from `widths` (default width per kind), no
    underground sections (tunnel/culvert). The width comes from the `width` tag, otherwise the default.

    Returns:
        [{"waterway", "width", "coords": [(x, y), ...]}] in local coordinates, direction as in OSM
    """
    result = []
    for element in osm_data:
        tags = element.get("tags") or {}
        kind = tags.get("waterway")
        if element.get("type") != "way" or kind not in widths:
            continue
        if any(tags.get(key) not in (None, "no") for key in UNDERGROUND_TAGS):
            continue
        coords = to_local(element.get("geometry") or [])
        if len(coords) < 2:
            continue
        result.append({"waterway": kind, "width": _parse_width(tags.get("width")) or widths[kind], "coords": coords})
    return result


def clip_line_to_bounds(coords: Sequence[Tuple[float, float]], bounds: Bounds) -> List[List[Tuple[float, float]]]:
    """Clips a line to the terrain; if it leaves the terrain and returns, several parts result."""
    clipped = LineString(coords).intersection(box(*bounds))
    if clipped.is_empty:
        return []
    parts = list(clipped.geoms) if isinstance(clipped, MultiLineString) else [clipped]
    return [list(part.coords) for part in parts if isinstance(part, LineString) and part.length > 0]


def cut_line_by_area(
    coords: Sequence[Tuple[float, float]], area: Optional[BaseGeometry], min_length: float = 1.0
) -> List[List[Tuple[float, float]]]:
    """
    Cuts away the parts of a line that lie in `area` (e.g. pond area): a stream ends at the bank.
    If it runs through the pond, two pieces result; leftover pieces shorter than `min_length` meters are dropped.
    """
    line = LineString(coords)
    if area is None or area.is_empty:
        return [list(coords)]
    rest = line.difference(area)
    if rest.is_empty:
        return []
    parts = list(rest.geoms) if hasattr(rest, "geoms") else [rest]
    return [list(part.coords) for part in parts if isinstance(part, LineString) and part.length >= min_length]


def _resample(coords: Sequence[Tuple[float, float]], spacing: float) -> np.ndarray:
    line = LineString(coords)
    count = max(2, int(round(line.length / spacing)) + 1)
    distances = np.linspace(0.0, line.length, count)
    return np.array([line.interpolate(d).coords[0] for d in distances])


def _perpendicular(points: np.ndarray) -> np.ndarray:
    tangents = np.gradient(points, axis=0)
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    norms[norms == 0] = 1.0
    return np.column_stack([-tangents[:, 1], tangents[:, 0]]) / norms[:, None]


def _snap_to_channel(points: np.ndarray, height_at: HeightAt, search: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pulls the line sideways (±search) onto the lowest point across the flow direction - the channel in the DGM1.
    The OSM line is often 1-2 m off; there the water surface would lie hidden under the bank. The offsets are
    smoothed (moving median, then mean) so the stream does not jump from edge to edge.

    Returns:
        (shifted points, channel floor height at these points)
    """
    perp = _perpendicular(points)
    offsets = np.linspace(-search, search, 9)
    samples = np.stack([np.asarray(height_at(*(points + perp * o).T), dtype=float) for o in offsets])
    best = offsets[np.argmin(samples, axis=0)]

    def smooth(values: np.ndarray, window: int, reducer) -> np.ndarray:
        half = window // 2
        padded = np.pad(values, half, mode="edge")
        return np.array([reducer(padded[i : i + window]) for i in range(len(values))])

    best = smooth(smooth(best, 5, np.median), 3, np.mean)
    snapped = points + perp * best[:, None]
    # Floor at the shifted position: minimum over a small neighborhood across (catches residual offset)
    fine = np.linspace(-0.75, 0.75, 4)
    bottom = np.stack([np.asarray(height_at(*(snapped + perp * o).T), dtype=float) for o in fine]).min(axis=0)
    return snapped, bottom


def build_river_nodes(
    coords: Sequence[Tuple[float, float]],
    height_at: HeightAt,
    width: float,
    depth: float,
    spacing: float = 10.0,
    lift: float = 0.2,
    search: float = 2.0,
) -> List[List[float]]:
    """
    River nodes [x, y, z, width, depth, 0, 0, 1] downstream.

    The nodes are pulled sideways onto the channel in the DGM1 (see _snap_to_channel). Water height =
    channel floor + `lift`, then as a running minimum downstream: the water only falls.
    If the terrain rises (fill, road, culvert), the water level stays below it and is invisible there.
    If the end lies higher than the start, the line is reversed (the OSM direction was then not the gradient).
    """
    points = _resample(coords, spacing)
    points, bottom = _snap_to_channel(points, height_at, search)
    if bottom[-1] > bottom[0] + 1.0:
        points, bottom = points[::-1], bottom[::-1]
    level = np.minimum.accumulate(bottom + lift)
    return [[float(x), float(y), float(z), float(width), float(depth), *NORMAL_UP] for (x, y), z in zip(points, level)]


def split_nodes(nodes: List[List[float]], max_nodes: int) -> List[List[List[float]]]:
    """Splits long node lists into pieces <= max_nodes; consecutive pieces share one node (gap-free)."""
    if len(nodes) <= max_nodes:
        return [nodes]
    chunks, start = [], 0
    while start < len(nodes) - 1:
        end = min(start + max_nodes, len(nodes))
        chunks.append(nodes[start:end])
        start = end - 1
    return chunks


IDENTITY_ROTATION = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def is_pond_area(tags: Dict[str, str]) -> bool:
    """
    Water area by OSM tags: natural water, basin or reservoir. Dry flood retention basins
    (`basin=detention`) are never water, but meadow (see landuse_mappings["meadow"]).
    """
    if tags.get("basin") == "detention":
        return False
    return tags.get("natural") == "water" or tags.get("landuse") in ("basin", "reservoir")


def select_pond_areas(landuse_polygons: Sequence[Dict], bounds: Bounds) -> List[BaseGeometry]:
    """Water areas (see is_pond_area) as geometries, clipped to the terrain (`bounds` = xmin, ymin, xmax, ymax)."""
    terrain_box = box(*bounds)
    areas = []
    for polygon in landuse_polygons:
        if not is_pond_area(polygon["osm_tags"]):
            continue
        geometry = polygon["geometry"].intersection(terrain_box)
        if not geometry.is_empty and geometry.area > 0:
            areas.append(geometry)
    return areas


def carve_pond_basins(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    areas: Sequence[BaseGeometry],
    depth: float = 0.5,
    slope_deg: float = 45.0,
) -> np.ndarray:
    """
    Lowers the terrain inside the water areas: all raster points in the polygon by `depth` meters, with an
    embankment of `slope_deg` degrees inward (lowering = distance to the edge * tan(angle), at most `depth`; at
    45 degrees that is 1 m lowering per meter inward, full depth after 0.5 m). Outside and on the edge nothing
    changes. If areas overlap, the larger lowering applies per point.

    heights[i, j] belongs to the world position (origin_x + j * square_size, origin_y + i * square_size).
    Returns a new heightmap, `heights` remains unchanged.
    """
    from shapely import contains_xy, distance, points

    rows, cols = heights.shape
    lowering = np.zeros_like(heights, dtype=float)
    slope = float(np.tan(np.radians(slope_deg)))
    for area in areas:
        if area is None or area.is_empty:
            continue
        min_x, min_y, max_x, max_y = area.bounds
        j0 = max(0, int(np.floor((min_x - origin_x) / square_size)))
        j1 = min(cols, int(np.ceil((max_x - origin_x) / square_size)) + 1)
        i0 = max(0, int(np.floor((min_y - origin_y) / square_size)))
        i1 = min(rows, int(np.ceil((max_y - origin_y) / square_size)) + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        xs, ys = np.meshgrid(origin_x + np.arange(j0, j1) * square_size, origin_y + np.arange(i0, i1) * square_size)
        inside = contains_xy(area, xs, ys)
        if not inside.any():
            continue
        edge_distance = distance(points(xs[inside], ys[inside]), area.boundary)
        window = lowering[i0:i1, j0:j1]
        window[inside] = np.maximum(window[inside], np.minimum(depth, edge_distance * slope))
    return heights - lowering


def pond_level(polygon: BaseGeometry, height_at: HeightAt, count: int = 3, rim_step: float = 1.0) -> float:
    """Water level: mean of the `count` lowest elevation points on the polygon edge (points every `rim_step` m)."""
    ring = polygon.exterior
    samples = max(count, int(np.ceil(ring.length / rim_step)))
    points = np.array([ring.interpolate(d).coords[0] for d in np.linspace(0.0, ring.length, samples, endpoint=False)])
    heights = np.sort(np.asarray(height_at(points[:, 0], points[:, 1]), dtype=float))
    return float(heights[:count].mean())


def _cells_covering(shape: BaseGeometry, cell: float) -> List[Tuple[float, float, float, float]]:
    """Raster cells of edge length `cell` that touch `shape`, merged into row rectangles."""
    prepared = prep(shape)
    min_x, min_y, max_x, max_y = shape.bounds
    rectangles = []
    y = min_y
    while y < max_y - 1e-9:
        run_start = None
        x = min_x
        while x < max_x - 1e-9:
            hit = prepared.intersects(box(x, y, x + cell, y + cell))
            if hit and run_start is None:
                run_start = x
            if not hit and run_start is not None:
                rectangles.append((run_start, y, x, y + cell))
                run_start = None
            x += cell
        if run_start is not None:
            rectangles.append((run_start, y, x, y + cell))
        y += cell
    return rectangles


def build_pond_blocks(
    polygon: BaseGeometry,
    height_at: HeightAt,
    depth: float = 3.0,
    cell: float = 6.0,
    margin: float = 2.0,
) -> List[Dict]:
    """
    `WaterBlock` boxes (position = center, surface at position.z; scale = width, length, depth) that
    cover the polygon plus `margin` meters of border without gaps.

    The water level applies to the whole water body: mean of the three lowest edge points (see pond_level). The
    blocks extend beyond the edge so that the water fills the whole hole where the terrain lies below the
    water level; where it is higher, the water is hidden. The cell size adapts to the pond size (at least
    1 m), so that even small ponds are filled cleanly.
    """
    if polygon is None or polygon.is_empty:
        return []
    polygon = polygon.buffer(0)
    pieces = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    blocks = []
    for piece in pieces:
        if piece.area < 1.0:
            continue
        level = pond_level(piece, height_at)
        step = min(cell, max(1.0, np.sqrt(piece.area) / 3.0))
        for x0, y0, x1, y1 in _cells_covering(piece.buffer(margin), step):
            blocks.append(
                {
                    "position": [(x0 + x1) / 2.0, (y0 + y1) / 2.0, level],
                    "scale": [x1 - x0, y1 - y0, float(depth)],
                    "rotationMatrix": list(IDENTITY_ROTATION),
                }
            )
    return blocks
