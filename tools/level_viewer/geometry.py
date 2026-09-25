"""
Pure numpy geometry builders for the viewer layers (no pyvista): road ribbons, oriented boxes, water boxes,
item transforms, spawn directions and terrain texture coordinates.

Meshes are returned as (points (n, 3), faces (m, k) int) with k = 3 or 4; ``to_vtk_faces`` converts the face
array into VTK's flat [k, i0, i1, ..., k, ...] layout.
"""

from typing import Optional, Tuple

import numpy as np

# Unit cube corners (x, y, z in -0.5..0.5) and its 6 quads (outward winding)
_UNIT_CORNERS = np.array(
    [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
     [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]]
)
BOX_QUADS = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
BOX_TOP_QUAD = 1  # index of the +Z face in BOX_QUADS


def to_vtk_faces(faces: np.ndarray) -> np.ndarray:
    """(m, k) face index array -> flat VTK cell array [k, i0, ..., k, ...]."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.hstack([np.full((len(faces), 1), faces.shape[1], dtype=np.int64), faces]).ravel()


def transform_points(local: np.ndarray, position, scale=(1.0, 1.0, 1.0), rotation: Optional[np.ndarray] = None) -> np.ndarray:
    """Local -> world: (local * scale) @ R + position, R rows = images of the local axes (BeamNG rotationMatrix)."""
    pts = np.asarray(local, dtype=float) * np.asarray(scale, dtype=float)
    if rotation is not None:
        pts = pts @ np.asarray(rotation, dtype=float)
    return pts + np.asarray(position, dtype=float)


def oriented_box(center, scale, rotation: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Box of size `scale` centered on `center` and rotated by `rotation` (Zone/Portal items)."""
    return transform_points(_UNIT_CORNERS, center, scale, rotation), BOX_QUADS.copy()


def water_block_box(position, scale) -> Tuple[np.ndarray, np.ndarray]:
    """WaterBlock: XY centered on position, water surface at position.z, scale = (width, length, depth downward)."""
    width, length, depth = (float(v) for v in scale)
    center = np.asarray(position, dtype=float) - [0.0, 0.0, depth / 2.0]
    return oriented_box(center, (width, length, depth))


def forward_direction(rotation: Optional[np.ndarray]) -> np.ndarray:
    """Facing direction of a spawned vehicle: local -Y, i.e. minus the second row of the rotationMatrix."""
    if rotation is None:
        return np.array([0.0, -1.0, 0.0])
    direction = -np.asarray(rotation, dtype=float)[1]
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-12 else np.array([0.0, -1.0, 0.0])


def heading_deg(direction) -> float:
    """Compass heading of a direction in degrees (0 = north/+Y, 90 = east/+X)."""
    return float(np.degrees(np.arctan2(direction[0], direction[1])) % 360.0)


def ribbon(nodes, z_offset: float = 0.0, width_index: int = 3, default_width: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flat band along a polyline (DecalRoad/River nodes [x, y, z, width, ...]).

    Each node gets a left and a right point at +-width/2 perpendicular to the XY tangent (central difference
    inside, one-sided at the ends); consecutive duplicate nodes are skipped. Points are ordered
    [left0, right0, left1, right1, ...]; faces are quads between consecutive node pairs.
    """
    arr = np.asarray(nodes, dtype=float)
    if arr.ndim != 2 or len(arr) < 2 or arr.shape[1] < 3:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = np.hypot(*(arr[1:, :2] - arr[:-1, :2]).T) > 1e-6
    arr = arr[keep]
    if len(arr) < 2:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)

    xy = arr[:, :2]
    tangent = np.empty_like(xy)
    tangent[1:-1] = xy[2:] - xy[:-2]
    tangent[0] = xy[1] - xy[0]
    tangent[-1] = xy[-1] - xy[-2]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # left of the direction of travel

    width = arr[:, width_index] if arr.shape[1] > width_index else np.full(len(arr), default_width)
    half = (width / 2.0)[:, None]
    z = arr[:, 2] + z_offset
    left = np.column_stack([xy + normal * half, z])
    right = np.column_stack([xy - normal * half, z])
    points = np.empty((2 * len(arr), 3))
    points[0::2], points[1::2] = left, right

    i = np.arange(len(arr) - 1) * 2
    faces = np.column_stack([i, i + 1, i + 3, i + 2])
    return points, faces


def road_z_offset(render_priority, base: float = 0.05, step: float = 0.005, max_priority: int = 20) -> float:
    """Height above the terrain for a DecalRoad: lower renderPriority is drawn on top in BeamNG, so it goes higher."""
    try:
        prio = float(render_priority)
    except (TypeError, ValueError):
        prio = max_priority
    return base + max(0.0, max_priority - prio) * step


def terrain_tcoords(x: np.ndarray, y: np.ndarray, x_min: float, y_max: float, size_x: float, size_y: float) -> np.ndarray:
    """
    Texture coordinates for an image placed with its top-left (row 0 = north) corner at (x_min, y_max).

    u grows east, v grows north (v = 0 at the south edge), which matches VTK's image origin at the bottom-left.
    """
    u = (np.asarray(x, dtype=float) - x_min) / size_x
    v = (np.asarray(y, dtype=float) - (y_max - size_y)) / size_y
    return np.column_stack([u.ravel(), v.ravel()])


def tile_sample_range(coords: np.ndarray, lower: float, upper: float, spacing: float) -> np.ndarray:
    """
    Indices of the grid samples that make up one photo tile [lower, upper] along one axis.

    Samples need not lie on the tile bounds (the terrain grid sits on half meters, the photo tiles on whole ones):
    each tile also takes the last sample before `lower`, so two neighbouring tiles share exactly one sample and
    their surfaces meet without a gap or an overlap.
    """
    coords = np.asarray(coords, dtype=float)
    return np.flatnonzero((coords > lower - spacing) & (coords <= upper))


def grid_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, hole: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quad mesh of a height grid: z[row, col] at (x[col], y[row]); point index = row * nx + col.

    Cells with a hole at any of their four corners are left out.
    """
    ny, nx = z.shape
    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    points = np.column_stack([xx.ravel(), yy.ravel(), np.asarray(z, dtype=float).ravel()])
    r, c = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    i = (r * nx + c).ravel()
    faces = np.column_stack([i, i + 1, i + nx + 1, i + nx])
    if hole is not None and np.any(hole):
        h = np.asarray(hole, dtype=bool).ravel()
        faces = faces[~(h[faces].any(axis=1))]
    return points, faces


def merge_meshes(meshes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenates [(points, faces, item_id), ...] (all faces with the same corner count) into one mesh.

    Returns (points, faces, cell_item_ids) with the item id repeated for every face of its mesh.
    """
    points, faces, ids = [], [], []
    offset = 0
    for pts, fcs, item_id in meshes:
        if len(pts) == 0 or len(fcs) == 0:
            continue
        points.append(np.asarray(pts, dtype=float))
        faces.append(np.asarray(fcs, dtype=np.int64) + offset)
        ids.append(np.full(len(fcs), item_id, dtype=np.int64))
        offset += len(pts)
    if not points:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.vstack(points), np.vstack(faces), np.concatenate(ids)
