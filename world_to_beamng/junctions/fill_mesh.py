"""
Fill mesh of the rounded junction corners, draped onto the terrain.

The mesh is purely visual (no collision: vehicles drive on the terrain, which is embedded under the fill anyway), so it
must lie ON the terrain surface instead of spanning it with large flat triangles. Every terrain cell under the fill is
split into four triangles around its centre: the cell corners take the grid heights, the centre the higher of the two
diagonal midpoints. Each of these triangles then lies inside ONE terrain triangle whichever diagonal BeamNG splits the
cell along, and its corners are on or above the terrain - so the mesh never dips below the ground and rises above it
only where a cell is not planar (millimetres on embedded ground). The triangles are clipped to the fill outline; new
points on the outline take the height of their triangle's plane. A small lift above that keeps the terrain from
flickering through; UVs are world-aligned so neighbouring fills continue seamlessly; one mesh per junction node.
"""

from typing import Dict, List, Sequence

import numpy as np

from ..walls.mesh_parts import MeshBuilder, unit_vector


def _cell_triangles(heights: np.ndarray, row: int, col: int, origin_x: float, origin_y: float, square_size: float) -> List[np.ndarray]:
    """The four (3, 3) triangles of a terrain cell around its centre (heights[row, col], x -> col, y -> row)."""
    x0, y0 = origin_x + col * square_size, origin_y + row * square_size
    x1, y1 = x0 + square_size, y0 + square_size
    h00, h01 = heights[row, col], heights[row, col + 1]
    h10, h11 = heights[row + 1, col], heights[row + 1, col + 1]
    p00, p01, p10, p11 = (x0, y0, h00), (x1, y0, h01), (x0, y1, h10), (x1, y1, h11)
    centre = (0.5 * (x0 + x1), 0.5 * (y0 + y1), max(0.5 * (h00 + h11), 0.5 * (h01 + h10)))
    return [np.array(t, dtype=float) for t in ((p00, p01, centre), (p01, p11, centre), (p11, p10, centre), (p10, p00, centre))]


def _plane_z(triangle: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Heights of the plane through `triangle` at the points `xy` (barycentric)."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = triangle
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    l1 = ((by - cy) * (xy[:, 0] - cx) + (cx - bx) * (xy[:, 1] - cy)) / det
    l2 = ((cy - ay) * (xy[:, 0] - cx) + (ax - cx) * (xy[:, 1] - cy)) / det
    return l1 * az + l2 * bz + (1.0 - l1 - l2) * cz


def _clip_to_fill(triangle: np.ndarray, fill) -> List[np.ndarray]:
    """The parts of a cell triangle inside the fill outline, as (3, 3) triangles on the triangle's plane."""
    import shapely
    from shapely.geometry import Polygon

    shape = Polygon(triangle[:, :2])
    if fill.contains(shape):
        return [triangle]
    part = fill.intersection(shape)
    if part.is_empty or part.area < 1e-8:
        return []
    result = []
    for piece in getattr(shapely.constrained_delaunay_triangles(part), "geoms", []):
        xy = np.asarray(piece.exterior.coords)[:3]
        if piece.area > 1e-10:
            result.append(np.column_stack([xy, _plane_z(triangle, xy)]))
    return result


def _draped_triangles(corner: Dict, heights: np.ndarray, origin_x: float, origin_y: float, square_size: float) -> List[np.ndarray]:
    import shapely
    from shapely.geometry import Polygon, box

    fill = Polygon(np.vstack([np.asarray(corner["corner_point"])[None, :2], np.asarray(corner["rim"])[:, :2]])).buffer(0)
    if fill.is_empty:
        return []
    shapely.prepare(fill)
    size_y, size_x = heights.shape
    minx, miny, maxx, maxy = fill.bounds
    col0 = max(0, int(np.floor((minx - origin_x) / square_size)))
    col1 = min(size_x - 2, int(np.floor((maxx - origin_x) / square_size)))
    row0 = max(0, int(np.floor((miny - origin_y) / square_size)))
    row1 = min(size_y - 2, int(np.floor((maxy - origin_y) / square_size)))
    triangles = []
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            x0, y0 = origin_x + col * square_size, origin_y + row * square_size
            if not fill.intersects(box(x0, y0, x0 + square_size, y0 + square_size)):
                continue
            for triangle in _cell_triangles(heights, row, col, origin_x, origin_y, square_size):
                triangles += _clip_to_fill(triangle, fill)
    return triangles


def build_junction_meshes(
    corners: Sequence[Dict],
    lift: float,
    tile_m: float,
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
) -> List[Dict]:
    """
    Args:
        corners: corner dicts of find_junction_corners() ("node", "corner_point", "rim", "surface")
        lift: height above the terrain surface, in meters
        tile_m: texture tile, in meters (u = x / tile_m, v = y / tile_m)
        heights, origin_x, origin_y, square_size: the finished terrain heightmap (heights[row, col], x -> col, y -> row)

    Returns:
        [{"id", "vertices", "uvs", "normals", "faces": {"<surface>_junction": [...]}}], one per junction node
    """
    by_node: Dict[tuple, List[Dict]] = {}
    for corner in corners:
        by_node.setdefault(tuple(np.round(np.asarray(corner["node"])[:2], 2)), []).append(corner)

    meshes = []
    for node_corners in by_node.values():
        builders: Dict[str, MeshBuilder] = {}
        for corner in node_corners:
            # own material per surface ("<surface>_junction"): a see-through gravel fill must not share its material
            # with an opaque gravel bridge deck ("<surface>_structure")
            builder = builders.setdefault(f"{corner['surface']}_junction", MeshBuilder())
            for triangle in _draped_triangles(corner, heights, origin_x, origin_y, square_size):
                tri = triangle + [0.0, 0.0, lift]
                normal = unit_vector(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
                if normal[2] < 0:
                    normal = [-c for c in normal]
                points = [list(map(float, v)) for v in tri]
                builder.triangle(points, [[v[0] / tile_m, v[1] / tile_m] for v in points], normal)
        if not any(b.faces for b in builders.values()):
            continue
        vertices, uvs, normals, faces = [], [], [], {}
        for material, builder in builders.items():
            offset = len(vertices)
            vertices += builder.vertices
            uvs += builder.uvs
            normals += builder.normals
            faces[material] = [[i + offset for i in face] for face in builder.faces]
        meshes.append({"id": f"junction_{len(meshes)}", "vertices": np.array(vertices, dtype=float),
                       "uvs": np.array(uvs, dtype=float), "normals": np.array(normals, dtype=float), "faces": faces})
    return meshes
