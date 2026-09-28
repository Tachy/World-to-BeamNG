"""Fill mesh of the junction corners: draped onto the terrain cells, lifted, world UVs, one mesh per node."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from world_to_beamng.junctions.fill_mesh import build_junction_meshes

LIFT, TILE = 0.02, 5.0


def _corner(node=(0.0, 0.0), surface="asphalt_road_standard", offset=(10.0, 10.0)):
    """Right-angle corner (r = 6) whose fill lies around (offset + (2.5..8.5, 3..9))."""
    ox, oy = offset
    angles = np.linspace(-np.pi / 2, -np.pi, 25)
    arc = np.column_stack([ox + 8.5 + 6 * np.cos(angles), oy + 9.0 + 6 * np.sin(angles), np.full(25, 100.0)])
    rim = np.vstack([[ox + 2.5 + 6.0, oy + 3.0, 100.0], arc, [ox + 2.5, oy + 3.0 + 6.0, 100.0]])
    return {"node": np.array([ox + node[0], oy + node[1], 100.0]), "corner_point": np.array([ox + 2.5, oy + 3.0, 100.0]),
            "arc": arc, "rim": rim, "surface": surface}


def _terrain(size=64):
    """Non-planar embedded-like ground: slope plus a gentle bump (heights[row, col], x -> col, y -> row, 1 m cells)."""
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    return 100.0 + 0.05 * cols + 0.03 * rows + 0.4 * np.exp(-((cols - 15.0) ** 2 + (rows - 16.0) ** 2) / 30.0)


def _build(corners, heights=None):
    heights = _terrain() if heights is None else heights
    return build_junction_meshes(corners, LIFT, TILE, heights, 0.0, 0.0, 1.0), heights


def _terrain_z(heights, x, y, diagonal):
    """Height of the terrain triangle at (x, y) for one of the two possible cell splits."""
    c0, r0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - c0, y - r0
    h00, h01, h10, h11 = heights[r0, c0], heights[r0, c0 + 1], heights[r0 + 1, c0], heights[r0 + 1, c0 + 1]
    if diagonal == "00-11":
        return h00 + fx * (h01 - h00) + fy * (h11 - h01) if fx >= fy else h00 + fy * (h10 - h00) + fx * (h11 - h10)
    return h00 + fx * (h01 - h00) + fy * (h10 - h00) if fx + fy <= 1 else h11 + (1 - fx) * (h10 - h11) + (1 - fy) * (h01 - h11)


def _mesh_z(mesh, x, y):
    """Height of the mesh at (x, y) (barycentric in the triangle containing it) or None."""
    v = mesh["vertices"]
    for faces in mesh["faces"].values():
        for a, b, c in faces:
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = v[a], v[b], v[c]
            det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(det) < 1e-12:
                continue
            l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det
            l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det
            l3 = 1 - l1 - l2
            if min(l1, l2, l3) >= -1e-9:
                return l1 * az + l2 * bz + l3 * cz
    return None


def _fill_polygon(corner):
    return Polygon(np.vstack([corner["corner_point"][None, :2], corner["rim"][:, :2]]))


def test_mesh_lies_on_the_terrain_for_both_cell_splits():
    corner = _corner()
    (mesh,), heights = _build([corner])
    fill = _fill_polygon(corner)
    rng = np.random.default_rng(7)
    minx, miny, maxx, maxy = fill.bounds
    checked = 0
    while checked < 300:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not fill.buffer(-0.05).contains(Point(x, y)):
            continue
        z = _mesh_z(mesh, x, y)
        assert z is not None, (x, y)
        for diagonal in ("00-11", "01-10"):
            gap = z - LIFT - _terrain_z(heights, x, y, diagonal)
            assert -1e-6 <= gap <= 0.03, (x, y, diagonal, gap)
        checked += 1


def test_mesh_covers_exactly_the_fill_area():
    corner = _corner()
    (mesh,), _ = _build([corner])
    v = mesh["vertices"]
    faces = np.array([f for fs in mesh["faces"].values() for f in fs])
    a, b, c = v[faces[:, 0], :2], v[faces[:, 1], :2], v[faces[:, 2], :2]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum()
    fill = _fill_polygon(corner)
    assert area == pytest.approx(fill.area, rel=0.01)
    assert all(fill.buffer(0.01).contains(Point(p)) for p in v[:, :2])


def test_one_mesh_per_node_faces_up_world_uvs_and_junction_materials():
    corners = [_corner(), _corner(surface="gravel_road", offset=(10.0, 10.0)), _corner(node=(30.0, 0.0), offset=(40.0, 10.0))]
    meshes, _ = _build(corners)
    assert len(meshes) == 2
    assert set(meshes[0]["faces"]) == {"asphalt_road_standard_junction", "gravel_road_junction"}
    for mesh in meshes:
        assert np.all(mesh["normals"][:, 2] > 0.9)
        assert np.allclose(mesh["uvs"], mesh["vertices"][:, :2] / TILE)


def test_no_corners_no_meshes():
    assert _build([])[0] == []
