"""Tests for world_to_beamng.tunnels.gallery_mesh: avalanche gallery open on the valley side (roof + columns)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.tunnels.gallery_mesh import build_gallery_mesh, build_galleries, resolve_open_side, valley_side

FLOOR, ROOF = "asphalt_road_standard", "tunnel_concrete"


def _straight_coords(length=60.0, z=500.0, n=13):
    return [(x, 0.0, z) for x in np.linspace(0.0, length, n)]


def test_valley_side_picks_the_lower_natural_terrain():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    # Terrain falls toward +y: when driving in +x direction, +y is the LEFT side (standard convention as in
    # offset_points(): left = direction rotated by +90° CCW = (-dy,dx); for direction=(1,0) that is (0,1) = +y).
    # So +y is the valley side -> left is downhill -> side < 0.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)

    side = valley_side(xy, ground_at, half_width=4.0)

    assert np.all(side < 0)


def test_valley_side_flips_when_the_slope_is_mirrored():
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    ground_at = lambda x, y: 500.0 + 2.0 * np.asarray(y, float)  # rises toward +y -> -y (right) is the valley side

    side = valley_side(xy, ground_at, half_width=4.0)

    assert np.all(side > 0)


def test_roof_and_floor_are_flat_at_the_given_heights():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, roof_thickness=0.35, floor_thickness=3.0,
    )
    v = mesh["vertices"]

    assert v[:, 2].min() == pytest.approx(497.0)  # floor(500) - floor thickness(3)
    assert v[:, 2].max() == pytest.approx(505.35)  # floor(500) + height(5) + roof thickness(0.35)


def test_faces_are_split_by_material():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(_straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF)

    assert set(mesh["faces"]) == {FLOOR, ROOF}
    assert len(mesh["faces"][FLOOR]) > 0 and len(mesh["faces"][ROOF]) > 0


def _off_grid_vertices_near(vertices, y, grid_step=5.0, tol=0.3):
    """Vertices near y whose X is NOT on the centerline point grid (multiples of grid_step) - floor/
    roof/wall faces only have vertices on the point grid, only columns insert vertices in between."""
    x, y_coord = vertices[:, 0], vertices[:, 1]
    on_grid = np.abs((x / grid_step) - np.round(x / grid_step)) < 0.01
    return np.sum((~on_grid) & (np.abs(y_coord - y) < tol))


def test_columns_are_on_the_valley_side_not_the_mountain_side():
    # Floor/roof/wall always span both edges (y=+4 and y=-4) and lie only on the 5 m centerline
    # point grid - only columns (here at x=5,15,...,55, exactly on the grid in this scenario, coinciding
    # with column_spacing=10) insert additional vertices EXACTLY at their x position. To distinguish that from
    # floor/roof/wall vertices (also at multiples of 5), deliberately do NOT choose column_spacing on the
    # 5 m grid here.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # falls toward +y -> +y is the valley side (left)
    mesh = build_gallery_mesh(_straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF, column_spacing=12.0)

    v = np.array(mesh["vertices"])
    assert _off_grid_vertices_near(v, y=4.0) > 0  # valley side: column vertices off the point grid
    assert _off_grid_vertices_near(v, y=-4.0) == 0  # mountain side: no columns


def test_wall_extends_wall_thickness_into_the_mountain():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # mountain side is -y (right)
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, wall_thickness=3.0, column_spacing=1000.0,  # no columns (would distort the edge)
    )

    v = np.array(mesh["vertices"])
    # Mountain side (y<0): the outer wall face extends to width/2 + wall_thickness = 4 + 3 = 7 m from the axis.
    assert v[:, 1].min() == pytest.approx(-7.0)
    # Valley side (y>0) stays at the plain carriageway width, width/2 = 4 m.
    assert v[:, 1].max() == pytest.approx(4.0)


def test_wall_is_flush_with_the_roof_top():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # mountain side is -y (right)
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, roof_thickness=0.5, column_spacing=1000.0,  # no columns (would distort the edge)
    )

    v = np.array(mesh["vertices"])
    mountain_side = np.abs(v[:, 1] + 4.0) < 6.0  # entire mountain side (wall extends to y=-9 with wall_thickness=5)
    assert v[mountain_side][:, 2].max() == pytest.approx(505.5)  # floor(500) + height(5) + roof thickness(0.5)


def test_curb_is_on_the_open_side_only():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # valley side (open) is +y
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, curb_height=0.5, curb_width=0.25, column_spacing=1000.0,
    )

    v = np.array(mesh["vertices"])
    at_valley_edge = np.abs(v[:, 1] - 4.0) < 0.01
    at_mountain_edge = np.abs(v[:, 1] + 4.0) < 0.01
    # Curb top edge (500.5 = floor 500 + curb height 0.5) only on the valley side, not on the mountain side.
    assert np.any(np.isclose(v[at_valley_edge][:, 2], 500.5))
    assert not np.any(np.isclose(v[at_mountain_edge][:, 2], 500.5))


def test_curb_does_not_widen_the_gallery_footprint():
    # The curb lies curb_width INSIDE the carriageway edge, so it does not extend beyond the previous width.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(
        _straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, curb_height=0.5, curb_width=0.25, wall_thickness=0.0, column_spacing=1000.0,
    )

    v = np.array(mesh["vertices"])
    assert v[:, 1].max() == pytest.approx(4.0)  # width / 2


def test_columns_sit_flush_on_top_of_the_curb_not_in_the_floor():
    """Regression: columns previously started at floor level inside the curb (Z overlap) - the base must
    now sit on the curb top edge, the top edge stays unchanged at the roof underside (the
    column becomes shorter by curb_height as a result)."""
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # +y is the valley side (open)
    mesh = build_gallery_mesh(
        _straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at,
        floor_material=FLOOR, roof_material=ROOF, column_spacing=12.0,
        curb_height=0.5, curb_width=0.4, column_size=0.4,
    )
    v = np.array(mesh["vertices"])

    # Column vertices: off the 5 m point grid (x) AND near the valley-side edge (y=4, outer column edge).
    off_grid = np.abs((v[:, 0] / 5.0) - np.round(v[:, 0] / 5.0)) >= 0.01
    near_valley_edge = np.abs(v[:, 1] - 4.0) < 0.5
    column_vertices = v[off_grid & near_valley_edge]

    assert len(column_vertices) > 0
    assert column_vertices[:, 2].min() == pytest.approx(500.5)  # curb top edge (floor 500 + 0.5), not 500
    assert column_vertices[:, 2].max() == pytest.approx(505.0)  # unchanged: floor(500) + height(5)


def test_columns_footprint_is_centered_on_the_curb_and_flush_with_the_roof_edge():
    """Regression: columns were previously centered on the carriageway edge (roof edge cut through the
    column center). Now centered on the curb centerline - with curb_width == column_size the outer
    column edge coincides exactly with the (unchanged) roof/carriageway edge."""
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(
        _straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at,
        floor_material=FLOOR, roof_material=ROOF, column_spacing=12.0,
        curb_height=0.5, curb_width=0.4, column_size=0.4,
    )
    v = np.array(mesh["vertices"])
    off_grid = np.abs((v[:, 0] / 5.0) - np.round(v[:, 0] / 5.0)) >= 0.01
    column_vertices = v[off_grid]

    assert len(column_vertices) > 0
    assert column_vertices[:, 1].max() == pytest.approx(4.0)  # = width/2 = roof/carriageway edge, no overhang
    assert column_vertices[:, 1].min() == pytest.approx(3.6)  # curb center (3.8) - half column width (0.2)


def test_ends_are_capped_with_outward_facing_faces():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(_straight_coords(z=500.0), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF)

    v, n = np.array(mesh["vertices"]), np.array(mesh["normals"])
    # End faces at the start (x=0, normal -x) and end (x=60, normal +x).
    start_faces = np.abs(v[:, 0]) < 1e-6
    end_faces = np.abs(v[:, 0] - 60.0) < 1e-6
    assert np.any(start_faces & (n[:, 0] < -0.99))
    assert np.any(end_faces & (n[:, 0] > 0.99))


def test_build_galleries_returns_one_mesh_per_gallery():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    galleries = [{"id": 1, "coords": _straight_coords(z=500.0), "width": 8.0, "floor_material": FLOOR}]

    meshes = build_galleries(galleries, ground_at, roof_material=ROOF, height=5.0)

    assert [m["id"] for m in meshes] == ["gallery_1"]


def test_build_galleries_skips_degenerate_galleries():
    ground_at = lambda x, y: np.full_like(np.asarray(x, float), 500.0)
    galleries = [{"id": 1, "coords": [(0.0, 0.0, 500.0)], "width": 8.0, "floor_material": FLOOR}]

    assert build_galleries(galleries, ground_at, roof_material=ROOF) == []


# --- resolve_open_side / open_side override ----------------------------------------------------------------------


def test_resolve_open_side_reads_the_avalanche_protector_tag():
    assert resolve_open_side({"avalanche_protector:left": "open"}) == "left"
    assert resolve_open_side({"avalanche_protector:right": "open"}) == "right"


def test_resolve_open_side_is_none_without_a_reliable_tag():
    assert resolve_open_side({}) is None
    assert resolve_open_side({"avalanche_protector:left": "no"}) is None


def test_open_side_override_ignores_ground_at_even_when_it_disagrees():
    # ground_at would put the valley side on +y (left) (see test_valley_side_picks_the_lower_natural_terrain) -
    # the tag must still win, since at an existing gallery the DGM shows the structure itself.
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    mesh = build_gallery_mesh(
        _straight_coords(length=60.0, z=500.0), width=8.0, height=5.0, ground_at=ground_at,
        floor_material=FLOOR, roof_material=ROOF, column_spacing=12.0, open_side="right",
    )

    v = np.array(mesh["vertices"])
    assert _off_grid_vertices_near(v, y=-4.0) > 0  # "right" = -y open -> columns there
    assert _off_grid_vertices_near(v, y=4.0) == 0


def test_build_galleries_uses_the_osm_tag_when_present():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)  # would return "left" without the tag
    galleries = [{
        "id": 1, "coords": _straight_coords(length=60.0, z=500.0), "width": 8.0, "floor_material": FLOOR,
        "osm_tags": {"avalanche_protector:right": "open"},
    }]

    meshes = build_galleries(galleries, ground_at, roof_material=ROOF, column_spacing=12.0)

    v = np.array(meshes[0]["vertices"])
    assert _off_grid_vertices_near(v, y=-4.0) > 0
    assert _off_grid_vertices_near(v, y=4.0) == 0


def _cap_faces(mesh, x, normal_x):
    v, n = mesh["vertices"], mesh["normals"]
    faces = [f for faces in mesh["faces"].values() for f in faces]
    return [f for f in faces if np.allclose(v[f][:, 0], x) and np.allclose(n[f[0]], [normal_x, 0.0, 0.0])]


def test_end_caps_can_be_left_out():
    ground_at = lambda x, y: 500.0 - 2.0 * np.asarray(y, float)
    kwargs = dict(width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR, roof_material=ROOF)

    capped = build_gallery_mesh(_straight_coords(), **kwargs)
    uncapped = build_gallery_mesh(_straight_coords(), cap_start=False, **kwargs)

    assert _cap_faces(capped, 0.0, -1.0) and not _cap_faces(uncapped, 0.0, -1.0)
    assert _cap_faces(uncapped, 60.0, 1.0)  # other end stays closed


def test_without_tag_the_whole_gallery_opens_to_the_majority_valley_side():
    # Terrain tips over at x = 42: before that the valley is on the right (-y), after that on the left (+y). The
    # gallery is still open toward ONE side over its whole length - the majority side (right), the mountain wall
    # continuously left.
    def ground_at(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return np.where(x < 42.0, 500.0 + 2.0 * y, 500.0 - 2.0 * y)

    mesh = build_gallery_mesh(
        _straight_coords(length=60.0, n=13), width=8.0, height=5.0, ground_at=ground_at, floor_material=FLOOR,
        roof_material=ROOF, wall_thickness=5.0,
    )
    y = mesh["vertices"][:, 1]

    assert np.any(np.isclose(y, 9.0))  # outer edge of the mountain wall on the left (4 + 5 m)
    assert not np.any(np.isclose(y, -9.0))  # no wall anywhere on the right


def _embedded_slope(valley_right: bool):
    """Slope as after embedding: flat up to 6 m beside the axis (carriageway + embankment fringe, with a tiny
    counter-slope as in the real export), outside of that the slope drops steeply toward the valley side."""
    sign = 1.0 if valley_right else -1.0  # driving in +x direction: right = -y, left = +y

    def ground_at(x, y):
        y = np.asarray(y, float)
        slope = 500.0 + sign * y  # valley_right: right (-y) is lower
        flat = 500.0 - 0.01 * sign * y  # counter-slope in the flat band: feigns the wrong side
        return np.where(np.abs(y) <= 6.0, flat, slope)

    return ground_at


@pytest.mark.parametrize("valley_right", [True, False])
def test_valley_side_looks_beyond_the_embedded_band(valley_right):
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])

    side = valley_side(xy, _embedded_slope(valley_right), half_width=3.25)

    assert np.all(side == (1.0 if valley_right else -1.0))


def test_untagged_gallery_on_an_embedded_slope_opens_to_the_valley():
    mesh = build_gallery_mesh(
        _straight_coords(length=60.0, n=13), width=6.5, height=5.0, ground_at=_embedded_slope(valley_right=True),
        floor_material=FLOOR, roof_material=ROOF, wall_thickness=5.0,
    )
    y = mesh["vertices"][:, 1]

    assert np.any(np.isclose(y, 3.25 + 5.0))  # mountain wall on the left (slope rises toward the left)
    assert not np.any(np.isclose(y, -(3.25 + 5.0)))


def test_gallery_open_side_prefers_the_tag_and_falls_back_to_the_terrain():
    from world_to_beamng.tunnels.gallery_mesh import gallery_open_side

    coords = _straight_coords(length=60.0, n=13)
    valley_right = _embedded_slope(valley_right=True)

    assert gallery_open_side({"avalanche_protector:left": "open"}, coords, valley_right, 6.5) == "left"
    assert gallery_open_side({}, coords, valley_right, 6.5) == "right"
    assert gallery_open_side({}, coords, _embedded_slope(valley_right=False), 6.5) == "left"


def test_build_galleries_uses_a_given_open_side():
    gallery = {"id": 9, "coords": _straight_coords(length=60.0, n=13), "width": 6.5, "floor_material": FLOOR,
               "osm_tags": {}, "open_side": "left"}

    mesh = build_galleries([gallery], _embedded_slope(valley_right=True), ROOF, wall_thickness=5.0)[0]
    y = mesh["vertices"][:, 1]

    assert np.any(np.isclose(y, -(3.25 + 5.0)))  # forced open on the left -> mountain wall on the right, despite terrain
    assert not np.any(np.isclose(y, 3.25 + 5.0))
