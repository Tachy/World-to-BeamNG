"""Tests for world_to_beamng.tunnels.tunnel_mesh: circular tunnel tube (default profile: 240° arc above the
carriageway, floor as a chord, radius/height derived from the width), chaining of the tunnel pieces."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math

import numpy as np
import pytest

from world_to_beamng.tunnels.tunnel_mesh import (
    build_tunnel_mesh,
    build_tunnels,
    chain_tunnel_pieces,
    resample_tunnel_coords,
    tunnel_crown_height,
    tunnel_radius,
)
from world_to_beamng.tunnels.tunnel_portal import plan_tunnels

WALL, FRAME, FLOOR = "tunnel_concrete", "tunnel_concrete", "asphalt_road_standard"


def _straight_coords(length=100.0, z=500.0, n=11):
    return [(x, 0.0, z) for x in np.linspace(0.0, length, n)]


def test_tunnel_radius_and_crown_height_follow_the_floor_width():
    # Floor chord = sqrt(3) * R with the 240°/120° split -> R = width / sqrt(3); chosen exactly so that R = 8
    width = 8.0 * math.sqrt(3.0)
    assert tunnel_radius(width) == pytest.approx(8.0)
    assert tunnel_crown_height(width) == pytest.approx(12.0)  # 1.5 * R


def test_tube_floor_is_flat_and_matches_the_road_width():
    # Check only the floor vertices (not mesh["vertices"] as a whole): the circular arc is wider than the
    # floor chord (with arc_segments=12 it bulges outward to ~R*cos(10°) > width/2) - that is correct
    # and not an error, see design spec section 5 (240° arc above a narrower floor chord).
    mesh = build_tunnel_mesh(_straight_coords(z=500.0), width=8.0, floor_material=FLOOR, wall_material=WALL)
    floor_idx = sorted({i for tri in mesh["faces"][FLOOR] for i in tri})
    floor_v = mesh["vertices"][floor_idx]

    assert floor_v[:, 2].min() == pytest.approx(500.0)  # floor = elevation profile
    assert floor_v[:, 1].min() == pytest.approx(-4.0) and floor_v[:, 1].max() == pytest.approx(4.0)  # floor width = 8 m


def test_crown_reaches_the_derived_height_above_the_floor():
    width = 8.0
    mesh = build_tunnel_mesh(_straight_coords(z=500.0), width=width, floor_material=FLOOR, wall_material=WALL, arc_segments=12)

    assert mesh["vertices"][:, 2].max() == pytest.approx(500.0 + tunnel_crown_height(width), abs=1e-6)


def test_tube_faces_are_split_by_material():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    assert set(mesh["faces"]) == {FLOOR, WALL}
    assert len(mesh["faces"][FLOOR]) == 2 * 2  # 2 segments, floor = 1 quad = 2 triangles per segment
    assert len(mesh["faces"][WALL]) == 2 * 6 * 2  # 6 arc strips per segment, 2 triangles each


def test_floor_normal_points_up_into_the_tube():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    assert (0.0, 0.0, 1.0) in {tuple(np.round(n, 2)) for n in mesh["normals"]}


def test_arc_normals_are_unit_length_and_do_not_point_straight_up():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    wall_normals = np.array([mesh["normals"][face[0]] for face in mesh["faces"][WALL]])
    assert np.allclose(np.linalg.norm(wall_normals, axis=1), 1.0, atol=1e-6)
    assert not np.any(np.all(np.isclose(wall_normals, [0.0, 0.0, 1.0], atol=1e-3), axis=1))


def test_floor_and_arc_share_exact_edge_vertices_even_on_a_curve():
    # arc_ring() uses exactly right[i]/left[i] at the floor edges like the floor mesh - otherwise a gap
    # between floor and arc would arise in a curve (different segment directions).
    coords = [(0.0, 0.0, 500.0), (20.0, 2.0, 500.0), (40.0, 0.0, 500.0)]
    mesh = build_tunnel_mesh(coords, width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    floor_idx = {i for tri in mesh["faces"][FLOOR] for i in tri}
    wall_idx = {i for tri in mesh["faces"][WALL] for i in tri}
    floor_points = {tuple(np.round(mesh["vertices"][i], 3)) for i in floor_idx}
    wall_points = {tuple(np.round(mesh["vertices"][i], 3)) for i in wall_idx}

    assert len(floor_points & wall_points) >= 4  # both floor-edge rings (start+end) are shared points


def test_resample_tunnel_coords_reduces_point_count_for_a_long_tunnel():
    coords = _straight_coords(length=17000.0, z=500.0, n=17001)  # 1 m spacing as from the normal pipeline

    resampled = resample_tunnel_coords(coords, step=10.0)

    assert len(resampled) < len(coords) / 5
    assert resampled[0] == pytest.approx(coords[0])
    assert resampled[-1][0] == pytest.approx(coords[-1][0], abs=1e-6)


def test_resample_tunnel_coords_leaves_short_tunnels_unchanged():
    coords = _straight_coords(length=5.0, z=500.0, n=6)

    assert resample_tunnel_coords(coords, step=10.0) == coords


def test_adjacent_segments_share_the_exact_same_ring_on_a_curve():
    # The rings are mitered at the centerline points: both segments at a bend use exactly
    # the same ring points - otherwise a gap opens between the tube segments in curves.
    coords = [(0.0, 0.0, 500.0), (20.0, 0.0, 500.0), (40.0, 8.0, 500.0)]
    mesh = build_tunnel_mesh(coords, width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)

    wall_idx = sorted({i for tri in mesh["faces"][WALL] for i in tri})
    wall_v = np.round(mesh["vertices"][wall_idx], 6)
    # Vertices at the bend (projection onto the axis near x=20) - there is only ONE ring of 7 points
    at_joint = {tuple(v) for v in wall_v if abs(v[0] - 20.0) < 2.0 and abs(v[1]) < 6.0}
    assert len(at_joint) == 7


def _piece(piece_id, coords, width=7.0):
    return {"id": piece_id, "coords": coords, "width": width, "floor_material": FLOOR}


def test_chain_tunnel_pieces_joins_split_pieces_into_one_tube_regardless_of_direction():
    a = _piece(1, [(0.0, 0.0, 500.0), (10.0, 0.0, 501.0)])
    b = _piece(2, [(20.0, 0.0, 502.0), (10.0, 0.0, 501.0)])  # digitized in reverse
    c = _piece(3, [(20.0, 0.0, 502.0), (30.0, 0.0, 503.0)])

    chains = chain_tunnel_pieces([b, c, a])

    assert len(chains) == 1
    xs = [p[0] for p in chains[0]["coords"]]
    assert xs in ([0.0, 10.0, 20.0, 30.0], [30.0, 20.0, 10.0, 0.0])


def test_chain_tunnel_pieces_ignores_a_different_tunnel_crossing_at_the_joint():
    # Footpath tunnel crosses the road tunnel in 2D and was split at the same point
    a = _piece(1, [(0.0, 0.0, 500.0), (10.0, 0.0, 500.0)])
    b = _piece(2, [(10.0, 0.0, 500.0), (20.0, 0.0, 500.0)])
    path_a = _piece(3, [(10.0, -10.0, 900.0), (10.0, 0.0, 900.0)], width=2.0)
    path_b = _piece(4, [(10.0, 0.0, 900.0), (10.0, 10.0, 900.0)], width=2.0)

    chains = chain_tunnel_pieces([a, path_a, b, path_b])

    assert sorted(len(c["coords"]) for c in chains) == [3, 3]


def test_chain_tunnel_pieces_does_not_join_at_a_three_way_joint_or_across_widths():
    a = _piece(1, [(0.0, 0.0, 500.0), (10.0, 0.0, 500.0)])
    b = _piece(2, [(10.0, 0.0, 500.0), (20.0, 0.0, 500.0)])
    c = _piece(3, [(10.0, 0.0, 500.0), (10.0, 10.0, 500.0)])
    d = _piece(4, [(20.0, 0.0, 500.0), (30.0, 0.0, 500.0)], width=3.0)

    assert len(chain_tunnel_pieces([a, b, c, d])) == 4


def _plans(tunnels):
    return plan_tunnels(tunnels, width_margin=1.5, segment_step=10.0, collar_ratio=0.1, flat_depth=1.5, length=3.5)


def test_build_tunnels_returns_one_tube_and_two_portal_blocks_per_chain():
    pieces = [_piece(42, _straight_coords(length=100.0, n=6)), _piece(43, [(100.0, 0.0, 500.0), (200.0, 0.0, 500.0)])]

    meshes = build_tunnels(_plans(pieces), WALL, FRAME)

    assert [m["id"] for m in meshes] == ["tunnel_42", "tunnel_42_portal_start", "tunnel_42_portal_end"]
    assert FLOOR in meshes[0]["faces"]
    assert FRAME in meshes[1]["faces"] and FRAME in meshes[2]["faces"]


def test_portal_block_opening_matches_the_first_tube_ring():
    plans = _plans([_piece(1, _straight_coords(length=100.0))])
    meshes = build_tunnels(plans, WALL, FRAME, arc_segments=12)
    tube, block = meshes[0], meshes[1]

    ring = {tuple(np.round(v, 5)) for v in tube["vertices"] if abs(v[0]) < 1e-9}
    block_front = {tuple(np.round(v, 5)) for v in block["vertices"] if abs(v[0]) < 1e-9}
    assert len(ring) == 13  # 12 arc strips -> 13 ring points
    assert ring <= block_front  # every ring edge of the tube is also an edge of the portal opening


def test_open_portal_collar_is_rectangular_with_a_tenth_of_the_diameter_at_its_thinnest_points():
    # Rectangular on the outside; left, right and top at the thinnest point wall thickness = diameter / 10
    plans = plan_tunnels([_piece(1, _straight_coords(length=100.0))], width_margin=1.5, segment_step=10.0, collar_ratio=0.1,
                         flat_depth=1.5, length=3.5, shell_ratio=1.0 / 15.0)
    portal = plans[0]["portals"][0]
    radius, crown = portal["radius"], portal["crown"]
    wall = 0.1 * 2.0 * radius
    block = build_tunnels(plans, WALL, FRAME)[1]
    v = block["vertices"]

    assert portal["collar"] == pytest.approx(wall)
    assert portal["half_width"] == pytest.approx(radius + wall)  # thinnest point at the side: circle at ±R
    assert portal["top_z"] == pytest.approx(500.0 + crown + wall)  # thinnest point at the top: above the crown
    assert portal["bottom_z"] == pytest.approx(500.0 - wall)
    assert v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(3.5)
    assert np.abs(v[:, 1]).max() == pytest.approx(radius + wall)
    assert v[:, 2].max() == pytest.approx(portal["top_z"]) and v[:, 2].min() == pytest.approx(portal["bottom_z"])

    # End face = rectangle minus clear cross-section, no triangle in the opening
    from shapely.geometry import Polygon, box

    from world_to_beamng.tunnels.tunnel_mesh import arc_cross_section

    front_area = 0.0
    for face in block["faces"][FRAME]:
        pts = v[face]
        if not (np.allclose(pts[:, 0], 0.0) and block["normals"][face[0]][0] < -0.99):
            continue
        (_, y0, z0), (_, y1, z1), (_, y2, z2) = pts
        front_area += abs((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)) / 2.0
        cy, cz = pts[:, 1].mean(), pts[:, 2].mean() - 500.0
        assert not (cz > 0.0 and math.hypot(cy, cz - radius / 2.0) < radius - 1e-6)
    frame = box(-radius - wall, -wall, radius + wall, crown + wall)
    assert front_area == pytest.approx(frame.difference(Polygon(arc_cross_section(radius, 12))).area, rel=1e-6)


def test_tilted_entrance_collar_front_follows_the_tilted_face_and_the_flat_zone_reaches_behind_its_top():
    plans = plan_tunnels([_piece(1, _straight_coords(length=100.0))], width_margin=1.5, segment_step=10.0, collar_ratio=0.1,
                         flat_depth=1.5, length=3.5, shell_ratio=1.0 / 15.0, tilt_deg=20.0)
    portal = plans[0]["portals"][0]
    tilt = math.tan(math.radians(20.0))
    block = build_tunnels(plans, WALL, FRAME)[1]
    v, n = block["vertices"], block["normals"]

    front = [f for f in block["faces"][FRAME] if n[f[0]][0] < -0.5]
    assert front
    for f in front:
        for x, _, z in v[f]:
            assert x == pytest.approx((z - 500.0) * tilt, abs=1e-6)
    assert portal["flat_depth"] == pytest.approx((portal["top_z"] - 500.0) * tilt)


def test_build_tunnels_skips_too_short_tunnels():
    tunnels = [_piece(1, [(0.0, 0.0, 500.0)])]

    assert build_tunnels(_plans(tunnels), WALL, FRAME) == []


def test_chain_tunnel_pieces_joins_ends_a_few_millimetres_apart():
    a = _piece(1, [(0.0, 0.0, 500.0), (10.0, 0.0, 500.0)])
    b = _piece(2, [(10.004, 0.003, 500.0), (20.0, 0.0, 500.0)])

    assert len(chain_tunnel_pieces([a, b])) == 1


SHELL = "tunnel_shell"


def _shell_mesh(width=8.0, thickness=1.0, length=100.0):
    return build_tunnel_mesh(
        _straight_coords(length=length, z=500.0), width=width, floor_material=FLOOR, wall_material=WALL,
        arc_segments=12, shell_thickness=thickness, shell_material=SHELL,
    )


def _shell_faces(mesh):
    v, n = mesh["vertices"], mesh["normals"]
    return [(v[f], n[f[0]]) for f in mesh["faces"][SHELL]]


def test_shell_is_a_concentric_cylinder_one_metre_outside_the_tube_with_a_floor_slab():
    # Tube as a cylinder with 1 m wall thickness: visibly solid from outside, may stand freely (no earth fill needed)
    width = 8.0
    radius = tunnel_radius(width)
    mesh = _shell_mesh(width)
    shell_v = np.array([p for tri, _ in _shell_faces(mesh) for p in tri])

    center_z = 500.0 + radius / 2.0
    arc = shell_v[shell_v[:, 2] > 500.0]  # above the floor only the outer arc (and the end rings)
    assert np.hypot(arc[:, 1], arc[:, 2] - center_z).max() == pytest.approx(radius + 1.0, abs=1e-6)
    assert shell_v[:, 2].max() == pytest.approx(500.0 + tunnel_crown_height(width) + 1.0, abs=1e-6)
    assert shell_v[:, 2].min() == pytest.approx(500.0 - 1.0)  # floor slab 1 m below the carriageway


def test_shell_outer_faces_point_away_from_the_tube_axis():
    radius = tunnel_radius(8.0)
    center_z = 500.0 + radius / 2.0
    for tri, normal in _shell_faces(_shell_mesh()):
        if abs(normal[0]) > 0.5:
            continue  # end rings
        c = tri.mean(axis=0)
        radial = np.array([0.0, c[1], c[2] - center_z])
        assert normal @ radial > 0.0


def test_shell_is_closed_by_a_ring_at_both_ends():
    from shapely.geometry import Polygon

    from world_to_beamng.tunnels.tunnel_mesh import arc_cross_section, shell_cross_section

    width, thickness = 8.0, 1.0
    radius = tunnel_radius(width)
    ring_area = Polygon(shell_cross_section(radius, 12, thickness)).area - Polygon(arc_cross_section(radius, 12)).area

    def area(tri):
        (_, y0, z0), (_, y1, z1), (_, y2, z2) = tri
        return abs((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0)) / 2.0

    faces = _shell_faces(_shell_mesh(width, thickness))
    start = [tri for tri, n in faces if np.allclose(tri[:, 0], 0.0) and n[0] < -0.99]
    end = [tri for tri, n in faces if np.allclose(tri[:, 0], 100.0) and n[0] > 0.99]
    assert sum(area(t) for t in start) == pytest.approx(ring_area, rel=1e-6)
    assert sum(area(t) for t in end) == pytest.approx(ring_area, rel=1e-6)


def test_tube_without_shell_stays_unchanged():
    mesh = build_tunnel_mesh(_straight_coords(n=3), width=8.0, floor_material=FLOOR, wall_material=WALL, arc_segments=6)
    assert set(mesh["faces"]) == {FLOOR, WALL}


def test_build_tunnels_gives_the_tube_a_shell_and_caps_only_ends_without_portal():
    # A shell end ring in the same plane as the collar end face would cause Z-fighting - only closed ends
    # (no open portal, e.g. in the middle of the mountain) get it.
    plans = plan_tunnels([_piece(1, _straight_coords(length=100.0))], width_margin=1.5, segment_step=10.0, collar_ratio=0.1,
                         flat_depth=1.5, length=3.5, shell_ratio=0.1)
    plans[0]["portals"][1]["open"] = False
    tube = build_tunnels(plans, "wall", "concrete")[0]
    v, n = tube["vertices"], tube["normals"]

    shell = tube["faces"]["concrete"]
    assert shell, "tube without outer shell"
    start_ring = [f for f in shell if np.allclose(v[f][:, 0], 0.0) and n[f[0]][0] < -0.99]
    end_ring = [f for f in shell if np.allclose(v[f][:, 0], 100.0) and n[f[0]][0] > 0.99]
    assert not start_ring and end_ring


def test_shell_thickness_is_a_tenth_of_the_tube_diameter():
    # Smaller tunnels get thinner walls: wall thickness : diameter = 1 : 10
    wide = plan_tunnels([_piece(1, _straight_coords(length=100.0), width=6.5)], width_margin=1.5, segment_step=10.0,
                        collar_ratio=0.1, flat_depth=1.5, length=3.5, shell_ratio=0.1)[0]
    narrow = plan_tunnels([_piece(2, _straight_coords(length=100.0), width=2.5)], width_margin=1.5, segment_step=10.0,
                          collar_ratio=0.1, flat_depth=1.5, length=3.5, shell_ratio=0.1)[0]

    for plan in (wide, narrow):
        assert plan["shell"] == pytest.approx(0.1 * 2.0 * plan["radius"])
        assert all(p["shell"] == pytest.approx(plan["shell"]) for p in plan["portals"])
    assert narrow["shell"] < wide["shell"]


def test_without_collar_the_portal_is_the_tube_end_ring_with_the_shell_thickness():
    # The portal should show the same wall thickness (1:15) as the tube: no collar, the tube closes itself
    plans = plan_tunnels([_piece(1, _straight_coords(length=100.0))], width_margin=1.5, segment_step=10.0, collar_ratio=0.0,
                         flat_depth=1.5, length=3.5, shell_ratio=1.0 / 15.0)
    meshes = build_tunnels(plans, "wall", "concrete")

    assert [m["id"] for m in meshes] == ["tunnel_1"]
    tube = meshes[0]
    v, n = tube["vertices"], tube["normals"]
    rings = [f for f in tube["faces"]["concrete"] if abs(n[f[0]][0]) > 0.99]
    assert any(np.allclose(v[f][:, 0], 0.0) for f in rings) and any(np.allclose(v[f][:, 0], 100.0) for f in rings)
    portal = plans[0]["portals"][0]
    assert portal["half_width"] == pytest.approx(portal["radius"] + portal["shell"])


TILT = math.tan(math.radians(20.0))


def test_tunnel_entrance_face_leans_20_degrees_into_the_mountain():
    # Portal front tilted by 20° toward the mountain side: at the floor on the portal plane, deeper in the mountain above
    width = 8.0
    radius = tunnel_radius(width)
    shell = radius * 2.0 / 15.0
    mesh = build_tunnel_mesh(
        _straight_coords(length=100.0, z=500.0), width=width, floor_material=FLOOR, wall_material=WALL, arc_segments=12,
        shell_thickness=shell, shell_material=SHELL, tilt_start=TILT,
    )
    v = mesh["vertices"]

    front = v[v[:, 0] < 5.0]  # vertices of the first ring (segment length 10 m)
    for x, _, z in front:
        assert x == pytest.approx((z - 500.0) * TILT, abs=1e-6)  # on the inclined plane
    crown = front[np.argmax(front[:, 2])]
    assert crown[0] == pytest.approx((tunnel_crown_height(width) + shell) * TILT, abs=1e-6)
    end = v[v[:, 0] > 95.0]
    assert np.allclose(end[:, 0], 100.0)  # other end perpendicular

    caps = [f for f in mesh["faces"][SHELL] if mesh["normals"][f[0]][0] < -0.5]
    assert caps
    for f in caps:
        assert mesh["normals"][f[0]] == pytest.approx([-math.cos(math.radians(20.0)), 0.0, math.sin(math.radians(20.0))])


def test_plan_tilts_only_tunnel_entrances_and_moves_the_flat_zone_behind_the_face():
    tunnel = _piece(1, _straight_coords(length=100.0))
    gallery = {"id": 2, "coords": [(-50.0, 0.0, 500.0), (0.0, 0.0, 500.0)], "width": 6.5, "floor_material": FLOOR,
               "osm_tags": {"covered": "yes"}}
    start, end = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, collar_ratio=0.0, flat_depth=1.5, length=3.5,
                              shell_ratio=1.0 / 15.0, tilt_deg=20.0, galleries=[gallery])[0]["portals"]

    assert start["kind"] == "gallery" and start["tilt"] == 0.0 and start["flat_depth"] == 1.5
    assert end["tilt"] == pytest.approx(TILT)
    face_depth = (end["crown"] + end["shell"]) * TILT  # this far the end face reaches into the mountain at the top
    assert end["flat_depth"] == pytest.approx(max(1.5, face_depth))
    assert end["length"] >= end["flat_depth"] + 1.5


def test_build_tunnels_tilts_the_tube_end_at_a_tunnel_entrance():
    plans = plan_tunnels([_piece(1, _straight_coords(length=100.0))], width_margin=1.5, segment_step=10.0, collar_ratio=0.0,
                         flat_depth=1.5, length=3.5, shell_ratio=1.0 / 15.0, tilt_deg=20.0)
    tube = build_tunnels(plans, "wall", "concrete")[0]
    v = tube["vertices"]

    top_front = v[(v[:, 0] < 5.0)][:, 0].max()
    assert top_front == pytest.approx((plans[0]["crown"] + plans[0]["shell"]) * TILT, abs=1e-6)


def test_collar_sides_are_at_least_the_minimum_side_width_top_stays_a_tenth():
    # At least collar_min_side on the sides (covers the hole cells at the portal step, grid diagonal 1.41 m), 1:10 top
    plans = plan_tunnels([_piece(1, _straight_coords(length=100.0))], width_margin=1.5, segment_step=10.0, collar_ratio=0.1,
                         flat_depth=1.5, length=3.5, shell_ratio=1.0 / 15.0, collar_min_side=1.5)
    portal = plans[0]["portals"][0]
    radius, crown = portal["radius"], portal["crown"]
    block = build_tunnels(plans, WALL, FRAME)[1]
    v = block["vertices"]

    assert portal["half_width"] == pytest.approx(radius + 1.5)
    assert portal["top_z"] == pytest.approx(500.0 + crown + 0.2 * radius)
    assert np.abs(v[:, 1]).max() == pytest.approx(radius + 1.5)
