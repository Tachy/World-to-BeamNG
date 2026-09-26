"""
Bridges from OSM lines (highway=* with bridge=*): concrete deck with a real road-bridge cross section
(carriageway with road material, a curb on both sides, on top of it a railing made of posts + handrail) and
rectangular support piers down to the natural terrain below.

The deck does NOT follow the terrain (unlike the walls) - its height comes from the linearly interpolated
bridge height profile (geometry/road_structures.py + geometry/polygon.py), which is already contained in the
passed `coords`. Only the piers reach down to the natural terrain below (`ground_at`). Curb and railing
follow the deck height profile, not the terrain.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, add_box_column, offset_points

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _arc_length(xy: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _interp_at(cum: np.ndarray, arr: np.ndarray, s: float):
    """Interpolates `arr` (1D or 2D, one value per point of `cum`) at the arc length `s`."""
    idx = max(1, min(int(np.searchsorted(cum, s)), len(cum) - 1))
    t = (s - cum[idx - 1]) / max(cum[idx] - cum[idx - 1], 1e-9)
    return arr[idx - 1] + t * (arr[idx] - arr[idx - 1])


def _direction_at(cum: np.ndarray, xy: np.ndarray, s: float) -> Tuple[float, float]:
    """Direction of travel (not normalized) at the arc length `s` - for add_box_column()'s `direction`, so that
    pier/post profiles are aligned relative to the bridge instead of axis-parallel to the world."""
    idx = max(1, min(int(np.searchsorted(cum, s)), len(xy) - 1))
    d = xy[idx] - xy[idx - 1]
    return float(d[0]), float(d[1])


def _build_edge_beam(xy_line: np.ndarray, top_z: np.ndarray, thickness: float, tile_m: float) -> MeshBuilder:
    """Thin, rectangular beam along `xy_line` (handrail): top/bottom plus both side faces.
    `top_z` gives the top edge per point of `xy_line` (so it follows the same height profile as the deck)."""
    edge_left, edge_right = offset_points(xy_line, thickness / 2.0, closed=False)
    bottom_z = top_z - thickness
    cum = _arc_length(xy_line)
    along = cum / tile_m
    across = thickness / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    builder = MeshBuilder()
    for i in range(len(xy_line) - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy_line[j] - xy_line[i]
        direction = direction / np.linalg.norm(direction)
        side_normal = [float(-direction[1]), float(direction[0]), 0.0]

        builder.quad(
            [p3(edge_left[i], top_z[i]), p3(edge_left[j], top_z[j]), p3(edge_right[j], top_z[j]), p3(edge_right[i], top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, 1.0],
        )
        builder.quad(
            [p3(edge_left[i], bottom_z[i]), p3(edge_right[i], bottom_z[i]), p3(edge_right[j], bottom_z[j]), p3(edge_left[j], bottom_z[j])],
            [[u0, 0.0], [u0, across], [u1, across], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        builder.quad(
            [p3(edge_left[i], bottom_z[i]), p3(edge_left[j], bottom_z[j]), p3(edge_left[j], top_z[j]), p3(edge_left[i], top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            side_normal,
        )
        builder.quad(
            [p3(edge_right[i], bottom_z[i]), p3(edge_right[j], bottom_z[j]), p3(edge_right[j], top_z[j]), p3(edge_right[i], top_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [-side_normal[0], -side_normal[1], 0.0],
        )
    return builder


def build_bridge_mesh(
    coords: Sequence[Tuple[float, float, float]],
    width: float,
    ground_at: HeightAt,
    deck_material: str,
    pier_material: str,
    railing_material: str,
    deck_thickness: float = 0.6,
    pier_spacing: float = 25.0,
    pier_width_fraction: float = 0.5,
    pier_depth_fraction: float = 0.5,
    pier_burial: float = 5.0,
    min_pier_clearance: float = 1.0,
    curb_width: float = 0.4,
    curb_height: float = 0.2,
    railing_height: float = 0.9,
    railing_post_spacing: float = 2.0,
    railing_post_size: float = 0.08,
    tile_m: float = 5.0,
    road_texture_length: float = 5.0,
    widths: Optional[Sequence[float]] = None,
) -> Dict:
    """
    Deck, curb, railing and pier mesh for a bridge along `coords` (already the
    bridge height profile, x,y,z per point). `widths` (per coordinate, default: `width` everywhere) lets the width
    change along the bridge - at the transition to a road of a different width (see
    geometry/road_width_transitions.py); deck, curbs and railing follow it.

    Cross section from outside to inside: railing (posts + handrail) - curb (curb_width/curb_height,
    pier_material) - carriageway (deck_material, exactly `width` wide). Curb and railing stand OUTSIDE the carriageway,
    so the deck slab is `width` + 2x curb_width wide. The carriageway UVs follow the DecalRoad layout (u across 0..1, v
    along in repeats of road_texture_length meters), so the road texture continues 1:1 from the approach.

    Piers: `pier_width_fraction` of the carriageway width across the road, `pier_depth_fraction` of that dimension along
    it, reaching `pier_burial` meters below the natural ground.

    Returns:
        {"vertices": (N,3), "uvs": (N,2), "normals": (N,3),
         "faces": {deck_material: [...], pier_material: [...], railing_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    top = points[:, 2]
    bottom = top - deck_thickness
    curb_top = top + curb_height

    # inner = carriageway edge (= curb inner face), outer = curb outer edge = deck slab edge
    half = (np.full(len(xy), width) if widths is None else np.asarray(widths, dtype=float)) / 2.0
    unit_left, unit_right = offset_points(xy, 1.0, closed=False)  # the miter offset is linear in the distance

    def offset_by(unit_edge, distance):
        return xy + (unit_edge - xy) * distance[:, None]

    inner_left, inner_right = offset_by(unit_left, half), offset_by(unit_right, half)
    outer_left, outer_right = offset_by(unit_left, half + curb_width), offset_by(unit_right, half + curb_width)
    width = float(np.mean(half) * 2.0)  # UV scale only

    cum = _arc_length(xy)
    along = cum / tile_m
    road_v = cum / road_texture_length
    deck_across = (width + 2.0 * curb_width) / tile_m
    curb_across = curb_width / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    deck_builder = MeshBuilder()
    pier_builder = MeshBuilder()
    for i in range(len(points) - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy[j] - xy[i]
        direction = direction / np.linalg.norm(direction)
        side_normal = [float(-direction[1]), float(direction[0]), 0.0]
        inward = [-side_normal[0], -side_normal[1], 0.0]

        # Carriageway (top side, between the curbs)
        deck_builder.quad(
            [p3(inner_left[i], top[i]), p3(inner_left[j], top[j]), p3(inner_right[j], top[j]), p3(inner_right[i], top[i])],
            [[0.0, road_v[i]], [0.0, road_v[j]], [1.0, road_v[j]], [1.0, road_v[i]]],
            [0.0, 0.0, 1.0],
        )
        # Underside (full width)
        deck_builder.quad(
            [p3(outer_left[i], bottom[i]), p3(outer_right[i], bottom[i]), p3(outer_right[j], bottom[j]), p3(outer_left[j], bottom[j])],
            [[u0, 0.0], [u0, deck_across], [u1, deck_across], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        # Fascia left/right (deck bottom edge up to carriageway level)
        deck_builder.quad(
            [p3(outer_left[i], bottom[i]), p3(outer_left[j], bottom[j]), p3(outer_left[j], top[j]), p3(outer_left[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, deck_thickness / tile_m], [u0, deck_thickness / tile_m]],
            side_normal,
        )
        deck_builder.quad(
            [p3(outer_right[i], bottom[i]), p3(outer_right[j], bottom[j]), p3(outer_right[j], top[j]), p3(outer_right[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, deck_thickness / tile_m], [u0, deck_thickness / tile_m]],
            [-side_normal[0], -side_normal[1], 0.0],
        )

        # Curb left: top, outer (continuation of the fascia) and inner face (toward the carriageway)
        pier_builder.quad(
            [p3(outer_left[i], curb_top[i]), p3(outer_left[j], curb_top[j]), p3(inner_left[j], curb_top[j]), p3(inner_left[i], curb_top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_across], [u0, curb_across]],
            [0.0, 0.0, 1.0],
        )
        pier_builder.quad(
            [p3(outer_left[i], top[i]), p3(outer_left[j], top[j]), p3(outer_left[j], curb_top[j]), p3(outer_left[i], curb_top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_height / tile_m], [u0, curb_height / tile_m]],
            side_normal,
        )
        pier_builder.quad(
            [p3(inner_left[i], top[i]), p3(inner_left[j], top[j]), p3(inner_left[j], curb_top[j]), p3(inner_left[i], curb_top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_height / tile_m], [u0, curb_height / tile_m]],
            inward,
        )
        # Curb right (mirrored)
        pier_builder.quad(
            [p3(inner_right[i], curb_top[i]), p3(inner_right[j], curb_top[j]), p3(outer_right[j], curb_top[j]), p3(outer_right[i], curb_top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_across], [u0, curb_across]],
            [0.0, 0.0, 1.0],
        )
        pier_builder.quad(
            [p3(outer_right[i], top[i]), p3(outer_right[j], top[j]), p3(outer_right[j], curb_top[j]), p3(outer_right[i], curb_top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_height / tile_m], [u0, curb_height / tile_m]],
            [-side_normal[0], -side_normal[1], 0.0],
        )
        pier_builder.quad(
            [p3(inner_right[i], top[i]), p3(inner_right[j], top[j]), p3(inner_right[j], curb_top[j]), p3(inner_right[i], curb_top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, curb_height / tile_m], [u0, curb_height / tile_m]],
            side_normal,
        )

    # End faces at both ends (deck full height + curb top on both sides)
    for index, sign, neighbour in ((0, -1.0, 1), (len(points) - 1, 1.0, len(points) - 2)):
        direction = xy[1] - xy[0] if index == 0 else xy[-1] - xy[neighbour]
        direction = direction / np.linalg.norm(direction)
        face_normal = [float(sign * direction[0]), float(sign * direction[1]), 0.0]
        deck_builder.quad(
            [p3(outer_left[index], bottom[index]), p3(outer_right[index], bottom[index]), p3(outer_right[index], top[index]), p3(outer_left[index], top[index])],
            [[0.0, 0.0], [deck_across, 0.0], [deck_across, deck_thickness / tile_m], [0.0, deck_thickness / tile_m]],
            face_normal,
        )
        for edge_out, edge_in in ((outer_left, inner_left), (outer_right, inner_right)):
            pier_builder.quad(
                [p3(edge_out[index], top[index]), p3(edge_in[index], top[index]), p3(edge_in[index], curb_top[index]), p3(edge_out[index], curb_top[index])],
                [[0.0, 0.0], [curb_across, 0.0], [curb_across, curb_height / tile_m], [0.0, curb_height / tile_m]],
                face_normal,
            )

    # Piers: every pier_spacing meters along the arc length, only if there is enough clearance above the terrain
    total_len = float(cum[-1])
    pier_positions = np.arange(pier_spacing, total_len, pier_spacing) if total_len > pier_spacing else np.array([])
    for s in pier_positions:
        cx, cy = _interp_at(cum, xy, s)
        deck_bottom_z = float(_interp_at(cum, bottom, s))
        ground_z = float(ground_at(np.array([cx]), np.array([cy]))[0])
        if deck_bottom_z - ground_z < min_pier_clearance:
            continue
        pier_width = float(_interp_at(cum, half * 2.0, s)) * pier_width_fraction
        add_box_column(
            pier_builder, cx, cy, ground_z - pier_burial, deck_bottom_z, pier_width * pier_depth_fraction, tile_m,
            direction=_direction_at(cum, xy, s), across=pier_width,
        )

    # Railing: posts + continuous handrail on both sides, on the curb top edge, centered on the curb (the mean of two
    # offset_points() results on the same normal equals an offset by the averaged distance)
    railing_builder = MeshBuilder()
    rail_top = curb_top + railing_height + railing_post_size / 2.0
    post_positions = np.arange(0.0, total_len + 1e-6, railing_post_spacing) if total_len > 0 else np.array([])
    for edge_xy in ((inner_left + outer_left) / 2.0, (inner_right + outer_right) / 2.0):
        for s in post_positions:
            px, py = _interp_at(cum, edge_xy, s)
            post_bottom_z = float(_interp_at(cum, curb_top, s))
            post_top_z = post_bottom_z + railing_height
            add_box_column(railing_builder, px, py, post_bottom_z, post_top_z, railing_post_size, tile_m, direction=_direction_at(cum, edge_xy, s))
        beam = _build_edge_beam(edge_xy, rail_top, railing_post_size, tile_m)
        railing_builder.vertices += beam.vertices
        railing_builder.uvs += beam.uvs
        railing_builder.normals += beam.normals
        offset = len(railing_builder.vertices) - len(beam.vertices)
        railing_builder.faces += [[a + offset, b + offset, c + offset] for a, b, c in beam.faces]

    def _merge(*builders: MeshBuilder):
        vertices, uvs, normals, faces = [], [], [], []
        for builder in builders:
            offset = len(vertices)
            vertices += builder.vertices
            uvs += builder.uvs
            normals += builder.normals
            faces.append([[a + offset, b + offset, c + offset] for a, b, c in builder.faces])
        return vertices, uvs, normals, faces

    all_vertices, all_uvs, all_normals, (deck_faces, pier_faces, railing_faces) = _merge(deck_builder, pier_builder, railing_builder)

    return {
        "vertices": np.array(all_vertices, dtype=float),
        "uvs": np.array(all_uvs, dtype=float),
        "normals": np.array(all_normals, dtype=float),
        "faces": {deck_material: deck_faces, pier_material: pier_faces, railing_material: railing_faces},
    }


def build_bridges(
    bridges: Sequence[Dict],
    ground_at: HeightAt,
    pier_material: str,
    railing_material: str,
    deck_thickness: float = 0.6,
    pier_spacing: float = 25.0,
    pier_width_fraction: float = 0.5,
    pier_depth_fraction: float = 0.5,
    pier_burial: float = 5.0,
    min_pier_clearance: float = 1.0,
    curb_width: float = 0.4,
    curb_height: float = 0.2,
    railing_height: float = 0.9,
    railing_post_spacing: float = 2.0,
    railing_post_size: float = 0.08,
    road_texture_length: float = 5.0,
) -> List[Dict]:
    """Mesh dicts for the DAE export, one per bridge (`bridges`: [{"id","coords","width","deck_material"}, ...];
    optional "widths": width per coordinate, see build_bridge_mesh())."""
    meshes = []
    for bridge in bridges:
        coords = bridge["coords"]
        if len(coords) < 2:
            continue
        mesh = build_bridge_mesh(
            coords, bridge["width"], ground_at, bridge["deck_material"], pier_material, railing_material,
            deck_thickness=deck_thickness, pier_spacing=pier_spacing,
            pier_width_fraction=pier_width_fraction, pier_depth_fraction=pier_depth_fraction, pier_burial=pier_burial,
            min_pier_clearance=min_pier_clearance,
            curb_width=curb_width, curb_height=curb_height, railing_height=railing_height,
            railing_post_spacing=railing_post_spacing, railing_post_size=railing_post_size,
            road_texture_length=road_texture_length, widths=bridge.get("widths"),
        )
        meshes.append({"id": f"bridge_{bridge['id']}", **mesh})
    return meshes
