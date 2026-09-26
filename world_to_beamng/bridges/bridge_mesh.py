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


def _pier_sites(xy, bottom, half, ground_at, pier_spacing, min_pier_clearance, pier_width_fraction) -> List[Tuple]:
    """Piers every pier_spacing meters along the arc length, only where there is enough clearance above the terrain:
    [(x, y, ground_z, deck_bottom_z, pier_width, direction)]."""
    cum = _arc_length(xy)
    total_len = float(cum[-1])
    positions = np.arange(pier_spacing, total_len, pier_spacing) if total_len > pier_spacing else np.array([])
    sites = []
    for s in positions:
        cx, cy = _interp_at(cum, xy, s)
        deck_bottom_z = float(_interp_at(cum, bottom, s))
        ground_z = float(ground_at(np.array([cx]), np.array([cy]))[0])
        if deck_bottom_z - ground_z < min_pier_clearance:
            continue
        pier_width = float(_interp_at(cum, half * 2.0, s)) * pier_width_fraction
        sites.append((float(cx), float(cy), ground_z, deck_bottom_z, pier_width, _direction_at(cum, xy, s)))
    return sites


def _add_pier(builder: MeshBuilder, site, pier_burial, pier_depth_fraction, tile_m) -> None:
    cx, cy, ground_z, deck_bottom_z, pier_width, direction = site
    add_box_column(
        builder, cx, cy, ground_z - pier_burial, deck_bottom_z, pier_width * pier_depth_fraction, tile_m,
        direction=direction, across=pier_width,
    )


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
    joined_left: Optional[Sequence[bool]] = None,
    joined_right: Optional[Sequence[bool]] = None,
    cap_start: bool = True,
    cap_end: bool = True,
    piers: bool = True,
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

    Several bridge ways as one structure (a lane split on a bridge, see build_bridge_group_mesh()):
    joined_left / joined_right (per coordinate) mark where the carriageway of another way lies right beside this side -
    there the side has no curb, railing or fascia; a flat slab strip of 2x curb_width, 1 cm below the carriageway,
    closes a narrow gap to the neighbour. cap_start / cap_end: end faces (not where the way meets the others);
    piers=False leaves the piers to the caller.

    Returns:
        {"vertices": (N,3), "uvs": (N,2), "normals": (N,3),
         "faces": {deck_material: [...], pier_material: [...], railing_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    top = points[:, 2]
    bottom = top - deck_thickness
    curb_top = top + curb_height
    count = len(xy)
    joined = {
        "left": np.zeros(count, dtype=bool) if joined_left is None else np.asarray(joined_left, dtype=bool),
        "right": np.zeros(count, dtype=bool) if joined_right is None else np.asarray(joined_right, dtype=bool),
    }

    # inner = carriageway edge (= curb inner face), outer = curb outer edge = deck slab edge (joined: the gap strip edge)
    half = (np.full(count, width) if widths is None else np.asarray(widths, dtype=float)) / 2.0
    unit = {}
    unit["left"], unit["right"] = offset_points(xy, 1.0, closed=False)  # the miter offset is linear in the distance

    def offset_by(side, distance):
        return xy + (unit[side] - xy) * distance[:, None]

    inner = {side: offset_by(side, half) for side in ("left", "right")}
    outer = {side: offset_by(side, half + np.where(joined[side], 2.0 * curb_width, curb_width)) for side in ("left", "right")}
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
    for i in range(count - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy[j] - xy[i]
        direction = direction / np.linalg.norm(direction)
        left_normal = [float(-direction[1]), float(direction[0]), 0.0]
        right_normal = [-left_normal[0], -left_normal[1], 0.0]

        # Carriageway (top side, between the curbs)
        deck_builder.quad(
            [p3(inner["left"][i], top[i]), p3(inner["left"][j], top[j]), p3(inner["right"][j], top[j]), p3(inner["right"][i], top[i])],
            [[0.0, road_v[i]], [0.0, road_v[j]], [1.0, road_v[j]], [1.0, road_v[i]]],
            [0.0, 0.0, 1.0],
        )
        # Underside (full width)
        deck_builder.quad(
            [p3(outer["left"][i], bottom[i]), p3(outer["right"][i], bottom[i]), p3(outer["right"][j], bottom[j]), p3(outer["left"][j], bottom[j])],
            [[u0, 0.0], [u0, deck_across], [u1, deck_across], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        for side, normal in (("left", left_normal), ("right", right_normal)):
            edge_in, edge_out = inner[side], outer[side]
            inward = [-normal[0], -normal[1], 0.0]
            if joined[side][i] and joined[side][j]:
                # Beside another carriageway: a flat strip just below the road surface closes the gap to it
                pier_builder.quad(
                    [p3(edge_in[i], top[i] - 0.01), p3(edge_in[j], top[j] - 0.01), p3(edge_out[j], top[j] - 0.01), p3(edge_out[i], top[i] - 0.01)],
                    [[u0, 0.0], [u1, 0.0], [u1, 2.0 * curb_across], [u0, 2.0 * curb_across]],
                    [0.0, 0.0, 1.0],
                )
                continue
            # Fascia (deck bottom edge up to carriageway level)
            deck_builder.quad(
                [p3(edge_out[i], bottom[i]), p3(edge_out[j], bottom[j]), p3(edge_out[j], top[j]), p3(edge_out[i], top[i])],
                [[u0, 0.0], [u1, 0.0], [u1, deck_thickness / tile_m], [u0, deck_thickness / tile_m]],
                normal,
            )
            # Curb: top, outer (continuation of the fascia) and inner face (toward the carriageway)
            pier_builder.quad(
                [p3(edge_out[i], curb_top[i]), p3(edge_out[j], curb_top[j]), p3(edge_in[j], curb_top[j]), p3(edge_in[i], curb_top[i])],
                [[u0, 0.0], [u1, 0.0], [u1, curb_across], [u0, curb_across]],
                [0.0, 0.0, 1.0],
            )
            pier_builder.quad(
                [p3(edge_out[i], top[i]), p3(edge_out[j], top[j]), p3(edge_out[j], curb_top[j]), p3(edge_out[i], curb_top[i])],
                [[u0, 0.0], [u1, 0.0], [u1, curb_height / tile_m], [u0, curb_height / tile_m]],
                normal,
            )
            pier_builder.quad(
                [p3(edge_in[i], top[i]), p3(edge_in[j], top[j]), p3(edge_in[j], curb_top[j]), p3(edge_in[i], curb_top[i])],
                [[u0, 0.0], [u1, 0.0], [u1, curb_height / tile_m], [u0, curb_height / tile_m]],
                inward,
            )

    # End faces (deck full height + curb top on the free sides), not where the way meets the others of its structure
    for index, sign, neighbour, wanted in ((0, -1.0, 1, cap_start), (count - 1, 1.0, count - 2, cap_end)):
        if not wanted:
            continue
        direction = xy[1] - xy[0] if index == 0 else xy[-1] - xy[neighbour]
        direction = direction / np.linalg.norm(direction)
        face_normal = [float(sign * direction[0]), float(sign * direction[1]), 0.0]
        deck_builder.quad(
            [p3(outer["left"][index], bottom[index]), p3(outer["right"][index], bottom[index]), p3(outer["right"][index], top[index]), p3(outer["left"][index], top[index])],
            [[0.0, 0.0], [deck_across, 0.0], [deck_across, deck_thickness / tile_m], [0.0, deck_thickness / tile_m]],
            face_normal,
        )
        for side in ("left", "right"):
            if joined[side][index]:
                continue
            pier_builder.quad(
                [p3(outer[side][index], top[index]), p3(inner[side][index], top[index]), p3(inner[side][index], curb_top[index]), p3(outer[side][index], curb_top[index])],
                [[0.0, 0.0], [curb_across, 0.0], [curb_across, curb_height / tile_m], [0.0, curb_height / tile_m]],
                face_normal,
            )

    total_len = float(cum[-1])
    if piers:
        for site in _pier_sites(xy, bottom, half, ground_at, pier_spacing, min_pier_clearance, pier_width_fraction):
            _add_pier(pier_builder, site, pier_burial, pier_depth_fraction, tile_m)

    # Railing: posts + continuous handrail on every free stretch of both sides, centered on the curb (the mean of two
    # offset_points() results on the same normal equals an offset by the averaged distance)
    railing_builder = MeshBuilder()
    rail_top = curb_top + railing_height + railing_post_size / 2.0
    post_positions = np.arange(0.0, total_len + 1e-6, railing_post_spacing) if total_len > 0 else np.array([])
    for side in ("left", "right"):
        edge_xy = (inner[side] + outer[side]) / 2.0
        free = ~joined[side]
        free_at = lambda s: bool(free[max(0, min(int(np.searchsorted(cum, s, side="right")) - 1, count - 1))]) and bool(
            free[max(0, min(int(np.searchsorted(cum, s)), count - 1))])
        for s in post_positions:
            if not free_at(s):
                continue
            px, py = _interp_at(cum, edge_xy, s)
            post_bottom_z = float(_interp_at(cum, curb_top, s))
            add_box_column(railing_builder, px, py, post_bottom_z, post_bottom_z + railing_height, railing_post_size, tile_m,
                           direction=_direction_at(cum, edge_xy, s))
        start = None
        for k in range(count + 1):
            if k < count and free[k]:
                start = k if start is None else start
                continue
            if start is not None and k - start >= 2:
                beam = _build_edge_beam(edge_xy[start:k], rail_top[start:k], railing_post_size, tile_m)
                offset = len(railing_builder.vertices)
                railing_builder.vertices += beam.vertices
                railing_builder.uvs += beam.uvs
                railing_builder.normals += beam.normals
                railing_builder.faces += [[a + offset, b + offset, c + offset] for a, b, c in beam.faces]
            start = None

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


def _densified(member: Dict, step: float) -> Dict:
    """The member with a point at least every `step` meters (coords and widths interpolated along the arc length)."""
    points = np.array(member["coords"], dtype=float)
    cum = _arc_length(points[:, :2])
    if cum[-1] <= 0.0:
        return member
    stations = np.unique(np.concatenate([cum, np.arange(0.0, cum[-1], step)]))
    coords = np.column_stack([np.interp(stations, cum, points[:, k]) for k in range(3)])
    widths = member.get("widths")
    result = {**member, "coords": [tuple(float(v) for v in c) for c in coords]}
    if widths is not None:
        result["widths"] = np.interp(stations, cum, np.asarray(widths, dtype=float))
    return result


def _carriageway_polygon(member: Dict):
    """Carriageway outline of a group member (flat ends), with its width profile."""
    from shapely.geometry import Polygon

    points = np.array(member["coords"], dtype=float)
    xy = points[:, :2]
    widths = member.get("widths")
    half = (np.full(len(xy), float(member["width"])) if widths is None else np.asarray(widths, dtype=float)) / 2.0
    unit_left, unit_right = offset_points(xy, 1.0, closed=False)
    left = xy + (unit_left - xy) * half[:, None]
    right = xy + (unit_right - xy) * half[:, None]
    return Polygon(np.vstack([left, right[::-1]])).buffer(0), xy, half, unit_left, unit_right


def build_bridge_group_mesh(
    members: Sequence[Dict],
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
    tile_m: float = 5.0,
    road_texture_length: float = 5.0,
    endpoint_tol: float = 0.5,
) -> Dict:
    """
    ONE bridge structure for several bridge ways of a lane split (see geometry/lane_splits.py): the trunk and its
    branches lie side by side in the same cross-section and move apart along their OSM courses. Every way keeps its own
    deck (build_bridge_mesh()); where another way's carriageway lies right beside one side (gap up to 2x curb_width),
    that side has no curb, railing or fascia and a flat slab strip closes the gap - once the carriageways have moved
    further apart, each side gets its own curb and railing. Ends that meet another way of the structure get no end
    face. Piers stand under every way; of two piers closer than pier_spacing / 2 only the first is kept.
    """
    from shapely.geometry import Point

    members = [_densified(m, 1.0) for m in members]  # the side masks switch per point: at most 1 m apart
    shapes = [_carriageway_polygon(m) for m in members]
    reach = 2.0 * curb_width + 0.05

    def joined_side(index, unit_side):
        polygon, xy, half, *_ = shapes[index]
        others = [shapes[k][0] for k in range(len(shapes)) if k != index]
        result = np.zeros(len(xy), dtype=bool)
        for i, (point, edge_unit) in enumerate(zip(xy, unit_side)):
            outward = edge_unit - point  # unit offset of the side (miter-scaled)
            edge = point + outward * half[i]
            probe = Point(*(edge + outward / max(np.linalg.norm(outward), 1e-9) * 0.01))
            result[i] = any(other.distance(probe) <= reach for other in others)
        return result

    def end_is_closed(index, end):
        xy, half = shapes[index][1], shapes[index][2]
        return any(
            k != index and np.linalg.norm(xy[end] - shapes[k][1][o]) <= max(half[end], shapes[k][2][o]) + endpoint_tol
            for k in range(len(shapes)) for o in (0, -1)
        )

    vertices, uvs, normals, faces = [], [], [], {}

    def add(mesh):
        offset = len(vertices)
        vertices.extend(mesh["vertices"].tolist())
        uvs.extend(mesh["uvs"].tolist())
        normals.extend(mesh["normals"].tolist())
        for material, part in mesh["faces"].items():
            faces.setdefault(material, []).extend([[a + offset, b + offset, c + offset] for a, b, c in part])

    sites = []
    for index, member in enumerate(members):
        _, xy, half, unit_left, unit_right = shapes[index]
        add(build_bridge_mesh(
            member["coords"], member["width"], ground_at, member["deck_material"], pier_material, railing_material,
            deck_thickness=deck_thickness, curb_width=curb_width, curb_height=curb_height, railing_height=railing_height,
            railing_post_spacing=railing_post_spacing, railing_post_size=railing_post_size, tile_m=tile_m,
            road_texture_length=road_texture_length, widths=member.get("widths"),
            joined_left=joined_side(index, unit_left), joined_right=joined_side(index, unit_right),
            cap_start=not end_is_closed(index, 0), cap_end=not end_is_closed(index, -1), piers=False,
        ))
        bottom = np.array(member["coords"], dtype=float)[:, 2] - deck_thickness
        for site in _pier_sites(xy, bottom, half, ground_at, pier_spacing, min_pier_clearance, pier_width_fraction):
            if all(np.hypot(site[0] - other[0], site[1] - other[1]) >= pier_spacing / 2.0 for other in sites):
                sites.append(site)
    pier_builder = MeshBuilder()
    for site in sites:
        _add_pier(pier_builder, site, pier_burial, pier_depth_fraction, tile_m)
    add({"vertices": np.array(pier_builder.vertices, dtype=float).reshape(-1, 3),
         "uvs": np.array(pier_builder.uvs, dtype=float).reshape(-1, 2),
         "normals": np.array(pier_builder.normals, dtype=float).reshape(-1, 3), "faces": {pier_material: pier_builder.faces}})
    return {
        "vertices": np.array(vertices, dtype=float), "uvs": np.array(uvs, dtype=float),
        "normals": np.array(normals, dtype=float), "faces": faces,
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
    optional "widths": width per coordinate, see build_bridge_mesh()). Bridges with the same optional "group" become
    ONE structure (build_bridge_group_mesh(), id "bridge_group_<group>")."""
    meshes = []
    groups: Dict[object, List[Dict]] = {}
    for bridge in bridges:
        if bridge.get("group") is not None and len(bridge["coords"]) >= 2:
            groups.setdefault(bridge["group"], []).append(bridge)
    for group, members in groups.items():
        mesh = build_bridge_group_mesh(
            members, ground_at, pier_material, railing_material,
            deck_thickness=deck_thickness, pier_spacing=pier_spacing,
            pier_width_fraction=pier_width_fraction, pier_depth_fraction=pier_depth_fraction, pier_burial=pier_burial,
            min_pier_clearance=min_pier_clearance,
            curb_width=curb_width, curb_height=curb_height, railing_height=railing_height,
            railing_post_spacing=railing_post_spacing, railing_post_size=railing_post_size,
            road_texture_length=road_texture_length,
        )
        meshes.append({"id": f"bridge_group_{group}", **mesh})
    for bridge in bridges:
        coords = bridge["coords"]
        if len(coords) < 2 or bridge.get("group") is not None:
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
