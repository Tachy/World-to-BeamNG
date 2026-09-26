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
    joined_left: Optional[Sequence[float]] = None,
    joined_right: Optional[Sequence[float]] = None,
    railing_left: bool = True,
    railing_right: bool = True,
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

    Parts of a bridge that splits (see build_bridge_group_mesh()): joined_left / joined_right give per coordinate how
    far the deck reaches beyond the carriageway edge to meet the neighbouring part (half the gap between the two
    carriageways, carriageway material, no curb), NaN where the side is free and has its curb as usual (the cut);
    railing_left / railing_right = False leaves out the railing of that side; cap_start / cap_end = False
    leaves out the end face where the part continues in another one; piers=False leaves the piers to the caller.

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
    reach = {
        side: np.full(count, np.nan) if value is None else np.asarray(value, dtype=float)
        for side, value in (("left", joined_left), ("right", joined_right))
    }
    joined = {side: np.isfinite(reach[side]) for side in reach}
    railing_on = {"left": railing_left, "right": railing_right}

    # inner = carriageway edge (= curb inner face), curb_edge = curb outer edge, surface = edge of the road surface
    # (reaches to the neighbouring part where joined), outer = deck slab edge
    half = (np.full(count, width) if widths is None else np.asarray(widths, dtype=float)) / 2.0
    unit = {}
    unit["left"], unit["right"] = offset_points(xy, 1.0, closed=False)  # the miter offset is linear in the distance

    def offset_by(side, distance):
        return xy + (unit[side] - xy) * distance[:, None]

    inner = {side: offset_by(side, half) for side in ("left", "right")}
    curb_edge = {side: offset_by(side, half + curb_width) for side in ("left", "right")}
    surface = {side: offset_by(side, half + np.where(joined[side], reach[side], 0.0)) for side in ("left", "right")}
    outer = {side: np.where(joined[side][:, None], surface[side], curb_edge[side]) for side in ("left", "right")}
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
        left_normal = [float(-direction[1]), float(direction[0]), 0.0]
        right_normal = [-left_normal[0], -left_normal[1], 0.0]

        # Carriageway (top side, between the curbs - up to the neighbouring part where joined)
        deck_builder.quad(
            [p3(surface["left"][i], top[i]), p3(surface["left"][j], top[j]), p3(surface["right"][j], top[j]), p3(surface["right"][i], top[i])],
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
            if joined[side][i] or joined[side][j]:
                continue  # meets the neighbouring part: no fascia, no curb (it begins at the first free point)
            edge_in, edge_out = inner[side], curb_edge[side]
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
                [-normal[0], -normal[1], 0.0],
            )

    # End faces at both ends (deck full height + curb top on the free sides)
    for index, sign, neighbour, wanted in ((0, -1.0, 1, cap_start), (len(points) - 1, 1.0, len(points) - 2, cap_end)):
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
                [p3(curb_edge[side][index], top[index]), p3(inner[side][index], top[index]), p3(inner[side][index], curb_top[index]), p3(curb_edge[side][index], curb_top[index])],
                [[0.0, 0.0], [curb_across, 0.0], [curb_across, curb_height / tile_m], [0.0, curb_height / tile_m]],
                face_normal,
            )

    # Where the cut begins (or ends) along the way: end face of the curb, facing the joined stretch
    for side in ("left", "right"):
        for k in range(count - 1):
            if joined[side][k] == joined[side][k + 1]:
                continue
            at = k + 1 if joined[side][k] else k  # the first free point after / the last free point before the joined stretch
            direction = xy[k + 1] - xy[k]
            direction = direction / np.linalg.norm(direction)
            sign = -1.0 if joined[side][k] else 1.0
            pier_builder.quad(
                [p3(curb_edge[side][at], top[at]), p3(inner[side][at], top[at]), p3(inner[side][at], curb_top[at]), p3(curb_edge[side][at], curb_top[at])],
                [[0.0, 0.0], [curb_across, 0.0], [curb_across, curb_height / tile_m], [0.0, curb_height / tile_m]],
                [float(sign * direction[0]), float(sign * direction[1]), 0.0],
            )

    total_len = float(cum[-1])
    if piers:
        for site in _pier_sites(xy, bottom, half, ground_at, pier_spacing, min_pier_clearance, pier_width_fraction):
            _add_pier(pier_builder, site, pier_burial, pier_depth_fraction, tile_m)

    # Railing: posts + continuous handrail on the free sides, centered on the curb (the mean of two offset_points()
    # results on the same normal equals an offset by the averaged distance)
    railing_builder = MeshBuilder()
    post_positions = np.arange(0.0, total_len + 1e-6, railing_post_spacing) if total_len > 0 else np.array([])
    for side in ("left", "right"):
        free = np.flatnonzero(~joined[side])
        if not railing_on[side] or len(free) < 2:
            continue
        first, last = int(free[0]), int(free[-1])  # the free stretch (a cut side is joined only towards the node)
        edge_xy = (inner[side] + curb_edge[side]) / 2.0
        rail_top = curb_top + railing_height + railing_post_size / 2.0
        for s in post_positions:
            if s < cum[first] - 1e-6 or s > cum[last] + 1e-6:
                continue
            px, py = _interp_at(cum, edge_xy, s)
            post_bottom_z = float(_interp_at(cum, curb_top, s))
            post_top_z = post_bottom_z + railing_height
            add_box_column(railing_builder, px, py, post_bottom_z, post_top_z, railing_post_size, tile_m, direction=_direction_at(cum, edge_xy, s))
        beam = _build_edge_beam(edge_xy[first:last + 1], rail_top[first:last + 1], railing_post_size, tile_m)
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


def _densified(member: Dict, step: float) -> Dict:
    """The member with a point at least every `step` meters (coords and widths interpolated along the arc length)."""
    points = np.array(member["coords"], dtype=float)
    cum = _arc_length(points[:, :2])
    if cum[-1] <= 0.0:
        return member
    stations = np.unique(np.concatenate([cum, np.arange(0.0, cum[-1], step)]))
    result = {**member, "coords": [tuple(float(v) for v in c) for c in np.column_stack([np.interp(stations, cum, points[:, k]) for k in range(3)])]}
    if member.get("widths") is not None:
        result["widths"] = np.interp(stations, cum, np.asarray(member["widths"], dtype=float))
    return result


def _cut_off_stem(member: Dict, node: np.ndarray, axis: np.ndarray, normal: np.ndarray, length: float, half_width: float):
    """
    The part of `member` beyond the stem (the rectangle `length` meters along `axis` from `node`, `half_width` to both
    sides), with the end that now touches the stem ("start"/"end", None if the member does not reach into it) - or
    (None, "all") if it lies completely inside. Also returns the (along, z) samples of its points inside the stem.
    """
    member = _densified(member, 1.0)
    points = np.array(member["coords"], dtype=float)
    rel = points[:, :2] - node
    along, lateral = rel @ axis, rel @ normal
    inside = (along > -1e-3) & (along < length - 1e-6) & (np.abs(lateral) <= half_width + 0.5)
    samples = [(float(a), float(z)) for a, z in zip(along[inside], points[inside, 2])]
    if along.max() <= 1e-3 or not inside.any():
        return member, None, []  # the trunk (before the node) or a way elsewhere
    if along.max() < length - 1e-6:
        return None, "all", samples  # ends inside the stem
    widths = member.get("widths")
    widths = None if widths is None else np.asarray(widths, dtype=float)
    reversed_member = bool(along[-1] < along[0])  # digitized towards the node: cut at its end
    if reversed_member:
        points, along = points[::-1], along[::-1]
        widths = None if widths is None else widths[::-1]
    k = int(np.argmax(along >= length))  # first point beyond the stem
    t = (length - along[k - 1]) / max(along[k] - along[k - 1], 1e-9) if k > 0 else 0.0
    cut_point = points[k - 1] + t * (points[k] - points[k - 1]) if k > 0 else points[0]
    duplicate = float(np.linalg.norm(points[k, :2] - cut_point[:2])) < 1e-6
    kept = points[k:] if duplicate else np.vstack([cut_point, points[k:]])
    kept_widths = None
    if widths is not None:
        cut_width = widths[k - 1] + t * (widths[k] - widths[k - 1]) if k > 0 else widths[0]
        kept_widths = widths[k:] if duplicate else np.concatenate([[cut_width], widths[k:]])
    if reversed_member:
        kept = kept[::-1]
        kept_widths = None if kept_widths is None else kept_widths[::-1]
    result = {**member, "coords": [tuple(float(v) for v in p) for p in kept]}
    if widths is not None:
        result["widths"] = kept_widths
    return result, ("end" if reversed_member else "start"), samples


def _carriageway_outline(coords, width, widths):
    """Carriageway polygon (flat ends) of a way with its width profile."""
    from shapely.geometry import Polygon

    points = np.array(coords, dtype=float)
    half = (np.full(len(points), float(width)) if widths is None else np.asarray(widths, dtype=float)) / 2.0
    left, right = offset_points(points[:, :2], 1.0, closed=False)
    xy = points[:, :2]
    return Polygon(np.vstack([xy + (left - xy) * half[:, None], (xy + (right - xy) * half[:, None])[::-1]])).buffer(0)


def _join_neighbours(parts, curb_width: float) -> None:
    """
    Sets joined_left / joined_right (see build_bridge_mesh()) of every part: where the carriageway of another part is
    closer than 2x curb_width to a side, there is no room for the two curbs yet - the side reaches to the middle of the
    gap (the deck is still one piece, only wider); from there on the side is free and has its curb (the cut). A side
    that meets another part anywhere gets no railing for now.
    """
    from shapely.geometry import LineString, Point

    # shrunk by 1 cm: a part that only continues this one end to end (the stem and its trunk or branches) is not beside it
    outlines = [_carriageway_outline(coords, width, widths).buffer(-0.01) for coords, width, widths, _, _ in parts]
    for index, (coords, width, widths, _, flags) in enumerate(parts):
        others = [outline for k, outline in enumerate(outlines) if k != index]
        if not others:
            continue
        points = np.array(coords, dtype=float)
        xy = points[:, :2]
        half = (np.full(len(xy), float(width)) if widths is None else np.asarray(widths, dtype=float)) / 2.0
        units = dict(zip(("left", "right"), offset_points(xy, 1.0, closed=False)))
        for side, unit in units.items():
            outward = unit - xy
            outward /= np.maximum(np.linalg.norm(outward, axis=1), 1e-9)[:, None]
            edge = xy + outward * half[:, None]
            reach = np.full(len(xy), np.nan)
            for k, (point, direction) in enumerate(zip(edge, outward)):
                ray = LineString([point, point + direction * 2.0 * curb_width])  # across the road, as far as two curbs
                hits = [ray.intersection(o) for o in others if ray.intersects(o)]
                if hits:
                    reach[k] = min(Point(*point).distance(hit) for hit in hits) / 2.0
            # an end that meets the next part (the cut line) follows its neighbour point - its corner may already
            # reach a little into the part before
            for end, neighbour, key in ((0, 1, "cap_start"), (-1, -2, "cap_end")):
                if flags.get(key) is False and len(reach) > 1:
                    reach[end] = reach[neighbour]
            if np.isfinite(reach).any():
                flags[f"joined_{side}"] = reach
                flags[f"railing_{side}"] = False


def build_bridge_group_mesh(
    members: Sequence[Dict],
    ground_at: HeightAt,
    pier_material: str,
    railing_material: str,
    stem: Optional[Dict] = None,
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
    ONE bridge for a lane split on a bridge (see geometry/lane_splits.py): the trunk's deck goes on past the node as one
    box of the trunk's width for stem["hold"] meters (the stem: {"node", "axis", "left_normal", "hold", "width",
    "deck_material"}), with curb and railing on its outer sides only. There it is cut like with a knife: each branch
    continues as its own deck, without curb and railing along the cut for now (the slab ends at its carriageway edge
    there) and with them on its outer side. The parts of the branches inside the stem are left out. Without a stem the members are
    built as separate bridges without end faces where they meet. Piers: one of two closer than pier_spacing / 2.
    """
    kwargs = dict(
        deck_thickness=deck_thickness, curb_width=curb_width, curb_height=curb_height, railing_height=railing_height,
        railing_post_spacing=railing_post_spacing, railing_post_size=railing_post_size, tile_m=tile_m,
        road_texture_length=road_texture_length, piers=False,
    )
    parts = []  # (coords, width, widths, deck_material, flags)
    if stem is not None:
        node = np.asarray(stem["node"], dtype=float)
        axis = np.asarray(stem["axis"], dtype=float)
        normal = np.asarray(stem["left_normal"], dtype=float)
        length, half_width = float(stem["hold"]), float(stem["width"]) / 2.0
        samples = []
        kept = []
        for member in members:
            rest, cut_end, inside = _cut_off_stem(member, node, axis, normal, length, half_width)
            samples += inside
            if rest is not None and len(rest["coords"]) >= 2:
                kept.append((rest, cut_end))
        reach = max((a for a, _ in samples), default=0.0)
        if samples and reach > 1.0:
            # stem heights from the branches running in it (their straight, held stretch)
            samples.sort()
            s_values = np.array([a for a, _ in samples])
            z_values = np.array([z for _, z in samples])
            stations = np.linspace(0.0, length, max(2, int(np.ceil(length)) + 1))
            stem_xy = node[None, :] + stations[:, None] * axis[None, :]
            stem_z = np.interp(stations, s_values, z_values)
            stem_part = ([(float(x), float(y), float(z)) for (x, y), z in zip(stem_xy, stem_z)], float(stem["width"]),
                         None, stem["deck_material"], {"cap_start": False, "cap_end": False})
        for rest, cut_end in kept:
            flags = {}
            if cut_end is not None:
                flags["cap_start" if cut_end == "start" else "cap_end"] = False
            parts.append((rest["coords"], rest["width"], rest.get("widths"), rest["deck_material"], flags))
        _join_neighbours(parts, curb_width)  # the stem's sides are outer sides: it takes no part in that
        if samples and reach > 1.0:
            parts.append(stem_part)
        # The trunk end at the node continues in the stem: no end face there
        for index, (coords, _, _, _, flags) in enumerate(parts):
            ends = np.array([coords[0][:2], coords[-1][:2]], dtype=float)
            for end, key in ((0, "cap_start"), (1, "cap_end")):
                if key not in flags and np.linalg.norm(ends[end] - node) <= endpoint_tol:
                    flags[key] = False
    else:
        for member in members:
            parts.append((member["coords"], member["width"], member.get("widths"), member["deck_material"], {}))
        for index, (coords, _, _, _, flags) in enumerate(parts):
            for end, key in ((0, "cap_start"), (-1, "cap_end")):
                point = np.asarray(coords[end][:2], dtype=float)
                if any(k != index and min(np.linalg.norm(point - np.asarray(c[e][:2], dtype=float)) for e in (0, -1)) <= endpoint_tol
                       for k, (c, *_rest) in enumerate(parts)):
                    flags[key] = False

    vertices, uvs, normals, faces = [], [], [], {}

    def add(mesh):
        offset = len(vertices)
        vertices.extend(np.asarray(mesh["vertices"]).tolist())
        uvs.extend(np.asarray(mesh["uvs"]).tolist())
        normals.extend(np.asarray(mesh["normals"]).tolist())
        for material, part in mesh["faces"].items():
            faces.setdefault(material, []).extend([[a + offset, b + offset, c + offset] for a, b, c in part])

    sites = []
    for coords, width, widths, deck_material, flags in parts:
        add(build_bridge_mesh(coords, width, ground_at, deck_material, pier_material, railing_material, widths=widths,
                              **kwargs, **flags))
        points = np.array(coords, dtype=float)
        half = (np.full(len(points), width) if widths is None else np.asarray(widths, dtype=float)) / 2.0
        for site in _pier_sites(points[:, :2], points[:, 2] - deck_thickness, half, ground_at, pier_spacing,
                                min_pier_clearance, pier_width_fraction):
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
    ONE structure (build_bridge_group_mesh() with the optional "stem" of a member, id "bridge_group_<group>")."""
    meshes = []
    groups: Dict[object, List[Dict]] = {}
    for bridge in bridges:
        if bridge.get("group") is not None and len(bridge["coords"]) >= 2:
            groups.setdefault(bridge["group"], []).append(bridge)
    for group, members in groups.items():
        stem = next((m["stem"] for m in members if m.get("stem")), None)
        mesh = build_bridge_group_mesh(
            members, ground_at, pier_material, railing_material, stem=stem,
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
