"""
Tunnel from OSM lines (highway=* with tunnel=yes/culvert/building_passage): circular tube - standard
tunnel profile, a 240° circular arc over a flat floor chord (road surface); the remaining 120° lie below the
chord and are not modeled (invisible invert) - along the linearly interpolated elevation profile (see
geometry/road_structures.py + geometry/polygon.py), with a portal structure at both ends (see
tunnels/tunnel_portal.py). The terrain above the tube and at the portal is shaped by terrain/tunnel_terrain.py.

In OSM a tunnel can consist of several consecutive ways (junction detection no longer splits tunnels,
see geometry/junctions.py::build_junction_network()). chain_tunnel_pieces() joins such pieces into
ONE continuous tube - otherwise each piece would get its own portals in the middle of the mountain.
"""

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..walls.mesh_parts import MeshBuilder, offset_points

ARC_SPAN_DEG = 240.0  # circular arc over the road surface
ARC_START_DEG = -30.0  # start angle (right floor edge), standard circle convention (0°=+x, CCW)
JOINT_TOLERANCE = 0.05  # how close two piece ends must come to count as a joint, in meters


def tunnel_radius(width: float) -> float:
    """Radius of the circular tunnel tube from the floor width (floor chord = sqrt(3)*R for the 240°/120° split)."""
    return width / math.sqrt(3.0)


def tunnel_crown_height(width: float) -> float:
    """Clear height (floor to crown apex) of a circular tunnel tube of the given floor width."""
    return 1.5 * tunnel_radius(width)


def arc_cross_section(radius: float, segments: int) -> List[Tuple[float, float]]:
    """
    (across, height) points of the 240° circular arc over the road surface, `segments` strips (segments+1 points), from
    the right floor edge (θ=-30°) over the crown (θ=90°) to the left floor edge (θ=210°). The floor is y=0, "across"
    is across the direction of travel (positive = right). The circle center is at (0, radius/2):
    a 240° arc whose chord at y=0 has half-width radius*cos(30°).
    """
    points = []
    for k in range(segments + 1):
        theta = math.radians(ARC_START_DEG + (k / segments) * ARC_SPAN_DEG)
        points.append((radius * math.cos(theta), radius / 2.0 + radius * math.sin(theta)))
    return points


def shell_cross_section(radius: float, segments: int, thickness: float) -> List[Tuple[float, float]]:
    """
    Outer contour (across, height) of the tube shell: 240° arc with radius radius + thickness around the same
    center (0, radius/2) as the inner arc, closed at the bottom by a floor slab `thickness` below the
    road surface. Winding: right arc foot over the crown to the left arc foot, then bottom left, bottom right.
    """
    outer = radius + thickness
    points = []
    for k in range(segments + 1):
        theta = math.radians(ARC_START_DEG + (k / segments) * ARC_SPAN_DEG)
        points.append((outer * math.cos(theta), radius / 2.0 + outer * math.sin(theta)))
    points.append((points[-1][0], -thickness))
    points.append((points[0][0], -thickness))
    return points


def resample_tunnel_coords(coords: Sequence[Tuple[float, float, float]], step: float) -> List[Tuple[float, float, float]]:
    """Thins out the (already linearly profiled) centerline to a fixed arc-length spacing (XYZ together,
    since the elevation profile is affine in arc length - see geometry/polygon.py::apply_structure_elevation_profiles()).
    Keeps the vertex count in check even for very long tunnels (e.g. 16.9 km)."""
    arr = np.array(coords, dtype=float)
    if len(arr) < 2:
        return list(coords)
    diffs = np.diff(arr[:, :2], axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < step:
        return list(coords)
    samples = np.linspace(0.0, total, max(2, int(np.ceil(total / step)) + 1))
    x = np.interp(samples, cum, arr[:, 0])
    y = np.interp(samples, cum, arr[:, 1])
    z = np.interp(samples, cum, arr[:, 2])
    return list(zip(x.tolist(), y.tolist(), z.tolist()))


def chain_tunnel_pieces(tunnels: Sequence[Dict]) -> List[Dict]:
    """
    Joins tunnel pieces that meet at an end point into continuous chains.

    Chaining only happens at unambiguous joints: exactly two matching piece ends (same width, same
    floor material) at the same point. Tunnels of a different kind at the same point do not count - e.g. a footway
    tunnel that branches off at the same OSM node. The
    direction of travel of individual pieces is reversed where needed.

    Args:
        tunnels: [{"id", "coords", "width", "floor_material"}, ...]

    Returns:
        [{"id" (of the first piece), "coords", "width", "floor_material"}, ...]
    """
    pieces = [t for t in tunnels if len(t["coords"]) >= 2]
    if not pieces:
        return []

    # Piece ends closer than JOINT_TOLERANCE form a joint (clipping at the map border
    # shifts end points by millimeters) - groups via union-find over all nearby pairs.
    end_refs = [(index, at_start) for index in range(len(pieces)) for at_start in (True, False)]
    end_xy = np.array([pieces[i]["coords"][0 if s else -1][:2] for i, s in end_refs], dtype=float)
    group = list(range(len(end_refs)))

    def find(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    for a, b in cKDTree(end_xy).query_pairs(JOINT_TOLERANCE):
        group[find(a)] = find(b)
    joint_of = {ref: find(i) for i, ref in enumerate(end_refs)}

    def key(index: int, at_start: bool) -> Tuple:
        return (joint_of[(index, at_start)], pieces[index]["width"], pieces[index]["floor_material"])

    ends = defaultdict(list)
    for index, at_start in end_refs:
        ends[key(index, at_start)].append((index, at_start))

    def partner(index: int, at_start: bool):
        joined = ends[key(index, at_start)]
        if len(joined) != 2:
            return None
        other = joined[0] if joined[1] == (index, at_start) else joined[1]
        if other[0] == index:
            return None  # piece closes on itself to form a ring
        return other

    visited = set()
    chains = []
    for first in range(len(pieces)):
        if first in visited:
            continue
        # Walk backwards to the free start of the chain (guarded against rings).
        head, head_at_start = first, True
        seen = {first}
        while True:
            prev = partner(head, head_at_start)
            if prev is None or prev[0] in seen:
                break
            head, head_at_start = prev[0], not prev[1]
            seen.add(head)

        # Forward: orient each piece so that it starts at the joint with its predecessor.
        coords: List[Tuple[float, float, float]] = []
        current, entry_at_start = head, head_at_start
        while current is not None and current not in visited:
            visited.add(current)
            piece_coords = [tuple(map(float, p)) for p in pieces[current]["coords"]]
            if not entry_at_start:
                piece_coords.reverse()
            coords.extend(piece_coords if not coords else piece_coords[1:])
            nxt = partner(current, not entry_at_start)
            if nxt is None:
                break
            current, entry_at_start = nxt[0], nxt[1]

        base = pieces[head]
        chains.append({"id": base["id"], "coords": coords, "width": base["width"], "floor_material": base["floor_material"]})
    return chains


def build_tunnel_mesh(
    coords: Sequence[Tuple[float, float, float]],
    width: float,
    floor_material: str,
    wall_material: str,
    arc_segments: int = 12,
    tile_m: float = 5.0,
    shell_thickness: float = 0.0,
    shell_material: str = None,
    cap_start: bool = True,
    cap_end: bool = True,
    tilt_start: float = 0.0,
    tilt_end: float = 0.0,
) -> Dict:
    """
    Tube mesh (floor + circular 240° arc above it) along `coords` (already the tunnel elevation profile).
    Radius and crown height follow from `width` (see tunnel_radius()/tunnel_crown_height()).

    With shell_thickness > 0 the tube gets an outer shell (see shell_cross_section()) including end rings at
    both ends: it is then a solid cylinder from the outside too and may stand freely in the terrain. cap_start/
    cap_end = False omits the end ring (a portal structure stands in the same plane there - otherwise z-fighting).
    tilt_start/tilt_end (tan of the tilt angle): the end face there is tilted toward the mountain side - a point at
    height h above the floor moves by h * tilt into the tube (the floor stays on the portal plane, see _end_shift()).

    The cross-section rings sit at the centerline points and are mitered there (like the floor edges from
    offset_points()): adjacent segments share exactly the same ring, so the tube is watertight even in curves.

    Returns:
        {"vertices", "uvs", "normals", "faces": {floor_material: [...], wall_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    floor_z = points[:, 2]
    radius = tunnel_radius(width)
    arc = arc_cross_section(radius, arc_segments)

    left, right = offset_points(xy, width / 2.0, closed=False)
    # Miter vector per centerline point (incl. miter extension), points to the right of the direction of travel
    miter_right = (right - xy) / (width / 2.0)
    rings = np.empty((len(points), arc_segments + 1, 3))
    for k, (across, height) in enumerate(arc):
        rings[:, k, :2] = xy + miter_right * across
        rings[:, k, 2] = floor_z + height
    rings[:, 0, :2] = right  # floor edges exactly like the floor mesh (no rounding gap)
    rings[:, arc_segments, :2] = left
    shift = _end_shift(xy, tilt_start, tilt_end)
    rings[:, :, :2] += shift[:, None, :] * (rings[:, :, 2:3] - floor_z[:, None, None])

    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(seg_len)]) / tile_m
    across_floor = width / tile_m
    across_arc = (radius * math.radians(ARC_SPAN_DEG)) / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    floor_builder = MeshBuilder()
    wall_builder = MeshBuilder()
    for i in range(len(points) - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy[j] - xy[i]
        direction = direction / np.linalg.norm(direction)
        perp_right = np.array([direction[1], -direction[0]])  # points "right" of the direction of travel

        # Floor (normal pointing up, into the tube interior)
        floor_builder.quad(
            [p3(left[i], floor_z[i]), p3(left[j], floor_z[j]), p3(right[j], floor_z[j]), p3(right[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across_floor], [u0, across_floor]],
            [0.0, 0.0, 1.0],
        )

        # Circular arc (240°) over the road surface, in arc_segments strips
        for k in range(arc_segments):
            theta_mid = math.radians(ARC_START_DEG + ((k + 0.5) / arc_segments) * ARC_SPAN_DEG)
            inward = [-math.cos(theta_mid) * perp_right[0], -math.cos(theta_mid) * perp_right[1], -math.sin(theta_mid)]
            v0 = (k / arc_segments) * across_arc
            v1 = ((k + 1) / arc_segments) * across_arc
            wall_builder.quad(
                [rings[i, k].tolist(), rings[j, k].tolist(), rings[j, k + 1].tolist(), rings[i, k + 1].tolist()],
                [[u0, v0], [u1, v0], [u1, v1], [u0, v1]],
                inward,
            )

    builders = [(floor_material, floor_builder), (wall_material, wall_builder)]
    if shell_thickness > 0.0:
        builders.append((shell_material or wall_material, _build_shell(xy, floor_z, miter_right, shift, radius, arc_segments, shell_thickness, tile_m, cap_start, cap_end)))

    vertices, uvs, normals, faces = [], [], [], {}
    for material, builder in builders:
        offset = len(vertices)
        vertices += builder.vertices
        uvs += builder.uvs
        normals += builder.normals
        faces.setdefault(material, []).extend([[a + offset, b + offset, c + offset] for a, b, c in builder.faces])

    return {
        "vertices": np.array(vertices, dtype=float),
        "uvs": np.array(uvs, dtype=float),
        "normals": np.array(normals, dtype=float),
        "faces": faces,
    }


def _end_shift(xy: np.ndarray, tilt_start: float, tilt_end: float) -> np.ndarray:
    """Horizontal shift per meter of height above the floor, per centerline point: at a tilted end tilt times the
    unit vector into the tube interior (at most so far that the crown stays in front of the neighboring ring), else 0."""
    shift = np.zeros_like(xy)
    for index, neighbour, tilt in ((0, 1, tilt_start), (len(xy) - 1, len(xy) - 2, tilt_end)):
        if tilt > 0.0 and len(xy) >= 2:
            inward = xy[neighbour] - xy[index]
            shift[index] = inward / np.linalg.norm(inward) * tilt
    return shift


def _build_shell(xy, floor_z, miter_right, shift, radius, arc_segments, thickness, tile_m, cap_start=True, cap_end=True) -> MeshBuilder:
    """Outer shell of the tube (jacket along the axis, normals pointing outward) plus end ring at the requested ends."""
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon

    profile = shell_cross_section(radius, arc_segments, thickness)
    center = np.array([0.0, radius / 2.0])
    builder = MeshBuilder()

    def world(i, across, height):
        x = xy[i, 0] + miter_right[i, 0] * across + shift[i, 0] * height
        y = xy[i, 1] + miter_right[i, 1] * across + shift[i, 1] * height
        return [float(x), float(y), float(floor_z[i] + height)]

    edge_len = [math.dist(profile[k], profile[(k + 1) % len(profile)]) for k in range(len(profile))]
    perimeter = np.concatenate([[0.0], np.cumsum(edge_len)]) / tile_m
    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(seg_len)]) / tile_m
    for i in range(len(xy) - 1):
        j = i + 1
        direction = (xy[j] - xy[i]) / np.linalg.norm(xy[j] - xy[i])
        perp_right = np.array([direction[1], -direction[0]])
        for k in range(len(profile)):
            a, b = np.array(profile[k]), np.array(profile[(k + 1) % len(profile)])
            normal_2d = np.array([b[1] - a[1], a[0] - b[0]])
            normal_2d /= np.linalg.norm(normal_2d)
            if normal_2d @ ((a + b) / 2.0 - center) < 0.0:
                normal_2d = -normal_2d
            builder.quad(
                [world(i, *a), world(j, *a), world(j, *b), world(i, *b)],
                [[along[i], perimeter[k]], [along[j], perimeter[k]], [along[j], perimeter[k + 1]], [along[i], perimeter[k + 1]]],
                [float(perp_right[0] * normal_2d[0]), float(perp_right[1] * normal_2d[0]), float(normal_2d[1])],
            )

    # End rings: outer contour minus clear cross-section, facing outward (away from the tunnel)
    ring = Polygon(profile).difference(Polygon(arc_cross_section(radius, arc_segments)))
    triangles = [list(t.exterior.coords)[:3] for t in constrained_delaunay_triangles(ring).geoms]
    for index, sign, cap in ((0, -1.0, cap_start), (len(xy) - 1, 1.0, cap_end)):
        if not cap:
            continue
        neighbour = 1 if index == 0 else index - 1
        axis = (xy[index] - xy[neighbour]) if index else (xy[neighbour] - xy[index])
        axis = axis / np.linalg.norm(axis)
        # tilted end face: normal pointing outward and up by the tilt angle
        tilt = float(np.linalg.norm(shift[index]))
        cos, sin = 1.0 / math.hypot(1.0, tilt), tilt / math.hypot(1.0, tilt)
        normal = [float(sign * axis[0] * cos), float(sign * axis[1] * cos), float(sin)]
        for tri in triangles:
            builder.triangle([world(index, c, h) for c, h in tri], [[c / tile_m, h / tile_m] for c, h in tri], normal)
    return builder


def build_tunnels(
    plans: Sequence[Dict], wall_material: str, portal_material: str, arc_segments: int = 12, transition_cover: float = 0.2
) -> List[Dict]:
    """
    Mesh dicts for the DAE export: per tunnel chain the tube (with outer shell from plan["shell"], material like the
    portals) plus one portal structure per open end.

    Args:
        plans: result of tunnel_portal.plan_tunnels() - portals with "top_z"/"bottom_z" already set
            (see terrain/tunnel_terrain.py::shape_terrain_for_tunnels())
        transition_cover: thickness of the solid cover slabs at the transition into a gallery, in meters
    """
    from .tunnel_portal import build_portal_block_mesh

    meshes = []
    for plan in plans:
        # Own portal structure only for collars with overhang or a gallery transition (faces between arch and gallery
        # cross-section). Without a collar the end ring of the tube is the portal (same wall thickness as the tube); with
        # a collar it is omitted (same plane as the collar end face -> z-fighting).
        open_portals = [p for p in plan["portals"] if p.get("open", True)]
        structures = [p for p in open_portals if p.get("kind") == "gallery" or p.get("collar", 0.0) > 0.0]
        collared = [p for p in open_portals if p.get("collar", 0.0) > 0.0]
        start, end = plan["portals"]
        tube = build_tunnel_mesh(
            plan["coords"], plan["tube_width"], plan["floor_material"], wall_material, arc_segments=arc_segments,
            shell_thickness=plan.get("shell", 0.0), shell_material=portal_material,
            cap_start=not any(p is start for p in collared), cap_end=not any(p is end for p in collared),
            tilt_start=start.get("tilt", 0.0) if start.get("open", True) else 0.0,
            tilt_end=end.get("tilt", 0.0) if end.get("open", True) else 0.0,
        )
        meshes.append({"id": f"tunnel_{plan['id']}", **tube})
        for portal in structures:
            block = build_portal_block_mesh(portal, portal_material, arc_segments=arc_segments, cover_thickness=transition_cover)
            meshes.append({"id": f"tunnel_{plan['id']}_portal_{portal['label']}", **block})
    return meshes
