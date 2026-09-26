"""
Portal at both ends of a tunnel tube, at the portal plane (OSM tunnel end, where the approach road or the
gallery connects): a concrete collar that is rectangular on the outside, TUNNEL_PORTAL_COLLAR_RATIO times the tube
diameter thick at its thinnest point (left, right, top), TUNNEL_PORTAL_LENGTH meters deep (without a collar: the end
ring of the tube shell). At tunnel entrances the end face is tilted toward the mountain side by
TUNNEL_PORTAL_TILT_DEG; at the transition into a gallery it is vertical, plus the faces between the tube arch and
the gallery cross-section (transition_regions()).

The portal covers the terrain holes without which the heightmap would block the opening: a grid cell that
spans the portal plane has road level at the front and cover height at the back - its sloped face would run across
the opening. terrain/tunnel_terrain.py therefore keeps the terrain at floor level up to TUNNEL_PORTAL_FLAT_DEPTH
behind the portal plane (hidden under the tube floor), removes the slope behind it in the footprint of the portal down
to just below its round outer contour, and turns the cells at the transition into holes. The size of the portal
depends only on the tube, never on the terrain.
"""

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder
from .tunnel_mesh import arc_cross_section, chain_tunnel_pieces, resample_tunnel_coords, tunnel_profile


def plan_tunnels(
    tunnels: Sequence[Dict],
    segment_step: float,
    flat_depth: float,
    length: float,
    galleries: Sequence[Dict] = None,
    gallery_height: float = 5.0,
    gallery_roof_thickness: float = 0.5,
    gallery_wall_thickness: float = 5.0,
    transition_tol: float = 0.5,
    shell_ratio: float = 0.0,
    tilt_deg: float = 0.0,
    collar_ratio: float = 0.0,
    collar_min_side: float = 0.0,
    curb_width: float = 0.4,
    curb_height: float = 0.2,
    edge_height: float = 4.2,
    max_arc_deg: float = 240.0,
) -> List[Dict]:
    """
    Chains the tunnel pieces (see tunnel_mesh.chain_tunnel_pieces()), thins out the centerline and fixes the
    two portals of each chain. Tube (tunnel_mesh.py), terrain (terrain/tunnel_terrain.py) and portal
    all work on this plan.

    Args:
        tunnels: [{"id", "coords", "width", "floor_material"}, ...] - "width" = carriageway width
        curb_width, curb_height, edge_height, max_arc_deg: tube cross-section, see tunnel_mesh.tunnel_profile()
        shell_ratio: wall thickness of the tube shell (tunnel_mesh.shell_cross_section()) relative to the
            tube diameter - smaller tunnels get thinner walls
        collar_ratio: portal collar, rectangular on the outside: wall thickness at the thinnest point (left, right,
            top) relative to the tube diameter; 0 = no collar, the end ring of the tube is the portal
        collar_min_side: collar at least this thick on the left/right (hole cells at the portal step reach up to one
            grid diagonal sideways beyond the tube radius and must stay hidden inside it), in meters
        length: depth of the collar into the mountain, in meters
        tilt_deg: tunnel entrances (not transitions into a gallery): end face tilted toward the mountain side by this
            many degrees (portal["tilt"] = tan). The flat portal zone then extends at least to behind the end face at
            the outer crown, so that the opening stays clear and the hole edge stays behind the sloped face.
        galleries: gallery inputs (as for build_galleries()); a portal that is at most transition_tol from an
            END POINT of a gallery centerline is a tunnel -> gallery transition (portal["kind"] ==
            "gallery"): same round portal as at a tunnel entrance, plus the faces between the tube arch and the
            gallery cross-section (see transition_regions()).

    Returns:
        [{"id", "coords", "road_width", "tube_width" (floor chord = road + 2 curbs), "radius", "center_z" (circle
        center above the floor), "crown", "curb_width", "curb_height", "edge_height", "max_arc_deg", "floor_material",
        "shell", "portals": [portal, portal]}, ...]
        portal: {"label", "xy", "axis" (unit vector into the tunnel interior), "floor_z", "radius", "center_z", "crown", "shell",
        "collar", "floor_width", "half_width", "length", "flat_depth", "top_z", "bottom_z", "open", "kind" ("open" |
        "gallery"; for "gallery" additionally "gallery_half_width", "gallery_height")}. Both kinds: round collar
        (outer radius radius + shell + collar = half_width; collar 0 = the end ring of the tube is the portal).
        Both dimensions depend only on the structure, not on the terrain. "open" is set by terrain/tunnel_terrain.py (an
        end in the middle of the mountain is not an open portal and gets no structure).
    """
    gallery_ends = []  # (x, y, road width, open_side, digitizing direction at the end) per gallery end point
    for gallery in galleries or []:
        gallery_coords = np.asarray(gallery["coords"], dtype=float)
        if len(gallery_coords) >= 2:
            for point, direction in ((gallery_coords[0], gallery_coords[1] - gallery_coords[0]), (gallery_coords[-1], gallery_coords[-1] - gallery_coords[-2])):
                gallery_ends.append((float(point[0]), float(point[1]), float(gallery["width"]), gallery.get("open_side"), direction[:2]))

    def gallery_at(x: float, y: float):
        for gx, gy, width, open_side, direction in gallery_ends:
            if np.hypot(gx - x, gy - y) <= transition_tol:
                return width, open_side, direction
        return None

    plans = []
    for chain in chain_tunnel_pieces(tunnels):
        coords = resample_tunnel_coords(chain["coords"], segment_step)
        if len(coords) < 2:
            continue
        road_width = chain["width"]
        tube_width = road_width + 2.0 * curb_width
        radius, center_z = tunnel_profile(road_width, curb_width, edge_height, max_arc_deg)
        crown = center_z + radius
        shell = shell_ratio * 2.0 * radius
        collar = collar_ratio * 2.0 * radius
        frame = collar if collar > 0.0 else shell  # wall thickness of the portal (collar or tube shell)
        tilt = math.tan(math.radians(tilt_deg))
        face_depth = (crown + frame) * tilt  # how far the tilted end face reaches into the mountain at its top edge
        points = np.asarray(coords, dtype=float)

        portals = []
        for label, index, neighbour in (("start", 0, 1), ("end", -1, -2)):
            axis = points[neighbour, :2] - points[index, :2]
            axis = axis / np.linalg.norm(axis)
            floor_z = float(points[index, 2])
            portals.append(
                {
                    "label": label,
                    "xy": (float(points[index, 0]), float(points[index, 1])),
                    "axis": (float(axis[0]), float(axis[1])),
                    "floor_z": floor_z,
                    "radius": radius,
                    "center_z": center_z,
                    "crown": crown,
                    "shell": shell,
                    "collar": collar,
                    "floor_width": tube_width,
                    "half_width": radius + (max(collar, collar_min_side) if collar > 0.0 else frame),
                    "collar_depth": length,
                    "length": face_depth + length,  # reach of the portal at the top edge (terrain, exclusion zones)
                    "flat_depth": max(flat_depth, face_depth),
                    "tilt": tilt,
                    "top_z": floor_z + crown + frame,
                    "bottom_z": floor_z - frame,
                    "open": True,
                    "kind": "open",
                }
            )
            gallery = gallery_at(*portals[-1]["xy"])
            if gallery is not None:
                gallery_width, open_side, direction = gallery
                portal = portals[-1]
                portal["kind"] = "gallery"
                portal["gallery_half_width"] = gallery_width / 2.0
                portal["gallery_height"] = gallery_height
                portal["gallery_roof"] = gallery_roof_thickness
                portal["gallery_wall"] = gallery_wall_thickness
                # Mountain wall in portal coordinates (+1 = right when looking into the tunnel interior, 0 = unknown): the
                # gallery sides follow their digitizing direction, which runs with or against the portal axis
                if open_side in ("left", "right"):
                    same = float(np.dot(direction, axis)) > 0.0
                    valley = (1 if open_side == "right" else -1) * (1 if same else -1)
                    portal["gallery_wall_side"] = -valley
                else:
                    portal["gallery_wall_side"] = 0
                # Transition: vertical end face (the rectangular cross-section of the gallery connects flush)
                portal["tilt"] = 0.0
                portal["flat_depth"] = flat_depth
                portal["length"] = length
        plans.append(
            {
                "id": chain["id"],
                "piece_ids": chain["piece_ids"],
                "coords": coords,
                "road_width": road_width,
                "tube_width": tube_width,
                "radius": radius,
                "center_z": center_z,
                "crown": crown,
                "curb_width": curb_width,
                "curb_height": curb_height,
                "edge_height": edge_height,
                "max_arc_deg": max_arc_deg,
                "floor_material": chain["floor_material"],
                "shell": shell,
                "portals": portals,
            }
        )
    return plans


def portal_local_coords(portal: Dict, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(along, across) relative to the portal: along = distance behind the portal plane (positive into the tunnel
    interior), across = lateral distance (positive = right, when looking into the tunnel interior)."""
    px, py = portal["xy"]
    ux, uy = portal["axis"]
    dx, dy = np.asarray(x, dtype=float) - px, np.asarray(y, dtype=float) - py
    return dx * ux + dy * uy, dx * uy - dy * ux


def portal_footprint(portal: Dict) -> List[Tuple[float, float]]:
    """Footprint of the portal (4 corners in world coordinates) - for exclusion zones (trees, vines)."""
    return [_world_xy(portal, a, c) for a, c in ((0.0, -portal["half_width"]), (0.0, portal["half_width"]), (portal["length"], portal["half_width"]), (portal["length"], -portal["half_width"]))]


def _world_xy(portal: Dict, along: float, across: float) -> Tuple[float, float]:
    px, py = portal["xy"]
    ux, uy = portal["axis"]
    return (px + ux * along + uy * across, py + uy * along - ux * across)


def transition_regions(
    radius: float,
    arc_segments: int,
    half_opening: float,
    opening_height: float,
    roof: float = 0.0,
    wall: float = 0.0,
    wall_side: int = 0,
    shell: float = 0.0,
    frame=None,
    center_z: float = None,
):
    """
    (step face, gallery side) of a transition portal as shapely areas (across, height), relative to the tube floor.
    Step face (facing the tube) = tube cross-section minus gallery body (clear cross-section ±half_opening x
    0..opening_height including roof `roof` and mountain wall `wall` on side `wall_side`, 0 = unknown: both sides):
    otherwise one would see outside from the tunnel next to the gallery; roof and wall are covered by the gallery end
    face (a face in the same plane and direction flickered). Gallery side (facing the gallery) = clear gallery
    cross-section minus tube including shell `shell` and portal collar `frame` (shapely area or None; the collar end
    face lies in the same plane): if the gallery extends beyond the arch, one would otherwise see outside from the
    gallery. center_z: height of the tube circle center above the floor (see tunnel_mesh.tunnel_profile()).
    """
    from .tunnel_mesh import shell_cross_section

    from shapely.geometry import Polygon, box

    tube = Polygon(arc_cross_section(radius, arc_segments, center_z))
    section = box(-half_opening, 0.0, half_opening, opening_height)
    top = opening_height + roof
    body = box(-half_opening, 0.0, half_opening, top)
    if wall > 0.0:
        if wall_side >= 0:
            body = body.union(box(half_opening, 0.0, half_opening + wall, top))
        if wall_side <= 0:
            body = body.union(box(-half_opening - wall, 0.0, -half_opening, top))
    tube_body = Polygon(shell_cross_section(radius, arc_segments, shell, center_z)) if shell > 0.0 else tube
    if frame is not None:
        tube_body = tube_body.union(frame)
    return tube.difference(body), section.difference(tube_body)


def _collar_frame(portal: Dict):
    """Outer contour of the portal collar (across, height relative to the tube floor): rectangle up to ±portal["half_width"]
    (at least collar_min_side thick at the sides), portal["collar"] above the crown at the top, equally deep below the
    road surface at the bottom."""
    from shapely.geometry import box

    crown, collar, half_width = portal["crown"], portal["collar"], portal["half_width"]
    return box(-half_width, -collar, half_width, crown + collar)


def _build_collar_mesh(portal: Dict, material: str, arc_segments: int, tile_m: float) -> Dict:
    """
    Portal collar, rectangular on the outside (_collar_frame()), portal["collar_depth"] meters deep. The end face lies
    in the plane of the tube end face tilted by portal["tilt"] (a point at height h moves by h * tilt into the
    mountain, see tunnel_mesh._end_shift()), the back face parallel behind it. End face = rectangle minus clear
    cross-section (hits exactly the first tube ring), back face = rectangle minus tube shell.
    """
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon

    from .tunnel_mesh import shell_cross_section

    floor_z, radius, depth = portal["floor_z"], portal["radius"], portal.get("collar_depth", portal["length"])
    shell, tilt, center_z = portal.get("shell", 0.0), portal.get("tilt", 0.0), portal.get("center_z")
    ux, uy = portal["axis"]
    frame = _collar_frame(portal)
    builder = MeshBuilder()
    cos, sin = 1.0 / math.hypot(1.0, tilt), tilt / math.hypot(1.0, tilt)

    def world(offset: float, across: float, height: float) -> List[float]:
        x, y = _world_xy(portal, offset + height * tilt, across)
        return [float(x), float(y), float(floor_z + height)]

    def face(shape, offset: float, normal) -> None:
        for tri in constrained_delaunay_triangles(shape).geoms:
            pts = list(tri.exterior.coords)[:3]
            builder.triangle([world(offset, c, h) for c, h in pts], [[c / tile_m, h / tile_m] for c, h in pts], normal)

    face(frame.difference(Polygon(arc_cross_section(radius, arc_segments, center_z))), 0.0, [-ux * cos, -uy * cos, sin])
    inner = (Polygon(shell_cross_section(radius, arc_segments, shell, center_z)) if shell > 0.0
             else Polygon(arc_cross_section(radius, arc_segments, center_z)))
    face(frame.difference(inner), depth, [ux * cos, uy * cos, -sin])

    # Jacket: four rectangular sides from the end face to the back face
    min_c, min_h, max_c, max_h = frame.bounds
    right = [uy, -ux, 0.0]
    left = [-uy, ux, 0.0]
    for a, b, normal in (
        ((max_c, min_h), (max_c, max_h), right),
        ((min_c, max_h), (min_c, min_h), left),
        ((max_c, max_h), (min_c, max_h), [-ux * sin, -uy * sin, cos]),
        ((min_c, min_h), (max_c, min_h), [ux * sin, uy * sin, -cos]),
    ):
        builder.quad(
            [world(0.0, *a), world(depth, *a), world(depth, *b), world(0.0, *b)],
            [[0.0, 0.0], [depth / tile_m, 0.0], [depth / tile_m, 1.0], [0.0, 1.0]],
            [float(v) for v in normal],
        )

    return {
        "vertices": np.array(builder.vertices, dtype=float),
        "uvs": np.array(builder.uvs, dtype=float),
        "normals": np.array(builder.normals, dtype=float),
        "faces": {material: builder.faces},
    }


def _add_slab(builder: MeshBuilder, portal: Dict, region, along_from: float, along_to: float, tile_m: float) -> None:
    """Solid slab: `region` (shapely area in (across, height) relative to the tube floor) from along_from to along_to
    (along the portal axis) - front and back face plus surrounding edges, visible from both sides."""
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    ux, uy = portal["axis"]
    right = (uy, -ux)

    def world(along: float, across: float, height: float) -> List[float]:
        x, y = _world_xy(portal, along, across)
        return [float(x), float(y), float(portal["floor_z"] + height)]

    polygons = [g for g in getattr(region, "geoms", [region]) if isinstance(g, Polygon) and not g.is_empty and g.area > 1e-9]
    for polygon in polygons:
        for tri in constrained_delaunay_triangles(polygon).geoms:
            pts = list(tri.exterior.coords)[:3]
            uvs = [[c / tile_m, h / tile_m] for c, h in pts]
            builder.triangle([world(along_from, c, h) for c, h in pts], uvs, [-ux, -uy, 0.0])
            builder.triangle([world(along_to, c, h) for c, h in pts], uvs, [ux, uy, 0.0])
        # Edges: outer ring counterclockwise, holes clockwise -> outward normal (dh, -dc) per edge
        oriented = orient(polygon, 1.0)
        for ring in [oriented.exterior, *oriented.interiors]:
            coords = list(ring.coords)
            for (c0, h0), (c1, h1) in zip(coords[:-1], coords[1:]):
                length = float(np.hypot(c1 - c0, h1 - h0))
                if length < 1e-9:
                    continue
                nc, nh = (h1 - h0) / length, -(c1 - c0) / length
                builder.quad(
                    [world(along_from, c0, h0), world(along_to, c0, h0), world(along_to, c1, h1), world(along_from, c1, h1)],
                    [[0.0, 0.0], [(along_to - along_from) / tile_m, 0.0], [(along_to - along_from) / tile_m, length / tile_m], [0.0, length / tile_m]],
                    [float(right[0] * nc), float(right[1] * nc), float(nh)],
                )


def build_portal_block_mesh(
    portal: Dict, material: str, arc_segments: int = 12, tile_m: float = 4.0, cover_thickness: float = 0.2
) -> Dict:
    """
    Portal structure of an open portal or transition: rectangular collar (only if portal["collar"] > 0, otherwise the
    end ring of the tube is the portal) plus - at the transition into a gallery - the faces between the tube arch and
    the gallery cross-section (transition_regions()) as solid slabs `cover_thickness` thick: the step extends from the
    portal plane into the tube, the gallery cover into the gallery.
    """
    parts = []
    if portal.get("collar", 0.0) > 0.0:
        parts.append(_build_collar_mesh(portal, material, arc_segments, tile_m))
    if portal.get("kind") == "gallery":
        builder = MeshBuilder()
        step, gallery_side = transition_regions(
            portal["radius"], arc_segments, portal["gallery_half_width"], portal["gallery_height"],
            roof=portal.get("gallery_roof", 0.0), wall=portal.get("gallery_wall", 0.0),
            wall_side=portal.get("gallery_wall_side", 0), shell=portal.get("shell", 0.0),
            frame=_collar_frame(portal) if portal.get("collar", 0.0) > 0.0 else None,
            center_z=portal.get("center_z"),
        )
        _add_slab(builder, portal, step, 0.0, cover_thickness, tile_m)
        _add_slab(builder, portal, gallery_side, -cover_thickness, 0.0, tile_m)
        parts.append({
            "vertices": np.array(builder.vertices, dtype=float).reshape(-1, 3),
            "uvs": np.array(builder.uvs, dtype=float).reshape(-1, 2),
            "normals": np.array(builder.normals, dtype=float).reshape(-1, 3),
            "faces": {material: builder.faces},
        })

    vertices, uvs, normals, faces = [], [], [], []
    for part in parts:
        offset = len(vertices)
        vertices += part["vertices"].tolist()
        uvs += part["uvs"].tolist()
        normals += part["normals"].tolist()
        faces += [[a + offset, b + offset, c + offset] for a, b, c in part["faces"][material]]
    return {
        "vertices": np.array(vertices, dtype=float).reshape(-1, 3),
        "uvs": np.array(uvs, dtype=float).reshape(-1, 2),
        "normals": np.array(normals, dtype=float).reshape(-1, 3),
        "faces": {material: faces},
    }
