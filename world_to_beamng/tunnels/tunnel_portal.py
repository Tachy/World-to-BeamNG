"""
Portalbauwerk an beiden Enden einer Tunnelröhre, von der Portalebene (OSM-Tunnelende, dort schließt die Zufahrt an)
TUNNEL_PORTAL_LENGTH Meter in den Berg:

- offenes Portal: runder Betonkragen um die Röhrenschale (Außenradius Radius + Schale + TUNNEL_PORTAL_WING);
- Übergang in eine Galerie: Stirnwand mit rechteckiger Galerie-Öffnung (Block über den Galerie-Querschnitt).

Das Bauwerk verdeckt die Terrain-Löcher, ohne die die Heightmap die Öffnung versperren würde: eine Rasterzelle, die
die Portalebene überspannt, hat vorne Straßenniveau und hinten Überdeckungshöhe - ihre schräge Fläche liefe quer
durch die Öffnung. terrain/tunnel_terrain.py hält deshalb das Gelände bis TUNNEL_PORTAL_FLAT_DEPTH hinter der
Portalebene auf Bodenhöhe (unter dem Röhrenboden verborgen), trägt den Hang dahinter im Grundriss des Bauwerks auf
dessen Außenkontur ab und macht die Zellen am Übergang zu Löchern. Die Größe des Bauwerks hängt nur von Röhre und
Galerie ab, nie vom Gelände.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder
from .tunnel_mesh import arc_cross_section, chain_tunnel_pieces, resample_tunnel_coords, tunnel_crown_height, tunnel_radius


TRANSITION_WALL_MARGIN = 0.2  # Stirnwand eines Übergangs reicht so weit über Röhrenschale bzw. Galeriedach hinaus, in Metern


def plan_tunnels(
    tunnels: Sequence[Dict],
    width_margin: float,
    segment_step: float,
    wing: float,
    flat_depth: float,
    length: float,
    galleries: Sequence[Dict] = None,
    gallery_height: float = 5.0,
    gallery_roof_thickness: float = 0.5,
    gallery_floor_thickness: float = 5.0,
    gallery_wall_thickness: float = 5.0,
    transition_tol: float = 0.5,
    shell_ratio: float = 0.0,
) -> List[Dict]:
    """
    Verkettet die Tunnel-Stücke (siehe tunnel_mesh.chain_tunnel_pieces()), dünnt die Centerline aus und legt die
    beiden Portale jeder Kette fest. Röhre (tunnel_mesh.py), Gelände (terrain/tunnel_terrain.py) und Portalblock
    arbeiten alle auf diesem Plan.

    Args:
        tunnels: [{"id", "coords", "width", "floor_material"}, ...]
        shell_ratio: Wandstärke der Röhrenschale (tunnel_mesh.shell_cross_section()) im Verhältnis zum
            Röhrendurchmesser - kleinere Tunnel bekommen dünnere Wände
        wing: so weit ragt der runde Portalkragen über die Röhrenschale hinaus, in Metern
        galleries: Galerie-Eingaben (wie build_galleries()); ein Portal, das höchstens transition_tol von einem
            ENDPUNKT einer Galerie-Centerline liegt, ist ein Übergang Tunnel -> Galerie (portal["kind"] ==
            "gallery"): Stirnwand mit Galerie-Öffnung statt offenes Portal ins Gelände, Block auf den ganzen
            Galerie-Querschnitt vergrößert.

    Returns:
        [{"id", "coords", "tube_width", "radius", "crown", "floor_material", "shell", "portals": [portal, portal]}, ...]
        portal: {"label", "xy", "axis" (Einheitsvektor ins Tunnelinnere), "floor_z", "radius", "crown", "shell",
        "collar", "floor_width", "half_width", "length", "flat_depth", "top_z", "bottom_z", "open", "kind" ("open" |
        "gallery"; bei "gallery" zusätzlich "gallery_half_width", "gallery_height")}. Offenes Portal: runder Kragen
        (Außenradius radius + shell + collar = half_width); Übergang in eine Galerie: Stirnwand bis top_z/bottom_z.
        Beide Maße hängen nur vom Bauwerk ab, nicht vom Gelände. "open" setzt terrain/tunnel_terrain.py (ein Ende
        mitten im Berg ist kein offenes Portal und bekommt kein Bauwerk).
    """
    gallery_ends = []  # (x, y, Fahrbahnbreite) je Galerie-Endpunkt
    for gallery in galleries or []:
        gallery_coords = gallery["coords"]
        if len(gallery_coords) >= 2:
            for point in (gallery_coords[0], gallery_coords[-1]):
                gallery_ends.append((float(point[0]), float(point[1]), float(gallery["width"])))

    def gallery_width_at(x: float, y: float):
        for gx, gy, width in gallery_ends:
            if np.hypot(gx - x, gy - y) <= transition_tol:
                return width
        return None

    plans = []
    for chain in chain_tunnel_pieces(tunnels):
        coords = resample_tunnel_coords(chain["coords"], segment_step)
        if len(coords) < 2:
            continue
        tube_width = chain["width"] + width_margin
        radius = tunnel_radius(tube_width)
        crown = tunnel_crown_height(tube_width)
        shell = shell_ratio * 2.0 * radius
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
                    "crown": crown,
                    "shell": shell,
                    "collar": wing,
                    "floor_width": tube_width,
                    "half_width": radius + shell + wing,
                    "length": length,
                    "flat_depth": flat_depth,
                    "top_z": floor_z + crown + shell + wing,
                    "bottom_z": floor_z - shell - wing,
                    "open": True,
                    "kind": "open",
                }
            )
            gallery_width = gallery_width_at(*portals[-1]["xy"])
            if gallery_width is not None:
                portal = portals[-1]
                portal["kind"] = "gallery"
                portal["gallery_half_width"] = gallery_width / 2.0
                portal["gallery_height"] = gallery_height
                # Stirnwand nur so groß wie Röhrenschale und Galerie-Querschnitt (ohne Kragenrand) plus Rand
                portal["half_width"] = max(radius + shell + TRANSITION_WALL_MARGIN, gallery_width / 2.0 + gallery_wall_thickness)
                portal["top_z"] = floor_z + max(crown + shell, gallery_height + gallery_roof_thickness) + TRANSITION_WALL_MARGIN
                portal["bottom_z"] = floor_z - max(shell, gallery_floor_thickness)
        plans.append(
            {
                "id": chain["id"],
                "coords": coords,
                "tube_width": tube_width,
                "radius": radius,
                "crown": crown,
                "floor_material": chain["floor_material"],
                "shell": shell,
                "portals": portals,
            }
        )
    return plans


def portal_local_coords(portal: Dict, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(along, across) relativ zum Portal: along = Abstand hinter der Portalebene (ins Tunnelinnere positiv),
    across = seitlicher Abstand (positiv = rechts, in Blickrichtung ins Tunnelinnere)."""
    px, py = portal["xy"]
    ux, uy = portal["axis"]
    dx, dy = np.asarray(x, dtype=float) - px, np.asarray(y, dtype=float) - py
    return dx * ux + dy * uy, dx * uy - dy * ux


def portal_footprint(portal: Dict) -> List[Tuple[float, float]]:
    """Grundriss des Portalblocks (4 Ecken in Weltkoordinaten) - für Ausschlusszonen (Bäume, Reben)."""
    return [_world_xy(portal, a, c) for a, c in ((0.0, -portal["half_width"]), (0.0, portal["half_width"]), (portal["length"], portal["half_width"]), (portal["length"], -portal["half_width"]))]


def _world_xy(portal: Dict, along: float, across: float) -> Tuple[float, float]:
    px, py = portal["xy"]
    ux, uy = portal["axis"]
    return (px + ux * along + uy * across, py + uy * along - ux * across)


def transition_wall_triangles(
    radius: float,
    arc_segments: int,
    half_width: float,
    bottom: float,
    top: float,
    half_opening: float,
    opening_height: float,
):
    """
    (Stirnwand, Stufenfläche) eines Übergangs-Portals als Dreieckslisten (across, height), relativ zum Röhrenboden.
    Durchgang = Röhrenquerschnitt (240°-Bogen über der Bodensehne) ∩ Galerie-Rechteck (±half_opening x
    0..opening_height). Die Stirnwand (zur Galerie) ist das Blockrechteck minus Durchgang, die Stufenfläche (zur Röhre)
    der Röhrenquerschnitt minus Durchgang. Ragt das Rechteck über den Bogen hinaus (Galerie breiter als der Tunnel,
    schmale Straße mit Krone unter der Galeriehöhe), schließt die Stirnwand diesen Teil - sonst bliebe dort ein Loch
    in den hohlen Block.
    """
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon, box

    def triangles(shape):
        return [list(tri.exterior.coords)[:3] for tri in constrained_delaunay_triangles(shape).geoms]

    tube = Polygon(arc_cross_section(radius, arc_segments))
    passage = tube.intersection(box(-half_opening, 0.0, half_opening, opening_height))
    wall = box(-half_width, bottom, half_width, top).difference(passage)
    return triangles(wall), triangles(tube.difference(passage))


def _build_collar_mesh(portal: Dict, material: str, arc_segments: int, tile_m: float) -> Dict:
    """
    Offenes Portal als runder Betonkragen: Außenkontur = Röhrenschale + portal["collar"] (tunnel_mesh.
    shell_cross_section()), von der Portalebene portal["length"] Meter in den Berg. Stirnseite = Kragenkontur minus
    lichter Röhrenquerschnitt (trifft exakt den ersten Röhrenring), Rückseite = Kragenkontur minus Röhrenschale.
    """
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon

    from .tunnel_mesh import shell_cross_section

    floor_z, radius, length = portal["floor_z"], portal["radius"], portal["length"]
    shell, collar = portal.get("shell", 0.0), portal.get("collar", 0.0)
    ux, uy = portal["axis"]
    outer = shell_cross_section(radius, arc_segments, shell + collar)
    center = np.array([0.0, radius / 2.0])
    builder = MeshBuilder()

    def world(along: float, across: float, height: float) -> List[float]:
        x, y = _world_xy(portal, along, across)
        return [float(x), float(y), float(floor_z + height)]

    def ring(shape, along: float, normal) -> None:
        for tri in constrained_delaunay_triangles(shape).geoms:
            pts = list(tri.exterior.coords)[:3]
            builder.triangle([world(along, c, h) for c, h in pts], [[c / tile_m, h / tile_m] for c, h in pts], normal)

    ring(Polygon(outer).difference(Polygon(arc_cross_section(radius, arc_segments))), 0.0, [-ux, -uy, 0.0])
    inner_shell = Polygon(shell_cross_section(radius, arc_segments, shell)) if shell > 0.0 else Polygon(arc_cross_section(radius, arc_segments))
    ring(Polygon(outer).difference(inner_shell), length, [ux, uy, 0.0])

    # Mantel des Kragens, Normalen vom Achsmittelpunkt weg
    right = np.array([uy, -ux])
    for k in range(len(outer)):
        a, b = np.array(outer[k]), np.array(outer[(k + 1) % len(outer)])
        normal_2d = np.array([b[1] - a[1], a[0] - b[0]])
        normal_2d /= np.linalg.norm(normal_2d)
        if normal_2d @ ((a + b) / 2.0 - center) < 0.0:
            normal_2d = -normal_2d
        builder.quad(
            [world(0.0, *a), world(length, *a), world(length, *b), world(0.0, *b)],
            [[0.0, a[0] / tile_m], [length / tile_m, a[0] / tile_m], [length / tile_m, b[0] / tile_m], [0.0, b[0] / tile_m]],
            [float(right[0] * normal_2d[0]), float(right[1] * normal_2d[0]), float(normal_2d[1])],
        )

    return {
        "vertices": np.array(builder.vertices, dtype=float),
        "uvs": np.array(builder.uvs, dtype=float),
        "normals": np.array(builder.normals, dtype=float),
        "faces": {material: builder.faces},
    }


def build_portal_block_mesh(portal: Dict, material: str, arc_segments: int = 12, tile_m: float = 4.0) -> Dict:
    """
    Betonblock des Portals: Stirnseite (Rechteck mit der 240°-Röhrenöffnung, dazu ein Streifen unter Bodenhöhe),
    zwei Seitenwände, Oberseite und Rückwand oberhalb der Röhre. Die Öffnung trifft exakt den ersten
    Querschnitts-Ring der Röhre (gleicher Bogen, gleiche Bodenhöhe) - die Röhre schließt nahtlos an.

    Übergangs-Portal (portal["kind"] == "gallery", siehe plan_tunnels()): Die Stirnseite ist eine geschlossene Wand
    mit rechteckiger Galerie-Öffnung; dazu eine zur Röhre gerichtete Stufenfläche zwischen Bogen und Rechteck.
    """
    floor_z = portal["floor_z"]
    radius, crown = portal["radius"], portal["crown"]
    half_width, length = portal["half_width"], portal["length"]
    top = portal["top_z"] - floor_z
    bottom = portal["bottom_z"] - floor_z
    ux, uy = portal["axis"]

    def world(along: float, across: float, height: float) -> List[float]:
        x, y = _world_xy(portal, along, across)
        return [float(x), float(y), float(floor_z + height)]

    outward = [-ux, -uy, 0.0]
    right = [uy, -ux, 0.0]
    left = [-uy, ux, 0.0]
    builder = MeshBuilder()

    def front(points: Sequence[Tuple[float, float]]):
        corners = [world(0.0, c, h) for c, h in points]
        uvs = [[c / tile_m, h / tile_m] for c, h in points]
        (builder.triangle if len(points) == 3 else builder.quad)(corners, uvs, outward)

    def outer_point(across: float, height: float) -> Tuple[float, float]:
        """Strahl von der Bodenmitte durch den Öffnungspunkt bis auf den Rechteck-Rand (Seiten oder Oberkante)."""
        if height <= 1e-9:
            return (half_width if across > 0 else -half_width, 0.0)
        scale = top / height
        if abs(across) > 1e-9:
            scale = min(scale, half_width / abs(across))
        return (across * scale, height * scale)

    def edge_of(point: Tuple[float, float]) -> str:
        if abs(point[1] - top) < 1e-6:
            return "top"
        return "right" if point[0] > 0 else "left"

    if portal.get("kind") != "gallery":
        return _build_collar_mesh(portal, material, arc_segments, tile_m)

    if portal.get("kind") == "gallery":
        # Übergang Tunnel -> Galerie: geschlossene Stirnwand mit rechteckiger Öffnung in Galeriegröße (Dach, Bergwand
        # und Sockel der Galerie schließen hier bündig an) ...
        wall, step = transition_wall_triangles(
            radius, arc_segments, half_width, bottom, top, portal["gallery_half_width"], portal["gallery_height"]
        )
        for tri in wall:
            front(tri)
        # ... und von innen der Querschnittssprung Röhrenbogen -> Rechteck, zur Röhre hin gerichtet
        for tri in step:
            builder.triangle([world(0.0, c, h) for c, h in tri], [[c / tile_m, h / tile_m] for c, h in tri], [ux, uy, 0.0])
    else:
        # Stirnseite: Ring zwischen Öffnung und Rechteck-Rand
        arc = arc_cross_section(radius, arc_segments)
        for k in range(arc_segments):
            inner_a, inner_b = arc[k], arc[k + 1]
            outer_a, outer_b = outer_point(*inner_a), outer_point(*inner_b)
            if edge_of(outer_a) == edge_of(outer_b):
                front([inner_a, inner_b, outer_b, outer_a])
                continue
            corner = (half_width if "right" in (edge_of(outer_a), edge_of(outer_b)) else -half_width, top)
            front([inner_a, inner_b, corner])
            front([inner_a, corner, outer_a])
            front([inner_b, outer_b, corner])
        # Streifen unter Bodenhöhe (falls das Gelände vor dem Portal tiefer liegt)
        front([(half_width, 0.0), (-half_width, 0.0), (-half_width, bottom), (half_width, bottom)])

    # Seitenwände
    for across, normal in ((half_width, right), (-half_width, left)):
        builder.quad(
            [world(0.0, across, bottom), world(length, across, bottom), world(length, across, top), world(0.0, across, top)],
            [[0.0, bottom / tile_m], [length / tile_m, bottom / tile_m], [length / tile_m, top / tile_m], [0.0, top / tile_m]],
            normal,
        )

    # Oberseite
    builder.quad(
        [world(0.0, -half_width, top), world(0.0, half_width, top), world(length, half_width, top), world(length, -half_width, top)],
        [[0.0, -half_width / tile_m], [0.0, half_width / tile_m], [length / tile_m, half_width / tile_m], [length / tile_m, -half_width / tile_m]],
        [0.0, 0.0, 1.0],
    )

    # Rückwand nur oberhalb der Röhre (die Röhre läuft hier weiter; ihre Krone kann bis zum Blockende etwas
    # steigen - daher etwas Luft über der Krone)
    back_bottom = crown + 0.5 + 0.1 * length
    if back_bottom < top - 0.05:
        builder.quad(
            [world(length, half_width, back_bottom), world(length, -half_width, back_bottom), world(length, -half_width, top), world(length, half_width, top)],
            [[half_width / tile_m, back_bottom / tile_m], [-half_width / tile_m, back_bottom / tile_m], [-half_width / tile_m, top / tile_m], [half_width / tile_m, top / tile_m]],
            [ux, uy, 0.0],
        )

    return {
        "vertices": np.array(builder.vertices, dtype=float),
        "uvs": np.array(builder.uvs, dtype=float),
        "normals": np.array(builder.normals, dtype=float),
        "faces": {material: builder.faces},
    }
