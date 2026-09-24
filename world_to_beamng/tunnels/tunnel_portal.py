"""
Portal an beiden Enden einer Tunnelröhre, an der Portalebene (OSM-Tunnelende, dort schließt die Zufahrt bzw. die
Galerie an): ein außen rechteckiger Betonkragen, an der dünnsten Stelle (links, rechts, oben) TUNNEL_PORTAL_COLLAR_RATIO
mal Röhrendurchmesser stark, TUNNEL_PORTAL_LENGTH Meter tief (ohne Kragen: der Stirnring der Röhrenschale). An
Tunneleingängen ist die Stirnseite um TUNNEL_PORTAL_TILT_DEG zur Bergseite gekippt; am Übergang in eine Galerie
senkrecht, dazu die Flächen zwischen Röhrenbogen und Galerie-Querschnitt (transition_regions()).

Das Portal verdeckt die Terrain-Löcher, ohne die die Heightmap die Öffnung versperren würde: eine Rasterzelle, die
die Portalebene überspannt, hat vorne Straßenniveau und hinten Überdeckungshöhe - ihre schräge Fläche liefe quer
durch die Öffnung. terrain/tunnel_terrain.py hält deshalb das Gelände bis TUNNEL_PORTAL_FLAT_DEPTH hinter der
Portalebene auf Bodenhöhe (unter dem Röhrenboden verborgen), trägt den Hang dahinter im Grundriss des Portals bis
knapp unter dessen runde Außenkontur ab und macht die Zellen am Übergang zu Löchern. Die Größe des Portals hängt nur
von der Röhre ab, nie vom Gelände.
"""

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder
from .tunnel_mesh import arc_cross_section, chain_tunnel_pieces, resample_tunnel_coords, tunnel_crown_height, tunnel_radius


def plan_tunnels(
    tunnels: Sequence[Dict],
    width_margin: float,
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
) -> List[Dict]:
    """
    Verkettet die Tunnel-Stücke (siehe tunnel_mesh.chain_tunnel_pieces()), dünnt die Centerline aus und legt die
    beiden Portale jeder Kette fest. Röhre (tunnel_mesh.py), Gelände (terrain/tunnel_terrain.py) und Portal
    arbeiten alle auf diesem Plan.

    Args:
        tunnels: [{"id", "coords", "width", "floor_material"}, ...]
        shell_ratio: Wandstärke der Röhrenschale (tunnel_mesh.shell_cross_section()) im Verhältnis zum
            Röhrendurchmesser - kleinere Tunnel bekommen dünnere Wände
        collar_ratio: Portalkragen, außen rechteckig: Wandstärke an der dünnsten Stelle (links, rechts, oben) im
            Verhältnis zum Röhrendurchmesser; 0 = kein Kragen, der Stirnring der Röhre ist das Portal
        collar_min_side: Kragen links/rechts mindestens so stark (Loch-Zellen an der Portalstufe reichen bis zu eine
            Rasterdiagonale seitlich über den Röhrenradius hinaus und müssen darin verborgen bleiben), in Metern
        length: Tiefe des Kragens in den Berg, in Metern
        tilt_deg: Tunneleingänge (nicht Übergänge in eine Galerie): Stirnseite um so viel Grad zur Bergseite gekippt
            (portal["tilt"] = tan). Die flache Portal-Zone reicht dann mindestens bis hinter die Stirnseite an der
            Außenkrone, damit Öffnung frei und Lochkante hinter der Schrägfläche bleiben.
        galleries: Galerie-Eingaben (wie build_galleries()); ein Portal, das höchstens transition_tol von einem
            ENDPUNKT einer Galerie-Centerline liegt, ist ein Übergang Tunnel -> Galerie (portal["kind"] ==
            "gallery"): gleiches rundes Portal wie am Tunneleingang, dazu die Flächen zwischen Röhrenbogen und
            Galerie-Querschnitt (siehe transition_regions()).

    Returns:
        [{"id", "coords", "tube_width", "radius", "crown", "floor_material", "shell", "portals": [portal, portal]}, ...]
        portal: {"label", "xy", "axis" (Einheitsvektor ins Tunnelinnere), "floor_z", "radius", "crown", "shell",
        "collar", "floor_width", "half_width", "length", "flat_depth", "top_z", "bottom_z", "open", "kind" ("open" |
        "gallery"; bei "gallery" zusätzlich "gallery_half_width", "gallery_height")}. Beide Arten: runder Kragen
        (Außenradius radius + shell + collar = half_width; collar 0 = der Stirnring der Röhre ist das Portal).
        Beide Maße hängen nur vom Bauwerk ab, nicht vom Gelände. "open" setzt terrain/tunnel_terrain.py (ein Ende
        mitten im Berg ist kein offenes Portal und bekommt kein Bauwerk).
    """
    gallery_ends = []  # (x, y, Fahrbahnbreite, open_side, Digitalisierungsrichtung am Ende) je Galerie-Endpunkt
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
        tube_width = chain["width"] + width_margin
        radius = tunnel_radius(tube_width)
        crown = tunnel_crown_height(tube_width)
        shell = shell_ratio * 2.0 * radius
        collar = collar_ratio * 2.0 * radius
        frame = collar if collar > 0.0 else shell  # Wandstärke des Portals (Kragen bzw. Röhrenschale)
        tilt = math.tan(math.radians(tilt_deg))
        face_depth = (crown + frame) * tilt  # so weit reicht die gekippte Stirnseite an ihrer Oberkante in den Berg
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
                    "collar": collar,
                    "floor_width": tube_width,
                    "half_width": radius + (max(collar, collar_min_side) if collar > 0.0 else frame),
                    "collar_depth": length,
                    "length": face_depth + length,  # Reichweite des Portals an der Oberkante (Gelände, Ausschlusszonen)
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
                # Bergwand in Portal-Koordinaten (+1 = rechts in Blickrichtung ins Tunnelinnere, 0 = unbekannt): die
                # Galerie-Seiten folgen ihrer Digitalisierungsrichtung, die gleich oder gegen die Portalachse läuft
                if open_side in ("left", "right"):
                    same = float(np.dot(direction, axis)) > 0.0
                    valley = (1 if open_side == "right" else -1) * (1 if same else -1)
                    portal["gallery_wall_side"] = -valley
                else:
                    portal["gallery_wall_side"] = 0
                # Übergang: senkrechte Stirnseite (der Rechteckquerschnitt der Galerie schließt stumpf an)
                portal["tilt"] = 0.0
                portal["flat_depth"] = flat_depth
                portal["length"] = length
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
    """Grundriss des Portals (4 Ecken in Weltkoordinaten) - für Ausschlusszonen (Bäume, Reben)."""
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
):
    """
    (Stufenfläche, Galerie-Seite) eines Übergangs-Portals als shapely-Flächen (across, height), relativ zum Röhrenboden.
    Stufenfläche (zur Röhre gerichtet) = Röhrenquerschnitt minus Galeriekörper (lichter Querschnitt ±half_opening x
    0..opening_height samt Dach `roof` und Bergwand `wall` auf Seite `wall_side`, 0 = unbekannt: beide Seiten): sonst
    sähe man aus dem Tunnel neben der Galerie ins Freie; Dach und Wand deckt die Galerie-Stirnfläche ab (eine Fläche
    in derselben Ebene und Richtung flackerte). Galerie-Seite (zur Galerie gerichtet) = lichter Galerie-Querschnitt
    minus Röhre samt Schale `shell` und Portalkragen `frame` (shapely-Fläche oder None; die Kragen-Stirnseite liegt in
    derselben Ebene): ragt die Galerie über den Bogen hinaus, sähe man sonst aus der Galerie ins Freie.
    """
    from .tunnel_mesh import shell_cross_section

    from shapely.geometry import Polygon, box

    tube = Polygon(arc_cross_section(radius, arc_segments))
    section = box(-half_opening, 0.0, half_opening, opening_height)
    top = opening_height + roof
    body = box(-half_opening, 0.0, half_opening, top)
    if wall > 0.0:
        if wall_side >= 0:
            body = body.union(box(half_opening, 0.0, half_opening + wall, top))
        if wall_side <= 0:
            body = body.union(box(-half_opening - wall, 0.0, -half_opening, top))
    tube_body = Polygon(shell_cross_section(radius, arc_segments, shell)) if shell > 0.0 else tube
    if frame is not None:
        tube_body = tube_body.union(frame)
    return tube.difference(body), section.difference(tube_body)


def _collar_frame(portal: Dict):
    """Außenkontur des Portalkragens (across, height relativ zum Röhrenboden): Rechteck bis ±portal["half_width"]
    (seitlich mindestens collar_min_side stark), oben portal["collar"] über der Krone, unten ebenso tief unter der
    Fahrbahn."""
    from shapely.geometry import box

    crown, collar, half_width = portal["crown"], portal["collar"], portal["half_width"]
    return box(-half_width, -collar, half_width, crown + collar)


def _build_collar_mesh(portal: Dict, material: str, arc_segments: int, tile_m: float) -> Dict:
    """
    Portalkragen, außen rechteckig (_collar_frame()), portal["collar_depth"] Meter tief. Die Stirnseite liegt in der
    um portal["tilt"] gekippten Ebene der Röhren-Stirnseite (ein Punkt in Höhe h rückt um h * tilt in den Berg, siehe
    tunnel_mesh._end_shift()), die Rückseite parallel dahinter. Stirnseite = Rechteck minus lichter Querschnitt
    (trifft exakt den ersten Röhrenring), Rückseite = Rechteck minus Röhrenschale.
    """
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon

    from .tunnel_mesh import shell_cross_section

    floor_z, radius, depth = portal["floor_z"], portal["radius"], portal.get("collar_depth", portal["length"])
    shell, tilt = portal.get("shell", 0.0), portal.get("tilt", 0.0)
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

    face(frame.difference(Polygon(arc_cross_section(radius, arc_segments))), 0.0, [-ux * cos, -uy * cos, sin])
    inner = Polygon(shell_cross_section(radius, arc_segments, shell)) if shell > 0.0 else Polygon(arc_cross_section(radius, arc_segments))
    face(frame.difference(inner), depth, [ux * cos, uy * cos, -sin])

    # Mantel: vier Rechteckseiten von der Stirn- zur Rückseite
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
    """Massive Platte: `region` (shapely-Fläche in (across, height) relativ zum Röhrenboden) von along_from bis along_to
    (entlang der Portalachse) - Vorder- und Rückseite plus umlaufende Kanten, von beiden Seiten sichtbar."""
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
        # Kanten: Außenring gegen den Uhrzeigersinn, Löcher im Uhrzeigersinn -> Außennormale (dh, -dc) je Kante
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
    Portalbauwerk eines offenen Portals bzw. Übergangs: rechteckiger Kragen (nur bei portal["collar"] > 0, sonst ist der
    Stirnring der Röhre das Portal) plus - beim Übergang in eine Galerie - die Flächen zwischen Röhrenbogen und
    Galerie-Querschnitt (transition_regions()) als massive Platten `cover_thickness` dick: die Stufe reicht von der
    Portalebene in die Röhre, die Galerie-Abdeckung in die Galerie.
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
