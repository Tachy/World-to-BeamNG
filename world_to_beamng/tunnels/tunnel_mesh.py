"""
Tunnel aus OSM-Linien (highway=* mit tunnel=yes/culvert/building_passage): kreisrunde Röhre - Standard-
Tunnelprofil, 240° Kreisbogen über einer flachen Bodensehne (Fahrbahn), die restlichen 120° liegen unterhalb der
Sehne und werden nicht modelliert (unsichtbare Sohle) - entlang des linear interpolierten Höhenprofils (siehe
geometry/road_structures.py + geometry/polygon.py), mit einem Portalbauwerk an beiden Enden (siehe
tunnels/tunnel_portal.py). Das Gelände über der Röhre und am Portal formt terrain/tunnel_terrain.py.

Ein Tunnel kann in OSM aus mehreren aneinandergereihten Ways bestehen (die Junction-Erkennung teilt Tunnel
nicht mehr, siehe geometry/junctions.py::build_junction_network()). chain_tunnel_pieces() fügt solche Stücke zu
EINER durchgehenden Röhre zusammen - sonst bekäme jedes Stück eigene Portale mitten im Berg.
"""

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..walls.mesh_parts import MeshBuilder, offset_points

ARC_SPAN_DEG = 240.0  # Kreisbogen über der Fahrbahn
ARC_START_DEG = -30.0  # Startwinkel (rechter Bodenrand), Standardkreis-Konvention (0°=+x, CCW)
JOINT_TOLERANCE = 0.05  # so nah müssen sich zwei Stück-Enden kommen, um als Stoß zu gelten, in Metern


def tunnel_radius(width: float) -> float:
    """Radius der kreisrunden Tunnelröhre aus der Bodenbreite (Bodensehne = sqrt(3)*R bei 240°/120°-Aufteilung)."""
    return width / math.sqrt(3.0)


def tunnel_crown_height(width: float) -> float:
    """Lichte Höhe (Boden bis Kronenscheitel) einer kreisrunden Tunnelröhre der gegebenen Bodenbreite."""
    return 1.5 * tunnel_radius(width)


def arc_cross_section(radius: float, segments: int) -> List[Tuple[float, float]]:
    """
    (across, height)-Punkte des 240°-Kreisbogens über der Fahrbahn, `segments` Streifen (segments+1 Punkte), vom
    rechten Bodenrand (θ=-30°) über die Krone (θ=90°) zum linken Bodenrand (θ=210°). Boden ist y=0, "across" ist
    quer zur Fahrtrichtung (positiv = rechts). Kreismittelpunkt liegt bei (0, radius/2) - siehe Design-Spec
    Abschnitt 5 für die Herleitung.
    """
    points = []
    for k in range(segments + 1):
        theta = math.radians(ARC_START_DEG + (k / segments) * ARC_SPAN_DEG)
        points.append((radius * math.cos(theta), radius / 2.0 + radius * math.sin(theta)))
    return points


def shell_cross_section(radius: float, segments: int, thickness: float) -> List[Tuple[float, float]]:
    """
    Außenkontur (across, height) der Röhrenschale: 240°-Bogen mit Radius radius + thickness um denselben
    Mittelpunkt (0, radius/2) wie der Innenbogen, unten geschlossen durch eine Bodenplatte `thickness` unter der
    Fahrbahn. Umlauf: rechter Bogenfuß über die Krone zum linken Bogenfuß, dann links unten, rechts unten.
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
    """Dünnt die (bereits linear profilierte) Centerline auf einen festen Bogenlängen-Abstand aus (XYZ gemeinsam,
    da das Höhenprofil affin in der Bogenlänge ist - siehe geometry/polygon.py::apply_structure_elevation_profiles()).
    Hält die Vertex-Zahl auch bei sehr langen Tunneln (z.B. 16,9 km) im Rahmen."""
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
    Fügt Tunnel-Stücke, die sich an einem Endpunkt treffen, zu durchgehenden Ketten zusammen.

    Verkettet wird nur an eindeutigen Stößen: genau zwei gleichartige Stück-Enden (gleiche Breite, gleiches
    Bodenmaterial) am selben Punkt. Andersartige Tunnel am selben Punkt zählen nicht mit - z.B. ein Fußweg-Tunnel,
    der am selben OSM-Knoten abzweigt. Die
    Laufrichtung einzelner Stücke wird bei Bedarf umgedreht.

    Args:
        tunnels: [{"id", "coords", "width", "floor_material"}, ...]

    Returns:
        [{"id" (des ersten Stücks), "coords", "width", "floor_material"}, ...]
    """
    pieces = [t for t in tunnels if len(t["coords"]) >= 2]
    if not pieces:
        return []

    # Stück-Enden, die näher als JOINT_TOLERANCE beieinanderliegen, bilden einen Stoß (Clipping am Kartenrand
    # verschiebt Endpunkte um Millimeter) - Gruppen per Union-Find über alle nahen Paare.
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
            return None  # Stück schließt sich selbst zum Ring
        return other

    visited = set()
    chains = []
    for first in range(len(pieces)):
        if first in visited:
            continue
        # Rückwärts bis zum freien Anfang der Kette laufen (mit Schutz gegen Ringe).
        head, head_at_start = first, True
        seen = {first}
        while True:
            prev = partner(head, head_at_start)
            if prev is None or prev[0] in seen:
                break
            head, head_at_start = prev[0], not prev[1]
            seen.add(head)

        # Vorwärts: jedes Stück so ausrichten, dass es am Stoß zum Vorgänger beginnt.
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
) -> Dict:
    """
    Röhren-Mesh (Boden + kreisrunder 240°-Bogen darüber) entlang `coords` (bereits das Tunnel-Höhenprofil).
    Radius und Kronenhöhe ergeben sich aus `width` (siehe tunnel_radius()/tunnel_crown_height()).

    Mit shell_thickness > 0 bekommt die Röhre eine Außenschale (siehe shell_cross_section()) samt Stirnringen an
    beiden Enden: sie ist dann auch von außen ein massiver Zylinder und darf frei im Gelände stehen. cap_start/
    cap_end = False lässt den Stirnring weg (dort steht ein Portalbauwerk in derselben Ebene - sonst Z-Fighting).

    Die Querschnitts-Ringe sitzen an den Centerline-Punkten und stehen dort auf Gehrung (wie die Bodenkanten aus
    offset_points()): benachbarte Segmente teilen sich exakt denselben Ring, die Röhre ist auch in Kurven dicht.

    Returns:
        {"vertices", "uvs", "normals", "faces": {floor_material: [...], wall_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    floor_z = points[:, 2]
    radius = tunnel_radius(width)
    arc = arc_cross_section(radius, arc_segments)

    left, right = offset_points(xy, width / 2.0, closed=False)
    # Gehrungs-Vektor je Centerline-Punkt (inkl. Gehrungs-Verlängerung), zeigt nach rechts der Laufrichtung
    miter_right = (right - xy) / (width / 2.0)
    rings = np.empty((len(points), arc_segments + 1, 3))
    for k, (across, height) in enumerate(arc):
        rings[:, k, :2] = xy + miter_right * across
        rings[:, k, 2] = floor_z + height
    rings[:, 0, :2] = right  # Bodenränder exakt wie das Boden-Mesh (kein Rundungsspalt)
    rings[:, arc_segments, :2] = left

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
        perp_right = np.array([direction[1], -direction[0]])  # zeigt "rechts" der Laufrichtung

        # Boden (Normale nach oben, ins Rohrinnere)
        floor_builder.quad(
            [p3(left[i], floor_z[i]), p3(left[j], floor_z[j]), p3(right[j], floor_z[j]), p3(right[i], floor_z[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across_floor], [u0, across_floor]],
            [0.0, 0.0, 1.0],
        )

        # Kreisbogen (240°) über der Fahrbahn, in arc_segments Streifen
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
        builders.append((shell_material or wall_material, _build_shell(xy, floor_z, miter_right, radius, arc_segments, shell_thickness, tile_m, cap_start, cap_end)))

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


def _build_shell(xy, floor_z, miter_right, radius, arc_segments, thickness, tile_m, cap_start=True, cap_end=True) -> MeshBuilder:
    """Außenschale der Röhre (Mantel entlang der Achse, Normalen nach außen) plus Stirnring an den gewünschten Enden."""
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import Polygon

    profile = shell_cross_section(radius, arc_segments, thickness)
    center = np.array([0.0, radius / 2.0])
    builder = MeshBuilder()

    def world(i, across, height):
        return [float(xy[i, 0] + miter_right[i, 0] * across), float(xy[i, 1] + miter_right[i, 1] * across), float(floor_z[i] + height)]

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

    # Stirnringe: Außenkontur minus lichter Querschnitt, nach außen (vom Tunnel weg) gerichtet
    ring = Polygon(profile).difference(Polygon(arc_cross_section(radius, arc_segments)))
    triangles = [list(t.exterior.coords)[:3] for t in constrained_delaunay_triangles(ring).geoms]
    for index, sign, cap in ((0, -1.0, cap_start), (len(xy) - 1, 1.0, cap_end)):
        if not cap:
            continue
        neighbour = 1 if index == 0 else index - 1
        axis = (xy[index] - xy[neighbour]) if index else (xy[neighbour] - xy[index])
        axis = axis / np.linalg.norm(axis)
        normal = [float(sign * axis[0]), float(sign * axis[1]), 0.0]
        for tri in triangles:
            builder.triangle([world(index, c, h) for c, h in tri], [[c / tile_m, h / tile_m] for c, h in tri], normal)
    return builder


def build_tunnels(plans: Sequence[Dict], wall_material: str, portal_material: str, arc_segments: int = 12) -> List[Dict]:
    """
    Mesh-Dicts für den DAE-Export: je Tunnel-Kette die Röhre (mit Außenschale aus plan["shell"], Material wie die
    Portale) plus ein Portalbauwerk je offenem Ende.

    Args:
        plans: Ergebnis von tunnel_portal.plan_tunnels() - Portale mit bereits gesetzter "top_z"/"bottom_z"
            (siehe terrain/tunnel_terrain.py::shape_terrain_for_tunnels())
    """
    from .tunnel_portal import build_portal_block_mesh

    meshes = []
    for plan in plans:
        start, end = plan["portals"]
        tube = build_tunnel_mesh(
            plan["coords"], plan["tube_width"], plan["floor_material"], wall_material, arc_segments=arc_segments,
            shell_thickness=plan.get("shell", 0.0), shell_material=portal_material,
            cap_start=not start.get("open", True), cap_end=not end.get("open", True),
        )
        meshes.append({"id": f"tunnel_{plan['id']}", **tube})
        for portal in [p for p in plan["portals"] if p.get("open", True)]:
            block = build_portal_block_mesh(portal, portal_material, arc_segments=arc_segments)
            meshes.append({"id": f"tunnel_{plan['id']}_portal_{portal['label']}", **block})
    return meshes
