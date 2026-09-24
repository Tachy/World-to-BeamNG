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
) -> Dict:
    """
    Röhren-Mesh (Boden + kreisrunder 240°-Bogen darüber) entlang `coords` (bereits das Tunnel-Höhenprofil).
    Radius und Kronenhöhe ergeben sich aus `width` (siehe tunnel_radius()/tunnel_crown_height()).

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

    all_vertices = floor_builder.vertices + wall_builder.vertices
    all_uvs = floor_builder.uvs + wall_builder.uvs
    all_normals = floor_builder.normals + wall_builder.normals
    wall_offset = len(floor_builder.vertices)
    wall_faces = [[a + wall_offset, b + wall_offset, c + wall_offset] for a, b, c in wall_builder.faces]

    return {
        "vertices": np.array(all_vertices, dtype=float),
        "uvs": np.array(all_uvs, dtype=float),
        "normals": np.array(all_normals, dtype=float),
        "faces": {floor_material: floor_builder.faces, wall_material: wall_faces},
    }


def build_tunnels(plans: Sequence[Dict], wall_material: str, portal_material: str, arc_segments: int = 12) -> List[Dict]:
    """
    Mesh-Dicts für den DAE-Export: je Tunnel-Kette die Röhre plus ein Portalbauwerk je offenem Ende.

    Args:
        plans: Ergebnis von tunnel_portal.plan_tunnels() - Portale mit bereits gesetzter "top_z"/"bottom_z"
            (siehe terrain/tunnel_terrain.py::shape_terrain_for_tunnels())
    """
    from .tunnel_portal import build_portal_block_mesh

    meshes = []
    for plan in plans:
        tube = build_tunnel_mesh(plan["coords"], plan["tube_width"], plan["floor_material"], wall_material, arc_segments=arc_segments)
        meshes.append({"id": f"tunnel_{plan['id']}", **tube})
        for portal in [p for p in plan["portals"] if p.get("open", True)]:
            block = build_portal_block_mesh(portal, portal_material, arc_segments=arc_segments)
            meshes.append({"id": f"tunnel_{plan['id']}_portal_{portal['label']}", **block})
    return meshes
