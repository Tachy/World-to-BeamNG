"""
Tunnel aus OSM-Linien (highway=* mit tunnel=yes/culvert/building_passage): kreisrunde Röhre - Standard-
Tunnelprofil, 240° Kreisbogen über einer flachen Bodensehne (Fahrbahn), die restlichen 120° liegen unterhalb der
Sehne und werden nicht modelliert (unsichtbare Sohle) - entlang des linear interpolierten Höhenprofils (siehe
geometry/road_structures.py + geometry/polygon.py), mit an die natürliche Hangneigung angepassten Portal-Rahmen
an beiden Enden (siehe Design-Spec Abschnitt 5). Die Röhre selbst hat rechtwinklige (nicht geschnittene) Enden -
die Schräge steckt im separaten, flachen Portal-Rahmen-Mesh, der die Öffnung umgibt (der Rahmen ist von außen
sichtbar, das Rohr-Ende dahinter nicht). Die Heightmap bleibt unverändert - die Röhre liegt "im Berg".
"""

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, offset_points
from .portal import portal_axial_shift, sample_slope_along_axis

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]

ARC_SPAN_DEG = 240.0  # Kreisbogen über der Fahrbahn
ARC_START_DEG = -30.0  # Startwinkel (rechter Bodenrand), Standardkreis-Konvention (0°=+x, CCW)


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

    Returns:
        {"vertices", "uvs", "normals", "faces": {floor_material: [...], wall_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    floor_z = points[:, 2]
    radius = tunnel_radius(width)
    arc = arc_cross_section(radius, arc_segments)

    left, right = offset_points(xy, width / 2.0, closed=False)

    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(seg_len)]) / tile_m
    across_floor = width / tile_m
    across_arc = (radius * math.radians(ARC_SPAN_DEG)) / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    def arc_ring(i: int, k: int, perp_right: np.ndarray) -> List[float]:
        """Weltposition des Ring-Punkts k (0=rechter Bodenrand, arc_segments=linker Bodenrand) an Centerline-Punkt
        i. Die beiden Bodenrand-Punkte sind exakt right[i]/left[i] (nahtlos zum Boden-Mesh), die Zwischenpunkte
        folgen dem Kreisbogen relativ zur Segment-Richtung (kleine Facette an Kurven statt Gehrung wie bei
        offset_points() - unauffällig bei der groben Tunnel-Resampling-Schrittweite)."""
        if k == 0:
            return p3(right[i], floor_z[i])
        if k == arc_segments:
            return p3(left[i], floor_z[i])
        ax, ay = arc[k]
        return [float(xy[i, 0] + perp_right[0] * ax), float(xy[i, 1] + perp_right[1] * ax), float(floor_z[i] + ay)]

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
                [arc_ring(i, k, perp_right), arc_ring(j, k, perp_right), arc_ring(j, k + 1, perp_right), arc_ring(i, k + 1, perp_right)],
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


def portal_frame_corners(
    xy_point: Tuple[float, float],
    axis_direction: Tuple[float, float],
    width: float,
    height: float,
    margin: float,
    floor_z: float,
    slope_along_axis: float,
) -> List[List[float]]:
    """
    4 Eckpunkte (Weltkoordinaten) eines Portal-Rahmen-Rings: unten-links, unten-rechts, oben-rechts, oben-links.
    `margin` vergrößert den Ring gegenüber der reinen Röhrenöffnung (0.0 = deckt genau die Öffnung ab). Die
    oberen Ecken sind entlang `axis_direction` verschoben (siehe portal.portal_axial_shift()), damit der Ring der
    natürlichen Hangneigung folgt statt rechtwinklig zur Achse zu stehen.
    """
    p = np.array(xy_point, dtype=float)
    axis = np.array(axis_direction, dtype=float)
    perp = np.array([-axis[1], axis[0]])
    half_w = width / 2.0 + margin

    bottom_shift = portal_axial_shift(0.0, slope_along_axis)
    top_shift = portal_axial_shift(height + margin, slope_along_axis)
    bottom = p + axis * bottom_shift
    top = p + axis * top_shift

    bl = [float(bottom[0] - perp[0] * half_w), float(bottom[1] - perp[1] * half_w), floor_z]
    br = [float(bottom[0] + perp[0] * half_w), float(bottom[1] + perp[1] * half_w), floor_z]
    tr = [float(top[0] + perp[0] * half_w), float(top[1] + perp[1] * half_w), floor_z + height + margin]
    tl = [float(top[0] - perp[0] * half_w), float(top[1] - perp[1] * half_w), floor_z + height + margin]
    return [bl, br, tr, tl]


def build_portal_frame_mesh(
    xy_point: Tuple[float, float],
    axis_direction: Tuple[float, float],
    width: float,
    height: float,
    floor_z: float,
    slope_along_axis: float,
    frame_margin: float,
    material: str,
) -> Dict:
    """Flacher Rahmen (4 Trapez-Flächen) um die Tunnelöffnung, an die Hangneigung angepasst (siehe portal_frame_corners())."""
    outer = portal_frame_corners(xy_point, axis_direction, width, height, frame_margin, floor_z, slope_along_axis)
    inner = portal_frame_corners(xy_point, axis_direction, width, height, 0.0, floor_z, slope_along_axis)
    axis = np.array(axis_direction, dtype=float)
    normal = [float(-axis[0]), float(-axis[1]), 0.0]  # zeigt vom Tunnelinneren weg (nach außen, sichtbare Seite)

    builder = MeshBuilder()
    for i in range(4):
        j = (i + 1) % 4
        builder.quad([outer[i], outer[j], inner[j], inner[i]], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], normal)

    return {
        "vertices": np.array(builder.vertices, dtype=float),
        "uvs": np.array(builder.uvs, dtype=float),
        "normals": np.array(builder.normals, dtype=float),
        "faces": {material: builder.faces},
    }


def build_tunnel(
    tunnel: Dict,
    ground_at: HeightAt,
    wall_material: str,
    frame_material: str,
    width_margin: float,
    arc_segments: int,
    segment_step: float,
    portal_slope_sample_dist: float,
    frame_margin: float,
) -> List[Dict]:
    """Tunnelröhre (Kreisbogen-Profil) + zwei Portal-Rahmen für einen Tunnel-Way (`tunnel`: {"id","coords","width","floor_material"})."""
    coords = resample_tunnel_coords(tunnel["coords"], segment_step)
    if len(coords) < 2:
        return []
    width = tunnel["width"] + width_margin
    crown_height = tunnel_crown_height(width)
    points = np.array(coords, dtype=float)

    tube = build_tunnel_mesh(coords, width, tunnel["floor_material"], wall_material, arc_segments=arc_segments)
    meshes = [{"id": f"tunnel_{tunnel['id']}", **tube}]

    for index, neighbour, label in ((0, 1, "start"), (len(points) - 1, len(points) - 2, "end")):
        direction = points[neighbour, :2] - points[index, :2]
        direction = direction / np.linalg.norm(direction)
        axis_direction = (float(direction[0]), float(direction[1]))
        slope = sample_slope_along_axis(ground_at, tuple(points[index, :2]), axis_direction, portal_slope_sample_dist)
        frame = build_portal_frame_mesh(
            tuple(points[index, :2]), axis_direction, width, crown_height, float(points[index, 2]), slope, frame_margin, frame_material
        )
        meshes.append({"id": f"tunnel_{tunnel['id']}_portal_{label}", **frame})
    return meshes


def build_tunnels(
    tunnels: Sequence[Dict],
    ground_at: HeightAt,
    wall_material: str,
    frame_material: str,
    width_margin: float = 1.5,
    arc_segments: int = 12,
    segment_step: float = 10.0,
    portal_slope_sample_dist: float = 5.0,
    frame_margin: float = 0.6,
) -> List[Dict]:
    """Mesh-Dicts für den DAE-Export, drei je Tunnel (`tunnels`: [{"id","coords","width","floor_material"}, ...])."""
    meshes = []
    for tunnel in tunnels:
        meshes.extend(build_tunnel(tunnel, ground_at, wall_material, frame_material, width_margin, arc_segments, segment_step, portal_slope_sample_dist, frame_margin))
    return meshes
