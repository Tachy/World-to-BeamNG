"""
Brücken aus OSM-Linien (highway=* mit bridge=*): Beton-Deck mit einem echten Straßenbrücken-Querschnitt
(Fahrbahn mit Straßenmaterial, beidseits ein Bordstein, darauf ein Geländer aus Pfosten + Handlauf) und
rechteckigen Stützpfeilern zum natürlichen Gelände darunter (siehe Design-Spec Abschnitt 4).

Das Deck folgt NICHT dem Gelände (im Gegensatz zu den Mauern) - seine Höhe kommt aus dem linear interpolierten
Brücken-Höhenprofil (geometry/road_structures.py + geometry/polygon.py), das schon in den übergebenen `coords`
steckt. Nur die Pfeiler reichen bis zum natürlichen Gelände darunter (`ground_at`). Bordstein und Geländer
folgen dem Deck-Höhenprofil, nicht dem Gelände.
"""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, add_box_column, offset_points

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _arc_length(xy: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _interp_at(cum: np.ndarray, arr: np.ndarray, s: float):
    """Interpoliert `arr` (1D oder 2D, ein Wert je Punkt von `cum`) an der Bogenlänge `s`."""
    idx = max(1, min(int(np.searchsorted(cum, s)), len(cum) - 1))
    t = (s - cum[idx - 1]) / max(cum[idx] - cum[idx - 1], 1e-9)
    return arr[idx - 1] + t * (arr[idx] - arr[idx - 1])


def _build_edge_beam(xy_line: np.ndarray, top_z: np.ndarray, thickness: float, tile_m: float) -> MeshBuilder:
    """Dünner, rechteckiger Balken entlang `xy_line` (Handlauf): Ober-/Unterseite plus beide Seitenflächen.
    `top_z` gibt die Oberkante je Punkt von `xy_line` an (folgt damit demselben Höhenprofil wie das Deck)."""
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
    pier_size: float = 1.5,
    min_pier_clearance: float = 1.0,
    curb_width: float = 0.25,
    curb_height: float = 0.15,
    railing_height: float = 0.9,
    railing_post_spacing: float = 2.0,
    railing_post_size: float = 0.08,
    tile_m: float = 5.0,
) -> Dict:
    """
    Deck-, Bordstein-, Geländer- und Pfeiler-Mesh für eine Brücke entlang `coords` (bereits das
    Brücken-Höhenprofil, x,y,z je Punkt).

    Querschnitt von außen nach innen: Geländer (Pfosten + Handlauf) - Bordstein (curb_width/curb_height,
    pier_material) - Fahrbahn (deck_material, um 2x curb_width schmaler als `width`).

    Returns:
        {"vertices": (N,3), "uvs": (N,2), "normals": (N,3),
         "faces": {deck_material: [...], pier_material: [...], railing_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    top = points[:, 2]
    bottom = top - deck_thickness
    curb_top = top + curb_height

    outer_left, outer_right = offset_points(xy, width / 2.0, closed=False)
    inner_left, inner_right = offset_points(xy, max(width / 2.0 - curb_width, 0.0), closed=False)

    cum = _arc_length(xy)
    along = cum / tile_m
    carriageway_across = max(width - 2.0 * curb_width, 0.0) / tile_m
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

        # Fahrbahn (Oberseite, zwischen den Bordsteinen)
        deck_builder.quad(
            [p3(inner_left[i], top[i]), p3(inner_left[j], top[j]), p3(inner_right[j], top[j]), p3(inner_right[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, carriageway_across], [u0, carriageway_across]],
            [0.0, 0.0, 1.0],
        )
        # Unterseite (volle Breite)
        deck_builder.quad(
            [p3(outer_left[i], bottom[i]), p3(outer_right[i], bottom[i]), p3(outer_right[j], bottom[j]), p3(outer_left[j], bottom[j])],
            [[u0, 0.0], [u0, width / tile_m], [u1, width / tile_m], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        # Fascia links/rechts (Deck-Unterkante bis Fahrbahnniveau)
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

        # Bordstein links: Oberseite, Außen- (Fortsetzung der Fascia) und Innenfläche (zur Fahrbahn hin)
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
        # Bordstein rechts (gespiegelt)
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

    # Stirnflächen an den beiden Enden (Deck volle Höhe + Bordstein-Aufsatz beidseits)
    for index, sign, neighbour in ((0, -1.0, 1), (len(points) - 1, 1.0, len(points) - 2)):
        direction = xy[1] - xy[0] if index == 0 else xy[-1] - xy[neighbour]
        direction = direction / np.linalg.norm(direction)
        face_normal = [float(sign * direction[0]), float(sign * direction[1]), 0.0]
        deck_builder.quad(
            [p3(outer_left[index], bottom[index]), p3(outer_right[index], bottom[index]), p3(outer_right[index], top[index]), p3(outer_left[index], top[index])],
            [[0.0, 0.0], [width / tile_m, 0.0], [width / tile_m, deck_thickness / tile_m], [0.0, deck_thickness / tile_m]],
            face_normal,
        )
        for edge_out, edge_in in ((outer_left, inner_left), (outer_right, inner_right)):
            pier_builder.quad(
                [p3(edge_out[index], top[index]), p3(edge_in[index], top[index]), p3(edge_in[index], curb_top[index]), p3(edge_out[index], curb_top[index])],
                [[0.0, 0.0], [curb_across, 0.0], [curb_across, curb_height / tile_m], [0.0, curb_height / tile_m]],
                face_normal,
            )

    # Pfeiler: alle pier_spacing Meter entlang der Bogenlänge, nur wenn ausreichend Abstand zum Gelände besteht
    total_len = float(cum[-1])
    pier_positions = np.arange(pier_spacing, total_len, pier_spacing) if total_len > pier_spacing else np.array([])
    for s in pier_positions:
        cx, cy = _interp_at(cum, xy, s)
        deck_bottom_z = float(_interp_at(cum, bottom, s))
        ground_z = float(ground_at(np.array([cx]), np.array([cy]))[0])
        if deck_bottom_z - ground_z < min_pier_clearance:
            continue
        add_box_column(pier_builder, cx, cy, ground_z, deck_bottom_z, pier_size, tile_m)

    # Geländer: Pfosten + durchlaufender Handlauf beidseits, auf der Bordstein-Oberkante
    railing_builder = MeshBuilder()
    rail_top = curb_top + railing_height + railing_post_size / 2.0
    post_positions = np.arange(0.0, total_len + 1e-6, railing_post_spacing) if total_len > 0 else np.array([])
    for edge_xy in (outer_left, outer_right):
        for s in post_positions:
            px, py = _interp_at(cum, edge_xy, s)
            post_bottom_z = float(_interp_at(cum, curb_top, s))
            post_top_z = post_bottom_z + railing_height
            add_box_column(railing_builder, px, py, post_bottom_z, post_top_z, railing_post_size, tile_m)
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
    pier_size: float = 1.5,
    min_pier_clearance: float = 1.0,
    curb_width: float = 0.25,
    curb_height: float = 0.15,
    railing_height: float = 0.9,
    railing_post_spacing: float = 2.0,
    railing_post_size: float = 0.08,
) -> List[Dict]:
    """Mesh-Dicts für den DAE-Export, eines je Brücke (`bridges`: [{"id","coords","width","deck_material"}, ...])."""
    meshes = []
    for bridge in bridges:
        coords = bridge["coords"]
        if len(coords) < 2:
            continue
        mesh = build_bridge_mesh(
            coords, bridge["width"], ground_at, bridge["deck_material"], pier_material, railing_material,
            deck_thickness=deck_thickness, pier_spacing=pier_spacing, pier_size=pier_size, min_pier_clearance=min_pier_clearance,
            curb_width=curb_width, curb_height=curb_height, railing_height=railing_height,
            railing_post_spacing=railing_post_spacing, railing_post_size=railing_post_size,
        )
        meshes.append({"id": f"bridge_{bridge['id']}", **mesh})
    return meshes
