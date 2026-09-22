"""
Brücken aus OSM-Linien (highway=* mit bridge=*): generisches Beton-Deck mit dem Fahrbahnmaterial der Straße
obenauf und rechteckigen Stützpfeilern zum natürlichen Gelände darunter (siehe Design-Spec Abschnitt 4).

Das Deck folgt NICHT dem Gelände (im Gegensatz zu den Mauern) - seine Höhe kommt aus dem linear interpolierten
Brücken-Höhenprofil (geometry/road_structures.py + geometry/polygon.py), das schon in den übergebenen `coords`
steckt. Nur die Pfeiler reichen bis zum natürlichen Gelände darunter (`ground_at`).
"""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..walls.mesh_parts import MeshBuilder, add_box_column, offset_points

HeightAt = Callable[[np.ndarray, np.ndarray], np.ndarray]


def build_bridge_mesh(
    coords: Sequence[Tuple[float, float, float]],
    width: float,
    ground_at: HeightAt,
    deck_material: str,
    pier_material: str,
    deck_thickness: float = 0.6,
    pier_spacing: float = 25.0,
    pier_size: float = 1.5,
    min_pier_clearance: float = 1.0,
    tile_m: float = 5.0,
) -> Dict:
    """
    Deck- und Pfeiler-Mesh für eine Brücke entlang `coords` (bereits das Brücken-Höhenprofil, x,y,z je Punkt).

    Returns:
        {"vertices": (N,3), "uvs": (N,2), "normals": (N,3), "faces": {deck_material: [...], pier_material: [...]}}
    """
    points = np.array(coords, dtype=float)
    xy = points[:, :2]
    top = points[:, 2]
    bottom = top - deck_thickness

    left, right = offset_points(xy, width / 2.0, closed=False)

    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    along = cum / tile_m
    across = width / tile_m

    def p3(pt_xy, z):
        return [float(pt_xy[0]), float(pt_xy[1]), float(z)]

    deck_builder = MeshBuilder()
    for i in range(len(points) - 1):
        j = i + 1
        u0, u1 = along[i], along[j]
        direction = xy[j] - xy[i]
        direction = direction / np.linalg.norm(direction)
        side_normal = [float(-direction[1]), float(direction[0]), 0.0]

        # Oberseite (Fahrbahn)
        deck_builder.quad(
            [p3(left[i], top[i]), p3(left[j], top[j]), p3(right[j], top[j]), p3(right[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, across], [u0, across]],
            [0.0, 0.0, 1.0],
        )
        # Unterseite
        deck_builder.quad(
            [p3(left[i], bottom[i]), p3(right[i], bottom[i]), p3(right[j], bottom[j]), p3(left[j], bottom[j])],
            [[u0, 0.0], [u0, across], [u1, across], [u1, 0.0]],
            [0.0, 0.0, -1.0],
        )
        # Fascia links/rechts
        deck_builder.quad(
            [p3(left[i], bottom[i]), p3(left[j], bottom[j]), p3(left[j], top[j]), p3(left[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, deck_thickness / tile_m], [u0, deck_thickness / tile_m]],
            side_normal,
        )
        deck_builder.quad(
            [p3(right[i], bottom[i]), p3(right[j], bottom[j]), p3(right[j], top[j]), p3(right[i], top[i])],
            [[u0, 0.0], [u1, 0.0], [u1, deck_thickness / tile_m], [u0, deck_thickness / tile_m]],
            [-side_normal[0], -side_normal[1], 0.0],
        )

    # Stirnflächen an den beiden Enden
    for index, sign, neighbour in ((0, -1.0, 1), (len(points) - 1, 1.0, len(points) - 2)):
        direction = xy[1] - xy[0] if index == 0 else xy[-1] - xy[neighbour]
        direction = direction / np.linalg.norm(direction)
        deck_builder.quad(
            [p3(left[index], bottom[index]), p3(right[index], bottom[index]), p3(right[index], top[index]), p3(left[index], top[index])],
            [[0.0, 0.0], [across, 0.0], [across, deck_thickness / tile_m], [0.0, deck_thickness / tile_m]],
            [float(sign * direction[0]), float(sign * direction[1]), 0.0],
        )

    # Pfeiler: alle pier_spacing Meter entlang der Bogenlänge, nur wenn ausreichend Abstand zum Gelände besteht
    pier_builder = MeshBuilder()
    total_len = float(cum[-1])
    pier_positions = np.arange(pier_spacing, total_len, pier_spacing) if total_len > pier_spacing else np.array([])
    for s in pier_positions:
        idx = max(1, min(int(np.searchsorted(cum, s)), len(points) - 1))
        t = (s - cum[idx - 1]) / max(cum[idx] - cum[idx - 1], 1e-9)
        cx = xy[idx - 1, 0] + t * (xy[idx, 0] - xy[idx - 1, 0])
        cy = xy[idx - 1, 1] + t * (xy[idx, 1] - xy[idx - 1, 1])
        deck_bottom_z = float(bottom[idx - 1] + t * (bottom[idx] - bottom[idx - 1]))
        ground_z = float(ground_at(np.array([cx]), np.array([cy]))[0])
        if deck_bottom_z - ground_z < min_pier_clearance:
            continue
        add_box_column(pier_builder, cx, cy, ground_z, deck_bottom_z, pier_size, tile_m)

    all_vertices = deck_builder.vertices + pier_builder.vertices
    all_uvs = deck_builder.uvs + pier_builder.uvs
    all_normals = deck_builder.normals + pier_builder.normals
    pier_offset = len(deck_builder.vertices)
    pier_faces = [[a + pier_offset, b + pier_offset, c + pier_offset] for a, b, c in pier_builder.faces]

    return {
        "vertices": np.array(all_vertices, dtype=float),
        "uvs": np.array(all_uvs, dtype=float),
        "normals": np.array(all_normals, dtype=float),
        "faces": {deck_material: deck_builder.faces, pier_material: pier_faces},
    }


def build_bridges(
    bridges: Sequence[Dict],
    ground_at: HeightAt,
    pier_material: str,
    deck_thickness: float = 0.6,
    pier_spacing: float = 25.0,
    pier_size: float = 1.5,
    min_pier_clearance: float = 1.0,
) -> List[Dict]:
    """Mesh-Dicts für den DAE-Export, eines je Brücke (`bridges`: [{"id","coords","width","deck_material"}, ...])."""
    meshes = []
    for bridge in bridges:
        coords = bridge["coords"]
        if len(coords) < 2:
            continue
        mesh = build_bridge_mesh(
            coords, bridge["width"], ground_at, bridge["deck_material"], pier_material,
            deck_thickness=deck_thickness, pier_spacing=pier_spacing, pier_size=pier_size, min_pier_clearance=min_pier_clearance,
        )
        meshes.append({"id": f"bridge_{bridge['id']}", **mesh})
    return meshes
