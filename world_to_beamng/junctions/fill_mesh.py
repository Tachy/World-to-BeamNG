"""
Fill mesh of the rounded junction corners: per corner a triangle fan from the corner point (the intersection of both
carriageway edges) to consecutive points of the fill outline (kerb A, arc, kerb B) - the fill region is star-shaped from
there, since the arc bulges towards it. Lifted slightly above the embedded terrain, world-aligned UVs so neighbouring fills continue seamlessly; one mesh per
junction node.
"""

from typing import Dict, List, Sequence

import numpy as np

from ..walls.mesh_parts import MeshBuilder, unit_vector


def build_junction_meshes(corners: Sequence[Dict], lift: float, tile_m: float) -> List[Dict]:
    """
    Args:
        corners: corner dicts of find_junction_corners() ("node", "corner_point", "rim" or "arc", "surface")
        lift: height above the embedded terrain, in meters
        tile_m: texture tile, in meters (u = x / tile_m, v = y / tile_m)

    Returns:
        [{"id", "vertices", "uvs", "normals", "faces": {"<surface>_structure": [...]}}], one per junction node
    """
    by_node: Dict[tuple, List[Dict]] = {}
    for corner in corners:
        by_node.setdefault(tuple(np.round(np.asarray(corner["node"])[:2], 2)), []).append(corner)

    meshes = []
    for number, node_corners in enumerate(by_node.values()):
        builders: Dict[str, MeshBuilder] = {}
        for corner in node_corners:
            builder = builders.setdefault(f"{corner['surface']}_structure", MeshBuilder())
            p = np.asarray(corner["corner_point"], dtype=float) + [0.0, 0.0, lift]
            rim = np.asarray(corner.get("rim", corner["arc"]), dtype=float) + [0.0, 0.0, lift]
            for a, b in zip(rim[:-1], rim[1:]):
                if np.linalg.norm(np.cross(a - p, b - p)) < 1e-9:
                    continue  # degenerate (outline point on the corner point)
                normal = unit_vector(np.cross(a - p, b - p))
                if normal[2] < 0:
                    normal = [-c for c in normal]
                tri = [list(map(float, v)) for v in (p, a, b)]
                builder.triangle(tri, [[v[0] / tile_m, v[1] / tile_m] for v in tri], normal)
        vertices, uvs, normals, faces = [], [], [], {}
        for material, builder in builders.items():
            offset = len(vertices)
            vertices += builder.vertices
            uvs += builder.uvs
            normals += builder.normals
            faces[material] = [[i + offset for i in face] for face in builder.faces]
        meshes.append({"id": f"junction_{number}", "vertices": np.array(vertices, dtype=float),
                       "uvs": np.array(uvs, dtype=float), "normals": np.array(normals, dtype=float), "faces": faces})
    return meshes
