"""
Road marking lines as thin mesh strips on structure floors (bridges, galleries, tunnels).

On structures a DecalRoad does not work (in tunnels BeamNG projects it onto the terrain above the tube, in galleries it
z-fights with the floor mesh), so the lines from geometry/road_markings.py become flat strips a few millimeters above
the floor. They use the same marking material and the same UV layout as a DecalRoad - u across the line 0..1, v along
the line in texture repeats (length / textureLength) - so the dash pattern matches the marking decals on the approach.
"""

from typing import Dict, List, Mapping, Sequence

import numpy as np

from ..walls.mesh_parts import MeshBuilder, offset_points


def build_marking_meshes(lines: Sequence[Dict], texture_lengths: Mapping[str, float], lift: float) -> List[Dict]:
    """
    One mesh per marking line.

    Args:
        lines: [{"name", "material", "nodes": [[x, y, z, width], ...]}, ...] (see terrain_workflow._road_marking_lines())
        texture_lengths: material -> textureLength of the marking decal, in meters
        lift: height of the strip above the line nodes, in meters

    Returns:
        [{"id", "vertices", "uvs", "normals", "faces": {material: [...]}}, ...]
    """
    meshes = []
    for line in lines:
        nodes = np.asarray(line["nodes"], dtype=float)
        if len(nodes) < 2:
            continue
        xy, z = nodes[:, :2], nodes[:, 2] + lift
        left, right = offset_points(xy, float(nodes[0, 3]) / 2.0, closed=False)
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        v = along / texture_lengths[line["material"]]

        builder = MeshBuilder()
        for i in range(len(nodes) - 1):
            j = i + 1
            builder.quad(
                [[*left[i], z[i]], [*left[j], z[j]], [*right[j], z[j]], [*right[i], z[i]]],
                [[0.0, v[i]], [0.0, v[j]], [1.0, v[j]], [1.0, v[i]]],
                [0.0, 0.0, 1.0],
            )
        meshes.append({
            "id": line["name"],
            "vertices": np.array(builder.vertices, dtype=float),
            "uvs": np.array(builder.uvs, dtype=float),
            "normals": np.array(builder.normals, dtype=float),
            "faces": {line["material"]: builder.faces},
        })
    return meshes
