"""
Stitching von Lücken zwischen Terrain und Böschungen.
"""

import numpy as np
from scipy.spatial import cKDTree

from .. import config


def stitch_terrain_gaps(
    vertex_manager,
    terrain_vertex_indices,
    road_slope_polygons_2d,
    stitch_radius=10.0,
):
    """Erzeugt zusätzliche Faces, um Lücken zwischen Terrain und Böschungen zu schließen.

    Ansatz: Für jede Böschungsaußenkante (links/rechts) pro Straße werden die Segmente
    gegen nahe Terrain-Vertices (vertex_types==0) gestitcht. Pro Segment entstehen zwei
    Dreiecke: slope_outer[i] -> terrain_a -> terrain_b und slope_outer[i] -> terrain_b -> slope_outer[i+1].
    """

    verts = np.asarray(vertex_manager.get_array())
    if len(verts) == 0 or not terrain_vertex_indices:
        return []

    terrain_vertex_indices = np.asarray(terrain_vertex_indices, dtype=int)
    terrain_xy = verts[terrain_vertex_indices][:, :2]
    terrain_tree = cKDTree(terrain_xy)

    def _stitch_side(indices, road_indices):
        faces = []
        if indices is None or len(indices) < 2:
            return faces
        for i in range(len(indices) - 1):
            v0_idx = int(indices[i])
            v1_idx = int(indices[i + 1])
            v0_xy = verts[v0_idx][:2]
            v1_xy = verts[v1_idx][:2]
            mid_xy = (v0_xy + v1_xy) * 0.5

            r0_idx = int(road_indices[i]) if road_indices is not None else None
            r1_idx = int(road_indices[i + 1]) if road_indices is not None else None

            width0 = (
                np.linalg.norm(verts[v0_idx] - verts[r0_idx])
                if r0_idx is not None
                else 0.0
            )
            width1 = (
                np.linalg.norm(verts[v1_idx] - verts[r1_idx])
                if r1_idx is not None
                else 0.0
            )

            dynamic_radius = max(stitch_radius, width0, width1)
            dynamic_radius += config.GRID_SPACING * 1.5

            d0, nn0 = terrain_tree.query(v0_xy, distance_upper_bound=dynamic_radius)
            d1, nn1 = terrain_tree.query(v1_xy, distance_upper_bound=dynamic_radius)
            dm, nnm = terrain_tree.query(mid_xy, distance_upper_bound=dynamic_radius)

            # Fallback: nimm global nächsten, falls im dynamischen Radius keiner gefunden wurde
            if np.isinf(d0):
                d0, nn0 = terrain_tree.query(v0_xy)
            if np.isinf(d1):
                d1, nn1 = terrain_tree.query(v1_xy)
            if np.isinf(dm):
                dm, nnm = terrain_tree.query(mid_xy)

            if np.isinf(d0) or np.isinf(d1) or np.isinf(dm):
                continue

            t0_idx = int(terrain_vertex_indices[nn0])
            t1_idx = int(terrain_vertex_indices[nn1])
            tm_idx = int(terrain_vertex_indices[nnm])

            # Sammle Dreiecke mit einfacher Entartungs-Prüfung
            tris = []
            if len({v0_idx, t0_idx, tm_idx}) == 3:
                tris.append([v0_idx, t0_idx, tm_idx])
            if len({v0_idx, tm_idx, v1_idx}) == 3:
                tris.append([v0_idx, tm_idx, v1_idx])
            if len({v1_idx, tm_idx, t1_idx}) == 3:
                tris.append([v1_idx, tm_idx, t1_idx])

            if not tris:
                # Fallback auf einfache Verbindung
                if t0_idx == t1_idx:
                    tris.append([v0_idx, t0_idx, v1_idx])
                else:
                    tris.append([v0_idx, t0_idx, t1_idx])
                    tris.append([v0_idx, t1_idx, v1_idx])

            faces.extend(tris)
        return faces

    stitch_faces = []
    for road_meta in road_slope_polygons_2d:
        slope_outer = road_meta.get("slope_outer_indices") or {}
        road_indices = road_meta.get("road_vertex_indices") or {}
        left_outer = slope_outer.get("left")
        right_outer = slope_outer.get("right")

        road_left = road_indices.get("left") if road_indices else None
        road_right = road_indices.get("right") if road_indices else None

        stitch_faces.extend(_stitch_side(left_outer, road_left))
        stitch_faces.extend(_stitch_side(right_outer, road_right))

    return stitch_faces
