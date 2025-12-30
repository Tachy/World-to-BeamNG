"""
Stitching von Lücken zwischen Terrain und Böschungen.
"""

from collections import defaultdict

import mapbox_earcut as earcut  # zwingend: schneller C++-Earclipper

import numpy as np
from scipy.spatial import cKDTree

from .. import config

ENABLE_FAN_STITCHING = False
MAX_CANDIDATES = 32  # Begrenze Fan-Punkte pro Böschungsknoten


def stitch_terrain_gaps(
    vertex_manager,
    terrain_vertex_indices,
    road_slope_polygons_2d,
    stitch_radius=10.0,
):
    """Erzeugt zusätzliche Faces, um Lücken zwischen Terrain und Böschungen zu schließen.

    Ansatz: Pro Böschungspunkt wird ein radialer Fan zu benachbarten Terrainpunkten erzeugt.
    """

    if not ENABLE_FAN_STITCHING:
        return []

    verts = np.asarray(vertex_manager.get_array())
    if len(verts) == 0 or not terrain_vertex_indices:
        return []

    terrain_vertex_indices = np.asarray(terrain_vertex_indices, dtype=int)
    terrain_xy = verts[terrain_vertex_indices][:, :2]
    terrain_tree = cKDTree(terrain_xy)

    def _stitch_side(indices, road_indices):
        faces = []
        if indices is None or len(indices) == 0:
            return faces

        for i in range(len(indices)):
            v_idx = int(indices[i])
            v_xy = verts[v_idx][:2]

            r_idx = int(road_indices[i]) if road_indices is not None else None
            width = (
                np.linalg.norm(verts[v_idx] - verts[r_idx])
                if r_idx is not None
                else 0.0
            )

            dynamic_radius = max(stitch_radius, width) + config.GRID_SPACING * 1.5

            cand_indices = terrain_tree.query_ball_point(v_xy, r=dynamic_radius)
            if len(cand_indices) == 0:
                # Fallback global nächster
                _, nn_fallback = terrain_tree.query(v_xy)
                cand_indices = [nn_fallback]

            cand_indices = np.unique(cand_indices)
            if len(cand_indices) < 2:
                continue

            cand_coords = terrain_xy[cand_indices]

            # Begrenze Kandidaten auf die nächsten MAX_CANDIDATES nach Distanz
            d2 = np.sum((cand_coords - v_xy) ** 2, axis=1)
            order_d = np.argsort(d2)
            if len(order_d) > MAX_CANDIDATES:
                order_d = order_d[:MAX_CANDIDATES]
            cand_indices = cand_indices[order_d]
            cand_coords = cand_coords[order_d]

            # Filter: nur Punkte im Halbraum Richtung Böschung (vom Straßenrand zum Böschungspunkt)
            if r_idx is not None:
                outward = verts[v_idx][:2] - verts[r_idx][:2]
                if np.dot(outward, outward) > 1e-9:
                    mask = np.einsum("ij,j->i", cand_coords - v_xy, outward) > 0.0
                    if np.any(mask):
                        cand_indices = cand_indices[mask]
                        cand_coords = cand_coords[mask]
            if len(cand_indices) < 2:
                continue

            angles = np.arctan2(
                cand_coords[:, 1] - v_xy[1], cand_coords[:, 0] - v_xy[0]
            )
            order = np.argsort(angles)
            ordered = cand_indices[order]

            t_indices = terrain_vertex_indices[ordered]
            if len(t_indices) < 2:
                continue

            t_indices = np.unique(t_indices)  # doppelte Targets entfernen
            if len(t_indices) < 2:
                continue

            # Fan schließen (wrap-around)
            for j in range(len(t_indices)):
                a = int(t_indices[j])
                b = int(t_indices[(j + 1) % len(t_indices)])
                if len({v_idx, a, b}) == 3:
                    faces.append([v_idx, a, b])

        # Fan-Stitching vorübergehend deaktiviert (Loch-Fixer übernimmt)
        return []

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


def fill_holes_by_boundary_loops(
    vertex_manager, terrain_faces, slope_faces, road_faces=None
):
    """Findet offene Boundary-Loops und füllt sie mit C++-Earcut."""

    verts = np.asarray(vertex_manager.get_array())
    if verts.size == 0:
        return []

    all_faces = []
    if terrain_faces is not None and len(terrain_faces):
        all_faces.extend(terrain_faces)
    if slope_faces is not None and len(slope_faces):
        all_faces.extend(slope_faces)
    if road_faces is not None and len(road_faces):
        all_faces.extend(road_faces)
    if not all_faces:
        return []

    faces_np = np.asarray(all_faces, dtype=np.int64)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        return []

    verts_xy = verts[:, :2]

    # Edge-Sammlung
    edges = np.vstack([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]])
    edges = np.sort(edges, axis=1)

    xs = verts[:, 0]
    ys = verts[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    tol = max(getattr(config, "GRID_SPACING", 1.0) * 0.5, 0.05)

    a_idx = edges[:, 0]
    b_idx = edges[:, 1]
    border_mask = (
        ((np.abs(xs[a_idx] - min_x) <= tol) & (np.abs(xs[b_idx] - min_x) <= tol))
        | ((np.abs(xs[a_idx] - max_x) <= tol) & (np.abs(xs[b_idx] - max_x) <= tol))
        | ((np.abs(ys[a_idx] - min_y) <= tol) & (np.abs(ys[b_idx] - min_y) <= tol))
        | ((np.abs(ys[a_idx] - max_y) <= tol) & (np.abs(ys[b_idx] - max_y) <= tol))
    )

    edges_filtered = edges[~border_mask]
    if edges_filtered.size == 0:
        return []

    unique_edges, counts = np.unique(edges_filtered, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return []

    boundary_edges = boundary_edges.astype(int)
    adj = defaultdict(list)
    edge_set = set()
    for a, b in boundary_edges:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)
        edge_set.add((a, b) if a < b else (b, a))

    def _angle(prev, current, nxt):
        base = verts_xy[current] - verts_xy[prev]
        tgt = verts_xy[nxt] - verts_xy[current]
        if not base.any():
            base_angle = 0.0
        else:
            base_angle = np.arctan2(base[1], base[0])
        ang = np.arctan2(tgt[1], tgt[0]) - base_angle
        return (ang + 2 * np.pi) % (2 * np.pi)

    def _trace_loop(start_a, start_b, edges_left):
        loop = [start_a, start_b]
        prev = start_a
        current = start_b
        guard = 0
        while True:
            guard += 1
            if guard > (len(boundary_edges) + 5):
                return None

            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                return None

            if len(neighbors) == 1:
                candidate = neighbors[0]
            else:
                candidate = min(neighbors, key=lambda n: _angle(prev, current, n))

            edge = (candidate, current) if candidate < current else (current, candidate)
            if edge not in edges_left:
                if candidate == loop[0]:
                    loop.append(candidate)
                return loop if candidate == loop[0] else None

            edges_left.remove(edge)
            loop.append(candidate)
            if candidate == loop[0]:
                return loop

            prev, current = current, candidate

    def _polygon_area2(indices):
        pts = verts_xy[indices]
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _max_edge_len(indices):
        pts = verts_xy[indices]
        rolled = np.roll(pts, -1, axis=0)
        d2 = np.sum((pts - rolled) ** 2, axis=1)
        return float(np.sqrt(d2.max())) if d2.size else 0.0

    def _is_convex(a, b, c):
        ab = verts_xy[b] - verts_xy[a]
        bc = verts_xy[c] - verts_xy[b]
        return (ab[0] * bc[1] - ab[1] * bc[0]) > 1e-9

    def _point_in_triangle(p, a, b, c):
        v0 = c - a
        v1 = b - a
        v2 = p - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-12:
            return False
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        return (u > 1e-8) and (v > 1e-8) and (u + v < 1 - 1e-8)

    def _triangulate_loop(loop_indices):
        if len(loop_indices) < 3:
            return []

        loop = list(loop_indices)
        if _polygon_area2(loop) < 0:
            loop.reverse()

        poly = verts_xy[loop].astype(np.float64)
        if poly.shape[0] < 3:
            return []

        ring_end = np.array([poly.shape[0]], dtype=np.uint32)
        tri_idx = earcut.triangulate_float64(poly, ring_end)
        if len(tri_idx) % 3 != 0:
            raise RuntimeError("Earcut gab keine gültige Dreiecksindizierung zurück")

        tris = []
        for k in range(0, len(tri_idx), 3):
            a = loop[tri_idx[k]]
            b = loop[tri_idx[k + 1]]
            c = loop[tri_idx[k + 2]]
            if len({a, b, c}) == 3:
                tris.append([int(a), int(b), int(c)])
        return tris

    loops = []
    edges_left = set(edge_set)
    while edges_left:
        a, b = edges_left.pop()
        loop = _trace_loop(a, b, edges_left)
        if loop is None:
            loop = _trace_loop(b, a, edges_left)
        if loop and len(loop) >= 4 and loop[0] == loop[-1]:
            loops.append(loop[:-1])

    filled_faces = []
    max_edge_limit = getattr(config, "HOLE_MAX_EDGE_LEN", None)
    max_area_limit = getattr(config, "HOLE_MAX_AREA", None)

    for loop in loops:
        if max_edge_limit is not None:
            if _max_edge_len(loop) > max_edge_limit:
                continue
        if max_area_limit is not None:
            if abs(_polygon_area2(loop)) > max_area_limit:
                continue

        tris = _triangulate_loop(loop)
        if tris:
            filled_faces.extend(tris)

    return filled_faces


def clean_faces_nonmanifold(faces, hole_start_idx=0):
    """Entfernt doppelte Faces und reduziert Kantenbelegung auf max. zwei Dreiecke.

    hole_start_idx: Faces mit Index >= hole_start_idx gelten als nachträglich hinzugefügt (z.B. Hole-Faces)
    und werden bei Kanten mit mehr als zwei belegten Dreiecken bevorzugt entfernt.
    """

    if not faces:
        return []

    faces_np = np.asarray(faces, dtype=np.int64)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        return faces

    # Duplikate entfernen
    faces_sorted = np.sort(faces_np, axis=1)
    _, unique_idx = np.unique(faces_sorted, axis=0, return_index=True)
    faces_np = faces_np[np.sort(unique_idx)]

    # Edge-Adjazenz aufbauen
    edge_to_faces = {}
    for fi, (a, b, c) in enumerate(faces_np):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(key, []).append(fi)

    remove = set()
    for face_ids in edge_to_faces.values():
        if len(face_ids) <= 2:
            continue
        # Bevorzuge Entfernen von Faces, die nach hole_start_idx liegen
        prioritized = sorted(
            face_ids, key=lambda idx: (idx < hole_start_idx, idx), reverse=True
        )
        # behalten die ersten zwei der inversen Priorität
        keep = set(prioritized[:2])
        for fid in prioritized[2:]:
            remove.add(fid)

    if not remove:
        return faces_np.tolist()

    keep_mask = np.ones(len(faces_np), dtype=bool)
    for fid in remove:
        if 0 <= fid < len(keep_mask):
            keep_mask[fid] = False

    cleaned = faces_np[keep_mask]
    return cleaned.tolist()
