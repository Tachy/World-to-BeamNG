"""
Face Cleanup Utilities.
"""

import os
import numpy as np
from world_to_beamng import config


def cleanup_duplicate_faces(all_faces):
    """Entfernt doppelte Faces (gleiche Vertex-Tripel, unabhaengig von Reihenfolge)."""
    if not all_faces:
        return []

    faces_np = np.array(all_faces, dtype=np.int64)

    # Sortiere Vertices in jedem Face fuer Vergleich
    faces_sorted = np.sort(faces_np, axis=1)

    # Finde eindeutige Faces
    _, unique_idx = np.unique(faces_sorted, axis=0, return_index=True)

    # Behalte Originale Reihenfolge (fuer korrekte Normalen)
    unique_faces = faces_np[np.sort(unique_idx)]

    num_removed = len(all_faces) - len(unique_faces)
    if num_removed > 0:
        print(f"  [OK] {num_removed} doppelte Faces entfernt")

    return unique_faces.tolist()


def enforce_ccw_up(faces, vertices):
    """Sorgt dafuer, dass Dreiecke mit +Z-Normalen (CCW nach oben) angeordnet sind."""
    if not faces:
        return faces
    faces_np = np.asarray(faces, dtype=np.int64)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        return faces

    verts_np = np.asarray(vertices, dtype=np.float64)
    tri_pts = verts_np[faces_np]
    normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    flip_idx = np.where(normals[:, 2] < 0)[0]
    if flip_idx.size:
        f1 = faces_np[flip_idx, 1].copy()
        f2 = faces_np[flip_idx, 2].copy()
        faces_np[flip_idx, 1] = f2
        faces_np[flip_idx, 2] = f1
    return faces_np.tolist()


def report_boundary_edges(faces, vertices, label="mesh", export_path=None):
    """Loggt offene und nicht-manifold Kanten (0-basiert). Optionaler Export der offenen Kanten als OBJ."""
    if not faces:
        print(f"  {label}: Keine Faces -> keine Kanten")
        return
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        print(f"  {label}: Nicht-dreieckige Faces uebersprungen")
        return

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)

    # Entferne Kanten, die vollstaendig auf dem aeusseren Bounding-Box-Rand liegen
    verts_np = np.asarray(vertices, dtype=np.float64)
    xs = verts_np[:, 0]
    ys = verts_np[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    tol = max(getattr(config, "GRID_SPACING", 1.0) * 0.5, 0.05)

    on_min_x = np.abs(xs - min_x) <= tol
    on_max_x = np.abs(xs - max_x) <= tol
    on_min_y = np.abs(ys - min_y) <= tol
    on_max_y = np.abs(ys - max_y) <= tol

    a_idx = edges[:, 0]
    b_idx = edges[:, 1]
    border_mask = (
        (on_min_x[a_idx] & on_min_x[b_idx])
        | (on_max_x[a_idx] & on_max_x[b_idx])
        | (on_min_y[a_idx] & on_min_y[b_idx])
        | (on_max_y[a_idx] & on_max_y[b_idx])
    )

    edges_filtered = edges[~border_mask]
    if edges_filtered.size == 0:
        print(f"  {label}: Keine Kanten nach Rand-Filter")
        return

    # Nutzung von np.unique ueber axis=0 vermeidet View-Probleme
    unique, counts = np.unique(edges_filtered, axis=0, return_counts=True)

    boundary_mask = counts == 1
    nonmanifold_mask = counts > 2
    num_boundary = int(np.count_nonzero(boundary_mask))
    num_nonmanifold = int(np.count_nonzero(nonmanifold_mask))

    print(f"  {label}: Boundary-Kanten={num_boundary}, Non-manifold={num_nonmanifold}")
    if num_boundary:
        sample = unique[boundary_mask][:10]
        print(f"    Beispiel offene Kanten (0-basiert): {sample.tolist()}")
        if export_path:
            try:
                edges_to_export = unique[boundary_mask]
                verts_np = np.asarray(vertices, dtype=np.float32)
                used_idx = np.unique(edges_to_export)

                # Duenne Quads aus zwei Dreiecken mit angehobenen Duplikaten, damit Flaeche sichtbar ist
                epsilon = 0.1  # Sichtbarer Offset

                # Mapping alt->neu (1-basiert fuer OBJ)
                new_idx = {int(old): i + 1 for i, old in enumerate(used_idx)}
                elev_idx = {}
                extra_vertices = []
                next_idx = len(used_idx) + 1

                for vid in used_idx:
                    v = verts_np[int(vid)].copy()
                    v[2] += epsilon
                    extra_vertices.append(v)
                    elev_idx[int(vid)] = next_idx
                    next_idx += 1

                faces = []
                for a, b in edges_to_export:
                    a = int(a)
                    b = int(b)
                    fa = new_idx[a]
                    fb = new_idx[b]
                    fa_up = elev_idx[a]
                    fb_up = elev_idx[b]
                    # Zwei Dreiecke pro Edge (duennes Band)
                    faces.append([fa_up, fa, fb])
                    faces.append([fa_up, fb, fb_up])

                mtl_path = (
                    export_path.replace(".obj", ".mtl")
                    if export_path.lower().endswith(".obj")
                    else None
                )

                with open(export_path, "w") as fobj:
                    if mtl_path:
                        fobj.write(f"mtllib {os.path.basename(mtl_path)}\n")
                    # Vertices (bestehende)
                    for vid in used_idx:
                        x, y, z = verts_np[int(vid)]
                        fobj.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")
                    # Angehoene Duplikate
                    for mv in extra_vertices:
                        fobj.write(f"v {mv[0]:.3f} {mv[1]:.3f} {mv[2]:.3f}\n")

                    fobj.write("o boundary_edges\n")
                    if mtl_path:
                        fobj.write("usemtl boundary_edges\n")
                    # Faces (duenne Baender ueber jeder Kante)
                    for fa, fb, fc in faces:
                        fobj.write(f"f {fa} {fb} {fc}\n")

                if mtl_path:
                    with open(mtl_path, "w") as fmtl:
                        fmtl.write("newmtl boundary_edges\n")
                        fmtl.write("Ka 1 0 0\nKd 1 0 0\nKs 0 0 0\nd 1.0\nillum 1\n")

                print(
                    f"    -> Boundary-Kanten als OBJ (Faces) exportiert nach {export_path}"
                )

            except Exception as exc:
                print(f"    [!] Export nach {export_path} fehlgeschlagen: {exc}")
