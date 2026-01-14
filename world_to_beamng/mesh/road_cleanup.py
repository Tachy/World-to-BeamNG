"""
Road-Face Cleanup für neues Tupel-Format.

Clippt Road-Faces an Grid-Grenzen und retrianguliert überstehende Dreiecke.
"""

import numpy as np
from shapely.geometry import Polygon, box


def clip_road_mesh_data(road_mesh_data, vertex_manager, grid_bounds_local):
    """
    Clippt strukturierte Road-Daten an Grid-Grenzen.

    Args:
        road_mesh_data: Liste von {'vertices': [v0,v1,v2], 'road_id': id, 'uvs': {v: (u,v), ...}}
        vertex_manager: VertexManager für Vertex-Zugriff
        grid_bounds_local: (min_x, max_x, min_y, max_y)

    Returns:
        Liste von geclippten Road-Daten (gleiche Struktur)
    """
    if not road_mesh_data or grid_bounds_local is None:
        return road_mesh_data

    min_x, max_x, min_y, max_y = grid_bounds_local
    bounds_box = box(min_x, min_y, max_x, max_y)

    data_to_keep = []
    data_to_clip = []

    # Phase 1: Sortiere Faces in "komplett drin", "komplett draußen", "teilweise"
    for face_data in road_mesh_data:
        v0_idx, v1_idx, v2_idx = face_data['vertices']

        v0 = vertex_manager.vertices[v0_idx]
        v1 = vertex_manager.vertices[v1_idx]
        v2 = vertex_manager.vertices[v2_idx]

        # Prüfe ob Vertices außerhalb liegen
        v0_outside = v0[0] < min_x or v0[0] > max_x or v0[1] < min_y or v0[1] > max_y
        v1_outside = v1[0] < min_x or v1[0] > max_x or v1[1] < min_y or v1[1] > max_y
        v2_outside = v2[0] < min_x or v2[0] > max_x or v2[1] < min_y or v2[1] > max_y

        outside_count = sum([v0_outside, v1_outside, v2_outside])

        if outside_count == 0:
            # Komplett innerhalb
            data_to_keep.append(face_data)
        elif outside_count == 3:
            # Komplett außerhalb - verwerfen
            pass
        else:
            # Teilweise überstehend - clippen
            data_to_clip.append((face_data, (v0, v1, v2)))

    if not data_to_clip:
        print(f"  [Road-Cleanup] Alle {len(data_to_keep)} Road-Faces innerhalb Grid")
        return data_to_keep

    print(f"  [Road-Cleanup] Clippe {len(data_to_clip)} teilweise überstehende Road-Faces...")

    # Phase 2: Clippe überstehende Faces
    clipped_count = 0
    for face_data, (v0, v1, v2) in data_to_clip:
        # Erstelle Face-Polygon (2D)
        face_poly = Polygon([(v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1])])

        # Clippe gegen Bounds
        try:
            clipped = face_poly.intersection(bounds_box)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        # Trianguliere geclipptes Polygon
        if clipped.geom_type == "Polygon":
            coords = list(clipped.exterior.coords[:-1])  # Ohne Duplikat

            if len(coords) < 3:
                continue

            # Einfache Fächer-Triangulation um ersten Punkt
            for i in range(1, len(coords) - 1):
                p0 = coords[0]
                p1 = coords[i]
                p2 = coords[i + 1]

                # Interpoliere Z-Werte aus Original-Triangle (baryzentrisch)
                z0 = _interpolate_z_from_triangle(p0, v0, v1, v2)
                z1 = _interpolate_z_from_triangle(p1, v0, v1, v2)
                z2 = _interpolate_z_from_triangle(p2, v0, v1, v2)

                # Füge neue Vertices hinzu
                idx0 = vertex_manager.add_vertex(p0[0], p0[1], z0)
                idx1 = vertex_manager.add_vertex(p1[0], p1[1], z1)
                idx2 = vertex_manager.add_vertex(p2[0], p2[1], z2)

                # Füge neue Face-Daten mit geclippten Vertices und UVs aus Original hinzu
                # UVs werden interpoliert wie Z-Werte
                road_id = face_data['road_id']
                orig_uvs = face_data['uvs']
                
                # Interpoliere UVs basierend auf baryzentrische Koordinaten
                # (vereinfacht: nutze UV vom ursprünglichen Vertex am nächsten)
                data_to_keep.append({
                    'vertices': [idx0, idx1, idx2],
                    'road_id': road_id,
                    'uvs': {
                        idx0: orig_uvs.get(v0_idx, (0.0, 0.0)),
                        idx1: orig_uvs.get(v1_idx, (0.0, 0.0)),
                        idx2: orig_uvs.get(v2_idx, (0.0, 0.0))
                    }
                })
                clipped_count += 1

    print(f"  [Road-Cleanup] {clipped_count} neue Dreiecke nach Clipping, {len(data_to_keep)} total")

    return data_to_keep


def _interpolate_z_from_triangle(point_2d, v0, v1, v2):
    """
    Interpoliert Z-Koordinate eines 2D-Punktes innerhalb eines Dreiecks.
    Nutzt baryzentrische Koordinaten.

    Args:
        point_2d: (x, y) Punkt
        v0, v1, v2: (x, y, z) Vertices des Dreiecks

    Returns:
        float: Interpolierte Z-Koordinate
    """
    px, py = point_2d

    # Baryzentrische Koordinaten berechnen
    x0, y0, z0 = v0[0], v0[1], v0[2]
    x1, y1, z1 = v1[0], v1[1], v1[2]
    x2, y2, z2 = v2[0], v2[1], v2[2]

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

    if abs(denom) < 1e-10:
        # Degeneriertes Dreieck → Durchschnitt der Z-Werte
        return (z0 + z1 + z2) / 3.0

    lambda0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    lambda1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    lambda2 = 1.0 - lambda0 - lambda1

    # Interpoliere Z
    z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2

    return z
