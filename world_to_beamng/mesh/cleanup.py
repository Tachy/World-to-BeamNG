"""
Cleanup-Funktionen für Mesh-Bereinigung.
"""

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import triangulate
from .. import config


def remove_road_faces_outside_bounds(mesh, vertex_manager):
    """
    Entfernt Straßen-Dreiecke, die komplett außerhalb der Grid-Bounds liegen.

    Args:
        mesh: Mesh-Instanz
        vertex_manager: VertexManager-Instanz

    Returns:
        Anzahl entfernter Faces
    """
    if not config.CLIP_ROAD_FACES_AT_BOUNDS:
        return 0

    bounds = config.GRID_BOUNDS_LOCAL
    if bounds is None:
        return 0

    min_x, max_x, min_y, max_y = bounds

    # Finde alle Road-Faces (Material != "terrain")
    # Da OSM-Mapper individuelle Material-Namen vergibt (residential_road, motorway, etc.),
    # müssen wir alle Non-Terrain-Faces als Roads betrachten
    all_faces = list(range(len(mesh.faces)))
    road_faces = []
    for face_idx in all_faces:
        props = mesh.face_props.get(face_idx, {})
        material = props.get("material", "terrain")
        if material != "terrain":
            road_faces.append(face_idx)

    print(f"  [Cleanup] Überprüfe {len(road_faces)} Straßen-Dreiecke auf Bounds-Ausschluss...")

    if not road_faces:
        return 0

    faces_to_remove = []

    for face_idx in road_faces:
        face = mesh.faces[face_idx]
        v0_idx, v1_idx, v2_idx = face

        # Hole Vertex-Koordinaten
        v0 = vertex_manager.vertices[v0_idx]
        v1 = vertex_manager.vertices[v1_idx]
        v2 = vertex_manager.vertices[v2_idx]

        # Prüfe ob ALLE drei Vertices außerhalb liegen
        v0_outside = v0[0] < min_x or v0[0] > max_x or v0[1] < min_y or v0[1] > max_y
        v1_outside = v1[0] < min_x or v1[0] > max_x or v1[1] < min_y or v1[1] > max_y
        v2_outside = v2[0] < min_x or v2[0] > max_x or v2[1] < min_y or v2[1] > max_y

        # Wenn alle drei Vertices außerhalb → Face entfernen
        if v0_outside and v1_outside and v2_outside:
            faces_to_remove.append(face_idx)

    if faces_to_remove:
        removed_count = mesh.remove_faces(faces_to_remove)
        print(f"  [Cleanup] {removed_count} Straßen-Dreiecke komplett außerhalb entfernt")

    # === TEIL 2: Clippe teilweise überstehende Dreiecke an Grid-Kante ===
    # Aktualisiere road_faces Liste (alte Indices sind ungültig nach remove_faces)
    all_faces = list(range(len(mesh.faces)))
    road_faces = []
    for face_idx in all_faces:
        props = mesh.face_props.get(face_idx, {})
        material = props.get("material", "terrain")
        if material != "terrain":
            road_faces.append(face_idx)

    # Grid-Bounds als Shapely Box
    bounds_box = box(min_x, min_y, max_x, max_y)

    faces_to_clip = []
    faces_to_clip_props = []

    for face_idx in road_faces:
        face = mesh.faces[face_idx]
        v0_idx, v1_idx, v2_idx = face

        v0 = vertex_manager.vertices[v0_idx]
        v1 = vertex_manager.vertices[v1_idx]
        v2 = vertex_manager.vertices[v2_idx]

        # Prüfe ob mindestens ein Vertex außerhalb liegt
        v0_outside = v0[0] < min_x or v0[0] > max_x or v0[1] < min_y or v0[1] > max_y
        v1_outside = v1[0] < min_x or v1[0] > max_x or v1[1] < min_y or v1[1] > max_y
        v2_outside = v2[0] < min_x or v2[0] > max_x or v2[1] < min_y or v2[1] > max_y

        # Wenn mindestens ein Vertex außerhalb (aber nicht alle) → clippen
        if v0_outside or v1_outside or v2_outside:
            faces_to_clip.append(face_idx)
            faces_to_clip_props.append(mesh.face_props.get(face_idx, {}))

    if faces_to_clip:
        print(f"  [Cleanup] Clippe {len(faces_to_clip)} teilweise überstehende Straßen-Dreiecke...")

        new_faces_data = []  # [(v0_idx, v1_idx, v2_idx, props), ...]

        for face_idx, props in zip(faces_to_clip, faces_to_clip_props):
            face = mesh.faces[face_idx]
            v0_idx, v1_idx, v2_idx = face

            v0 = vertex_manager.vertices[v0_idx]
            v1 = vertex_manager.vertices[v1_idx]
            v2 = vertex_manager.vertices[v2_idx]

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
                coords = list(clipped.exterior.coords[:-1])  # Ohne Duplikat am Ende

                if len(coords) < 3:
                    continue

                # Einfache Fächer-Triangulation um ersten Punkt
                # Interpoliere Z-Koordinaten aus Original-Face (baryzentrischer Ansatz)
                for i in range(1, len(coords) - 1):
                    p0 = coords[0]
                    p1 = coords[i]
                    p2 = coords[i + 1]

                    # Interpoliere Z-Werte aus Original-Triangle
                    z0 = _interpolate_z_from_triangle(p0, v0, v1, v2)
                    z1 = _interpolate_z_from_triangle(p1, v0, v1, v2)
                    z2 = _interpolate_z_from_triangle(p2, v0, v1, v2)

                    # Füge Vertices hinzu
                    idx0 = vertex_manager.add_vertex(p0[0], p0[1], z0)
                    idx1 = vertex_manager.add_vertex(p1[0], p1[1], z1)
                    idx2 = vertex_manager.add_vertex(p2[0], p2[1], z2)

                    new_faces_data.append((idx0, idx1, idx2, props))

        # Entferne alte geclippte Faces
        mesh.remove_faces(faces_to_clip)

        # Füge neue Faces hinzu
        for v0, v1, v2, props in new_faces_data:
            mesh.add_face(v0, v1, v2, **props)

        print(f"  [Cleanup] {len(new_faces_data)} neue Dreiecke nach Clipping eingefügt")

    return removed_count


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
