"""
Lokales Stitching: Loch-Suche entlang Search-Circles um Centerlines.

Statt globaler Boundary-Suche: Pro Centerline-Sample-Punkt einen Search-Circle
anlegen und nur dort nach Loch-Polygonen suchen.
"""

import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


def find_boundary_polygons_in_circle(
    centerline_point,
    search_radius,
    vertex_manager,
    terrain_faces,
    slope_faces,
    terrain_vertex_indices,
    debug=False,
):
    """
    Findet Boundary-Polygone in einem Search-Circle um einen Centerline-Punkt.

    Verwendet Face-basierte Edge-Klassifikation (Option A) und Connected Components
    um mehrere unabhängige Löcher pro Circle zu finden.

    Args:
        centerline_point: (x, y, z) - Mittelpunkt des Search-Circles
        search_radius: Radius in Metern (typisch 10.0m)
        vertex_manager: VertexManager mit allen Mesh-Vertices
        terrain_faces: Liste von [v0, v1, v2] Terrain-Faces
        slope_faces: Liste von [v0, v1, v2] Slope/Road-Faces
        terrain_vertex_indices: Set oder Liste der Terrain-Vertex-Indices
        debug: Optionaler Debug-Flag für Ausgaben

    Returns:
        Liste von Polygon-Dicts:
        [
            {
                'vertices': [v0, v1, ..., vN],  # Geschlossener Ring
                'coords': [(x,y,z), ...],
                'terrain_count': int,
                'slope_count': int
            },
            ...
        ]
    """
    cx, cy, cz = centerline_point

    # Hole alle Vertices
    verts = np.asarray(vertex_manager.get_array())
    if len(verts) == 0:
        return []

    # KDTree für schnelle räumliche Suche (nur XY)
    kdtree = cKDTree(verts[:, :2])

    # Finde alle Vertices im Search-Circle
    circle_vertex_indices = kdtree.query_ball_point([cx, cy], r=search_radius)

    if len(circle_vertex_indices) < 3:
        return []

    circle_vertex_set = set(circle_vertex_indices)

    # Kombiniere alle Faces
    all_faces = terrain_faces + slope_faces

    # Finde Faces deren Vertices im Circle sind (mindestens 2 von 3)
    faces_in_circle = []
    face_indices_in_circle = []
    terrain_face_count = 0
    slope_face_count = 0

    for face_idx, face in enumerate(all_faces):
        vertices_in_circle = sum(1 for v in face if v in circle_vertex_set)
        if vertices_in_circle >= 2:
            faces_in_circle.append(face)
            face_indices_in_circle.append(face_idx)
            if face_idx < len(terrain_faces):
                terrain_face_count += 1
            else:
                slope_face_count += 1

    if len(faces_in_circle) < 2:
        return []

    # Baue lokale Edge-Map
    edge_to_faces = defaultdict(list)
    for local_idx, face in enumerate(faces_in_circle):
        global_face_idx = face_indices_in_circle[local_idx]
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            edge_to_faces[edge].append(global_face_idx)

    # Finde Boundary-Edges
    # - Fall A: nur 1 Face (klassische Außengrenze)
    # - Fall B: genau 2 Faces mit unterschiedlichem Material (Terrain vs. Slope/Road)
    boundary_edges = []
    boundary_edges_mixed = 0

    for edge, face_list in edge_to_faces.items():
        v1, v2 = edge
        if v1 not in circle_vertex_set or v2 not in circle_vertex_set:
            continue

        if len(face_list) == 1:
            boundary_edges.append(edge)
        elif len(face_list) == 2:
            f1, f2 = face_list
            f1_is_terrain = f1 < len(terrain_faces)
            f2_is_terrain = f2 < len(terrain_faces)
            if f1_is_terrain != f2_is_terrain:
                boundary_edges.append(edge)
                boundary_edges_mixed += 1

    if len(boundary_edges) < 3:
        return []

    # Finde Connected Components
    components = _find_connected_components(boundary_edges)

    if not components:
        return []

    # Baue Polygone aus Components
    terrain_face_set = set(range(len(terrain_faces)))
    polygons = []

    for component_path in components:
        polygon = _build_polygon_from_component(
            component_path,
            edge_to_faces,
            terrain_face_set,
            verts,
            debug=debug,
        )
        if polygon:
            polygons.append(polygon)

    if debug:
        # Kompakte Debug-Ausgabe für ersten Kreis
        print(
            "  [debug] circle verts/faces(both)/bnd_edges(all/mixed)/components/polys:",
            len(circle_vertex_indices),
            len(faces_in_circle),
            f"T{terrain_face_count}/S{slope_face_count}",
            len(boundary_edges),
            boundary_edges_mixed,
            len(components),
            len(polygons),
        )
        if boundary_edges:
            sample_edges = list(boundary_edges[:5]) if isinstance(boundary_edges, list) else list(boundary_edges)[:5]
            print("  [debug] sample boundary edges:", sample_edges)
    return polygons


def _find_connected_components(boundary_edges):
    """Findet alle zusammenhängenden Komponenten in Edge-Liste."""
    if not boundary_edges:
        return []

    # Baue Adjacency-List
    adj = defaultdict(list)
    for v1, v2 in boundary_edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    visited_vertices = set()
    components = []

    for start_v in adj.keys():
        if start_v in visited_vertices:
            continue

        # Graph-Walking für diese Component
        component_path = [start_v]
        visited_vertices.add(start_v)
        current = start_v
        prev = None

        while True:
            neighbors = [n for n in adj[current] if n != prev and n not in visited_vertices]

            if not neighbors:
                # Prüfe ob Loop geschlossen werden kann
                if len(component_path) > 2 and start_v in adj[current]:
                    # Geschlossener Loop
                    pass
                break

            next_v = neighbors[0]
            visited_vertices.add(next_v)
            component_path.append(next_v)
            prev = current
            current = next_v

        if len(component_path) >= 3:
            components.append(component_path)

    return components


def _build_polygon_from_component(component_path, edge_to_faces, terrain_face_set, verts, debug=False):
    """Baut geschlossenes Polygon aus Component-Pfad mit Terrain/Slope-Separierung."""

    # Klassifiziere Edges als Terrain oder Slope basierend auf Face-Typ
    terrain_edges = []
    slope_edges = []

    for i in range(len(component_path)):
        v1 = component_path[i]
        v2 = component_path[(i + 1) % len(component_path)]
        edge = tuple(sorted([v1, v2]))

        if edge not in edge_to_faces:
            continue

        face_list = edge_to_faces[edge]
        has_terrain = any(f in terrain_face_set for f in face_list)
        has_slope = any(f not in terrain_face_set for f in face_list)

        # Falls Mixed-Kante: in beide Listen aufnehmen, damit beide Seiten einen Pfad haben
        if has_terrain:
            terrain_edges.append((v1, v2))  # Orientiert!
        if has_slope:
            slope_edges.append((v1, v2))

    if not terrain_edges and not slope_edges:
        if debug:
            print(
                "  [debug] component skipped (terrain_edges, slope_edges, len_path):",
                len(terrain_edges),
                len(slope_edges),
                len(component_path),
            )
        return None  # Kein verwertbarer Rand

    # Baue geordnete Pfade
    terrain_path = _walk_oriented_edges(terrain_edges) if terrain_edges else []
    slope_path = _walk_oriented_edges(slope_edges) if slope_edges else []

    if terrain_path and slope_path:
        # Geschlossenes Polygon: Terrain + Slope (reversed)
        polygon_vertices = terrain_path + slope_path[::-1]
    elif terrain_path:
        polygon_vertices = terrain_path
    elif slope_path:
        polygon_vertices = slope_path
    else:
        return None

    # Konvertiere zu Koordinaten
    coords = [tuple(verts[v]) for v in polygon_vertices]

    return {
        "vertices": polygon_vertices,
        "coords": coords,
        "terrain_count": len(terrain_path),
        "slope_count": len(slope_path),
        "centerline_point": tuple(verts[component_path[0]]) if component_path else None,
    }


def _walk_oriented_edges(oriented_edges):
    """Baut geordneten Pfad aus orientierten Edges."""
    if not oriented_edges:
        return []

    # Baue Adjacency von v1 -> v2
    adj = defaultdict(list)
    for v1, v2 in oriented_edges:
        adj[v1].append(v2)

    # Starte bei erstem Vertex
    start_v = oriented_edges[0][0]
    path = [start_v]
    current = start_v
    visited_edges = set()

    while True:
        # Finde ausgehende Edge
        found = False
        for next_v in adj[current]:
            edge = (current, next_v)
            if edge not in visited_edges:
                visited_edges.add(edge)
                path.append(next_v)
                current = next_v
                found = True
                break

        if not found:
            break

    return path


def export_boundary_polygons_to_obj(
    polygons,
    centerline_point,
    output_path="cache/boundary_polygons_local.obj",
    search_radius=None,
    circle_segments=64,
):
    """
    Exportiert Boundary-Polygone als OBJ zur Visualisierung im DAE-Viewer.

    Args:
        polygons: Liste von Polygon-Dicts
        centerline_point: (x, y, z) - Centerline-Sample-Punkt
        output_path: Ausgabepfad für OBJ-Datei
        search_radius: Optionaler Radius für Visualisierung des Suchkreises
        circle_segments: Anzahl Segmente für Kreis-Approximation
    """
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Boundary-Polygone (lokales Stitching)\n")
        f.write(f"# Centerline: ({centerline_point[0]:.2f}, {centerline_point[1]:.2f}, {centerline_point[2]:.2f})\n")
        f.write(f"# Polygone: {len(polygons)}\n\n")

        obj_vertex_idx = 1

        for poly_idx, poly in enumerate(polygons):
            coords = poly.get("coords", [])
            terrain_count = poly.get("terrain_count", 0)
            slope_count = poly.get("slope_count", 0)

            f.write(f"\n# Polygon {poly_idx + 1} (Terrain: {terrain_count}, Slope: {slope_count})\n")

            # Schreibe Vertices
            for coord in coords:
                f.write(f"v {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

            # Schreibe geschlossene Linie
            if len(coords) >= 3:
                indices = " ".join(str(obj_vertex_idx + i) for i in range(len(coords)))
                f.write(f"l {indices} {obj_vertex_idx}\n")  # Schließe Loop

            obj_vertex_idx += len(coords)

        # Zeichne den Suchkreis als Polylinie (immer, damit sichtbar im Debug-Layer)
        sr = search_radius if search_radius is not None else 10.0
        segs = max(8, int(circle_segments) if circle_segments is not None else 64)
        f.write(f"\n# Search-Circle (r={sr:.2f})\n")
        cx, cy, cz = centerline_point
        circle_vertices = []
        for i in range(segs):
            angle = 2.0 * np.pi * i / segs
            x = cx + sr * np.cos(angle)
            y = cy + sr * np.sin(angle)
            circle_vertices.append((x, y, cz))
            f.write(f"v {x:.6f} {y:.6f} {cz:.6f}\n")
        indices = " ".join(str(obj_vertex_idx + i) for i in range(len(circle_vertices)))
        f.write(f"l {indices} {obj_vertex_idx}\n")
        obj_vertex_idx += len(circle_vertices)

        # Markiere Centerline-Punkt
        f.write(f"\n# Centerline-Punkt\n")
        f.write(f"v {centerline_point[0]:.6f} {centerline_point[1]:.6f} {centerline_point[2]:.6f}\n")
