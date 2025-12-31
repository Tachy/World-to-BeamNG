"""
Stitching von Luecken zwischen Terrain und Boeschungen.
Globale Loch-Suche: Findet alle Boundary-Polygone auf der Map.
"""

from collections import defaultdict

import numpy as np

from .. import config


def stitch_terrain_gaps(
    vertex_manager,
    terrain_vertex_indices,
    road_slope_polygons_2d,
    terrain_faces,
    slope_faces,
    stitch_radius=10.0,
):
    """Findet globale Loch-Polygone und exportiert sie zur Visualisierung.

    Globaler Ansatz:
    - Finde alle Boundary-Edges (Loecher) auf der kompletten Map
    - Baue geschlossene Polygone daraus
    - Exportiere als OBJ (Linienzuege) fuer debug_road_viewer
    - Noch KEIN Earcut/Fuellen
    """

    verts = np.asarray(vertex_manager.get_array())
    if len(verts) == 0:
        return []

    print(f"  Suche global nach Loch-Polygonen...")

    # Konvertiere Faces zu numpy
    all_existing_faces = terrain_faces + slope_faces
    all_faces_np = np.array(all_existing_faces, dtype=np.int32)

    # Baue globale Edge-Map
    print(f"  Baue globale Edge-Map...")
    edge_to_faces = defaultdict(list)
    for face_idx, face in enumerate(all_faces_np):
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            edge_to_faces[edge].append(face_idx)

    # Finde Boundary-Edges (nur 1 Face = Loch-Rand)
    boundary_edges = []
    for edge, face_list in edge_to_faces.items():
        if len(face_list) == 1:
            boundary_edges.append(edge)

    print(f"  [OK] {len(boundary_edges)} Boundary-Edges gefunden")

    if len(boundary_edges) == 0:
        print("  [!] Keine Loecher gefunden")
        return []

    # Klassifiziere Vertices: Terrain vs Slope
    terrain_vertex_set = set(terrain_vertex_indices)
    slope_vertex_set = set()
    for poly_data in road_slope_polygons_2d:
        slope_outer = poly_data.get("slope_outer_indices") or {}
        if slope_outer.get("left"):
            slope_vertex_set.update(slope_outer["left"])
        if slope_outer.get("right"):
            slope_vertex_set.update(slope_outer["right"])

    # Baue Polygone aus Boundary-Edges
    print(f"  Baue Loch-Polygone...")
    hole_polygons = _build_hole_polygons(
        boundary_edges, terrain_vertex_set, slope_vertex_set
    )

    print(f"  [OK] {len(hole_polygons)} Loch-Polygone gefunden")

    # Exportiere als OBJ fuer visuelle Inspektion
    if hole_polygons:
        _export_hole_polygons_obj(hole_polygons, verts)

    # Noch kein Stitching - nur Analyse
    return []


def _build_hole_polygons(edges, terrain_vertex_set, slope_vertex_set):
    """Baut geschlossene Polygone aus Boundary-Edges - findet zusammenhängende Komponenten."""
    if not edges:
        return []

    # Baue Adjacency-List
    adj = defaultdict(list)
    for v1, v2 in edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    # DEBUG: Vertex-Degree-Statistik
    degrees = [len(neighbors) for neighbors in adj.values()]
    print(
        f"    DEBUG: Vertex-Degrees: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/len(degrees):.1f}"
    )
    degree_1 = sum(1 for d in degrees if d == 1)
    degree_2 = sum(1 for d in degrees if d == 2)
    degree_3plus = sum(1 for d in degrees if d >= 3)
    print(
        f"    DEBUG: Degree-1 (Endpunkte): {degree_1}, Degree-2 (Pfad): {degree_2}, Degree-3+ (Verzweigungen): {degree_3plus}"
    )

    # Finde zusammenhängende Komponenten (Connected Components) - OPTIMIERT
    visited_vertices = set()
    ordered_paths = []

    for start_v in adj.keys():
        if start_v in visited_vertices:
            continue

        # BFS um zusammenhängende Komponente zu finden und DIREKT als Pfad zu bauen
        path = [start_v]
        visited_vertices.add(start_v)
        current = start_v
        prev = None

        # Folge dem Pfad entlang der Edges
        while True:
            neighbors = [
                n for n in adj[current] if n != prev and n not in visited_vertices
            ]

            if not neighbors:
                # Keine unbesuchten Nachbarn mehr
                # Pruefe ob wir zum Start zurueckkoennen (Loop)
                if len(path) > 2 and start_v in adj[current]:
                    # Geschlossener Loop
                    pass
                break

            # Nimm ersten unbesuchten Nachbarn
            next_v = neighbors[0]
            visited_vertices.add(next_v)
            path.append(next_v)

            prev = current
            current = next_v

        if len(path) >= 3:
            ordered_paths.append(path)

    print(f"    DEBUG: {len(ordered_paths)} geordnete Pfade erstellt (optimiert)")

    # Filtere Pfade: nur Terrain+Slope Mix
    mixed_paths = []
    terrain_only = []
    slope_only = []

    for path in ordered_paths:
        has_terrain = any(v in terrain_vertex_set for v in path)
        has_slope = any(v in slope_vertex_set for v in path)

        if has_terrain and has_slope:
            mixed_paths.append(path)
        elif has_terrain:
            terrain_only.append(path)
        elif has_slope:
            slope_only.append(path)

    print(
        f"    DEBUG: {len(mixed_paths)} Pfade mit Mix, {len(terrain_only)} nur Terrain, {len(slope_only)} nur Slope"
    )

    # Exportiere ALLE Pfade zur Visualisierung
    all_boundary_paths = mixed_paths + terrain_only + slope_only
    return all_boundary_paths


def _export_hole_polygons_obj(polygons, verts):
    """Exportiert Loch-Polygone als OBJ fuer visuelle Inspektion.

    Exportiert alle Boundary-Komponenten (nicht nur Loops):
    - Vertices werden in sortierter Reihenfolge geschrieben
    - Lines verbinden benachbarte Vertices in der Komponente
    """
    output_path = "lochpolygone.obj"
    mtl_path = "lochpolygone.mtl"

    print(f"  Exportiere {len(polygons)} Boundary-Komponenten nach {output_path}...")

    with open(output_path, "w") as f:
        f.write("# Boundary-Komponenten (Loch-Ränder)\n")
        f.write(f"mtllib {mtl_path}\n\n")

        # Schreibe alle Vertices
        vertex_index_map = {}
        obj_vertex_idx = 1

        for poly in polygons:
            for v_idx in poly:
                if v_idx not in vertex_index_map:
                    vertex_index_map[v_idx] = obj_vertex_idx
                    v = verts[v_idx]
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    obj_vertex_idx += 1

        f.write("\n")
        f.write("usemtl boundary_edges\n")

        # Schreibe Komponenten als Linienzuege (Punktwolken verbunden)
        for poly in polygons:
            # Einfach alle Punkte hintereinander verbinden
            f.write("l")
            for v_idx in poly:
                f.write(f" {vertex_index_map[v_idx]}")
            # Schliesse zurueck zum ersten Punkt
            if len(poly) > 1:
                f.write(f" {vertex_index_map[poly[0]]}")
            f.write("\n")

    # Schreibe MTL
    with open(mtl_path, "w") as f:
        f.write("# Material fuer Boundary-Komponenten\n")
        f.write("newmtl boundary_edges\n")
        f.write("Ka 1.0 0.0 0.0\n")  # Rot
        f.write("Kd 1.0 0.0 0.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("d 1.0\n")

    print(f"  [OK] Exportiert: {output_path} und {mtl_path}")
