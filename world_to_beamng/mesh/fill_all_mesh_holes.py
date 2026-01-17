"""
Finde und schließe ALLE Löcher im Mesh (äußere Grenzen + innere Inseln).
OPTIMIERT für große Meshes mit KDTree und NumPy-Vectorisierung.

STRATEGIE:
1. Extrahiere ALLE Boundary-Loops (nicht nur Tile-Kanten)
2. Pro Loop: Identifiziere Kontur-Typ (äußer vs. innen)
3. Sortiere Vertices CCW pro Loop
4. Verbinde benachbarte Vertices ohne Edge mit Hilfsdreiecken
5. Validiere Edge-Längen (Warnung bei zu langen Edges)

PERFORMANCE:
- Edge-Map: Set (O(1) statt O(n))
- Räumliche Suche: KDTree (O(log n) statt O(n))
- Centroid: NumPy-Vectorisiert (O(n) statt O(n²))
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from collections import defaultdict
from scipy.spatial import cKDTree

from .vertex_manager import VertexManager


def find_all_boundary_loops(
    faces_list: List[Tuple[int, int, int]],
    vertex_manager: VertexManager,
) -> List[Dict]:
    """
    Finde ALLE geschlossenen Boundary-Loops (äußerer Rand + innere Inseln).

    OPTIMIERUNGEN:
    - Set-basierte Edge-Verwaltung (O(1) Lookup)
    - Minimale Speicherallokation
    - NumPy für Batch-Operationen

    Args:
        faces_list: List von Face-Tupeln (v0, v1, v2)
        vertex_manager: VertexManager

    Returns:
        Liste von Loop-Dicts (siehe fill_all_mesh_holes)
    """
    print("  [Hole-Filling] Suche alle Boundary-Loops...")

    if len(faces_list) == 0:
        return []

    # === STEP 1: Baue Edge-Set (O(1) Lookup) ===
    edge_count_map = defaultdict(int)  # {edge: count}

    for face in faces_list:
        v0, v1, v2 = face
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]
        for edge in edges:
            edge_count_map[edge] += 1

    # === STEP 2: Extrahiere NUR Boundary-Edges (count == 1) ===
    boundary_edges = set(e for e, count in edge_count_map.items() if count == 1)

    if len(boundary_edges) == 0:
        print(f"    [i] Keine Boundary-Edges gefunden (Mesh ist geschlossen)")
        return []

    print(f"    [i] {len(boundary_edges)} Boundary-Edges gefunden")

    # === STEP 3: Baue Adjacency-Map nur für Boundary-Edges ===
    adjacency = defaultdict(list)
    for v1, v2 in boundary_edges:
        adjacency[v1].append((v2, (v1, v2)))
        adjacency[v2].append((v1, (v2, v1)))

    # === STEP 4: Trace Loops (modifiziert für Robustheit) ===
    loops = []
    used_edges = set()
    vertices = vertex_manager.vertices  # Cache einmal

    for start_edge in boundary_edges:
        if start_edge in used_edges:
            continue

        loop_vertices = []
        loop_edges = []
        current_edge = start_edge
        max_iterations = len(boundary_edges) + 10

        for iteration in range(max_iterations):
            v1, v2 = current_edge
            loop_vertices.append(v1)
            loop_edges.append(current_edge)
            used_edges.add(current_edge)

            # Finde nächste Edge (von v2 weg)
            next_candidates = [(next_v, edge) for next_v, edge in adjacency[v2] if edge not in used_edges]

            if len(next_candidates) == 0:
                break

            next_v, next_edge = next_candidates[0]
            current_edge = next_edge

        if len(loop_vertices) < 3:
            continue

        # === STEP 5: Berechne signed area (NumPy-Vectorisiert) ===
        coords_idx = np.array(loop_vertices, dtype=np.int32)
        coords_2d = vertices[coords_idx, :2]

        # Shoelace-Formel (vectorisiert)
        x = coords_2d[:, 0]
        y = coords_2d[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        signed_area = 0.5 * np.sum(x * y_next - x_next * y)

        is_outer = signed_area > 0

        loops.append(
            {
                "vertices": loop_vertices,
                "edges": loop_edges,
                "area": abs(signed_area),
                "is_outer": is_outer,
                "signed_area": signed_area,
                "centroid": coords_2d.mean(axis=0),  # Cachen
                "centroid_distances": None,  # Lazy-computed
            }
        )

    print(f"    [i] {len(loops)} Loops gefunden")
    return loops


def fill_boundary_loop_holes(
    loop_dict: Dict,
    vertex_manager: VertexManager,
    mesh_obj,
    existing_edges: Set[Tuple[int, int]],
    max_edge_length: float = 100.0,
) -> int:
    """
    Schließe Lücken in einem einzelnen Boundary-Loop (OPTIMIERT).

    OPTIMIERUNGEN:
    - Passed-in existing_edges (O(1) Lookup, nicht neu-aufgebaut)
    - KDTree für innenliegende Punkt-Suche
    - NumPy für Entfernungsberechnungen
    - Batch-Distanz-Berechnung statt Schleife

    Args:
        loop_dict: Loop-Dictionary (aus find_all_boundary_loops)
        vertex_manager: VertexManager
        mesh_obj: Mesh mit .faces und .face_props
        existing_edges: Set aller bestehenden Edges (cached)
        max_edge_length: Max. Edge-Länge

    Returns:
        Anzahl eingefügter Faces
    """
    loop_vertices = loop_dict["vertices"]
    centroid = loop_dict["centroid"]

    vertices = vertex_manager.vertices
    new_faces = 0

    # === EARLY EXIT: Vertex-Set nur einmal building ===
    loop_vertex_set = set(loop_vertices)
    all_vertex_count = len(vertices)

    # === KDTree für innenliegende Kandidaten ===
    # Baue KDTree NUR einmal pro Loop
    all_coords_2d = vertices[:, :2]
    tree = cKDTree(all_coords_2d)

    # Finde Kandidaten: in Nähe des Centroids (~5x Loop-Durchmesser)
    loop_coords = vertices[np.array(loop_vertices), :2]
    loop_radii = np.linalg.norm(loop_coords - centroid, axis=1)
    avg_radius = loop_radii.mean()
    search_radius = max(avg_radius * 1.5, 50.0)  # Min 50m

    candidate_indices = tree.query_ball_point(centroid, r=search_radius)
    candidate_set = set(c for c in candidate_indices if c not in loop_vertex_set)

    if len(candidate_set) == 0:
        # Fallback: Nimm die gesamte Vertez-Liste
        candidate_set = set(range(all_vertex_count)) - loop_vertex_set

    # === Prüfe benachbarte Vertex-Paare (O(loop_size)) ===
    for i in range(len(loop_vertices)):
        v1 = loop_vertices[i]
        v2 = loop_vertices[(i + 1) % len(loop_vertices)]

        edge = (min(v1, v2), max(v1, v2))

        if edge in existing_edges:
            continue  # Edge existiert bereits

        # === LÜCKE! Berechne beste innenliegende Vertex (KDTree-basiert) ===
        v1_pos = vertices[v1, :2]
        v2_pos = vertices[v2, :2]
        mid_pos = 0.5 * (v1_pos + v2_pos)
        edge_length = np.linalg.norm(v1_pos - v2_pos)

        if edge_length > max_edge_length:
            print(f"        [!] Sehr lange Edge: {edge_length:.1f}m (v{v1} → v{v2})")

        # === Vectorisierte Distanz-Berechnung ===
        if len(candidate_set) == 0:
            continue

        candidate_array = np.array(list(candidate_set), dtype=np.int32)
        candidate_coords = vertices[candidate_array, :2]

        # Distanzen zum Midpoint
        distances = np.linalg.norm(candidate_coords - mid_pos, axis=1)

        # Finde Minimum (innenliegender Punkt)
        best_local_idx = np.argmin(distances)
        best_vertex = candidate_array[best_local_idx]

        # Erstelle Dreieck
        face = (v1, v2, best_vertex)
        mesh_obj.faces.append(face)
        mesh_obj.face_props[len(mesh_obj.faces) - 1] = {"material": "terrain"}

        new_faces += 1

    return new_faces


def fill_all_mesh_holes(
    mesh_obj,
    vertex_manager: VertexManager,
    max_edge_length: float = 100.0,
) -> int:
    """
    Hauptfunktion: Finde und schließe ALLE Mesh-Holes (OPTIMIERT).

    OPTIMIERUNGEN:
    - Shared existing_edges Set (rebuild nur einmal)
    - Loop-Processing mit lokalem KDTree
    - Caching von Centroiden
    - Early-exits bei Trivialfällen

    Args:
        mesh_obj: Mesh mit .faces und .face_props
        vertex_manager: VertexManager
        max_edge_length: Max. Edge-Länge

    Returns:
        Gesamtanzahl eingefügter Faces
    """
    # === Kompatibilität ===
    if hasattr(mesh_obj, "faces"):
        faces_list = mesh_obj.faces
    else:
        faces_list = mesh_obj.get("faces", [])

    if len(faces_list) == 0:
        return 0

    # === Baue shared existing_edges Set (einmal für alle Loops) ===
    existing_edges = set()
    for face in faces_list:
        v0, v1, v2 = face
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]
        for edge in edges:
            existing_edges.add(edge)

    # === Finde alle Loops ===
    loops = find_all_boundary_loops(faces_list, vertex_manager)

    if len(loops) == 0:
        return 0

    # === Fülle jeden Loop (mit cached existing_edges) ===
    total_faces = 0
    for loop_idx, loop_dict in enumerate(loops):
        loop_type = "OUTER" if loop_dict["is_outer"] else "ISLAND"
        vert_count = len(loop_dict["vertices"])
        print(f"  [Loop {loop_idx}] ({loop_type}, {vert_count} Verts) - fülle Lücken...")

        new_in_loop = fill_boundary_loop_holes(
            loop_dict,
            vertex_manager,
            mesh_obj,
            existing_edges=existing_edges,
            max_edge_length=max_edge_length,
        )

        if new_in_loop > 0:
            print(f"    → {new_in_loop} Faces eingefügt")

        total_faces += new_in_loop

    return total_faces
