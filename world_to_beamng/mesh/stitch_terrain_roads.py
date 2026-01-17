"""
Vervollständige die äußere Tile-Kante mit Dreiecken.

PROBLEM: Wo Straßen die Tile-Grenze schneiden, entstehen Lücken:
- Terrain-Vertices auf der Kante
- Road-Vertices auf der Kante
- Aber keine Edge dazwischen → Lücke

LÖSUNG:
1. Gehe den Umfang entlang (4 Seiten)
2. Sortiere Vertices pro Seite
3. Prüfe ob benachbarte Vertices verbunden sind
4. Falls nicht: Füge Hilfsdreieck ein
"""

import numpy as np
from typing import List, Tuple
from collections import defaultdict

from .vertex_manager import VertexManager


def stitch_terrain_to_roads(
    terrain_mesh,
    vertex_manager: VertexManager,
    tile_bounds: Tuple[float, float, float, float] = None,
) -> int:
    """
    Vervollständige die äußere Tile-Kante mit Dreiecken.

    Args:
        terrain_mesh: Mesh-Objekt oder Dict mit Faces und face_props
        vertex_manager: VertexManager
        tile_bounds: (x_min, x_max, y_min, y_max)

    Returns:
        Anzahl neuer Faces
    """
    print(f"  [Tile-Kante vervollständigen] Prüfe Umfang...")

    if tile_bounds is None:
        print(f"    [!] Keine tile_bounds angegeben")
        return 0

    x_min, x_max, y_min, y_max = tile_bounds
    tolerance = 2.0  # 2m Toleranz für "auf Kante" (exakt Grid-Spacing)

    # === Kompatibilität ===
    if hasattr(terrain_mesh, "faces"):
        faces_list = terrain_mesh.faces
        face_props_dict = getattr(terrain_mesh, "face_props", {})
    else:
        faces_list = terrain_mesh.get("faces", [])
        face_props_dict = terrain_mesh.get("face_props", {})

    vertices = vertex_manager.vertices

    # === STEP 1: Baue Edge-Set UND Adjacency-Map ===
    existing_edges = set()  # {(v1, v2), ...} normalisiert
    adjacency = defaultdict(set)  # {v_idx: {neighbor_indices}} - VIEL SCHNELLER als Edge-Iteration!
    vertices_in_edges = set()  # Alle Vertices die in irgendeiner Edge vorkommen

    for face in faces_list:
        v0, v1, v2 = face
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]
        for edge in edges:
            e1, e2 = edge
            existing_edges.add(edge)
            adjacency[e1].add(e2)
            adjacency[e2].add(e1)
            vertices_in_edges.add(e1)
            vertices_in_edges.add(e2)

    # === STEP 2: Pro Seite Vertices sammeln und sortieren ===
    def get_side_vertices(side: str) -> List[int]:
        """Sammle alle Vertices EXAKT auf einer Seite (keine Toleranz!)."""
        side_verts = []

        # NUR Vertices aus Edges prüfen
        for v_idx in vertices_in_edges:
            v = vertices[v_idx]
            x, y = v[0], v[1]

            on_side = False
            # EXAKTE Koordinaten-Prüfung (keine Toleranz!)
            if side == "north" and y == y_max:
                on_side = True
            elif side == "south" and y == y_min:
                on_side = True
            elif side == "east" and x == x_max:
                on_side = True
            elif side == "west" and x == x_min:
                on_side = True

            if on_side:
                side_verts.append(v_idx)

        return side_verts

    def sort_side_vertices(side_verts: List[int], side: str) -> List[int]:
        """Sortiere Vertices entlang einer Seite."""
        if side in ["north", "south"]:
            # Sortiere nach X (links → rechts)
            return sorted(side_verts, key=lambda v: vertices[v, 0])
        else:  # east/west
            # Sortiere nach Y (unten → oben)
            return sorted(side_verts, key=lambda v: vertices[v, 1])

    # === STEP 3: Für jede Seite Lücken finden und füllen ===
    new_faces = 0
    tile_center_2d = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])

    for side in ["north", "south", "east", "west"]:

        side_verts = get_side_vertices(side)
        if len(side_verts) < 2:
            print(f"      Nur {len(side_verts)} Vertices - überspringe")
            continue

        sorted_verts = sort_side_vertices(side_verts, side)

        # Prüfe ob benachbarte Vertices verbunden sind
        gaps_found = 0
        for i in range(len(sorted_verts) - 1):
            v1 = sorted_verts[i]
            v2 = sorted_verts[i + 1]

            # Ist diese Edge vorhanden?
            edge = (min(v1, v2), max(v1, v2))

            if edge not in existing_edges:
                # LÜCKE! Füge Hilfsdreieck ein
                v1_pos = vertices[v1]
                v2_pos = vertices[v2]
                mid_pos = (v1_pos[:2] + v2_pos[:2]) / 2.0

                # Finde innenliegenden Nachbarn mit DIREKTEM Adjacency-Zugriff (O(1) statt O(E))
                all_neighbors = adjacency[v1] | adjacency[v2]

                best_vertex = None
                best_score = float("inf")  # Kombinierter Score: distance

                # Vorgecachte innenliegende Richtung pro Seite
                to_center = tile_center_2d - mid_pos
                norm_center = np.linalg.norm(to_center)

                if norm_center > 1e-6:
                    to_center_normalized = to_center / norm_center

                    for v_candidate in all_neighbors:
                        if v_candidate == v1 or v_candidate == v2:
                            continue

                        v_pos = vertices[v_candidate, :2]
                        to_candidate = v_pos - mid_pos
                        dist_to_mid = np.linalg.norm(to_candidate)

                        if dist_to_mid < 1e-6:
                            continue

                        # Dot-Product für Innen-Prüfung
                        dot = np.dot(to_candidate, to_center_normalized) / dist_to_mid

                        # NUR innenliegende Candidates
                        if dot > 0:
                            # Score: Kleinere Distanz = besser
                            if dist_to_mid < best_score:
                                best_score = dist_to_mid
                                best_vertex = v_candidate

                # Falls kein passender Vertex gefunden: Ignoriere diese Lücke
                if best_vertex is None:
                    continue

                # Erstelle Dreieck mit Material "terrain"
                face = (v1, v2, best_vertex)

                if hasattr(terrain_mesh, "add_face"):
                    terrain_mesh.add_face(face[0], face[1], face[2], material="terrain")
                else:
                    terrain_mesh["faces"].append(face)
                    if "face_props" not in terrain_mesh:
                        terrain_mesh["face_props"] = {}
                    face_idx = len(terrain_mesh["faces"]) - 1
                    terrain_mesh["face_props"][face_idx] = {"material": "terrain"}

                new_faces += 1
                gaps_found += 1

        if gaps_found > 0:
            print(f"      {gaps_found} Lücken gefüllt")
        else:
            print(f"      ✓ Keine Lücken")

    print(f"  [✓] {new_faces} Dreiecke eingefügt (Tile-Kante vervollständigt)")
    # === VERIFIKATION: Prüfe dass Tile-Kante jetzt lückenlos verbunden ist ===
    print(f"  [Verifikation] Prüfe Tile-Kanten-Durchgängigkeit...")

    # Rebuild existing_edges mit neuen Faces
    existing_edges_new = set()
    for face in faces_list:
        v0, v1, v2 = face
        edges = [
            (min(v0, v1), max(v0, v1)),
            (min(v1, v2), max(v1, v2)),
            (min(v2, v0), max(v2, v0)),
        ]
        for edge in edges:
            existing_edges_new.add(edge)

    all_verified = True
    for side in ["north", "south", "east", "west"]:
        side_verts = get_side_vertices(side)
        if len(side_verts) < 2:
            continue

        sorted_verts = sort_side_vertices(side_verts, side)

        # Prüfe JEDES Paar benachbarter Vertices
        gaps_remaining = 0
        for i in range(len(sorted_verts) - 1):
            v1 = sorted_verts[i]
            v2 = sorted_verts[i + 1]
            edge = (min(v1, v2), max(v1, v2))

            if edge not in existing_edges_new:
                gaps_remaining += 1
                all_verified = False

        if gaps_remaining > 0:
            print(f"    [{side.upper()}] ✗ {gaps_remaining} Lücken NOCH VORHANDEN!")

    if all_verified:
        print(f"  [✓✓] Alle Tile-Kanten lückenlos verbunden!\n")
    else:
        print(f"  [!] WARNUNG: Nicht alle Lücken gefüllt!\n")

    return new_faces
