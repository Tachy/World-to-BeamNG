"""
Lokales Stitching: Loch-Suche entlang Search-Circles um Centerlines.

Statt globaler Boundary-Suche: Pro Centerline-Sample-Punkt einen Search-Circle
anlegen und nur dort nach Loch-Polygonen suchen.
"""

import numpy as np
from .. import config
from scipy.spatial import cKDTree
from mapbox_earcut import triangulate_float64
from collections import defaultdict
from ..utils.debug_exporter import DebugNetworkExporter


def find_boundary_polygons_in_circle(
    centerline_point,
    centerline_geometry,
    search_radius,
    road_width,
    vertex_manager,
    mesh,
    terrain_vertex_indices,
    cached_verts=None,
    cached_kdtree=None,
    cached_face_materials=None,
    cached_vertex_to_faces=None,
    cached_terrain_face_indices=None,
    debug=False,
):
    """
    Findet Boundary-Polygone in einem Search-Circle um einen Centerline-Punkt.

    Verwendet Face-basierte Edge-Klassifikation (Option A) und Connected Components
    um mehrere unabhängige Löcher pro Circle zu finden.

    Args:
        centerline_point: (x, y, z) - Mittelpunkt des Search-Circles
        centerline_geometry: (N, 3) Array - Komplette Centerline Punkte
        search_radius: Radius in Metern (dynamisch: road_width + GRID_SPACING*2.5)
        road_width: Breite der aktuellen Straße in Metern (für Merge-Threshold-Berechnung)
        vertex_manager: VertexManager mit allen Mesh-Vertices
        mesh: Mesh-Instanz mit terrain_faces und slope_faces
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

    # Hole alle Vertices (optional aus Cache)
    verts = np.asarray(cached_verts) if cached_verts is not None else np.asarray(vertex_manager.get_array())
    if len(verts) == 0:
        return []

    # KDTree für schnelle räumliche Suche (nur XY)
    kdtree = cached_kdtree if cached_kdtree is not None else cKDTree(verts[:, :2])
    face_materials = (
        cached_face_materials
        if cached_face_materials is not None
        else [mesh.face_props.get(idx, {}).get("material") for idx in range(len(mesh.faces))]
    )
    vertex_to_faces = cached_vertex_to_faces if cached_vertex_to_faces is not None else {}

    # Finde alle Vertices im Search-Circle
    circle_vertex_indices = kdtree.query_ball_point([cx, cy], r=search_radius)

    if len(circle_vertex_indices) < 3:
        return []

    circle_vertex_set = set(circle_vertex_indices)

    # Filter Centerline-Segmente in Reichweite (einmal pro Circle, nicht pro Edge)
    cl2d = np.asarray(centerline_geometry[:, :2], dtype=float)
    centerline_segments = []
    seg_start_arr = None
    seg_vec_arr = None
    seg_mid_arr = None
    if len(cl2d) >= 2:
        seg_start = cl2d[:-1]
        seg_end = cl2d[1:]
        mids = 0.5 * (seg_start + seg_end)
        dx = mids[:, 0] - cx
        dy = mids[:, 1] - cy
        dist2 = dx * dx + dy * dy
        max_r2 = (search_radius + 5.0) * (search_radius + 5.0)
        mask = dist2 <= max_r2
        if np.any(mask):
            centerline_segments = list(zip(seg_start[mask], seg_end[mask]))
            seg_start_arr = seg_start[mask]
            seg_vec_arr = seg_end[mask] - seg_start[mask]
            seg_mid_arr = mids[mask]

    # Finde relevante Face-Indizes über Vertex->Faces Mapping
    candidate_face_indices = set()
    for v in circle_vertex_indices:
        candidate_face_indices.update(vertex_to_faces.get(v, ()))

    if not candidate_face_indices:
        return []

    # Finde Faces deren Vertices im Circle sind (mindestens 2 von 3) - vektorisiert
    candidate_list = list(candidate_face_indices)
    candidate_faces = [mesh.faces[idx] for idx in candidate_list]

    # Vektorisiert: Zähle Vertices im Circle pro Face
    faces_in_circle = []
    face_indices_in_circle = []
    terrain_face_count = 0
    slope_face_count = 0

    for i, face in enumerate(candidate_faces):
        # Schneller: Set intersection statt sum mit generator
        vertices_in_count = len(circle_vertex_set.intersection(face))
        if vertices_in_count >= 2:
            face_idx = candidate_list[i]
            faces_in_circle.append(face)
            face_indices_in_circle.append(face_idx)
            mat = face_materials[face_idx] if face_idx < len(face_materials) else None
            if mat == "terrain":
                terrain_face_count += 1
            elif mat != "terrain":
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
    boundary_edges_single_edges = []
    boundary_edges_mixed = 0
    boundary_edges_single = 0

    for edge, face_list in edge_to_faces.items():
        v1, v2 = edge
        # WICHTIG: BEIDE Vertices müssen im Circle sein!
        # Verhindert, dass Circle-Rand-Edges als Boundaries erkannt werden
        if v1 not in circle_vertex_set or v2 not in circle_vertex_set:
            continue

        if len(face_list) == 1:
            boundary_edges_single_edges.append(edge)
            boundary_edges_single += 1
        elif len(face_list) == 2:
            f1, f2 = face_list
            f1_mat = face_materials[f1] if f1 < len(face_materials) else None
            f2_mat = face_materials[f2] if f2 < len(face_materials) else None
            f1_is_terrain = f1_mat == "terrain"
            f2_is_terrain = f2_mat == "terrain"
            # Boundary wenn ein Face Terrain ist und das andere nicht (=Straße/Slope/etc.)
            if f1_is_terrain != f2_is_terrain:
                boundary_edges_mixed += 1

    # Skip, wenn keine offenen Boundary-Edges vorhanden sind
    if len(boundary_edges_single_edges) < 3:
        return []

    # Verwende ALLE Boundary-Edges (keine Material-Trennung!)
    boundary_edges = boundary_edges_single_edges

    # Finde Connected Components (ohne Centerline-Überquerungen)
    components = _find_connected_components(
        boundary_edges,
        verts,
        seg_start_arr,
        seg_vec_arr,
        seg_mid_arr,
        centerline_point,
    )

    if not components:
        return []

    # Merge Components die am Kreisrand getrennt sind
    dynamic_merge_threshold = max(1.5, search_radius / 3.0)
    if len(components) > 2:
        components = _merge_nearby_components(components, verts, merge_threshold=dynamic_merge_threshold)

    # Baue Polygone aus Components
    terrain_face_set = {idx for idx, mat in enumerate(face_materials) if mat == "terrain"}
    polygons = []

    for component_edges in components:
        polygon = _build_polygon_from_component(
            component_edges,
            edge_to_faces,
            terrain_face_set,
            verts,
            debug=debug,
        )
        if polygon:
            polygons.append(polygon)

    # Trianguliere die Polygone und füge neue Faces hinzu
    if polygons:
        new_faces = _triangulate_polygons(polygons, verts, mesh, debug=debug)

    return polygons


def _edges_cross_centerline_vectorized(edges_arr, verts, seg_start_arr, seg_vec_arr, seg_mid_arr, centerline_point):
    """Vectorized Side-of-Line Test für viele Edges gegen vorgefilterte Segmente."""
    if seg_start_arr is None or seg_vec_arr is None or seg_mid_arr is None or len(seg_start_arr) == 0:
        return np.zeros(len(edges_arr), dtype=bool)

    # Edge-Endpunkte holen
    p1 = verts[edges_arr[:, 0], :2]
    p2 = verts[edges_arr[:, 1], :2]

    # Nutze Edge-Midpoints um das nächstgelegene Centerline-Segment zu wählen
    edge_mid = 0.5 * (p1 + p2)  # (N,2)
    # Distanzen zu Segment-Mittelpunkten
    diff = edge_mid[:, None, :] - seg_mid_arr[None, :, :]  # (N,S,2)
    dist2 = np.einsum("nsi,nsi->ns", diff, diff)  # (N,S)
    nearest_idx = np.argmin(dist2, axis=1)  # (N,)

    # Hole passende Segment-Starts und -Vektoren
    seg_starts = seg_start_arr[nearest_idx]
    seg_vecs = seg_vec_arr[nearest_idx]

    # Vektoren von Segmentstart zu Edge-Points
    to_p1 = p1 - seg_starts
    to_p2 = p2 - seg_starts

    # 2D Cross Products (z-Komponente)
    cross_p1 = seg_vecs[:, 0] * to_p1[:, 1] - seg_vecs[:, 1] * to_p1[:, 0]
    cross_p2 = seg_vecs[:, 0] * to_p2[:, 1] - seg_vecs[:, 1] * to_p2[:, 0]

    return cross_p1 * cross_p2 < 0


def _find_connected_components(boundary_edges, verts, seg_start_arr, seg_vec_arr, seg_mid_arr, centerline_point):
    """
    Findet alle zusammenhängenden Komponenten in Edge-Liste.

    Verwendet DFS um alle zusammenhängenden Edge-Gruppen zu finden.
    Edges die die Centerline überqueren werden ignoriert.
    Jede Komponente wird als Liste von Edges zurückgegeben.
    """
    if not boundary_edges:
        return []

    # Vectorized Centerline-Crossing Filter
    edges_arr = np.array(boundary_edges, dtype=int)
    crosses = _edges_cross_centerline_vectorized(
        edges_arr, verts, seg_start_arr, seg_vec_arr, seg_mid_arr, centerline_point
    )
    valid_edges_arr = edges_arr[~crosses]

    # Baue Adjacency-List (ungerichtet)
    adj = defaultdict(list)
    for v1, v2 in valid_edges_arr:
        adj[v1].append(v2)
        adj[v2].append(v1)

    # Finde Connected Components (Vertex-basiert) - Sammle Edges direkt
    visited_vertices = set()
    components_edges = []

    for start_v in adj.keys():
        if start_v in visited_vertices:
            continue

        # DFS um alle verbundenen Vertices zu finden
        component_verts = set()
        stack = [start_v]

        while stack:
            v = stack.pop()
            if v in component_verts:
                continue

            component_verts.add(v)
            visited_vertices.add(v)

            for neighbor in adj[v]:
                if neighbor not in component_verts:
                    stack.append(neighbor)

        if len(component_verts) >= 3:  # Mindestens 3 Vertices für ein Polygon
            # Sammle Edges für diese Component (schnell mit Set-Membership)
            comp_edges = []
            comp_verts_set = component_verts  # Bereits ein Set!
            for v1, v2 in valid_edges_arr:
                if v1 in comp_verts_set and v2 in comp_verts_set:
                    comp_edges.append((v1, v2))

            if len(comp_edges) >= 3:  # Mindestens 3 Edges für ein Polygon
                components_edges.append(comp_edges)

    return components_edges


def _merge_nearby_components(components_edges, verts, merge_threshold=2.0):
    """
    Merged Components die am Kreisrand durch fehlende Faces getrennt sind.

    Findet Endpoints zwischen Components, die nahe beieinander liegen,
    und fügt synthetische Edges hinzu, um sie zu verbinden.

    Args:
        components_edges: Liste von Edge-Listen (eine pro Component)
        verts: Vertex-Array
        merge_threshold: Maximale Distanz zwischen Endpoints für Merge (in Metern)

    Returns:
        Neue Liste von Components (evtl. gemerged)
    """
    if len(components_edges) <= 2:
        return components_edges

    # Baue Adjacency für jede Component um Endpoints zu finden
    def find_endpoints(comp_edges):
        adj = defaultdict(set)
        for v1, v2 in comp_edges:
            adj[v1].add(v2)
            adj[v2].add(v1)

        endpoints = []
        for v, neighbors in adj.items():
            if len(neighbors) == 1:  # Endpunkt
                endpoints.append(v)
        return endpoints

    # Finde alle Endpoints pro Component (nur einmal!)
    component_endpoints = [find_endpoints(comp) for comp in components_edges]

    # Finde Merge-Kandidaten: Endpoints zwischen verschiedenen Components
    # Optimiert: Berechne nur die Distances die wir brauchen
    merge_pairs = []  # Liste von (comp_i_idx, comp_j_idx, v_i, v_j, distance)

    for i in range(len(components_edges)):
        eps_i = component_endpoints[i]
        if not eps_i:  # Skip wenn keine Endpoints
            continue
            
        for j in range(i + 1, len(components_edges)):
            eps_j = component_endpoints[j]
            if not eps_j:  # Skip wenn keine Endpoints
                continue

            # Finde nächstgelegene Endpoints zwischen i und j
            for ep_i in eps_i:
                pi = verts[ep_i]
                for ep_j in eps_j:
                    pj = verts[ep_j]
                    dist = np.linalg.norm(np.array(pi) - np.array(pj))

                    if dist <= merge_threshold:
                        merge_pairs.append((i, j, ep_i, ep_j, dist))

    # Early exit wenn keine Merges möglich
    if not merge_pairs:
        return components_edges

    # Sortiere nach Distanz für greedy Union-Find (merge closest first)
    merge_pairs.sort(key=lambda x: x[4])

    # Union-Find für Component-Merging
    parent = list(range(len(components_edges)))

    def find(x):
        # Iterativ statt rekursiv (verhindert Stack Overflow)
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    # Merge Components und füge synthetische Edges hinzu
    synthetic_edges = []
    for i, j, ep_i, ep_j, dist in merge_pairs:
        if union(i, j):
            synthetic_edges.append((ep_i, ep_j))

    # Baue neue Component-Liste (optimiert mit Vertex-Caching)
    merged_components = defaultdict(list)
    merged_components_verts = defaultdict(set)  # Cache für Vertex-Sets
    
    for idx, comp_edges in enumerate(components_edges):
        root = find(idx)
        merged_components[root].extend(comp_edges)
        # Sammle Vertices für diesen Component
        for v1, v2 in comp_edges:
            merged_components_verts[root].add(v1)
            merged_components_verts[root].add(v2)

    # Füge synthetische Edges hinzu (optimiert mit gecachten Vertex-Sets)
    for edge in synthetic_edges:
        v1, v2 = edge
        # Finde Component der v1/v2 gehört (schnell mit Cache)
        for root, vertices in merged_components_verts.items():
            if v1 in vertices or v2 in vertices:
                merged_components[root].append(edge)
                merged_components_verts[root].add(v1)
                merged_components_verts[root].add(v2)
                break

    return list(merged_components.values())


def _build_complete_boundary_polygons(
    terrain_components,
    road_components,
    edge_to_faces,
    face_materials,
    verts,
    seg_start_arr,
    seg_vec_arr,
    seg_mid_arr,
    centerline_point,
    debug=False,
):
    """
    Baut vollständige Boundary-Polygone aus Terrain+Road Component-Paaren.

    Ein Gap wird NUR geschlossen, wenn BEIDE Seiten vorhanden sind:
    - Terrain-Edge (grün): Terrain-zu-Gap Boundary
    - Road-Edge (rot): Road-zu-Gap Boundary

    Verhindert "Terrain-Only" oder "Road-Only" Stitching, das zu Müll-Faces führt.

    WICHTIG: Paart nur Components auf DERSELBEN Seite der Centerline!

    Args:
        terrain_components: Liste von Terrain-Component-Edge-Listen
        road_components: Liste von Road-Component-Edge-Lists
        edge_to_faces: Edge->Face Mapping
        face_materials: Liste von Face-Materials
        verts: Vertex-Array
        seg_start_arr: Centerline Segment-Starts (für Side-of-Line Test)
        seg_vec_arr: Centerline Segment-Vektoren
        seg_mid_arr: Centerline Segment-Mittelpunkte
        centerline_point: Aktueller Centerline-Punkt
        debug: Debug-Flag

    Returns:
        Liste von Polygon-Dicts (geschlossene Boundaries)
    """
    polygons = []

    # Wenn nur eine Seite vorhanden ist → KEINE Polygone erstellen!
    if not terrain_components or not road_components:
        print(f"    ⚠️  Unvollständige Boundary: Terrain={len(terrain_components)}, Road={len(road_components)}")
        print(f"    → KEIN Stitching (verhindert Müll-Faces)")
        return polygons

    # Baue Terrain-Face-Set
    terrain_face_set = {idx for idx, mat in enumerate(face_materials) if mat == "terrain"}

    # Pre-compute Side-of-Line für alle Components (Cache für Performance)
    terrain_sides = []
    road_sides = []
    terrain_endpoints_cache = []
    road_endpoints_cache = []

    for terrain_comp in terrain_components:
        side = _get_component_side(terrain_comp, verts, seg_start_arr, seg_vec_arr, seg_mid_arr)
        terrain_sides.append(side)
        terrain_endpoints_cache.append(_find_component_endpoints(terrain_comp, verts))

    for road_comp in road_components:
        side = _get_component_side(road_comp, verts, seg_start_arr, seg_vec_arr, seg_mid_arr)
        road_sides.append(side)
        road_endpoints_cache.append(_find_component_endpoints(road_comp, verts))

    # Strategie: Kombiniere jeweils nächstgelegene Terrain+Road Paare
    used_terrain = set()
    used_road = set()

    # Fast path: Wenn nur 1 Terrain und 1 Road auf gleicher Seite → direkt paaren
    if len(terrain_components) == 1 and len(road_components) == 1:
        terrain_side = terrain_sides[0]
        road_side = road_sides[0]
        if _check_same_side_cached(terrain_side, road_side):
            # Direkt paaren ohne Distanz-Berechnung
            terrain_endpoints = terrain_endpoints_cache[0]
            road_endpoints = road_endpoints_cache[0]
            closing_edges = _find_closing_edges(terrain_endpoints, road_endpoints, verts)
            combined_edges = terrain_components[0] + closing_edges + road_components[0]
            polygon = _build_polygon_from_component(combined_edges, edge_to_faces, terrain_face_set, verts, debug=debug)
            if polygon:
                polygons.append(polygon)
            print(
                f"    ✅ {len(polygons)} vollständige Boundary-Polygone erstellt (aus {len(terrain_components)} Terrain + {len(road_components)} Road)"
            )
            return polygons

    for t_idx, terrain_comp in enumerate(terrain_components):
        if t_idx in used_terrain:
            continue

        # Finde nächstgelegene Road-Component
        best_road_idx = None
        best_dist = float("inf")

        terrain_side = terrain_sides[t_idx]

        for r_idx, road_comp in enumerate(road_components):
            if r_idx in used_road:
                continue

            # WICHTIG: Nur paaren, wenn auf DERSELBEN Seite der Centerline!
            road_side = road_sides[r_idx]
            if not _check_same_side_cached(terrain_side, road_side):
                continue

            # Berechne minimale Distanz zwischen Components
            min_dist = _compute_component_distance(terrain_comp, road_comp, verts)

            if min_dist < best_dist:
                best_dist = min_dist
                best_road_idx = r_idx

                # Early exit: Wenn Distanz sehr klein (< 2m), ist das gut genug
                if best_dist < 2.0:
                    break

        if best_road_idx is not None:
            # WICHTIG: Füge Closing-Edges zwischen Terrain und Road Endpoints hinzu
            # Das verbindet die beiden Components am Kreisrand
            terrain_comp_with_closing = terrain_comp.copy()
            road_comp_copy = road_components[best_road_idx].copy()

            # Hole Endpoints aus Cache (statt neu berechnen)
            terrain_endpoints = terrain_endpoints_cache[t_idx]
            road_endpoints = road_endpoints_cache[best_road_idx]

            # Füge Closing-Edges zwischen nächstgelegenen Endpoints hinzu
            closing_edges = _find_closing_edges(terrain_endpoints, road_endpoints, verts)

            # Kombiniere: Terrain + Closing + Road
            combined_edges = terrain_comp_with_closing + closing_edges + road_comp_copy

            # Baue Polygon aus kombinierten Edges (jetzt geschlossen!)
            polygon = _build_polygon_from_component(combined_edges, edge_to_faces, terrain_face_set, verts, debug=debug)

            if polygon:
                polygons.append(polygon)
                used_terrain.add(t_idx)
                used_road.add(best_road_idx)

    return polygons


def _compute_component_distance(comp1_edges, comp2_edges, verts):
    """Berechne minimale Distanz zwischen zwei Components (vectorized)."""
    # Sample maximal erste 2 Punkte pro Component für Speed
    sample1 = min(2, len(comp1_edges))
    sample2 = min(2, len(comp2_edges))

    if sample1 == 0 or sample2 == 0:
        return float("inf")

    # Hole Vertices vektorisiert
    p1_indices = [comp1_edges[i][0] for i in range(sample1)]
    p2_indices = [comp2_edges[i][0] for i in range(sample2)]

    p1_coords = verts[p1_indices, :2]  # (N1, 2)
    p2_coords = verts[p2_indices, :2]  # (N2, 2)

    # Berechne alle Distanzen auf einmal (Broadcasting)
    diff = p1_coords[:, None, :] - p2_coords[None, :, :]  # (N1, N2, 2)
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)  # (N1, N2)

    return float(np.sqrt(np.min(dist2)))


def _get_component_side(comp_edges, verts, seg_start_arr, seg_vec_arr, seg_mid_arr):
    """Berechne durchschnittlichen Side-of-Line für Component (cachebar)."""
    if seg_start_arr is None or len(seg_start_arr) == 0:
        return 0.0

    # Sample maximal erste 3 Edges (statt 5) für Speed
    sample_count = min(3, len(comp_edges))
    if sample_count == 0:
        return 0.0

    # Vectorized: Hole alle Sample-Vertices auf einmal
    sample_verts = verts[[comp_edges[i][0] for i in range(sample_count)], :2]

    # Finde nächste Segmente für alle Samples
    diff = sample_verts[:, None, :] - seg_mid_arr[None, :, :]  # (N, S, 2)
    dist2 = np.einsum("nsi,nsi->ns", diff, diff)  # (N, S)
    nearest_indices = np.argmin(dist2, axis=1)  # (N,)

    # Berechne Cross-Products vektorisiert
    seg_starts = seg_start_arr[nearest_indices]  # (N, 2)
    seg_vecs = seg_vec_arr[nearest_indices]  # (N, 2)
    to_p = sample_verts - seg_starts  # (N, 2)
    cross_products = seg_vecs[:, 0] * to_p[:, 1] - seg_vecs[:, 1] * to_p[:, 0]  # (N,)

    return float(np.mean(cross_products))


def _check_same_side_cached(side1, side2):
    """Prüft ob zwei gecachte Sides auf derselben Seite liegen."""
    # Toleranz für Components nahe Centerline
    if abs(side1) < 0.1 or abs(side2) < 0.1:
        return True
    return (side1 * side2) > 0


def _components_on_same_side(
    comp1_edges, comp2_edges, verts, seg_start_arr, seg_vec_arr, seg_mid_arr, centerline_point
):
    """
    Prüft ob zwei Components auf derselben Seite der Centerline liegen.

    DEPRECATED: Verwende stattdessen _get_component_side() + _check_same_side_cached()
    für bessere Performance bei mehrfachen Aufrufen.

    Args:
        comp1_edges: Edges der ersten Component
        comp2_edges: Edges der zweiten Component
        verts: Vertex-Array
        seg_start_arr, seg_vec_arr, seg_mid_arr: Centerline Segment-Daten
        centerline_point: Aktueller Centerline-Punkt

    Returns:
        True wenn beide Components auf derselben Seite liegen
    """
    if seg_start_arr is None or seg_vec_arr is None or seg_mid_arr is None or len(seg_start_arr) == 0:
        return True

    side1 = _get_component_side(comp1_edges, verts, seg_start_arr, seg_vec_arr, seg_mid_arr)
    side2 = _get_component_side(comp2_edges, verts, seg_start_arr, seg_vec_arr, seg_mid_arr)

    return _check_same_side_cached(side1, side2)


def _find_component_endpoints(component_edges, verts):
    """
    Finde Endpoints einer Component (Vertices mit nur 1 Nachbar).

    Returns:
        Liste von Vertex-Indices (Endpoints)
    """
    adj = defaultdict(set)
    for v1, v2 in component_edges:
        adj[v1].add(v2)
        adj[v2].add(v1)

    endpoints = []
    for v, neighbors in adj.items():
        if len(neighbors) == 1:
            endpoints.append(v)

    return endpoints


def _find_closing_edges(endpoints1, endpoints2, verts):
    """
    Finde Closing-Edges zwischen zwei Endpoint-Listen.

    Strategie:
    - Finde nächstgelegene Paare zwischen endpoints1 und endpoints2
    - Erstelle Edges zwischen diesen Paaren
    - Maximal 2 Closing-Edges (beide Seiten des Gaps)

    Args:
        endpoints1: Liste von Vertex-Indices (z.B. Terrain-Endpoints)
        endpoints2: Liste von Vertex-Indices (z.B. Road-Endpoints)
        verts: Vertex-Array

    Returns:
        Liste von (v1, v2) Closing-Edges
    """
    if not endpoints1 or not endpoints2:
        return []

    closing_edges = []
    used_ep1 = set()
    used_ep2 = set()

    # Finde bis zu 2 Paare (beide Seiten)
    for _ in range(min(2, len(endpoints1), len(endpoints2))):
        best_dist = float("inf")
        best_pair = None

        for ep1 in endpoints1:
            if ep1 in used_ep1:
                continue
            p1 = verts[ep1][:2]

            for ep2 in endpoints2:
                if ep2 in used_ep2:
                    continue
                p2 = verts[ep2][:2]

                dist = np.linalg.norm(p1 - p2)
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (ep1, ep2)

        if best_pair:
            closing_edges.append(best_pair)
            used_ep1.add(best_pair[0])
            used_ep2.add(best_pair[1])

    return closing_edges


def _build_polygon_from_component(component_edges, edge_to_faces, terrain_face_set, verts, debug=False):
    """
    Baut geschlossenes Polygon aus Component-Edges.

    WICHTIG: Alle Edges einer Component bilden EINEN zusammenhängenden Pfad,
    der über Terrain UND Slope geht (wechselt zwischen beiden Seiten der Straße).

    Args:
        component_edges: Liste von (v1, v2) Edges in dieser Komponente
        edge_to_faces: Dict mapping edge -> face_list
        terrain_face_set: Set von Terrain-Face-Indices
        verts: Vertex-Array
        debug: Debug-Flag

    Returns:
        Polygon-Dict oder None
    """
    if not component_edges:
        return None

    # Baue EINEN zusammenhängenden Pfad aus allen Edges
    # (KEINE Trennung nach Terrain/Slope - das zerreißt zusammenhängende Komponenten!)
    polygon_vertices = _build_ordered_path_from_edges(component_edges)

    if len(polygon_vertices) < 3:
        return None

    # Entferne Duplikate am Anfang/Ende falls Loop geschlossen
    if len(polygon_vertices) > 1 and polygon_vertices[0] == polygon_vertices[-1]:
        polygon_vertices = polygon_vertices[:-1]

    if len(polygon_vertices) < 3:
        return None

    # Zähle Terrain/Slope-Edges für Statistik
    terrain_count = 0
    slope_count = 0
    for v1, v2 in component_edges:
        edge = tuple(sorted([v1, v2]))
        if edge in edge_to_faces:
            face_list = edge_to_faces[edge]
            has_terrain = any(f in terrain_face_set for f in face_list)
            has_slope = any(f not in terrain_face_set for f in face_list)
            if has_terrain:
                terrain_count += 1
            if has_slope:
                slope_count += 1

    # Konvertiere zu Koordinaten
    coords = [tuple(verts[v]) for v in polygon_vertices]

    return {
        "vertices": polygon_vertices,
        "coords": coords,
        "terrain_count": terrain_count,
        "slope_count": slope_count,
        "centerline_point": tuple(verts[polygon_vertices[0]]) if polygon_vertices else None,
    }


def _build_ordered_path_from_edges(edges):
    """
    Baut geordneten Pfad aus ungeordneten Edges.

    Verwendet einen verbesserten Algorithmus der auch bei Gabeln funktioniert:
    - Findet Start-Vertex (Vertex mit nur 1 Nachbar, oder beliebiger wenn Loop)
    - Baut Pfad durch Greedy-Auswahl noch nicht besuchter Edges
    - Behandelt Gabeln durch Priorisierung von unbesuchten Edges

    Args:
        edges: Liste von (v1, v2) Tuples (können orientiert oder unorientiert sein)

    Returns:
        Liste von Vertex-Indices in Reihenfolge
    """
    if not edges:
        return []

    # Baue Adjacency-List und Edge-Set gleichzeitig (für Speed)
    adj = defaultdict(set)
    edge_set = set()

    for v1, v2 in edges:
        adj[v1].add(v2)
        adj[v2].add(v1)
        edge_set.add(tuple(sorted([v1, v2])))

    # Finde Start-Vertex: Vertex mit nur 1 Nachbar (Endpunkt), sonst beliebig
    start_v = None
    for v, neighbors in adj.items():
        if len(neighbors) == 1:
            start_v = v
            break

    if start_v is None:
        start_v = next(iter(adj)) if adj else edges[0][0]

    # Baue Pfad mit Edge-Tracking (optimiert)
    path = [start_v]
    current = start_v
    prev = None

    while True:
        # Schneller: Direkt in Adjacency schauen statt Liste bauen
        neighbors = adj[current]

        # Finde ersten unbenutzten Nachbarn
        next_v = None
        for neighbor in neighbors:
            if neighbor == prev:  # Verhindere Rückwärts-Edge
                continue

            edge = tuple(sorted([current, neighbor]))
            if edge in edge_set:
                next_v = neighbor
                edge_set.discard(edge)  # Remove aus Set (schneller als used_edges)
                break

        if next_v is None:
            break

        path.append(next_v)
        prev = current
        current = next_v

    return path


def _triangulate_polygons(polygons, verts, mesh, debug=False):
    """
    Trianguliert Boundary-Polygone mit mapbox-earcut (Ear Clipping) und
    fügt die Faces direkt per mesh.add_face() hinzu.

    Args:
        polygons: Liste von Polygon-Dicts mit 'vertices' und 'coords'
        verts: Vertex-Array (NumPy)
        mesh: Mesh-Instanz (verwendet mesh.add_face() für Deduplizierung)
        debug: Debug-Flag

    Returns:
        Liste der neu erzeugten Face-Indices
    """
    if not polygons:
        return []

    new_faces = []

    for poly_idx, poly in enumerate(polygons):
        polygon_vertices = poly.get("vertices", [])

        if len(polygon_vertices) < 3:
            continue

        # Extrahiere 2D-Koordinaten (x, y) für Earcut (direkte numpy indexierung)
        polygon_verts_arr = np.array(polygon_vertices, dtype=np.int32)
        coords_3d = verts[polygon_verts_arr]  # Schneller als list comprehension
        coords_2d = coords_3d[:, :2]  # Nur X, Y

        # Earcut erwartet Nx2 float64 Array und ring_end_indices (End-Index jedes Rings)
        coords_arr = np.asarray(coords_2d, dtype=np.float64)
        ring_end_indices = np.array([len(coords_arr)], dtype=np.uint32)

        try:
            # Ear Clipping Triangulation (liefert Vertex-Indizes bezogen auf coords_2d)
            indices = triangulate_float64(coords_arr, ring_end_indices)

            if indices.size == 0:
                continue

            # Mappe Triangles zurück auf globale Vertex-Indizes und füge hinzu
            for i in range(0, len(indices), 3):
                v0_local, v1_local, v2_local = indices[i], indices[i + 1], indices[i + 2]
                v0_global = polygon_vertices[v0_local]
                v1_global = polygon_vertices[v1_local]
                v2_global = polygon_vertices[v2_local]

                face_idx = mesh.add_face(v0_global, v1_global, v2_global, material="terrain")
                new_faces.append(face_idx)

            if debug:
                print(f"  [Triangulation] Polygon {poly_idx}: {len(indices) // 3} Dreiecke erzeugt (earcut)")

        except Exception as e:
            if debug:
                print(f"  [!] Triangulation Polygon {poly_idx} fehlgeschlagen: {e}")
            continue

    return new_faces


def export_boundary_polygons_to_json(
    polygons,
    centerline_point,
    search_radius=None,
    circle_segments=64,
):
    """
    Exportiert Boundary-Polygone als JSON-Struktur für debug_network.json.

    Args:
        polygons: Liste von Polygon-Dicts
        centerline_point: (x, y, z) - Centerline-Sample-Punkt
        search_radius: Optionaler Radius für Visualisierung des Suchkreises
        circle_segments: Anzahl Segmente für Kreis-Approximation

    Returns:
        Liste von Boundary-Polygon-Dicts für debug_network.json
    """
    result = []

    # Exportiere Boundary-Polygone
    for poly_idx, poly in enumerate(polygons):
        coords = poly.get("coords", [])
        terrain_count = poly.get("terrain_count", 0)
        slope_count = poly.get("slope_count", 0)

        if len(coords) >= 3:
            result.append(
                {
                    "type": "boundary",
                    "coords": [[float(c[0]), float(c[1]), float(c[2])] for c in coords],
                    "color": [1.0, 0.0, 1.0],  # Magenta
                    "terrain_count": terrain_count,
                    "slope_count": slope_count,
                }
            )

    # Exportiere Suchkreis
    if search_radius is not None:
        sr = search_radius
        segs = max(8, int(circle_segments) if circle_segments is not None else 64)
        cx, cy, cz = centerline_point

        circle_coords = []
        for i in range(segs):
            angle = 2.0 * np.pi * i / segs
            x = cx + sr * np.cos(angle)
            y = cy + sr * np.sin(angle)
            circle_coords.append([float(x), float(y), float(cz)])

        result.append(
            {
                "type": "search_circle",
                "coords": circle_coords,
                "color": [1.0, 0.0, 0.0],  # Rot
                "radius": float(sr),
                "center": [float(cx), float(cy), float(cz)],
            }
        )

    return result


def _export_search_circle_to_debug(centerline_point, search_radius, sample_rate=1):
    """
    Exportiert einen einzelnen Suchkreis zum Debug-Exporter (Singleton).

    Zeichnet einen Kreis am centerline_point mit gegebenem radius.
    Farbe: Gelb (unterscheidet sich von Component-Lines).

    Args:
        centerline_point: (x, y, z) - Mittelpunkt des Suchkreises
        search_radius: Radius in Metern
        sample_rate: Nur jeder N-te Sample wird exportiert (default: 1 = alle)
    """
    # Nutze Hash der Centerline-Koordinaten als Pseudo-Counter
    cx, cy, cz = centerline_point
    coord_hash = hash((round(cx, 2), round(cy, 2)))

    # Nur jeden N-ten Sample exportieren
    if coord_hash % sample_rate != 0:
        return

    exporter = DebugNetworkExporter.get_instance()

    # Generiere Kreis-Punkte (32 Segmente für glatten Kreis)
    segs = 32
    circle_coords = []
    for i in range(segs + 1):  # +1 um Kreis zu schließen
        angle = 2.0 * np.pi * i / segs
        x = cx + search_radius * np.cos(angle)
        y = cy + search_radius * np.sin(angle)
        circle_coords.append((float(x), float(y), float(cz)))

    # Exportiere als Component-Line (gelbe Farbe)
    exporter.add_component_line(
        coords=circle_coords,
        color=[1.0, 1.0, 0.0],  # Gelb
        label=f"search_circle_{cx:.1f}_{cy:.1f}",
        line_width=2.0,
    )


def _export_boundary_polygons_to_debug(polygons, centerline_point, search_radius=None, sample_rate=100):
    """
    Exportiert Boundary-Polygone zum Debug-Exporter (Singleton).

    Args:
        polygons: Liste von Polygon-Dicts
        centerline_point: (x, y, z) - Centerline-Sample-Punkt
        search_radius: Optionaler Suchradius
        sample_rate: Nur jeder N-te Sample wird exportiert (default: 100)
    """
    # Nutze Hash der Centerline-Koordinaten als Pseudo-Counter
    cx, cy, cz = centerline_point
    coord_hash = hash((round(cx, 2), round(cy, 2)))

    # Nur jeden N-ten Sample exportieren
    if coord_hash % sample_rate != 0:
        return

    exporter = DebugNetworkExporter.get_instance()

    # Exportiere Boundary-Polygone
    for poly_idx, poly in enumerate(polygons):
        coords = poly.get("coords", [])
        terrain_count = poly.get("terrain_count", 0)
        slope_count = poly.get("slope_count", 0)

        if len(coords) >= 3:
            # Konvertiere zu Float-Listen (nicht NumPy)
            coords_list = [[float(c[0]), float(c[1]), float(c[2])] for c in coords]

            exporter.add_boundary(
                {
                    "type": "boundary",
                    "coords": coords_list,
                    "color": [1.0, 0.0, 1.0],  # Magenta
                    "terrain_count": int(terrain_count),
                    "slope_count": int(slope_count),
                }
            )

    # Suchkreis wird separat über _export_search_circle_to_debug() exportiert (gelb)
    # NICHT hier (verhindert Duplikation)


def _export_component_lines_to_debug(components_edges, verts, centerline_point, component_type="auto", sample_rate=100):
    """
    Exportiert Connected Component Linien zum Debug-Exporter (Singleton).

    Verwendet einen Sample-Counter um nur jeden N-ten Sample zu exportieren.

    Args:
        components_edges: Liste von Edge-Listen (eine pro Component)
        verts: Vertex-Array
        centerline_point: (x, y, z) - Centerline-Sample-Punkt (für Zähler)
        component_type: "terrain" (grün), "road" (rot), oder "auto" (alternierend)
        sample_rate: Nur jeder N-te Sample wird exportiert (default: 100)
    """
    # Nutze Hash der Centerline-Koordinaten als Pseudo-Counter
    cx, cy, cz = centerline_point
    coord_hash = hash((round(cx, 2), round(cy, 2)))

    # Nur jeden N-ten Sample exportieren
    if coord_hash % sample_rate != 0:
        return

    exporter = DebugNetworkExporter.get_instance()

    for comp_idx, comp_edges in enumerate(components_edges):
        # Baue geordneten Pfad aus Edges
        ordered_path = _build_ordered_path_from_edges(comp_edges)

        if len(ordered_path) < 2:
            continue

        # Konvertiere Vertex-Indizes zu Koordinaten
        coords = [tuple(verts[v]) for v in ordered_path]

        # Bestimme Farbe und Label basierend auf component_type
        if component_type == "terrain":
            color = [0.2, 0.8, 0.2]  # Grün für Terrain
            label = f"component_terrain_{comp_idx}"
        elif component_type == "road":
            color = [0.8, 0.2, 0.2]  # Rot für Straße
            label = f"component_road_{comp_idx}"
        else:  # "auto" - alternierend
            if comp_idx == 0:
                color = [0.2, 0.8, 0.2]  # Grün
                label = "component_terrain"
            elif comp_idx == 1:
                color = [0.8, 0.2, 0.2]  # Rot
                label = "component_road"
            else:
                color = [0.2, 0.2, 0.8 + (comp_idx * 0.1) % 0.4]
                label = f"component_{comp_idx}"

        exporter.add_component_line(
            coords=coords,
            color=color,
            label=label,
            line_width=3.0,
        )
