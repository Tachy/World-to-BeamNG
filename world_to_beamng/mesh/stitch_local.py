"""
Lokales Stitching: Loch-Suche entlang Search-Circles um Centerlines.

Statt globaler Boundary-Suche: Pro Centerline-Sample-Punkt einen Search-Circle
anlegen und nur dort nach Loch-Polygonen suchen.
"""

import numpy as np
from scipy.spatial import cKDTree
from mapbox_earcut import triangulate_float64
from collections import defaultdict

from world_to_beamng import config

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
        search_radius: Radius in Metern (typisch 10.0m)
        vertex_manager: VertexManager mit allen Mesh-Vertices
        mesh: Mesh-Instanz mit terrain_faces und slope_faces
        road_width: Breite der Straße in Metern
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
    cx, cy, _ = centerline_point

    # Hole alle Vertices (optional aus Cache)
    verts = np.asarray(cached_verts) if cached_verts is not None else np.asarray(vertex_manager.get_array())
    if len(verts) == 0:
        return []

    # KDTree für schnelle räumliche Suche (nur XY)
    kdtree = cached_kdtree if cached_kdtree is not None else cKDTree(verts[:, :2])
    face_materials = (
        cached_face_materials
        if cached_face_materials is not None
        else {idx: mesh.face_props.get(idx, {}).get("material") for idx in range(len(mesh.faces))}
    )
    vertex_to_faces = cached_vertex_to_faces if cached_vertex_to_faces is not None else {}

    # Finde alle Vertices im Search-Circle
    circle_vertex_indices = kdtree.query_ball_point([cx, cy], r=search_radius)

    if len(circle_vertex_indices) < 3:
        return []

    circle_vertex_set = set(circle_vertex_indices)

    # Filter Centerline-Segmente in Reichweite (einmal pro Circle, nicht pro Edge)
    centerline_segments = []
    seg_start_arr = None
    seg_vec_arr = None
    seg_mid_arr = None

    if centerline_geometry is not None and len(centerline_geometry) >= 2:
        cl2d = np.asarray(centerline_geometry[:, :2], dtype=float)
        if len(cl2d) >= 2:
            seg_start = cl2d[:-1]
            seg_end = cl2d[1:]
            mids = 0.5 * (seg_start + seg_end)
            dx = mids[:, 0] - cx
            dy = mids[:, 1] - cy
            dist2 = dx * dx + dy * dy
            # WICHTIG: Großzügiger Filter - nehme alle Segmente die den Circle überschneiden könnten!
            # Segment-Länge kann größer als search_radius sein, deshalb: radius + max_segment_length
            max_r2 = (search_radius * 2.0 + 20.0) ** 2  # Großzügig: 2x Radius + 20m Puffer
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

    # Finde Faces deren Vertices im Circle sind (mindestens 2 von 3)
    faces_in_circle = []
    face_indices_in_circle = []
    terrain_face_count = 0
    slope_face_count = 0

    for face_idx in candidate_face_indices:
        face = mesh.faces[face_idx]
        vertices_in_circle = sum(1 for v in face if v in circle_vertex_set)
        if vertices_in_circle >= 2:
            faces_in_circle.append(face)
            face_indices_in_circle.append(face_idx)
            mat = face_materials.get(face_idx)  # ← Dict-Zugriff statt Index
            if mat == "terrain":
                terrain_face_count += 1
            elif mat in ("slope", "road"):
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
        if v1 not in circle_vertex_set or v2 not in circle_vertex_set:
            continue

        if len(face_list) == 1:
            boundary_edges_single_edges.append(edge)
            boundary_edges_single += 1
        elif len(face_list) == 2:
            f1, f2 = face_list
            f1_mat = face_materials.get(f1)  # ← Dict-Zugriff statt Index
            f2_mat = face_materials.get(f2)  # ← Dict-Zugriff statt Index
            f1_is_terrain = f1_mat == "terrain"
            f2_is_terrain = f2_mat == "terrain"
            if f1_is_terrain != f2_is_terrain:
                boundary_edges_mixed += 1

    # Skip, wenn keine offenen Boundary-Edges vorhanden sind
    if len(boundary_edges_single_edges) < 3:
        return []

    # Komponenten nur aus offenen Rändern bilden (keine Material-Wechsel-Kanten)
    boundary_edges = boundary_edges_single_edges

    # Baue terrain_face_set VOR Component-Bildung (wird für Face-Type-Separation benötigt)
    terrain_face_set = (
        cached_terrain_face_indices
        if cached_terrain_face_indices is not None
        else {idx for idx, mat in enumerate(face_materials) if mat == "terrain"}
    )

    # Finde Connected Components (ohne Centerline-Überquerungen, getrennt nach Face-Typ)
    components = _find_connected_components(
        boundary_edges,
        verts,
        seg_start_arr,
        seg_vec_arr,
        seg_mid_arr,
        centerline_point,
        edge_to_faces,
        terrain_face_set,
    )

    _export_component_lines_to_debug(
        components, verts, centerline_point, edge_to_faces, terrain_face_set, sample_rate=1
    )

    if not components:
        return []

    # Merge Components die am Kreisrand getrennt sind
    if len(components) >= 2:
        if centerline_geometry is not None:
            # Roads: Merge mit Crossing-Check (verhindert Merging über die Straße)
            dynamic_merge_threshold = max(1.5, config.GRID_SPACING * 2.5)  # Roads: 5m bei 2m Grid
            components = _merge_nearby_components(
                components,
                verts,
                edge_to_faces,
                terrain_face_set,
                merge_threshold=dynamic_merge_threshold,
                centerline_geom=centerline_geometry,
            )
        else:
            # Junctions: Merge OHNE Crossing-Check (keine Centerline verfügbar)
            # Nutze gleichen Threshold wie Roads - Components sollen sich auf gleicher Seite mergen
            dynamic_merge_threshold = max(1.5, config.GRID_SPACING * 2.5)  # 5m bei 2m Grid
            components = _merge_nearby_components(
                components,
                verts,
                edge_to_faces,
                terrain_face_set,
                merge_threshold=dynamic_merge_threshold,
                centerline_geom=None,
            )

    # Filtere offene Components (die keinen Partner gefunden haben)
    # Nur geschlossene Polygone triangulieren
    closed_components = []
    for comp in components:
        # Prüfe ob Component geschlossen ist (keine Endpunkte)
        adj = defaultdict(set)
        for v1, v2 in comp:
            adj[v1].add(v2)
            adj[v2].add(v1)

        # Zähle Endpunkte (Vertices mit nur 1 Nachbar)
        endpoint_count = sum(1 for neighbors in adj.values() if len(neighbors) == 1)

        if endpoint_count == 0:
            # Geschlossen - behalten
            closed_components.append(comp)

    components = closed_components

    # Baue Polygone aus Components
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

    # Export Boundary-Polygone vor Triangulation (für Debug)

    # if polygons:
    #     _export_boundary_polygons_to_debug(polygons, centerline_point, search_radius=search_radius, sample_rate=1)
    # Trianguliere die Polygone und füge neue Faces hinzu
    _export_search_circle_to_debug(centerline_point, search_radius=search_radius, sample_rate=1)

    new_faces = []
    if polygons:
        new_faces = _triangulate_polygons(polygons, verts, mesh, debug=debug)

    # Aktualisiere optionale Caches sofort, damit der nächste Circle aktuelle Daten sieht
    if new_faces:
        for face_idx in new_faces:
            if cached_face_materials is not None:
                cached_face_materials[face_idx] = mesh.face_props.get(face_idx, {}).get("material")

            if cached_vertex_to_faces is not None:
                face = mesh.faces[face_idx]
                for v in face:
                    cached_vertex_to_faces.setdefault(v, []).append(face_idx)

            if cached_terrain_face_indices is not None:
                mat = mesh.face_props.get(face_idx, {}).get("material")
                if mat == "terrain":
                    cached_terrain_face_indices.add(face_idx)

    return


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


def _find_connected_components(
    boundary_edges, verts, seg_start_arr, seg_vec_arr, seg_mid_arr, centerline_point, edge_to_faces, terrain_face_set
):
    """
    Findet alle zusammenhängenden Komponenten in Edge-Liste.

    Verwendet DFS um alle zusammenhängenden Edge-Gruppen zu finden.
    Edges die die Centerline überqueren werden ignoriert.
    WICHTIG: Edges werden nach Face-Typ getrennt (terrain vs road)!
    Components mit unterschiedlichen Face-Typen werden getrennt gehalten,
    auch wenn sie topologisch zusammenhängen.

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

    # Klassifiziere jede Edge nach Face-Typ (terrain vs road)
    edge_types = {}  # (v1, v2) → "terrain" oder "road"
    terrain_edge_count = 0
    road_edge_count = 0
    for v1, v2 in valid_edges_arr:
        edge_faces = edge_to_faces.get((v1, v2), []) or edge_to_faces.get((v2, v1), [])
        if edge_faces:
            is_terrain = any(f in terrain_face_set for f in edge_faces)
            edge_key = tuple(sorted([v1, v2]))
            edge_types[edge_key] = "terrain" if is_terrain else "road"
            if is_terrain:
                terrain_edge_count += 1
            else:
                road_edge_count += 1

    # Baue Adjacency-List (ungerichtet) - aber nur für Edges des GLEICHEN Typs
    # Jeder Vertex kann mehrere Adjacencies haben, gruppiert nach Edge-Typ
    adj_terrain = defaultdict(list)  # Nur terrain-Edges
    adj_road = defaultdict(list)  # Nur road-Edges

    for v1, v2 in valid_edges_arr:
        edge_key = tuple(sorted([v1, v2]))
        edge_type = edge_types.get(edge_key)

        if edge_type == "terrain":
            adj_terrain[v1].append(v2)
            adj_terrain[v2].append(v1)
        elif edge_type == "road":
            adj_road[v1].append(v2)
            adj_road[v2].append(v1)

    # Finde Connected Components für jeden Typ separat
    components_edges = []

    # Terrain-Components (mit separatem visited_vertices für Terrain!)
    visited_vertices_terrain = set()
    terrain_components_found = 0
    for start_v in adj_terrain.keys():
        if start_v in visited_vertices_terrain:
            continue

        # DFS um alle verbundenen Vertices zu finden (nur über terrain-Edges)
        component_verts = set()
        stack = [start_v]

        while stack:
            v = stack.pop()
            if v in component_verts:
                continue

            component_verts.add(v)
            visited_vertices_terrain.add(v)

            for neighbor in adj_terrain[v]:
                if neighbor not in component_verts:
                    stack.append(neighbor)

        if len(component_verts) >= 3:
            # Extrahiere die zugehörigen terrain-Edges
            comp_edges = []
            for v1, v2 in valid_edges_arr:
                edge_key = tuple(sorted([v1, v2]))
                if v1 in component_verts and v2 in component_verts and edge_types.get(edge_key) == "terrain":
                    comp_edges.append((v1, v2))

            if len(comp_edges) >= 1:
                terrain_components_found += 1
                components_edges.append(comp_edges)

    # Road-Components (mit separatem visited_vertices für Road!)
    visited_vertices_road = set()
    road_components_found = 0
    road_components_checked = 0
    for start_v in adj_road.keys():
        if start_v in visited_vertices_road:
            continue

        # DFS um alle verbundenen Vertices zu finden (nur über road-Edges)
        component_verts = set()
        stack = [start_v]

        while stack:
            v = stack.pop()
            if v in component_verts:
                continue

            component_verts.add(v)
            visited_vertices_road.add(v)

            for neighbor in adj_road[v]:
                if neighbor not in component_verts:
                    stack.append(neighbor)

        if len(component_verts) >= 3:
            # Extrahiere die zugehörigen road-Edges
            comp_edges = []
            for v1, v2 in valid_edges_arr:
                edge_key = tuple(sorted([v1, v2]))
                if v1 in component_verts and v2 in component_verts and edge_types.get(edge_key) == "road":
                    comp_edges.append((v1, v2))

            if len(comp_edges) >= 1:
                road_components_found += 1
                components_edges.append(comp_edges)

    return components_edges


def _merge_nearby_components(
    components_edges, verts, edge_to_faces, terrain_face_set, merge_threshold=2.0, centerline_geom=None
):
    """
    Merged Components die am Kreisrand durch fehlende Faces getrennt sind.

    Findet Endpoints zwischen Components, die nahe beieinander liegen,
    und fügt synthetische Edges hinzu, um sie zu verbinden.

    WICHTIG: Nur Road+Terrain Paare werden gemergt (nicht Road+Road oder Terrain+Terrain)!
    Component-Typ wird basierend auf den angrenzenden Faces bestimmt.

    WICHTIG: Mutual-Nearest-Neighbor Matching verhindert falsche Verbindungen!

    Args:
        components_edges: Liste von Edge-Listen (eine pro Component)
        verts: Vertex-Array
        edge_to_faces: Dict mapping edge (v1, v2) or (v2, v1) → face_list
        terrain_face_set: Set von Terrain-Face-Indices
        merge_threshold: Maximale Distanz zwischen Endpoints für Merge (in Metern)
        centerline_geom: UNUSED (nur für API-Kompatibilität)

    Returns:
        Neue Liste von Components (evtl. gemerged)
    """
    if len(components_edges) < 2:
        return components_edges

    # Klassifiziere Components nach Face-Typ
    # Terrain: Component-Edges grenzen an Terrain-Faces
    # Road: Component-Edges grenzen an Non-Terrain-Faces
    component_types = []

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

    # Finde alle Endpoints pro Component ZUERST (bevor wir Typen klassifizieren)
    component_endpoints = []
    for comp in components_edges:
        eps = find_endpoints(comp)
        component_endpoints.append(eps)

    for comp_idx, comp in enumerate(components_edges):
        # Sammle alle Face-Types für diese Component
        # Ignoriere Edges ohne Faces (synthetische Edges aus Merge)
        face_types = set()
        edges_with_faces = 0
        for v1, v2 in comp:
            # Prüfe beide Richtungen
            edge_faces = edge_to_faces.get((v1, v2), []) or edge_to_faces.get((v2, v1), [])
            if not edge_faces:
                # Synthetische Edge - ignorieren
                continue

            edges_with_faces += 1
            for face_idx in edge_faces:
                if face_idx in terrain_face_set:
                    face_types.add("terrain")
                else:
                    face_types.add("road")

        # Klassifizierung basierend auf vorhandenen Faces
        # Keine Faces (nur synthetische Edges) → ignorieren diese Component
        if edges_with_faces == 0:
            # Nur synthetische Edges - skip diese Component
            component_types.append("synthetic")
        elif face_types == {"terrain"}:
            component_types.append("terrain")
        else:
            component_types.append("road")

    # Finde Merge-Kandidaten: Endpoints zwischen verschiedenen Components
    merge_pairs = []  # Liste von (comp_i_idx, comp_j_idx, v_i, v_j, distance)

    for i in range(len(components_edges)):
        for j in range(i + 1, len(components_edges)):
            # WICHTIG: Nur Road+Terrain Paare mergen, nicht Road+Road oder Terrain+Terrain!
            type_i = component_types[i]
            type_j = component_types[j]
            if type_i == type_j:  # Beide gleicher Typ - skip
                continue

            eps_i = component_endpoints[i]
            eps_j = component_endpoints[j]

            for ep_i in eps_i:
                for ep_j in eps_j:
                    pi = verts[ep_i]
                    pj = verts[ep_j]
                    dist = np.linalg.norm(np.array(pi[:2]) - np.array(pj[:2]))

                    if dist <= merge_threshold:
                        merge_pairs.append((i, j, ep_i, ep_j, dist))

    if not merge_pairs:
        return components_edges

    # MUTUAL-NEAREST-NEIGHBOR MATCHING
    # Nur verbinden wenn beide Components sich gegenseitig als nächste wählen

    # Schritt 1: Finde für jede Component die nächste Component des anderen Typs
    nearest_neighbor = {}  # comp_idx → (nearest_comp_idx, ep_i, ep_j, dist)

    for i in range(len(components_edges)):
        best_dist = float("inf")
        best_match = None

        # Suche die nächste Component des anderen Typs in merge_pairs
        for ci, cj, ep_i, ep_j, dist in merge_pairs:
            if ci == i:
                # i ist erste Component, ep_i gehört zu i
                other_comp = cj
                if dist < best_dist:
                    best_dist = dist
                    best_match = (other_comp, ep_i, ep_j, dist)
            elif cj == i:
                # i ist zweite Component, ep_j gehört zu i - tausche Endpunkte!
                other_comp = ci
                if dist < best_dist:
                    best_dist = dist
                    best_match = (other_comp, ep_j, ep_i, dist)  # Getauscht!

        if best_match:
            nearest_neighbor[i] = best_match

    # Schritt 2: Finde mutual nearest neighbors
    mutual_pairs = []
    processed = set()

    for i, (j, ep_i, ep_j, dist_ij) in nearest_neighbor.items():
        if i in processed or j in processed:
            continue

        # Prüfe ob j auch i als nächsten hat
        if j in nearest_neighbor:
            j_nearest, j_ep_i, j_ep_j, dist_ji = nearest_neighbor[j]
            if j_nearest == i:
                # Mutual match!
                mutual_pairs.append((i, j, ep_i, ep_j, dist_ij))
                processed.add(i)
                processed.add(j)

    # Union-Find: Merge nur die mutual pairs
    parent = list(range(len(components_edges)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    # Merge Components
    synthetic_edges_with_comps = []
    for i, j, ep_i, ep_j, dist in mutual_pairs:
        # WICHTIG: Prüfe ob die Endpunkte bereits in der gleichen Adjacency sind!
        # Das passiert bei überschneidenden Suchkreisen - dann ist bereits eine Kante vorhanden

        eps_i = component_endpoints[i]
        eps_j = component_endpoints[j]

        # Baue Adjacency für beide Components um zu prüfen ob bereits verbunden
        adj_i = defaultdict(set)
        for v1, v2 in components_edges[i]:
            adj_i[v1].add(v2)
            adj_i[v2].add(v1)

        adj_j = defaultdict(set)
        for v1, v2 in components_edges[j]:
            adj_j[v1].add(v2)
            adj_j[v2].add(v1)

        # Prüfe: Sind ep_i und ep_j bereits direkt verbunden?
        already_connected = ep_j in adj_i.get(ep_i, set()) or ep_i in adj_j.get(ep_j, set())

        if union(i, j):
            # Finde die anderen Endpunkte
            other_ep_i = [e for e in eps_i if e != ep_i][0] if len(eps_i) > 1 else None
            other_ep_j = [e for e in eps_j if e != ep_j][0] if len(eps_j) > 1 else None

            if not already_connected and ep_i != ep_j:
                # ep_i und ep_j sind NICHT verbunden UND verschieden - verbinde sie
                synthetic_edges_with_comps.append(((ep_i, ep_j), i, j))

            # Verbinde auch die anderen Endpunkte (falls vorhanden)
            if other_ep_i is not None and other_ep_j is not None and other_ep_i != other_ep_j:
                synthetic_edges_with_comps.append(((other_ep_i, other_ep_j), i, j))

    # Baue neue Component-Liste (Union-Find Ergebnis)
    merged_components = defaultdict(list)
    for idx, comp_edges in enumerate(components_edges):
        root = find(idx)
        merged_components[root].extend(comp_edges)

    # Füge synthetische Edges hinzu
    for edge, comp_i, comp_j in synthetic_edges_with_comps:
        root = find(comp_i)
        merged_components[root].append(edge)

    return list(merged_components.values())


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
    - Schließt automatisch den Loop wenn möglich

    Args:
        edges: Liste von (v1, v2) Tuples (können orientiert oder unorientiert sein)

    Returns:
        Liste von Vertex-Indices in Reihenfolge (geschlossen wenn möglich)
    """
    if not edges:
        return []

    # Baue Adjacency-List (ungerichtet) - WICHTIG: Set statt List für Deduplication!
    adj = defaultdict(set)

    for v1, v2 in edges:
        adj[v1].add(v2)
        adj[v2].add(v1)

    # Finde Start-Vertex: Vertex mit nur 1 Nachbar (Endpunkt), sonst beliebig
    start_v = None
    for v, neighbors in adj.items():
        if len(neighbors) == 1:
            start_v = v
            break

    if start_v is None:
        start_v = list(adj.keys())[0] if adj else edges[0][0]

    # Baue Pfad mit Edge-Tracking
    path = [start_v]
    used_edges = set()
    current = start_v

    while True:
        # Finde unbenutzte ausgehende Edges
        available_neighbors = []
        for neighbor in adj[current]:
            # Normalisiere Edge für Vergleich (ungerichtet)
            edge = tuple(sorted([current, neighbor]))

            if edge not in used_edges:
                available_neighbors.append(neighbor)

        if not available_neighbors:
            break

        # Wähle nächsten Nachbarn (Greedy: ersten verfügbaren)
        next_v = available_neighbors[0]

        # Markiere Edge als benutzt (normalisiert)
        edge = tuple(sorted([current, next_v]))
        used_edges.add(edge)

        path.append(next_v)
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

    HINWEIS: UVs werden NICHT hier berechnet (zu langsam pro Face!),
    sondern am Ende per mesh.compute_terrain_uvs_batch() für alle Terrain+Stitch-Faces.
    """
    if not polygons:
        return []

    new_faces = []

    for poly_idx, poly in enumerate(polygons):
        polygon_vertices = poly.get("vertices", [])

        if len(polygon_vertices) < 3:
            continue

        # Extrahiere 2D-Koordinaten (x, y) für Earcut
        coords_3d = np.array([verts[v] for v in polygon_vertices])
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

                # KEINE UV-Berechnung hier (Performance!)
                # UVs werden später per mesh.compute_terrain_uvs_batch() gesetzt
                face_idx = mesh.add_face(v0_global, v1_global, v2_global, material="terrain")
                new_faces.append(face_idx)

            if debug:
                print(f"  [Triangulation] Polygon {poly_idx}: {len(indices) // 3} Dreiecke erzeugt (earcut)")

        except Exception as e:
            if debug:
                print(f"  [!] Triangulation Polygon {poly_idx} fehlgeschlagen: {e}")
            continue

    return new_faces


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
        color=[0.0, 0.0, 1.0],  # Blau
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
        sample_rate: Nur jeder N-te Sample wird exportiert (default: 100, sample_rate=1 exportiert alle)
    """
    # Nutze Hash der Centerline-Koordinaten als Pseudo-Counter
    cx, cy, cz = centerline_point
    coord_hash = hash((round(cx, 2), round(cy, 2)))

    # Nur jeden N-ten Sample exportieren (wenn sample_rate > 1)
    if sample_rate > 1 and coord_hash % sample_rate != 0:
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
                    "color": [1.0, 0.0, 0.0],  # Magenta
                    "terrain_count": int(terrain_count),
                    "slope_count": int(slope_count),
                }
            )

    # Suchkreis wird separat über _export_search_circle_to_debug() exportiert (gelb)
    # NICHT hier (verhindert Duplikation)


def _export_component_lines_to_debug(
    components_edges, verts, centerline_point, edge_to_faces, terrain_face_set, sample_rate=100
):
    """
    Exportiert Connected Component Linien zum Debug-Exporter (Singleton).

    Klassifizierung basiert auf angrenzenden Faces:
    - Grün: Terrain-Edges (grenzen an Terrain-Faces)
    - Rot: Road-Edges (grenzen an Non-Terrain-Faces)
    - Synthetische Edges (ohne Faces) werden ignoriert

    Args:
        components_edges: Liste von Edge-Listen (eine pro Component)
        verts: Vertex-Array
        centerline_point: (x, y, z) - Centerline-Sample-Punkt (für Zähler)
        edge_to_faces: Dict mapping edge (v1, v2) or (v2, v1) → face_list
        terrain_face_set: Set von Terrain-Face-Indices
        sample_rate: Nur jeder N-te Sample wird exportiert (default: 100)
    """
    cx, cy, cz = centerline_point
    coord_hash = hash((round(cx, 2), round(cy, 2)))

    if coord_hash % sample_rate != 0:
        return

    exporter = DebugNetworkExporter.get_instance()

    # Für jede Component: Zeichne Terrain-Edges und Road-Edges separat
    for comp_idx, comp_edges in enumerate(components_edges):
        terrain_edges = []
        road_edges = []

        for v1, v2 in comp_edges:
            edge_faces = edge_to_faces.get((v1, v2), []) or edge_to_faces.get((v2, v1), [])
            if not edge_faces:
                # Synthetische Edge - ignorieren
                continue

            # Klassifiziere diese Edge nach ihrem Face-Typ
            is_terrain = any(f in terrain_face_set for f in edge_faces)

            if is_terrain:
                terrain_edges.append((v1, v2))
            else:
                road_edges.append((v1, v2))

        # Exportiere Terrain-Edges (grün)
        if terrain_edges:
            ordered_path = _build_ordered_path_from_edges(terrain_edges)
            if len(ordered_path) >= 2:
                coords = [tuple(verts[v]) for v in ordered_path]
                exporter.add_component_line(
                    coords=coords,
                    color=[0.2, 0.8, 0.2],  # Grün
                    label=f"component_terrain_{comp_idx}",
                    line_width=3.0,
                )

        # Exportiere Road-Edges (rot)
        if road_edges:
            ordered_path = _build_ordered_path_from_edges(road_edges)
            if len(ordered_path) >= 2:
                coords = [tuple(verts[v]) for v in ordered_path]
                exporter.add_component_line(
                    coords=coords,
                    color=[0.8, 0.2, 0.2],  # Rot
                    label=f"component_road_{comp_idx}",
                    line_width=3.0,
                )


# UV-Berechnung wurde komplett entfernt - erfolgt jetzt per mesh.compute_terrain_uvs_batch()
# am Ende der Mesh-Erstellung für maximale Performance (vektorisiert über alle Faces).
