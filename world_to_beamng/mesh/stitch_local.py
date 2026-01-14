"""
Lokales Stitching: Loch-Suche entlang Search-Circles um Centerlines.

Statt globaler Boundary-Suche: Pro Centerline-Sample-Punkt einen Search-Circle
anlegen und nur dort nach Loch-Polygonen suchen.
"""

import numpy as np
from scipy.spatial import cKDTree
from mapbox_earcut import triangulate_float64
from collections import defaultdict


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
            mat = face_materials[face_idx] if face_idx < len(face_materials) else None
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
            f1_mat = face_materials[f1] if f1 < len(face_materials) else None
            f2_mat = face_materials[f2] if f2 < len(face_materials) else None
            f1_is_terrain = f1_mat == "terrain"
            f2_is_terrain = f2_mat == "terrain"
            if f1_is_terrain != f2_is_terrain:
                boundary_edges_mixed += 1

    # Skip, wenn keine offenen Boundary-Edges vorhanden sind
    if len(boundary_edges_single_edges) < 3:
        return []

    # Komponenten nur aus offenen Rändern bilden (keine Material-Wechsel-Kanten)
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
    # Skaliere merge_threshold proportional zum search_radius
    # search_radius = road_width + GRID_SPACING*2.5 (dynamisch), merge_threshold ~ 1/3 davon
    dynamic_merge_threshold = max(1.5, search_radius / 3.0)  # Heuristic: 1/3 des search_radius
    if len(components) > 2:
        components = _merge_nearby_components(components, verts, merge_threshold=dynamic_merge_threshold)

    # Baue Polygone aus Components
    terrain_face_set = (
        cached_terrain_face_indices
        if cached_terrain_face_indices is not None
        else {idx for idx, mat in enumerate(face_materials) if mat == "terrain"}
    )
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

    # Trianguliere die Polygone und füge neue Faces hinzu
    if polygons:
        _triangulate_polygons(polygons, verts, mesh, debug=debug)

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

    # Finde Connected Components (Vertex-basiert)
    visited_vertices = set()
    components_vertices = []

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

        if len(component_verts) >= 3:
            components_vertices.append(component_verts)

    # Für jede Komponente: Extrahiere die zugehörigen Edges
    components_edges = []
    for comp_verts in components_vertices:
        comp_edges = []
        for v1, v2 in valid_edges_arr:
            if v1 in comp_verts and v2 in comp_verts:
                comp_edges.append((v1, v2))

        if len(comp_edges) >= 3:
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

    # Finde alle Endpoints pro Component
    component_endpoints = []
    for comp in components_edges:
        eps = find_endpoints(comp)
        component_endpoints.append(eps)

    # Finde Merge-Kandidaten: Endpoints zwischen verschiedenen Components
    merge_pairs = []  # Liste von (comp_i_idx, comp_j_idx, v_i, v_j, distance)

    for i in range(len(components_edges)):
        for j in range(i + 1, len(components_edges)):
            eps_i = component_endpoints[i]
            eps_j = component_endpoints[j]

            for ep_i in eps_i:
                for ep_j in eps_j:
                    pi = verts[ep_i]
                    pj = verts[ep_j]
                    dist = np.linalg.norm(np.array(pi) - np.array(pj))

                    if dist <= merge_threshold:
                        merge_pairs.append((i, j, ep_i, ep_j, dist))

    if not merge_pairs:
        return components_edges

    # Sortiere nach Distanz (merge closest first)
    merge_pairs.sort(key=lambda x: x[4])

    # Union-Find Struktur zum Tracken welche Components gemerged wurden
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

    # Merge Components und füge synthetische Edges hinzu
    synthetic_edges = []
    for i, j, ep_i, ep_j, dist in merge_pairs:
        if union(i, j):
            synthetic_edges.append((ep_i, ep_j))

    # Baue neue Component-Liste
    merged_components = defaultdict(list)
    for idx, comp_edges in enumerate(components_edges):
        root = find(idx)
        merged_components[root].extend(comp_edges)

    # Füge synthetische Edges hinzu
    for edge in synthetic_edges:
        # Finde welcher Component diese Edge gehört
        v1, v2 = edge
        # Finde Components die v1 oder v2 enthalten
        for root, edges in merged_components.items():
            # Prüfe ob v1 oder v2 in diesen Edges vorkommen
            vertices_in_comp = set()
            for e_v1, e_v2 in edges:
                vertices_in_comp.add(e_v1)
                vertices_in_comp.add(e_v2)

            if v1 in vertices_in_comp or v2 in vertices_in_comp:
                edges.append(edge)
                break

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


# UV-Berechnung wurde komplett entfernt - erfolgt jetzt per mesh.compute_terrain_uvs_batch()
# am Ende der Mesh-Erstellung für maximale Performance (vektorisiert über alle Faces).
