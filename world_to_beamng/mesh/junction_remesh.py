"""
Lokales Remeshing für Junction-Punkte.

Algorithmus:
1. Definiere Suchkreis (Radius) um Junction-Point
2. Sammle alle Vertices und Faces im Radius
3. Projiziere auf XY-Ebene und wende unary_union an
4. Trianguliere neu mit exakten Boundary-Constraints
5. Rekonstruiere Z-Werte durch Ebenen-Interpolation
6. Stitche mit Umgebungs-Mesh
"""

import numpy as np
from scipy.spatial import cKDTree, Delaunay
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union, triangulate

from .. import config


def collect_nearby_geometry(junction_pos, vertices, faces, radius=15.0):
    """
    Sammelt alle Vertices und Faces im Suchradius um einen Junction-Point.
    WICHTIG: 'faces' sollte schon gefiltert sein (nur Straßen-Faces!)

    Args:
        junction_pos: (x, y, z) Position des Junction-Points
        vertices: Nx3 Array aller Mesh-Vertices
        faces: Mx3 Array aller Face-Indizes (sollte nur Straßen-Faces sein!)
        radius: Suchradius in Metern (default 10m)

    Returns:
        Dict mit:
        - nearby_vertex_indices: Indizes der Vertices im Radius
        - nearby_face_indices: Indizes der Faces im Radius
        - nearby_vertices_3d: Die 3D-Koordinaten der Vertices
        - boundary_edge_indices: Kanten am Rand des Suchradius
        - center_point: Junction-Position (x, y)
    """
    import time

    t_start = time.time()

    center_xy = np.array(junction_pos[:2])  # Nur X, Y

    # OPTIMIERUNG: Reduziere Suchbereich - nicht jede Vertex einzeln, sondern:
    # Finde FACE-Centroids im Radius, dann nimm alle Vertices dieser Faces
    # Das ist viel schneller als KDTree auf allen Vertices!

    # Berechne Centroids aller Faces
    face_centroids = vertices[faces, :2].mean(axis=1)  # Nx2 array of (x,y) centroids

    # KDTree EINMAL auf Face-Centroids (nicht auf Vertices!)
    centroid_tree = cKDTree(face_centroids)
    candidate_face_indices = centroid_tree.query_ball_point(center_xy, radius)

    # Sammle ALLE Vertices aus diesen Faces
    nearby_vertex_set = set()
    for face_idx in candidate_face_indices:
        for v_idx in faces[face_idx]:
            nearby_vertex_set.add(v_idx)

    nearby_vertex_indices = np.array(sorted(nearby_vertex_set), dtype=np.int32)

    if len(nearby_vertex_indices) == 0:
        print(f"    [Remesh] FEHLER: Keine Vertices gefunden!")
        return None

    # Nearby-Face-Indizes sind einfach die Kandidaten
    nearby_face_indices = np.array(candidate_face_indices, dtype=np.int32)

    # 3. Identifiziere Boundary-Edges (Kanten die nur teilweise im Radius sind)
    boundary_edges = []
    for face_idx in nearby_face_indices:
        face = faces[face_idx]
        for i in range(3):
            v1_idx = face[i]
            v2_idx = face[(i + 1) % 3]

            v1_in = v1_idx in nearby_vertex_set
            v2_in = v2_idx in nearby_vertex_set

            # Boundary: eine Vertex im Radius, eine außerhalb
            if v1_in != v2_in:
                boundary_edges.append((v1_idx, v2_idx))

    return {
        "center_point": center_xy,
        "junction_pos_3d": junction_pos,
        "nearby_vertex_indices": nearby_vertex_indices,
        "nearby_face_indices": nearby_face_indices,
        "nearby_vertices_3d": vertices[nearby_vertex_indices],
        "boundary_edges": boundary_edges,
        "all_nearby_vertex_indices": nearby_vertex_indices,
        "radius": radius,
    }


def analyze_junction_geometry(geometry_data, vertices, faces):
    """
    Analysiert die Geometrie um einen Junction herum (für Debug/Visualisierung).

    Zeigt:
    - Wie viele Straßen-Meshes beteiligt sind
    - Größe der betroffenen Fläche
    - Z-Variation in der Region
    """
    if geometry_data is None:
        return None

    nearby_v_indices = geometry_data["nearby_vertex_indices"]
    nearby_f_indices = geometry_data["nearby_face_indices"]

    # Z-Statistik
    z_values = vertices[nearby_v_indices, 2]
    z_min, z_max, z_mean = z_values.min(), z_values.max(), z_values.mean()
    z_range = z_max - z_min

    # Flächenstatistik (2D Projection)
    v_2d = vertices[nearby_v_indices, :2]
    x_range = v_2d[:, 0].max() - v_2d[:, 0].min()
    y_range = v_2d[:, 1].max() - v_2d[:, 1].min()
    area_bounding_box = x_range * y_range

    return {
        "z_min": z_min,
        "z_max": z_max,
        "z_mean": z_mean,
        "z_range": z_range,
        "bbox_area": area_bounding_box,
        "face_count": len(nearby_f_indices),
    }


def prepare_remesh_data(geometry_data, vertices, faces):
    """
    Bereitet die Daten für 2D-Projektion und Remeshing vor.

    Returns:
        Dict mit Polygonen und Z-Informationen für nächsten Schritt
    """
    if geometry_data is None:
        return None

    nearby_v_indices = geometry_data["nearby_vertex_indices"]
    nearby_f_indices = geometry_data["nearby_face_indices"]
    center_point = geometry_data["center_point"]

    # 2D-Projektion: Alle Faces als Shapely-Polygone
    all_polygons_2d = []
    all_face_indices = []  # Speichere Original-Face-Indizes

    for f_idx in nearby_f_indices:
        face = faces[f_idx]
        face_vertices = vertices[face]

        # 2D-Koordinaten
        xy_coords = face_vertices[:, :2]

        try:
            polygon = Polygon(xy_coords)
            if polygon.is_valid and polygon.area > 1e-6:  # Nur gültige Polygone
                all_polygons_2d.append(polygon)
                all_face_indices.append(f_idx)  # Speichere den Index!
        except Exception as e:
            pass  # Ungültiges Polygon ignorieren

    if len(all_polygons_2d) == 0:
        print(f"    [Remesh] FEHLER: Keine gültigen Polygone gefunden!")
        return None

    # OPTIMIZATION: Skip overlap filtering entirely!
    # unary_union() below will handle all polygon merging automatically
    # This eliminates the O(n²) or even O(n log n) check - unary_union is efficient
    polygons_2d = all_polygons_2d

    # Boundary-Edges als Constraints (für später)
    boundary_edges_2d = []
    for v1_idx, v2_idx in geometry_data["boundary_edges"]:
        p1 = vertices[v1_idx, :2]
        p2 = vertices[v2_idx, :2]
        boundary_edges_2d.append((p1, p2))

    return {
        "polygons_2d": polygons_2d,
        "boundary_edges_2d": boundary_edges_2d,
        "center_point": center_point,
        "radius": geometry_data["radius"],
        "original_data": geometry_data,
        "original_vertices_3d": vertices[nearby_v_indices],  # Speichere Original-Vertices
    }


def merge_and_triangulate(remesh_data):
    """
    Führt Shapely unary_union durch und trianguliert die resultierende Geometrie.

    Args:
        remesh_data: Dict aus prepare_remesh_data()

    Returns:
        Dict mit:
        - merged_polygon: Das Resultat der unary_union
        - new_vertices_2d: Triangulierte Vertices
        - new_faces: Triangulierte Face-Indizes
        - boundary_polygon: Polygon für Boundary-Constraints
    """
    if remesh_data is None or len(remesh_data["polygons_2d"]) == 0:
        print(f"    [Remesh] FEHLER: Keine Polygone für Union!")
        return None

    polygons_2d = remesh_data["polygons_2d"]

    # 1. Shapely unary_union für alle Polygone
    merged = unary_union(polygons_2d)

    # merged kann jetzt Polygon oder MultiPolygon sein
    if isinstance(merged, MultiPolygon):
        # Nimm den größten Part
        largest = max(merged.geoms, key=lambda p: p.area)
        merged_polygon = largest
    else:
        merged_polygon = merged

    # Kleiner Puffer, damit vereinfachte Union-Kanten Randdreiecke nicht abschneiden
    merged_polygon = merged_polygon.buffer(0.001)

    # FILTER: Welche der ursprünglichen Faces überlappen wirklich mit dem Union-Polygon?
    # Das ist der echte Junction-Bereich (7-14m), nicht die ganzen 18m!
    overlapping_polygons = []
    overlapping_count = 0

    for i, poly in enumerate(polygons_2d):
        if poly.intersects(merged_polygon):
            overlapping_polygons.append(poly)
            overlapping_count += 1

    # Nutze nur die überlappenden Polygone - und mache NOCHMAL unary_union
    # Das eliminiert Rauschen von weit entfernten Faces!
    if overlapping_count > 0 and overlapping_count < len(polygons_2d):
        merged_filtered = unary_union(overlapping_polygons)
        if isinstance(merged_filtered, MultiPolygon):
            merged_polygon = max(merged_filtered.geoms, key=lambda p: p.area)
        else:
            merged_polygon = merged_filtered

    # 2. Extrahiere die äußere Boundary - und snappe sie zu Original-Vertices
    boundary_coords_raw = np.array(merged_polygon.exterior.coords[:-1])  # Letzter Punkt = erster

    # Snappe die Boundary-Coords zu den nächsten Nearby-Vertices (innerhalb 1m)
    original_data = remesh_data["original_data"]
    nearby_v_indices = original_data["nearby_vertex_indices"]
    nearby_vertices_xy = remesh_data["original_vertices_3d"][:, :2]

    from scipy.spatial import cKDTree

    nearby_tree = cKDTree(nearby_vertices_xy)

    boundary_coords = []
    boundary_vertex_global_indices = []
    for bc_raw in boundary_coords_raw:
        dist, idx_in_nearby = nearby_tree.query(bc_raw, k=1)
        if dist < 1.0:  # 1m - großzügiger Threshold für Shapely-vereinfachte Coords
            # Snape zur nächsten Nearby-Vertex
            bc_snapped = nearby_vertices_xy[idx_in_nearby]
            global_idx = nearby_v_indices[idx_in_nearby]
            boundary_coords.append(bc_snapped)
            boundary_vertex_global_indices.append(global_idx)
        else:
            # Kein Match - nutze raw Coord (sollte selten passieren)
            boundary_coords.append(bc_raw)

    boundary_coords = np.array(boundary_coords, dtype=np.float32)

    # 3. Trianguliere das Polygon robust mit shapely.triangulate (respektiert Concavity)
    try:
        tri_polys = triangulate(merged_polygon)

        vertices = []
        vertex_map = {}
        faces = []

        for tri_poly in tri_polys:
            if not tri_poly.is_valid or tri_poly.area <= 1e-9:
                continue
            # Sicherstellen, dass das Dreieck vollständig im Polygon liegt
            if not merged_polygon.covers(tri_poly):
                continue

            coords = np.array(tri_poly.exterior.coords[:-1])
            face_idx = []
            for xy in coords:
                key = (float(xy[0]), float(xy[1]))
                idx = vertex_map.get(key)
                if idx is None:
                    idx = len(vertices)
                    vertices.append([xy[0], xy[1]])
                    vertex_map[key] = idx
                face_idx.append(idx)
            if len(face_idx) == 3:
                faces.append(face_idx)

        new_vertices_2d = np.array(vertices, dtype=np.float32)
        new_faces = np.array(faces, dtype=np.int32)

        # SNAP: Passe Triangulations-Vertices an Original-Nearby-Vertices an
        # Dafür baue einen KDTree mit allen nearby-Vertices
        original_data = remesh_data["original_data"]
        all_nearby_v_indices = original_data.get("all_nearby_vertex_indices", np.array([]))

        # Dictionary für gesnappte Vertices (Index → Original Z-Wert)
        snapped_vertex_z_values = {}

        if len(all_nearby_v_indices) > 0:
            # Wir brauchen die echten Vertices - hole aus remesh_data
            nearby_xyz_from_data = remesh_data["original_vertices_3d"]  # Das ist vertices[nearby_v_indices]
            nearby_xy = nearby_xyz_from_data[:, :2]
            nearby_z = nearby_xyz_from_data[:, 2]
            nearby_tree = cKDTree(nearby_xy)

            # Für jede neue Triangulations-Vertex: snap an nähere Original-Vertex (5cm)
            snap_dist_threshold = 0.05

            for i, tri_v in enumerate(new_vertices_2d):
                dist, idx = nearby_tree.query(tri_v, k=1)
                if dist < snap_dist_threshold:
                    # Snap zur Original-Vertex (nur XY!)
                    # Z-Wert wird später durch reconstruct_z_values berechnet
                    new_vertices_2d[i] = nearby_xy[idx]
                    # NICHT den alten Z-Wert übernehmen - der ist falsch!
                    # Stattdessen markieren wir diese Vertex als "gesnapped" für XY
                    snapped_vertex_z_values[i] = None  # Markierung: Z wird später berechnet
    except Exception as e:
        print(f"    [Remesh] FEHLER bei shapely.triangulate: {e}")
        return None

    # Übergebe die gesnappten Z-Werte an reconstruct_z_values
    return {
        "merged_polygon": merged_polygon,
        "boundary_coords": boundary_coords,
        "boundary_vertex_global_indices": boundary_vertex_global_indices,  # NEU: Original-Indizes der Boundary-Vertices
        "new_vertices_2d": new_vertices_2d,
        "new_faces": new_faces,
        "remesh_data": remesh_data,
        "snapped_vertex_z_values": snapped_vertex_z_values,  # WICHTIG: Übergebe die gesnappten Z-Werte
    }


def reconstruct_z_values(triangulation_data, original_vertices, original_faces=None):
    """
    Rekonstruiert Z-Werte für die neuen Vertices der Triangulation.

    Methode:
    1. Nutze gesnappte Z-Werte wo möglich
    2. Nutze Ebenen-Interpolation von Original-Faces
    3. Fallback: IDW-Interpolation von Vertices

    Args:
        triangulation_data: Dict aus merge_and_triangulate()
        original_vertices: Alle Original-Mesh-Vertices (Nx3)
        original_faces: Alle Original-Mesh-Faces (optional, für Ebenen-Interpolation)

    Returns:
        Dict mit neuen 3D-Vertices für die Triangulation
    """
    if triangulation_data is None:
        return None

    new_vertices_2d = triangulation_data["new_vertices_2d"]
    boundary_coords = triangulation_data["boundary_coords"]
    remesh_data = triangulation_data["remesh_data"]

    # Hole Z-Werte aus den Original-Vertices im Radius
    original_data = remesh_data["original_data"]
    nearby_v_indices = original_data["nearby_vertex_indices"]
    original_z_values = original_vertices[nearby_v_indices, 2]
    z_mean = np.mean(original_z_values)
    z_min = np.min(original_z_values)
    z_max = np.max(original_z_values)

    # Erstelle XY→Z Dictionary für exaktes Matching mit ALLEN nearby-Vertices
    all_nearby_v_indices = original_data.get("all_nearby_vertex_indices", np.array([]))
    all_nearby_xy_to_z = {}
    if len(all_nearby_v_indices) > 0:
        all_nearby_vertices_3d = original_vertices[all_nearby_v_indices]
        # Erstelle Dictionary: (x, y) → z (auf mm genau gerundet für Matching)
        for i, v in enumerate(all_nearby_vertices_3d):
            # Runde auf 0.001m (1mm) für robustes Matching
            key = (round(v[0], 3), round(v[1], 3))
            all_nearby_xy_to_z[key] = v[2]  # Exakter Z-Wert

    # Zusätzlich: KDTree für räumliche Nähe als Fallback
    if len(all_nearby_v_indices) > 0:
        nearby_xy = all_nearby_vertices_3d[:, :2]
        nearby_z = all_nearby_vertices_3d[:, 2]
        nearby_tree = cKDTree(nearby_xy)
    else:
        nearby_tree = None

    # KDTree auf Original-Vertices für Z-Interpolation
    original_xy = original_vertices[nearby_v_indices, :2]
    z_tree = cKDTree(original_xy)

    # Für jeden neuen Vertex: Finde nächste Original-Vertices und interpoliere Z
    new_vertices_3d = []
    snap_threshold = 0.15  # 15cm Snapping-Toleranz für räumliche Nähe

    # Hole gesnappte Z-Werte aus triangulation_data falls vorhanden
    snapped_vertex_z_values = triangulation_data.get("snapped_vertex_z_values", {})

    for i, vertex_2d in enumerate(new_vertices_2d):
        z_value = None

        # 0. PRIORITÄT: Nutze bereits gesnappte Z-Werte (aus der Triangulation)
        # ABER: Wenn der Wert None ist, dann wurde XY gesnapped aber Z soll interpoliert werden!
        if i in snapped_vertex_z_values:
            z_value = snapped_vertex_z_values[i]
            # Wenn z_value == None, dann fällt es durch zur Interpolation

        # 1. Prüfe EXAKTES XY-Matching mit Nearby-Vertices (auf mm genau)
        if z_value is None and all_nearby_xy_to_z:
            key = (round(vertex_2d[0], 3), round(vertex_2d[1], 3))
            if key in all_nearby_xy_to_z:
                z_value = all_nearby_xy_to_z[key]

        # 2. Prüfe räumliche Nähe zu Nearby-Vertices (Fallback)
        if z_value is None and nearby_tree is not None:
            dist, idx = nearby_tree.query(vertex_2d, k=1)
            if dist < snap_threshold:
                # Snap an nächsten Nearby-Vertex
                z_value = nearby_z[idx]

        # 3. Fallback: Interpolation aus nächsten Vertices
        if z_value is None:
            # Nutze mehr Vertices für bessere Interpolation (nicht nur 3)
            k_neighbors = min(8, len(original_xy))  # Nutze 8 nächste Vertices statt 3
            distances, indices = z_tree.query(vertex_2d, k=k_neighbors)

            if np.min(distances) < 1e-6:  # Vertex ist sehr nah an Original
                original_idx = indices if isinstance(indices, np.integer) else indices[0]
                z_value = original_z_values[original_idx]
            else:
                # IDW-Interpolation mit stärkerer Gewichtung auf nähere Vertices
                if isinstance(distances, np.ndarray):
                    # Nutze stärkere inverse distance weighting (power 2 statt 1)
                    weights = 1.0 / np.power(distances + 1e-8, 2)
                else:
                    weights = np.array([1.0 / np.power(distances + 1e-8, 2)])
                weights /= np.sum(weights)

                if isinstance(indices, np.ndarray):
                    z_values = original_z_values[indices]
                else:
                    z_values = original_z_values[[indices]]
                z_value = np.sum(weights * z_values)

        new_vertices_3d.append([vertex_2d[0], vertex_2d[1], z_value])

    new_vertices_3d = np.array(new_vertices_3d, dtype=np.float32)
    nearby_vertices = original_vertices[nearby_v_indices]

    # Übernimm Z-Werte von Nearby-Vertices, die nahe XY-Koordinaten haben
    nearby_vertices_xy = nearby_vertices[:, :2]  # Nur XY
    nearby_tree_xy = cKDTree(nearby_vertices_xy)

    for i in range(len(new_vertices_3d)):
        new_xy = new_vertices_3d[i, :2]

        # Suche nach Nearby-Vertex mit naher XY-Koordinate (< 5cm Abstand)
        # Das ist der gleiche Threshold wie beim XY-Snapping
        dist, idx_in_nearby = nearby_tree_xy.query(new_xy, k=1)
        if dist < 0.05:  # 5cm = Snapping-Threshold
            # Übernimm die Z-Koordinate des Nearby-Vertex
            nearby_z = nearby_vertices[idx_in_nearby, 2]
            old_z = new_vertices_3d[i, 2]

            if abs(old_z - nearby_z) > 0.0001:  # Nur wenn Unterschied > 0.1mm
                new_vertices_3d[i, 2] = nearby_z

    # Interpoliere auch Z-Werte für Boundary-Coords (für Debug-Export)
    boundary_z_values = []
    for boundary_pt in boundary_coords:
        distances, indices = z_tree.query(boundary_pt, k=min(3, len(original_xy)))
        if np.min(distances) < 1e-6:
            original_idx = indices if isinstance(indices, np.integer) else indices[0]
            z_value = original_z_values[original_idx]
        else:
            if isinstance(distances, np.ndarray):
                weights = 1.0 / (distances + 1e-8)
            else:
                weights = np.array([1.0 / (distances + 1e-8)])
            weights /= np.sum(weights)
            if isinstance(indices, np.ndarray):
                z_vals = original_z_values[indices]
            else:
                z_vals = original_z_values[[indices]]
            z_value = np.sum(weights * z_vals)
        boundary_z_values.append(z_value)

    boundary_z_values = np.array(boundary_z_values, dtype=np.float32)

    return {
        "new_vertices_3d": new_vertices_3d,
        "new_faces": triangulation_data["new_faces"],
        "triangulation_data": triangulation_data,
        "z_mean": float(z_mean),
        "boundary_z_values": boundary_z_values,
    }
