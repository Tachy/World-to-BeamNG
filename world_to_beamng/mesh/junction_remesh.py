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
from shapely.ops import unary_union


def collect_nearby_geometry(
    junction_pos, vertices, faces, radius=10.0
):
    """
    Sammelt alle Vertices und Faces im Suchradius um einen Junction-Point.
    
    Args:
        junction_pos: (x, y, z) Position des Junction-Points
        vertices: Nx3 Array aller Mesh-Vertices
        faces: Mx3 Array aller Face-Indizes
        radius: Suchradius in Metern (default 10m)
    
    Returns:
        Dict mit:
        - nearby_vertex_indices: Indizes der Vertices im Radius
        - nearby_face_indices: Indizes der Faces im Radius
        - nearby_vertices_3d: Die 3D-Koordinaten der Vertices
        - boundary_edge_indices: Kanten am Rand des Suchradius
        - center_point: Junction-Position (x, y)
    """
    
    center_xy = np.array(junction_pos[:2])  # Nur X, Y
    
    # 1. KDTree: Finde alle Vertices im Radius (XY-Projektion)
    vertices_xy = vertices[:, :2]
    tree = cKDTree(vertices_xy)
    nearby_vertex_indices = tree.query_ball_point(center_xy, radius)
    nearby_vertex_indices = np.array(nearby_vertex_indices)
    
    print(f"  [Remesh] Junction: {len(nearby_vertex_indices)} Vertices im Radius {radius}m")
    
    if len(nearby_vertex_indices) == 0:
        print(f"    [Remesh] FEHLER: Keine Vertices gefunden! Radius zu klein?")
        return None
    
    # 2. Finde alle Faces, die mindestens einen Vertex im Radius haben
    nearby_vertex_set = set(nearby_vertex_indices)
    nearby_face_indices = []
    
    for face_idx, face in enumerate(faces):
        if any(v_idx in nearby_vertex_set for v_idx in face):
            nearby_face_indices.append(face_idx)
    
    nearby_face_indices = np.array(nearby_face_indices)
    print(f"    [Remesh] {len(nearby_face_indices)} Faces mit Vertices im Radius")
    
    # 3. Identifiziere Boundary-Edges (Kanten die nur teilweise im Radius sind)
    boundary_edges = []
    for face_idx, face in enumerate(faces[nearby_face_indices]):
        for i in range(3):
            v1_idx = face[i]
            v2_idx = face[(i + 1) % 3]
            
            v1_in = v1_idx in nearby_vertex_set
            v2_in = v2_idx in nearby_vertex_set
            
            # Boundary: eine Vertex im Radius, eine außerhalb
            if v1_in != v2_in:
                boundary_edges.append((v1_idx, v2_idx))
    
    print(f"    [Remesh] {len(boundary_edges)} Boundary-Edges")
    
    return {
        "center_point": center_xy,
        "junction_pos_3d": junction_pos,
        "nearby_vertex_indices": nearby_vertex_indices,
        "nearby_face_indices": nearby_face_indices,
        "nearby_vertices_3d": vertices[nearby_vertex_indices],
        "boundary_edges": boundary_edges,
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
    
    print(f"    [Remesh] Geometrie-Analyse:")
    print(f"      - Z-Bereich: {z_min:.2f} bis {z_max:.2f} m (Variation: {z_range:.2f}m)")
    print(f"      - 2D Bounding Box: {x_range:.2f}m x {y_range:.2f}m = {area_bounding_box:.2f}m²")
    print(f"      - Faces im Radius: {len(nearby_f_indices)}")
    
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
    polygons_2d = []
    z_values_per_face = []  # Speichere Z-Werte für Interpolation
    
    for f_idx in nearby_f_indices:
        face = faces[f_idx]
        face_vertices = vertices[face]
        
        # 2D-Koordinaten
        xy_coords = face_vertices[:, :2]
        # Z-Werte für Interpolation
        z_coords = face_vertices[:, 2]
        
        try:
            polygon = Polygon(xy_coords)
            if polygon.is_valid and polygon.area > 1e-6:  # Nur gültige Polygone
                polygons_2d.append(polygon)
                z_values_per_face.append(z_coords.mean())  # Mittlerer Z pro Face
        except Exception as e:
            pass  # Ungültiges Polygon ignorieren
    
    print(f"    [Remesh] {len(polygons_2d)} gültige 2D-Polygone")
    
    if len(polygons_2d) == 0:
        print(f"    [Remesh] FEHLER: Keine gültigen Polygone gefunden!")
        return None
    
    # Boundary-Edges als Constraints (für später)
    boundary_edges_2d = []
    for v1_idx, v2_idx in geometry_data["boundary_edges"]:
        p1 = vertices[v1_idx, :2]
        p2 = vertices[v2_idx, :2]
        boundary_edges_2d.append((p1, p2))
    
    return {
        "polygons_2d": polygons_2d,
        "z_values_per_face": z_values_per_face,
        "boundary_edges_2d": boundary_edges_2d,
        "center_point": center_point,
        "radius": geometry_data["radius"],
        "original_data": geometry_data,
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
    print(f"    [Remesh] Führe unary_union auf {len(polygons_2d)} Polygonen durch...")
    merged = unary_union(polygons_2d)
    
    # merged kann jetzt Polygon oder MultiPolygon sein
    if isinstance(merged, MultiPolygon):
        print(f"    [Remesh] Resultat: MultiPolygon mit {len(merged.geoms)} Teilen")
        # Nimm den größten Part
        largest = max(merged.geoms, key=lambda p: p.area)
        merged_polygon = largest
        print(f"    [Remesh] Nutze größtes Polygon ({largest.area:.2f} m²)")
    else:
        merged_polygon = merged
        print(f"    [Remesh] Resultat: Single Polygon ({merged.area:.2f} m²)")
    
    # 2. Extrahiere die äußere Boundary als Vertices
    boundary_coords = np.array(merged_polygon.exterior.coords[:-1])  # Letzter Punkt = erster
    print(f"    [Remesh] Boundary: {len(boundary_coords)} Vertices")
    
    # 3. Trianguliere mit Scipy Delaunay
    print(f"    [Remesh] Trianguliere mit Delaunay...")
    try:
        tri = Delaunay(boundary_coords)
        new_faces = tri.simplices
        new_vertices_2d = tri.points
        print(f"    [Remesh] Triangulation: {len(new_vertices_2d)} Vertices, {len(new_faces)} Faces")
    except Exception as e:
        print(f"    [Remesh] FEHLER bei Delaunay: {e}")
        return None
    
    return {
        "merged_polygon": merged_polygon,
        "boundary_coords": boundary_coords,
        "new_vertices_2d": new_vertices_2d,
        "new_faces": new_faces,
        "remesh_data": remesh_data,
    }


def reconstruct_z_values(triangulation_data, original_vertices):
    """
    Rekonstruiert Z-Werte für die neuen Vertices der Triangulation.
    
    Methode: Benutze die mittlere Z-Höhe der nächsten Original-Vertices im Radius.
    
    Args:
        triangulation_data: Dict aus merge_and_triangulate()
        original_vertices: Alle Original-Mesh-Vertices (Nx3)
    
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
    
    print(f"    [Remesh] Z-Rekonstruktion:")
    print(f"      - Original Z-Werte im Radius: min={z_min:.2f}, max={z_max:.2f}, mean={z_mean:.2f}")
    
    # KDTree auf Original-Vertices für Z-Interpolation
    original_xy = original_vertices[nearby_v_indices, :2]
    z_tree = cKDTree(original_xy)
    
    # Für jeden neuen Vertex: Finde nächste Original-Vertices und interpoliere Z
    new_vertices_3d = []
    for vertex_2d in new_vertices_2d:
        # Finde 3 nächste Original-Vertices
        distances, indices = z_tree.query(vertex_2d, k=min(3, len(original_xy)))
        
        # Gewichtete Interpolation (inverse distance weighting)
        if np.min(distances) < 1e-6:  # Vertex ist sehr nah an Original
            # Nutze exakte Z
            original_idx = indices if isinstance(indices, np.integer) else indices[0]
            z_value = original_z_values[original_idx]
        else:
            # IDW-Interpolation
            if isinstance(distances, np.ndarray):
                weights = 1.0 / (distances + 1e-8)
            else:
                weights = np.array([1.0 / (distances + 1e-8)])
            weights /= np.sum(weights)
            
            if isinstance(indices, np.ndarray):
                z_values = original_z_values[indices]
            else:
                z_values = original_z_values[[indices]]
            z_value = np.sum(weights * z_values)
        
        new_vertices_3d.append([vertex_2d[0], vertex_2d[1], z_value])
    
    new_vertices_3d = np.array(new_vertices_3d, dtype=np.float32)
    print(f"    [Remesh] 3D-Rekonstruktion: {len(new_vertices_3d)} Vertices")
    
    return {
        "new_vertices_3d": new_vertices_3d,
        "new_faces": triangulation_data["new_faces"],
        "triangulation_data": triangulation_data,
    }


def remesh_single_junction(junction_idx, junction, vertices, faces, vertex_manager):
    """
    Remesht einen einzelnen Junction komplett.
    
    Pipeline:
    1. Sammle Geometrie im Radius
    2. Bereite 2D-Projektion vor
    3. Merge und Triangulate mit Delaunay
    4. Rekonstruiere Z-Werte
    5. Integriere neue Vertices im VertexManager
    6. Gebe neue Face-Indizes zurück
    
    Args:
        junction_idx: Index des Junction in junctions-Liste
        junction: Junction-Dict mit 'position' etc.
        vertices: Alle Mesh-Vertices (Nx3)
        faces: Alle Mesh-Faces (Mx3)
        vertex_manager: VertexManager-Instanz für neue Vertices
    
    Returns:
        Dict mit:
        - success: bool
        - new_faces: Liste der neuen Face-Indizes
        - statistics: Debug-Info
        
        oder None bei Fehler
    """
    
    try:
        # 1. Sammle Geometrie
        geometry_data = collect_nearby_geometry(
            junction["position"],
            vertices,
            faces,
            radius=10.0
        )
        
        if geometry_data is None:
            return {
                "success": False,
                "new_faces": [],
                "statistics": "Keine Geometrie im Radius",
            }
        
        # 2. Bereite Remesh-Daten vor
        remesh_data = prepare_remesh_data(geometry_data, vertices, faces)
        
        if remesh_data is None:
            return {
                "success": False,
                "new_faces": [],
                "statistics": "Konnte Remesh-Daten nicht vorbereiten",
            }
        
        # 3. Merge und Triangulate
        triangulation_data = merge_and_triangulate(remesh_data)
        
        if triangulation_data is None:
            return {
                "success": False,
                "new_faces": [],
                "statistics": "Triangulation fehlgeschlagen",
            }
        
        # 4. Rekonstruiere Z-Werte
        result_data = reconstruct_z_values(triangulation_data, vertices)
        
        if result_data is None:
            return {
                "success": False,
                "new_faces": [],
                "statistics": "Z-Rekonstruktion fehlgeschlagen",
            }
        
        # 5. Integriere neue Vertices im VertexManager
        new_vertices_3d = result_data["new_vertices_3d"]
        new_faces = result_data["new_faces"]
        
        # Füge Vertices hinzu und erhalte die neuen Indizes
        vertex_indices = []
        for vertex in new_vertices_3d:
            idx = vertex_manager.add_vertex(vertex[0], vertex[1], vertex[2])
            vertex_indices.append(idx)
        
        # 6. Konvertiere Face-Indizes auf neue Vertex-Indizes
        remeshed_faces = []
        for face in new_faces:
            remeshed_face = [
                vertex_indices[face[0]],
                vertex_indices[face[1]],
                vertex_indices[face[2]]
            ]
            remeshed_faces.append(remeshed_face)
        
        return {
            "success": True,
            "new_faces": remeshed_faces,
            "new_vertices_count": len(new_vertices_3d),
            "new_faces_count": len(remeshed_faces),
            "nearby_vertices": len(geometry_data["nearby_vertex_indices"]),
            "nearby_faces": len(geometry_data["nearby_face_indices"]),
        }
        
    except Exception as e:
        print(f"    [Remesh] FEHLER bei Junction {junction_idx}: {e}")
        return {
            "success": False,
            "new_faces": [],
            "statistics": str(e),
        }
