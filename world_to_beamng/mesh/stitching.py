"""
Stitching von Lücken zwischen Terrain und Böschungen.
Edge-basierter Ansatz: Verbindet Slope-Außenkanten mit nächstem Terrain.
"""

import mapbox_earcut as earcut
import numpy as np
from scipy.spatial import cKDTree


def stitch_terrain_gaps(
    vertex_manager,
    terrain_vertex_indices,
    road_slope_polygons_2d,
    terrain_faces,
    slope_faces,
    stitch_radius=10.0,
):
    """Füllt Lücken zwischen Slope-Außenkanten und nächstem Terrain mit earcut.

    Neuer Ansatz:
    - Für jede Slope-Außenkante
    - Finde nächste Terrain-Vertices
    - Bilde Polygon aus Slope-Edge + Terrain-Vertices
    - Trianguliere mit earcut
    - Gibt neue Faces als Terrain-Faces zurück
    """

    verts = np.asarray(vertex_manager.get_array())
    if len(verts) == 0:
        return []

    # Baue KDTree für Terrain-Vertices (schneller Zugriff)
    if not terrain_vertex_indices:
        print("  ⚠ Keine Terrain-Vertices für Stitching")
        return []
    
    terrain_indices_np = np.asarray(terrain_vertex_indices, dtype=int)
    terrain_verts = verts[terrain_indices_np][:, :2]
    terrain_tree = cKDTree(terrain_verts)
    
    print(f"  Terrain KDTree mit {len(terrain_indices_np)} Vertices erstellt")

    # Sammle alle Slope-Außenkanten mit ihren Koordinaten
    all_slope_edges = []
    for poly_data in road_slope_polygons_2d:
        slope_outer = poly_data.get("slope_outer_indices") or {}
        
        left_outer = slope_outer.get("left")
        right_outer = slope_outer.get("right")
        
        if left_outer and len(left_outer) >= 2:
            # Jede Edge als (v1, v2, center_xy)
            for i in range(len(left_outer) - 1):
                v1, v2 = left_outer[i], left_outer[i + 1]
                center = (verts[v1][:2] + verts[v2][:2]) / 2
                all_slope_edges.append((v1, v2, center))
        
        if right_outer and len(right_outer) >= 2:
            for i in range(len(right_outer) - 1):
                v1, v2 = right_outer[i], right_outer[i + 1]
                center = (verts[v1][:2] + verts[v2][:2]) / 2
                all_slope_edges.append((v1, v2, center))
    
    if len(all_slope_edges) == 0:
        print("  ⚠ Keine Slope-Außenkanten gefunden")
        return []
    
    print(f"  {len(all_slope_edges)} Slope-Außenkanten gefunden")

    all_stitch_faces = []
    
    # Für jede Slope-Edge: Finde nächste Terrain-Vertices und trianguliere
    for v1, v2, center in all_slope_edges:
        # Finde Terrain-Vertices in der Nähe dieser Edge
        candidates_local = terrain_tree.query_ball_point(center, r=stitch_radius)
        
        if len(candidates_local) < 1:
            continue
        
        # Konvertiere zu globalen Indices
        candidates = terrain_indices_np[candidates_local]
        
        # Baue Polygon: v1 -> v2 -> Terrain-Vertices
        edge_mid = (verts[v1][:2] + verts[v2][:2]) / 2
        
        # Filtere Terrain-Vertices
        terrain_candidates_sorted = []
        for t_idx in candidates:
            # Nur Vertices nehmen die nicht v1 oder v2 sind
            if t_idx == v1 or t_idx == v2:
                continue
            terrain_candidates_sorted.append(t_idx)
        
        if len(terrain_candidates_sorted) == 0:
            continue
        
        # Sortiere Terrain-Vertices nach Abstand zu Edge-Mitte
        terrain_coords = verts[terrain_candidates_sorted][:, :2]
        distances = np.linalg.norm(terrain_coords - edge_mid, axis=1)
        sorted_indices = np.argsort(distances)
        
        # Nimm nur die nächsten N Terrain-Vertices
        max_terrain_verts = min(5, len(sorted_indices))
        nearest_terrain = [terrain_candidates_sorted[i] for i in sorted_indices[:max_terrain_verts]]
        
        # Polygon für earcut
        polygon_indices = [v1, v2] + nearest_terrain
        
        if len(polygon_indices) < 3:
            continue
        
        # Trianguliere
        try:
            polygon_coords = verts[polygon_indices][:, :2]
            ring_indices = np.array([len(polygon_coords)], dtype=np.uint32)
            tri_indices = earcut.triangulate_float64(polygon_coords, ring_indices)
            
            if len(tri_indices) == 0:
                continue
            
            tri_indices = tri_indices.reshape(-1, 3)
            
            # Konvertiere zu globalen Indices
            for tri in tri_indices:
                face = [polygon_indices[tri[0]], polygon_indices[tri[1]], polygon_indices[tri[2]]]
                if len(set(face)) == 3:
                    all_stitch_faces.append(face)
        except Exception:
            pass
    
    print(f"  ✓ {len(all_stitch_faces)} Stitch-Faces erzeugt")
    return all_stitch_faces
