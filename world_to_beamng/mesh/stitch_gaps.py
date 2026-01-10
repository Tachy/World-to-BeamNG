"""
Stitching von Terrain-Lücken entlang Centerlines.
Iteriert über alle Straßen und füllt Löcher in Suchkreisen.
"""

import numpy as np
from shapely.geometry import LineString
from scipy.spatial import cKDTree

from .. import config
from ..config import OSM_MAPPER
from .stitch_local import find_boundary_polygons_in_circle


def stitch_all_gaps(
    road_data_for_classification,
    vertex_manager,
    mesh,
    terrain_vertex_indices,
    junction_points=None,
    filter_road_id=None,
    filter_junction_id=None,
):
    """
    Iteriert über alle Centerlines und stitcht Terrain-Lücken.

    Für jede Straße: Sample entlang der Centerline, suche in Suchkreisen
    nach Boundary-Polygonen und trianguliere sie mit Terrain-Faces.
    """
    print("  Stitche Terrain-Gaps entlang Centerlines...")
    if filter_road_id is not None:
        print(f"  [Filter] Nur Straße mit road_id={filter_road_id}")
    if filter_junction_id is not None:
        print(f"  [Filter] Nur Junction mit id={filter_junction_id}")

    # DEBUG: Prüfe ob Daten vorhanden sind
    print(
        f"  [DEBUG] road_data_for_classification: {len(road_data_for_classification) if road_data_for_classification else 0} Einträge"
    )
    if not road_data_for_classification:
        return []

    road_count = 0

    # Static Vertex-Cache und KDTree (Vertices werden nicht geändert)
    verts_cache = np.asarray(vertex_manager.get_array())
    kdtree_cache = cKDTree(verts_cache[:, :2]) if len(verts_cache) else None

    # Face-/Material-/Adjacency-Caches (werden nach neuen Faces inkrementell aktualisiert)
    vertex_to_faces, face_materials, terrain_face_indices = _build_face_caches(mesh)
    total_face_count = len(mesh.faces)

    for road_info in road_data_for_classification:
        road_count += 1
        if road_count % 50 == 0:
            print(f"  Verarbeite Straße #{road_count}...")

        road_id = road_info.get("road_id", road_info.get("id"))
        if filter_road_id is not None and str(road_id) != str(filter_road_id):
            continue

        # Hole Centerline
        trimmed_centerline = road_info.get("trimmed_centerline", [])

        if trimmed_centerline is None or len(trimmed_centerline) < 2:
            continue

        # Berechne dynamische Stitching-Parameter basierend auf Straßenbreite
        osm_tags = road_info.get("osm_tags", {})
        road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
        # Skaliere Search-Radius proportional zu Straßenbreite + Grid-Spacing-abhängiger Buffer
        # Formel: road_width + GRID_SPACING*2.5 (bei 2m Grid: road_width + 5m)
        dynamic_search_radius = road_width + config.GRID_SPACING * 2.5
        dynamic_sample_spacing = road_width + config.GRID_SPACING * 2.5  # Identisch mit Suchradius

        centerline_3d = np.array(trimmed_centerline, dtype=float)
        centerline_points = centerline_3d[:, :2]
        centerline_linestring = LineString(centerline_points)

        if not centerline_linestring.is_valid or centerline_linestring.length == 0:
            continue

        # Sample entlang Centerline
        total_length = centerline_linestring.length
        num_samples = max(2, int(np.ceil(total_length / dynamic_sample_spacing)) + 1)
        sample_distances = np.linspace(0, total_length, num_samples)

        # Verarbeite Sample-Punkte
        for distance in sample_distances:
            sample_pt_2d = centerline_linestring.interpolate(distance)

            # Z-Koordinate interpolieren
            distance_frac = distance / total_length if total_length > 0 else 0
            z_idx = int(distance_frac * (len(centerline_3d) - 1))
            z = centerline_3d[z_idx, 2]

            centerline_sample = np.array([sample_pt_2d.x, sample_pt_2d.y, z])

            # Finde Boundaries in Suchkreis (mit dynamischen Parametern)
            find_boundary_polygons_in_circle(
                centerline_point=centerline_sample,
                centerline_geometry=centerline_3d,
                search_radius=dynamic_search_radius,
                road_width=road_width,
                vertex_manager=vertex_manager,
                mesh=mesh,
                terrain_vertex_indices=terrain_vertex_indices,
                cached_verts=verts_cache,
                cached_kdtree=kdtree_cache,
                cached_face_materials=face_materials,
                cached_vertex_to_faces=vertex_to_faces,
                cached_terrain_face_indices=terrain_face_indices,
                debug=False,
            )

            # Neue Faces in Cache übernehmen (Triangulation fügt Terrain-Faces hinzu)
            if len(mesh.faces) > total_face_count:
                for global_idx in range(total_face_count, len(mesh.faces)):
                    face = mesh.faces[global_idx]
                    mat = mesh.face_props.get(global_idx, {}).get("material")
                    face_materials.append(mat)
                    # Adjacency ergänzen
                    for v in face:
                        vertex_to_faces.setdefault(v, []).append(global_idx)
                    if mat == "terrain":
                        terrain_face_indices.add(global_idx)
                total_face_count = len(mesh.faces)

    # Junction-Punkte mit Suchkreis prüfen
    if junction_points:
        print(f"  Verarbeite {len(junction_points)} Junction-Punkte...")
        for jp_idx, jp in enumerate(junction_points):
            jp_id = None
            coords = None
            connected_road_tags = []

            if isinstance(jp, dict):
                jp_id = jp.get("id")
                coords = jp.get("pos")
                if coords is None:
                    coords = jp.get("position")
                if coords is None:
                    coords = jp.get("coords")
                connected_road_tags = jp.get("connected_road_tags", [])
            elif isinstance(jp, (list, tuple)) and len(jp) == 2 and hasattr(jp[1], "__len__"):
                # Form (id, coords)
                jp_id = jp[0]
                coords = jp[1]
            else:
                coords = jp

            if jp_id is None:
                jp_id = jp_idx

            if filter_junction_id is not None and str(jp_id) != str(filter_junction_id):
                continue

            if coords is None or len(coords) < 2:
                continue

            # Berechne dynamischen Junction-Search-Radius:
            # Formel: max_road_width * 1.3 + GRID_SPACING*2.5 (bei 2m Grid: max_width * 1.3 + 5m)
            max_road_width = 4.0  # Fallback: 4m default
            if connected_road_tags:
                road_widths = [OSM_MAPPER.get_road_properties(tags)["width"] for tags in connected_road_tags]
                max_road_width = max(road_widths) if road_widths else max_road_width

            junction_search_radius = max_road_width * 1.3 + config.GRID_SPACING * 2.5

            if len(coords) == 2:
                centerline_sample = np.array([coords[0], coords[1], 0.0], dtype=float)
            else:
                centerline_sample = np.array(coords, dtype=float)

            _ = find_boundary_polygons_in_circle(
                centerline_point=centerline_sample,
                centerline_geometry=np.array([centerline_sample, centerline_sample]),
                search_radius=junction_search_radius,
                road_width=max_road_width,
                vertex_manager=vertex_manager,
                mesh=mesh,
                terrain_vertex_indices=terrain_vertex_indices,
                cached_verts=verts_cache,
                cached_kdtree=kdtree_cache,
                cached_face_materials=face_materials,
                cached_vertex_to_faces=vertex_to_faces,
                cached_terrain_face_indices=terrain_face_indices,
                debug=False,
            )

            if len(mesh.faces) > total_face_count:
                for global_idx in range(total_face_count, len(mesh.faces)):
                    face = mesh.faces[global_idx]
                    mat = mesh.face_props.get(global_idx, {}).get("material")
                    face_materials.append(mat)
                    for v in face:
                        vertex_to_faces.setdefault(v, []).append(global_idx)
                    if mat == "terrain":
                        terrain_face_indices.add(global_idx)
                total_face_count = len(mesh.faces)

    # Production-Rückgabe: Keine explizite Rückgabe nötig (Faces sind direkt ins Mesh eingefügt)
    return []


def _build_face_caches(mesh):
    """Erzeuge Face-Material-Liste, Terrain-Index-Set und Vertex->Faces-Adjacency."""
    face_materials = []
    vertex_to_faces = {}
    terrain_face_indices = set()

    for idx, face in enumerate(mesh.faces):
        mat = mesh.face_props.get(idx, {}).get("material")
        face_materials.append(mat)
        for v in face:
            vertex_to_faces.setdefault(v, []).append(idx)
        if mat == "terrain":
            terrain_face_indices.add(idx)

    return vertex_to_faces, face_materials, terrain_face_indices
