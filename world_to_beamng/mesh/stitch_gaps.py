"""
Stitching von Terrain-Lücken entlang Centerlines.
Iteriert über alle Straßen und füllt Löcher in Suchkreisen.
"""

import numpy as np
from shapely.geometry import LineString
from scipy.spatial import cKDTree

from .. import config
from .fill_all_mesh_holes import fill_all_mesh_holes
from ..config import OSM_MAPPER
from .stitch_local import find_boundary_polygons_in_circle
from ..utils.debug_exporter import DebugNetworkExporter


def stitch_all_gaps(
    road_data_for_classification,
    vertex_manager,
    mesh,
    terrain_vertex_indices,
    junction_points=None,
    filter_road_id=None,
    filter_junction_id=None,
    debug_stop_at_road=None,
    debug_stop_at_circle=None,
):
    """
    Iteriert über alle Centerlines und stitcht Terrain-Lücken.

    Für jede Straße: Sample entlang der Centerline, suche in Suchkreisen
    nach Boundary-Polygonen und trianguliere sie mit Terrain-Faces.

    Args:
        debug_stop_at_road: Optional int - Stoppe nach dieser Straßennummer (1-basiert)
        debug_stop_at_circle: Optional int - Stoppe nach diesem Suchkreis (1-basiert) bei debug_stop_at_road
                             Alle bis dahin gefundenen Polygone werden mit _export_boundary_polygons_to_debug geloggt
    """
    print("  Stitche Terrain-Gaps entlang Centerlines...")
    if filter_road_id is not None:
        print(f"  [Filter] Nur Straße mit road_id={filter_road_id}")
    if filter_junction_id is not None:
        print(f"  [Filter] Nur Junction mit id={filter_junction_id}")
    if debug_stop_at_road is not None:
        print(f"  [DEBUG] Stoppe nach Straße #{debug_stop_at_road}, Suchkreis #{debug_stop_at_circle}")

    if not road_data_for_classification:
        return []

    road_count = 0
    valid_road_count = 0

    verts_cache = np.asarray(vertex_manager.get_array())
    kdtree_cache = cKDTree(verts_cache[:, :2]) if len(verts_cache) else None
    vertex_to_faces, face_materials, terrain_face_indices = _build_face_caches(mesh)

    # Junction-Punkte mit Suchkreis prüfen (ZUERST vor Straßen)
    if junction_points:
        print(f"  Verarbeite {len(junction_points)} Junction-Punkte...")
        for jp_idx, jp in enumerate(junction_points):
            coords = None
            connected_road_tags = []

            if isinstance(jp, dict):
                coords = jp.get("pos")
                if coords is None:
                    coords = jp.get("position")
                if coords is None:
                    coords = jp.get("coords")
                connected_road_tags = jp.get("connected_road_tags", [])
            elif isinstance(jp, (list, tuple)) and len(jp) == 2 and hasattr(jp[1], "__len__"):
                coords = jp[1]
            else:
                coords = jp

            if filter_junction_id is not None and int(jp_idx) != int(filter_junction_id):
                continue
            if coords is None or len(coords) < 2:
                continue

            max_road_width = 4.0
            if connected_road_tags:
                road_widths = [OSM_MAPPER.get_road_properties(tags)["width"] for tags in connected_road_tags]
                max_road_width = max(road_widths) if road_widths else max_road_width

            junction_search_radius = max_road_width * 1.6 + config.GRID_SPACING * 2.5
            centerline_sample = np.array([coords[0], coords[1], 0.0] if len(coords) == 2 else coords, dtype=float)

            debug_exporter = DebugNetworkExporter.get_instance()
            debug_exporter.add_junction(
                {
                    "id": jp_idx,
                    "position": centerline_sample.tolist(),
                    "road_indices": [],
                    "label": f"Junction_{jp_idx}",
                }
            )

            find_boundary_polygons_in_circle(
                centerline_point=centerline_sample,
                centerline_geometry=None,
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

    for road_info in road_data_for_classification:
        road_count += 1

        road_id = road_info.get("road_id", road_info.get("id"))
        if filter_road_id is not None and str(road_id) != str(filter_road_id):
            continue

        trimmed_centerline = road_info.get("trimmed_centerline", [])
        if trimmed_centerline is None or len(trimmed_centerline) < 2:
            continue

        osm_tags = road_info.get("osm_tags", {})
        road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]
        dynamic_search_radius = road_width / 2 + config.GRID_SPACING * 2.5
        dynamic_sample_spacing = road_width / 2 + config.GRID_SPACING * 2.5

        centerline_3d = np.array(trimmed_centerline, dtype=float)
        centerline_linestring = LineString(centerline_3d[:, :2])

        if not centerline_linestring.is_valid or centerline_linestring.length == 0:
            continue

        valid_road_count += 1
        if debug_stop_at_road is not None:
            print(f"  [DEBUG] Verarbeite gültige Straße #{valid_road_count}...")
        elif valid_road_count % 50 == 0:
            print(f"  Verarbeite Straße #{valid_road_count}...")

        total_length = centerline_linestring.length
        num_samples = max(2, int(np.ceil(total_length / dynamic_sample_spacing)) + 1)
        sample_distances = np.linspace(0, total_length, num_samples)

        circle_count = 0
        for distance in sample_distances:
            circle_count += 1
            if filter_road_id is not None or debug_stop_at_road is not None and valid_road_count == debug_stop_at_road:
                print(f"  [DEBUG] Suchkreis #{circle_count}...")

            sample_pt_2d = centerline_linestring.interpolate(distance)
            distance_frac = distance / total_length if total_length > 0 else 0
            z_idx = int(distance_frac * (len(centerline_3d) - 1))
            z = centerline_3d[z_idx, 2]
            centerline_sample = np.array([sample_pt_2d.x, sample_pt_2d.y, z])

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

            if filter_road_id is not None or debug_stop_at_road is not None and debug_stop_at_circle is not None:
                if (
                    filter_road_id is not None
                    and circle_count == debug_stop_at_circle
                    or valid_road_count == debug_stop_at_road
                    and circle_count == debug_stop_at_circle
                ):
                    print(f"  [DEBUG] Stoppe bei Straße #{valid_road_count}, Suchkreis #{circle_count}")
                    return []

    # === STEP 3: Schließe verbliebene Löcher (Inseln, etc.) ===
    holes_filled = 0
    if config.FILL_ALL_MESH_HOLES:
        print(f"\n  [Hole-Filling] Schließe alle verbliebenen Mesh-Holes...")
        holes_filled = fill_all_mesh_holes(
            mesh_obj=mesh,
            vertex_manager=vertex_manager,
            max_edge_length=config.FILL_HOLES_MAX_EDGE_LENGTH,
        )
        print(f"  [✓] {holes_filled} Faces durch Hole-Filling eingefügt\n")
    else:
        print(f"  [i] Hole-Filling deaktiviert (config.FILL_ALL_MESH_HOLES=False)")

    return []


def _build_face_caches(mesh):
    """Erzeuge Face-Material-Dict, Terrain-Index-Set und Vertex->Faces-Adjacency."""
    face_materials = {}
    vertex_to_faces = {}
    terrain_face_indices = set()

    for idx, face in enumerate(mesh.faces):
        mat = mesh.face_props.get(idx, {}).get("material")
        face_materials[idx] = mat
        for v in face:
            vertex_to_faces.setdefault(v, []).append(idx)
        if mat == "terrain":
            terrain_face_indices.add(idx)

    return vertex_to_faces, face_materials, terrain_face_indices
