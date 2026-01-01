"""
WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator

Refactored Version mit modularer Architektur.
Main Entry Point für die Anwendung.

Benötigte Pakete:
  pip install requests numpy scipy pyproj pyvista shapely rtree

Alle Abhängigkeiten sind ERFORDERLICH - kein Fallback!!
"""

import sys
import time
import os
import glob
import gc
import copy
import numpy as np

# UTF-8 Encoding fuer Windows Console (fuer Unicode Zeichen in Status-Bar)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Importiere alle Module
from world_to_beamng import config
from world_to_beamng.terrain.elevation import load_height_data, get_height_data_hash
from world_to_beamng.terrain.grid import create_terrain_grid
from world_to_beamng.osm.parser import (
    calculate_bbox_from_height_data,
    extract_roads_from_osm,
)
from world_to_beamng.osm.downloader import get_osm_data
from world_to_beamng.geometry.polygon import get_road_polygons, clip_road_polygons
from world_to_beamng.geometry.junctions import (
    detect_junctions_in_centerlines,
    mark_junction_endpoints,
    junction_stats,
)
from world_to_beamng.mesh.junction_remesh import (
    collect_nearby_geometry,
    analyze_junction_geometry,
    prepare_remesh_data,
    merge_and_triangulate,
    reconstruct_z_values,
    remesh_single_junction,
)
from world_to_beamng.geometry.vertices import classify_grid_vertices
from world_to_beamng.mesh.road_mesh import generate_road_mesh_strips
from world_to_beamng.mesh.vertex_manager import VertexManager
from world_to_beamng.mesh.terrain_mesh import generate_full_grid_mesh
from world_to_beamng.mesh.cleanup import cleanup_duplicate_faces
from world_to_beamng.mesh.stitching import (
    stitch_terrain_gaps,
)
from world_to_beamng.io.obj import (
    save_unified_obj,
    save_roads_obj,
    save_ebene1_roads,
    save_ebene2_centerlines_junctions,
)


def enforce_ccw_up(faces, vertices):
    """Sorgt dafür, dass Dreiecke mit +Z-Normalen (CCW nach oben) angeordnet sind."""
    if not faces:
        return faces
    faces_np = np.asarray(faces, dtype=np.int64)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        return faces

    verts_np = np.asarray(vertices, dtype=np.float64)
    tri_pts = verts_np[faces_np]
    normals = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    flip_idx = np.where(normals[:, 2] < 0)[0]
    if flip_idx.size:
        f1 = faces_np[flip_idx, 1].copy()
        f2 = faces_np[flip_idx, 2].copy()
        faces_np[flip_idx, 1] = f2
        faces_np[flip_idx, 2] = f1
    return faces_np.tolist()


def report_boundary_edges(faces, vertices, label="mesh", export_path=None):
    """Loggt offene und nicht-manifold Kanten (0-basiert). Optionaler Export der offenen Kanten als OBJ."""
    if not faces:
        print(f"  {label}: Keine Faces -> keine Kanten")
        return
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        print(f"  {label}: Nicht-dreieckige Faces uebersprungen")
        return

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)

    # Entferne Kanten, die vollständig auf dem äußeren Bounding-Box-Rand liegen
    verts_np = np.asarray(vertices, dtype=np.float64)
    xs = verts_np[:, 0]
    ys = verts_np[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    tol = max(getattr(config, "GRID_SPACING", 1.0) * 0.5, 0.05)

    on_min_x = np.abs(xs - min_x) <= tol
    on_max_x = np.abs(xs - max_x) <= tol
    on_min_y = np.abs(ys - min_y) <= tol
    on_max_y = np.abs(ys - max_y) <= tol

    a_idx = edges[:, 0]
    b_idx = edges[:, 1]
    border_mask = (
        (on_min_x[a_idx] & on_min_x[b_idx])
        | (on_max_x[a_idx] & on_max_x[b_idx])
        | (on_min_y[a_idx] & on_min_y[b_idx])
        | (on_max_y[a_idx] & on_max_y[b_idx])
    )

    edges_filtered = edges[~border_mask]
    if edges_filtered.size == 0:
        print(f"  {label}: Keine Kanten nach Rand-Filter")
        return

    # Nutzung von np.unique über axis=0 vermeidet View-Probleme
    unique, counts = np.unique(edges_filtered, axis=0, return_counts=True)

    boundary_mask = counts == 1
    nonmanifold_mask = counts > 2
    num_boundary = int(np.count_nonzero(boundary_mask))
    num_nonmanifold = int(np.count_nonzero(nonmanifold_mask))

    print(f"  {label}: Boundary-Kanten={num_boundary}, Non-manifold={num_nonmanifold}")
    if num_boundary:
        sample = unique[boundary_mask][:10]
        print(f"    Beispiel offene Kanten (0-basiert): {sample.tolist()}")
        if export_path:
            try:
                edges_to_export = unique[boundary_mask]
                verts_np = np.asarray(vertices, dtype=np.float32)
                used_idx = np.unique(edges_to_export)

                # Dünne Quads aus zwei Dreiecken mit angehobenen Duplikaten, damit Fläche sichtbar ist
                epsilon = 0.1  # Sichtbarer Offset

                # Mapping alt->neu (1-basiert für OBJ)
                new_idx = {int(old): i + 1 for i, old in enumerate(used_idx)}
                elev_idx = {}
                extra_vertices = []
                next_idx = len(used_idx) + 1

                for vid in used_idx:
                    v = verts_np[int(vid)].copy()
                    v[2] += epsilon
                    extra_vertices.append(v)
                    elev_idx[int(vid)] = next_idx
                    next_idx += 1

                faces = []
                for a, b in edges_to_export:
                    a = int(a)
                    b = int(b)
                    fa = new_idx[a]
                    fb = new_idx[b]
                    fa_up = elev_idx[a]
                    fb_up = elev_idx[b]
                    # Zwei Dreiecke pro Edge (dünnes Band)
                    faces.append([fa_up, fa, fb])
                    faces.append([fa_up, fb, fb_up])

                mtl_path = (
                    export_path.replace(".obj", ".mtl")
                    if export_path.lower().endswith(".obj")
                    else None
                )

                with open(export_path, "w") as fobj:
                    if mtl_path:
                        fobj.write(f"mtllib {os.path.basename(mtl_path)}\n")
                    # Vertices (bestehende)
                    for vid in used_idx:
                        x, y, z = verts_np[int(vid)]
                        fobj.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")
                    # Angehoene Duplikate
                    for mv in extra_vertices:
                        fobj.write(f"v {mv[0]:.3f} {mv[1]:.3f} {mv[2]:.3f}\n")

                    fobj.write("o boundary_edges\n")
                    if mtl_path:
                        fobj.write("usemtl boundary_edges\n")
                    # Faces (dünne Bänder über jeder Kante)
                    for fa, fb, fc in faces:
                        fobj.write(f"f {fa} {fb} {fc}\n")

                if mtl_path:
                    with open(mtl_path, "w") as fmtl:
                        fmtl.write("newmtl boundary_edges\n")
                        fmtl.write("Ka 1 0 0\nKd 1 0 0\nKs 0 0 0\nd 1.0\nillum 1\n")

                print(
                    f"    -> Boundary-Kanten als OBJ (Faces) exportiert nach {export_path}"
                )
            except Exception as exc:
                print(f"    [!] Export nach {export_path} fehlgeschlagen: {exc}")


def main():
    """Hauptfunktion der Anwendung - koordiniert alle Module."""
    
    import time as time_module  # Umbenennen um Konflikt zu vermeiden
    
    # === Command-Line Arguments ===
    import argparse
    parser = argparse.ArgumentParser(description='World-to-BeamNG Straßen-Generator')
    parser.add_argument(
        '--junction-id',
        type=int,
        default=None,
        help='Optional: Nur diese Junction-ID remeshen (für Debugging/Profiling)'
    )
    args = parser.parse_args()
    
    debug_junction_id = args.junction_id
    if debug_junction_id is not None:
        print(f"[DEBUG] Junction-Remeshing nur für Junction #{debug_junction_id}")

    # Reset globale Zustände
    config.LOCAL_OFFSET = None
    config.BBOX = None
    config.GRID_BOUNDS_LOCAL = None

    start_time = time_module.time()
    timings = {}

    # ===== SCHRITT 1: Lade Höhendaten =====
    print("=" * 60)
    print("WORLD-TO-BEAMNG - OSM zu BeamNG Strassen-Generator")
    print("=" * 60)

    print("\n[1] Lade Hoehendaten...")
    step_start = time.time()
    height_points, height_elevations = load_height_data()
    timings["1_Höhendaten_laden"] = time.time() - step_start

    # ===== SCHRITT 2: Berechne BBOX =====
    print("\n[2] Berechne BBOX aus Hoehendaten...")
    step_start = time.time()
    config.BBOX = calculate_bbox_from_height_data(height_points)

    # WICHTIG: Setze Local Offset SOFORT (vor allen weiteren Transformationen)
    if config.LOCAL_OFFSET is None:
        config.LOCAL_OFFSET = (
            height_points[0, 0],
            height_points[0, 1],
            height_elevations[0],
        )
        print(f"  LOCAL_OFFSET gesetzt: {config.LOCAL_OFFSET}")

    # Transformiere height_points zu lokalen Koordinaten
    ox, oy, oz = config.LOCAL_OFFSET
    height_points[:, 0] -= ox
    height_points[:, 1] -= oy
    height_elevations = height_elevations - oz  # Auch Z-Koordinaten transformieren!
    print(f"  [OK] height_points + elevations zu lokalen Koordinaten transformiert")

    timings["2_BBOX_berechnen"] = time.time() - step_start

    # ===== SCHRITT 3: Prüfe OSM-Daten-Cache =====
    print("\n[3] Pruefe OSM-Daten-Cache...")
    step_start = time.time()
    height_hash = get_height_data_hash()
    if not height_hash:
        height_hash = "no_files"

    cache_height_hash_path = os.path.join(config.CACHE_DIR, "height_data_hash.txt")

    # Prüfe ob height-data geändert wurde
    need_reload = False
    if os.path.exists(cache_height_hash_path):
        with open(cache_height_hash_path, "r") as f:
            cached_hash = f.read().strip()
        if cached_hash != height_hash:
            print("  [!] Hoehendaten haben sich geaendert - lade OSM-Daten neu")
            need_reload = True
    else:
        need_reload = True

    # Speichere aktuellen Hash
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(cache_height_hash_path, "w") as f:
        f.write(height_hash)

    # Lösche alte Caches wenn nötig
    if need_reload:
        old_caches = glob.glob(os.path.join(config.CACHE_DIR, "osm_all_*.json"))
        old_caches += glob.glob(os.path.join(config.CACHE_DIR, "elevations_*.json"))
        for cache in old_caches:
            try:
                os.remove(cache)
                print(f"  Alter Cache geloescht: {os.path.basename(cache)}")
            except:
                pass

    osm_elements = get_osm_data(config.BBOX)
    timings["3_OSM_Daten_holen"] = time.time() - step_start
    if not osm_elements:
        print("Keine Daten gefunden.")
        return

    # ===== SCHRITT 4: Extrahiere Straßen =====
    print("\n[4] Extrahiere Strassen aus OSM-Daten...")
    step_start = time.time()
    roads = extract_roads_from_osm(osm_elements)
    timings["4_Straßen_extrahieren"] = time.time() - step_start
    if not roads:
        print("Keine Strassen gefunden.")
        return

    # ===== SCHRITT 5: Erstelle Terrain-Grid =====
    print("\n[5] Erstelle Terrain-Grid...")
    step_start = time.time()
    grid_points, grid_elevations, nx, ny = create_terrain_grid(
        height_points, height_elevations, grid_spacing=config.GRID_SPACING
    )
    timings["5_Grid_erstellen"] = time.time() - step_start

    # LOCAL_OFFSET wurde bereits in Schritt 2 gesetzt

    # Speichere Grid-Bounds für Clipping
    config.GRID_BOUNDS_LOCAL = (
        grid_points[:, 0].min(),
        grid_points[:, 0].max(),
        grid_points[:, 1].min(),
        grid_points[:, 1].max(),
    )
    print(
        f"  Grid Bounds (lokal): X=[{config.GRID_BOUNDS_LOCAL[0]:.1f}, {config.GRID_BOUNDS_LOCAL[1]:.1f}], Y=[{config.GRID_BOUNDS_LOCAL[2]:.1f}, {config.GRID_BOUNDS_LOCAL[3]:.1f}]"
    )

    # ===== SCHRITT 6: Extrahiere Straßen-Polygone =====
    step_start = time.time()
    print(f"\n[6] Extrahiere {len(roads)} Strassen-Polygone...")
    road_polygons = get_road_polygons(
        roads, config.BBOX, height_points, height_elevations
    )
    print(f"  [OK] {len(road_polygons)} Strassen-Polygone extrahiert")

    # Clippe Straßen-Polygone am Grid-Rand (vor Mesh-Generierung!)
    if config.ROAD_CLIP_MARGIN > 0:
        print(
            f"  Clippe Straßen-Polygone am Grid-Rand ({config.ROAD_CLIP_MARGIN}m Margin)..."
        )
        road_polygons = clip_road_polygons(
            road_polygons, config.GRID_BOUNDS_LOCAL, margin=config.ROAD_CLIP_MARGIN
        )
        print(f"  [OK] {len(road_polygons)} Strassen nach Clipping")

    timings["6_Straßen_Polygone"] = time.time() - step_start

    # ===== SCHRITT 6a: Erkenne Straßen-Junctions in Centerlines =====
    step_start = time.time()
    print("\n[6a] Erkenne Strassen-Junctions in Centerlines...")
    junctions = detect_junctions_in_centerlines(road_polygons)
    road_polygons = mark_junction_endpoints(road_polygons, junctions)
    junction_stats(junctions, road_polygons)

    # Backup der unkürzten Centerlines vor Truncation (für Debug/Analyse)
    road_polygons_full = copy.deepcopy(road_polygons)
    timings["6a_Junctions_erkennen"] = time.time() - step_start

    # ===== SCHRITT 7: Generiere Straßen-Mesh =====
    print("\n[7] Generiere Strassen-Mesh-Streifen (mit Junctions)...")
    step_start = time.time()

    # Initialisiere VertexManager
    print("  Initialisiere zentrale Vertex-Verwaltung...")
    vertex_manager = VertexManager(tolerance=0.01)  # 1cm Toleranz für präzises Snapping
    print(f"    [OK] VertexManager bereit (Toleranz: 1cm)")
    timings["6b_VertexManager_Init"] = time.time() - step_start
    (
        road_faces,
        road_face_to_idx,
        slope_faces,
        road_slope_polygons_2d,
        original_to_mesh_idx,
    ) = generate_road_mesh_strips(
        road_polygons, height_points, height_elevations, vertex_manager
    )

    # Initialisiere Combined-Lists für spätere Verwendung
    combined_junction_faces = []

    print(
        f"  [OK] {len(road_slope_polygons_2d)} 2D-Polygone für Grid-Klassifizierung extrahiert"
    )
    print(
        f"  [OK] {vertex_manager.get_count()} Vertices gesamt (inkl. Straßen+Böschungen+Junctions+Connectors)"
    )
    timings["7_Straßen_Mesh"] = time.time() - step_start

    # Initialisiere Combined-Lists für spätere Verwendung
    combined_junction_faces = []

    # ===== SCHRITT 7a-7e: ÜBERSPRUNGEN (neuer Algorithmus in Planung) =====
    if False:
        # ===== SCHRITT 7a: Extrahiere Junction-Vertices aus Road-Mesh =====
        print("\n[7a] Baue Junction-Polygone aus Road-Mesh-Vertices...")
        step_start = time.time()

        # Extrahiere die Edge-Vertices der gekürzten Straßen
        junction_polys_dict = extract_junction_vertices_from_mesh(
            junctions, road_polygons, vertex_manager
        )

        # Konvertiere zu junction_polys-Format
        junction_polys = []
        for junction_idx, junction_data in junction_polys_dict.items():
            junction = junctions[junction_idx]
            junction_polys.append(
                {
                    "vertices_2d": junction_data["vertices_2d"],
                    "vertices_3d": junction_data["vertices_3d"],
                    "center": junction_data["center"],
                    "road_indices": junction_data["road_indices"],
                    "road_edge_data": junction_data.get(
                        "road_edge_data", {}
                    ),  # WICHTIG für Connectoren!
                    "type": (
                        "t_junction"
                        if len(junction_data["road_indices"]) == 3
                        else "cross"
                    ),
                    "num_roads": len(junction_data["road_indices"]),
                }
            )

        print(
            f"  [OK] {len(junction_polys)} Junction-Polygone aus Mesh-Vertices gebaut"
        )

        # ===== SCHRITT 7b: Baue Connector-Quads =====
        print("\n[7b] Baue Connector-Quads...")
        truncation_distance = 10.0
        connector_polys = build_junction_connectors(
            junction_polys,
            junctions,
            road_polygons,
            truncation_distance,
            config.ROAD_WIDTH,
        )
        print(f"  [OK] {len(connector_polys)} Connector-Quads gebaut")
        if connector_polys:
            print(
                f"    -> Erstes Connector: vertices_3d = {connector_polys[0].get('vertices_3d', 'NONE')}"
            )

        # ===== SCHRITT 7c: Meshe Junction-Polygone (separates Meshing) =====
        print("\n[7c] Meshe Junction-Quads (Fan-Triangulation)...")

        junction_faces, junction_face_indices = add_junction_polygons_to_mesh(
            vertex_manager, junction_polys
        )
        print(f"  [OK] {len(junction_faces)} Junction-Faces generiert")

        # WICHTIG: Speichere Junction-Faces SEPARAT für Material-Export!
        combined_junction_faces.extend(junction_faces)

        # ===== SCHRITT 7d: Meshe Connector-Quads (separates Meshing) =====
        print("\n[7d] Meshe Connector-Quads (Fan-Triangulation)...")
        connector_faces = connectors_to_faces(connector_polys, vertex_manager)

        # Integriere in road_faces VOR CCW-Normalisierung
        road_faces.extend(junction_faces)
        road_faces.extend(connector_faces)
        print(
            f"  [OK] {len(road_faces)} road_faces gesamt (inkl. Straßen+Junctions+Connectors)"
        )
        print(f"  [OK] {vertex_manager.get_count()} Vertices nach Junctions/Connectors")
        timings["7ab_Junction_Connector_Mesh"] = time.time() - step_start

        # ===== SCHRITT 7e: Füge Junction/Connector-Polygone für Terrain-Ausschnitt hinzu =====
        print(
            "\n[7e] Füge Junction+Connector-Polygone für Terrain-Klassifizierung hinzu..."
        )
        for junction_poly in junction_polys:
            vertices_2d = junction_poly.get("vertices_2d", [])
            if len(vertices_2d) >= 3:
                road_slope_polygons_2d.append(
                    {
                        "road_polygon": vertices_2d,
                        "slope_polygon": vertices_2d,  # Keine separaten Böschungen
                        "original_coords": [],
                        "road_vertex_indices": {"left": [], "right": []},
                        "slope_outer_indices": {"left": [], "right": []},
                    }
                )

        for connector_poly in connector_polys:
            vertices_2d = connector_poly.get("vertices_2d", [])
            if len(vertices_2d) >= 3:
                road_slope_polygons_2d.append(
                    {
                        "road_polygon": vertices_2d,
                        "slope_polygon": vertices_2d,  # Keine separaten Böschungen (vorerst)
                        "original_coords": [],
                        "road_vertex_indices": {"left": [], "right": []},
                        "slope_outer_indices": {"left": [], "right": []},
                    }
                )

        print(
            f"  [OK] {len(road_slope_polygons_2d)} Polygone gesamt fuer Terrain-Ausschnitt"
        )
    else:
        # Initialisiere leere Listen
        junction_polys = []
        connector_polys = []

    # ===== SCHRITT 8: Klassifiziere Grid-Vertices =====
    print("\n[8] Klassifiziere Grid-Vertices (Schneide Strassen aus Terrain)...")
    step_start = time.time()
    vertex_types, modified_heights = classify_grid_vertices(
        grid_points, grid_elevations, road_slope_polygons_2d
    )
    timings["8_Vertex_Klassifizierung"] = time.time() - step_start

    # ===== SCHRITT 9: Regeneriere Terrain-Mesh (mit Straßenausschnitten) =====
    step_start = time.time()
    print("\n[9] Regeneriere Terrain-Grid-Mesh (mit ausgeschnittenen Strassen)...")
    # WICHTIG: VertexManager dedupliziert automatisch - Terrain-Vertices werden wiederverwendet!
    terrain_faces_final, terrain_vertex_indices = generate_full_grid_mesh(
        grid_points,
        modified_heights,
        vertex_types,
        nx,
        ny,
        vertex_manager,
        dedup=False,
    )
    print(f"  [OK] {vertex_manager.get_count()} Vertices final (gesamt)")
    timings["9_Terrain_Grid_Final"] = time.time() - step_start

    # ===== SCHRITT 10: Junction Remeshing (mit sauberem Terrain) =====
    print("\n[10] Junction Remeshing mit lokaler Delaunay-Triangulation...")
    step_start = time.time()

    # Konvertiere zu NumPy für effiziente Operationen
    road_faces_array = (
        np.array(road_faces, dtype=np.int32)
        if road_faces
        else np.empty((0, 3), dtype=np.int32)
    )
    road_face_to_idx_array = (
        np.array(road_face_to_idx, dtype=np.int32)
        if road_face_to_idx
        else np.empty(0, dtype=np.int32)
    )

    # === CRITICAL: Übergebe nur road_faces_array (schon gefiltert!)
    # Keine zusätzliche Mask nötig - die Filterung ist bereits erfolgt
    print(f"  [Info] {len(road_faces_array)} Straßen-Faces für Remeshing verfügbar")

    remesh_boundaries = []
    remesh_radius_circles = []
    remesh_stats = {"success": 0, "failed": 0}

    # Remeshe Junctions mit mindestens drei Centerline-Verbindungen
    # Zähle Verbindungstypen: 'mid' zählt als 2 Richtungen, sonst Anzahl der Einträge
    def _connection_count(j):
        ct = j.get("connection_types", {}) or {}
        count = 0
        for types in ct.values():
            if "mid" in types:
                count += 2
            else:
                count += len(types)
        # Fallback: wenn keine connection_types hinterlegt, nutze road_indices-Länge
        if count == 0:
            count = len(j.get("road_indices", []))
        return count

    remesh_candidates = [
        (idx, j) for idx, j in enumerate(junctions) if _connection_count(j) >= 3
    ]
    
    # === DEBUG: Filtere nur auf eine Junction wenn --junction-id gegeben ===
    if debug_junction_id is not None:
        remesh_candidates = [
            (idx, j) for idx, j in remesh_candidates if idx == debug_junction_id
        ]
        if not remesh_candidates:
            print(f"  [WARN] Junction #{debug_junction_id} nicht gefunden oder hat <3 Verbindungen!")
        else:
            print(f"  [DEBUG] Remeshe nur Junction #{debug_junction_id}")

    t_loop_start = time_module.time()
    processed_count = 0
    total_junctions = len(remesh_candidates)
    
    # OPTIMIZATION: Cacle all_vertices - nur updaten wenn neue Vertices hinzugekommen
    all_vertices = vertex_manager.get_array()
    vertices_version = 0
    
    for junction_idx, junction in remesh_candidates:
        t_iter = time_module.time()
        
        # Aktualisiere all_vertices nur wenn Vertices hinzugekommen sind
        current_version = vertex_manager.get_count()  # Zähle Vertices
        if current_version != vertices_version:
            t_getarr = time_module.time()
            all_vertices = vertex_manager.get_array()
            t_getarr_done = time_module.time()
            vertices_version = current_version
        else:
            t_getarr = time_module.time()
            t_getarr_done = time_module.time()

        t_remesh = time_module.time()
        result = remesh_single_junction(
            junction_idx, junction, all_vertices, road_faces_array, vertex_manager
        )
        t_remesh_done = time_module.time()

        if result is not None and result["success"]:
            t_process = time_module.time()
            
            faces_to_remove = result.get("faces_to_remove", [])
            if faces_to_remove:
                # NumPy boolean indexing für effiziente Filterung
                keep_mask = np.ones(len(road_faces_array), dtype=bool)
                keep_mask[faces_to_remove] = False
                road_faces_array = road_faces_array[keep_mask]
                road_face_to_idx_array = road_face_to_idx_array[keep_mask]

            t_remove = time_module.time()
            
            # Integriere neue Faces mit NumPy concatenate
            new_faces = np.array(result["new_faces"], dtype=np.int32)
            new_face_idx = np.full(len(new_faces), -1, dtype=np.int32)
            road_faces_array = (
                np.vstack([road_faces_array, new_faces])
                if len(road_faces_array) > 0
                else new_faces
            )
            road_face_to_idx_array = np.concatenate(
                [road_face_to_idx_array, new_face_idx]
            )
            
            t_vstack = time_module.time()
            
            remesh_stats["success"] += 1
            boundary_coords = result.get("boundary_coords_3d")
            if boundary_coords is not None:
                remesh_boundaries.append(boundary_coords)

            circle = result.get("search_radius_circle")
            if circle:
                remesh_radius_circles.append(
                    {"junction_idx": junction_idx, "circle": circle}
                )
            
            processed_count += 1
        else:
            remesh_stats["failed"] += 1
            print(f"  [FEHLER] Junction {junction_idx} remesh fehlgeschlagen")

        # Fortschritt alle 100 Junctions oder am Ende melden
        if total_junctions > 0 and (processed_count % 100 == 0 or processed_count == total_junctions):
            print(f"  [Fortschritt] Schritt 10: {processed_count}/{total_junctions} Junctions")

    # Konvertiere zurück zu Listen für Kompatibilität mit nachfolgendem Code
    road_faces = road_faces_array.tolist()
    road_face_to_idx = road_face_to_idx_array.tolist()

    timings["10_Junction_Remesh"] = time.time() - step_start

    # Initialisiere Combined-Lists für spätere Verwendung
    combined_junction_faces = []

    # ===== SCHRITT 11: Normalisiere CCW-Orientierung =====
    step_start = time.time()
    print("\n[9a] Normalisiere CCW-Orientierung fuer alle Faces...")
    all_vertices_combined = np.asarray(vertex_manager.get_array())

    terrain_faces_final = enforce_ccw_up(terrain_faces_final, all_vertices_combined)
    road_faces = enforce_ccw_up(road_faces, all_vertices_combined)
    slope_faces = enforce_ccw_up(slope_faces, all_vertices_combined)
    print(f"  [OK] CCW-Orientierung sichergestellt")
    timings["9a_CCW_Normalization"] = time.time() - step_start

    # ===== SCHRITT 12: Stiching zwischen Terrain und Böschungen =====
    if config.HOLE_CHECK_ENABLED and len(slope_faces) > 0:
        step_start = time.time()
        print("\n[12] Fuelle Luecken zwischen Terrain und Boeschungen (Stitching)...")
        stitch_faces = stitch_terrain_gaps(
            vertex_manager,
            terrain_vertex_indices,
            road_slope_polygons_2d,
            terrain_faces_final,
            slope_faces,
            stitch_radius=10.0,
        )
        terrain_faces_final.extend(stitch_faces)
        print(f"  [OK] {len(stitch_faces)} Stitch-Faces hinzugefuegt")
        timings["12_Terrain_Stitching"] = time.time() - step_start
    else:
        reason = (
            "HOLE_CHECK_ENABLED=False"
            if not config.HOLE_CHECK_ENABLED
            else "Boeschungen deaktiviert"
        )
        print(f"\n[12] Stitching SKIP ({reason})")
        timings["12_Terrain_Stitching"] = 0.0

    # ===== SCHRITT 13: Hole finale Vertex-Daten =====
    step_start = time.time()
    print("\n[13] Extrahiere finale Vertex-Daten...")

    # Cleanup: Entferne doppelte Faces
    terrain_faces_final = cleanup_duplicate_faces(terrain_faces_final)
    road_faces = cleanup_duplicate_faces(road_faces)
    slope_faces = cleanup_duplicate_faces(slope_faces)

    all_vertices_combined = np.asarray(vertex_manager.get_array())
    total_vertex_count = len(all_vertices_combined)

    # WICHTIG: Junction+Connector-Faces sind bereits in road_faces integriert (Schritt 7a/7b)
    # Sie durchlaufen damit automatisch CCW-Normalisierung, cleanup, etc.
    print(f"\n[10b] Road-Faces bereits komplett (inkl. Junctions+Connectors)...")
    combined_road_faces = (
        road_faces  # Enthält bereits: Straßen + Junction-Quads + Connectors
    )
    # combined_junction_faces wird bereits gefüllt in Schritt 7c, NICHT hier leer setzen!

    # Faces sind bereits CCW-orientiert (Schritt 10a)
    combined_terrain_faces = terrain_faces_final
    combined_slope_faces = slope_faces

    if config.HOLE_CHECK_ENABLED:
        print("  Kanten-Check (0-basiert, vor Export)...")
        combined_all_faces = (
            combined_terrain_faces
            + combined_slope_faces
            + combined_road_faces
            + combined_junction_faces
        )
        export_path = config.BOUNDARY_EDGES_EXPORT or "boundary_edges.obj"
        report_boundary_edges(
            combined_all_faces,
            all_vertices_combined,
            label="Gesamtmesh",
            export_path=export_path,
        )

    print(f"  [OK] {total_vertex_count:,} Vertices gesamt (dedupliziert)")
    print(f"    • Terrain: {len(combined_terrain_faces):,} Faces")
    print(f"    • Strassen (inkl. Junctions): {len(combined_road_faces):,} Faces")
    print(f"    • Boeschungen: {len(combined_slope_faces):,} Faces")
    timings["10_Vertices_Finalisieren"] = time.time() - step_start

    # ===== SCHRITT 11: Terrain-Simplification (deaktiviert) =====
    step_start = time.time()
    if config.TERRAIN_REDUCTION > 0:
        print(
            "\n[11] Terrain-Simplification aktuell deaktiviert (zentraler VertexManager) -> bitte Terrain_REDUCTION=0 lassen."
        )
    else:
        print("\n[11] Überspringe Terrain-Simplification (TERRAIN_REDUCTION = 0)...")
    timings["11_Terrain_Simplification"] = time.time() - step_start

    # ===== SCHRITT 14: Bereite Faces für Export vor =====
    step_start = time.time()
    print("\n[12] Bereite Faces fuer OBJ-Export vor...")

    terrain_faces_final = np.array(combined_terrain_faces, dtype=np.int32)
    road_faces_array = np.array(combined_road_faces, dtype=np.int32)
    slope_faces_array = np.array(combined_slope_faces, dtype=np.int32)
    junction_faces_array = np.array(combined_junction_faces, dtype=np.int32)

    # OBJ erwartet 1-basierte Indices
    terrain_faces_final = terrain_faces_final + 1
    road_faces_final = road_faces_array + 1
    slope_faces_final = slope_faces_array + 1
    junction_faces_final = junction_faces_array + 1

    print(f"    • Terrain: {len(terrain_faces_final):,} Faces")
    print(f"    • Strassen: {len(road_faces_final):,} Faces")
    print(f"    • Boeschungen: {len(slope_faces_final):,} Faces")
    print(f"    • Junctions: {len(junction_faces_final):,} Faces")

    timings["12_Faces_Vorbereiten"] = time.time() - step_start

    # ===== SCHRITT 15: Exportiere Meshes als OBJ =====
    print("\n[13] Exportiere Meshes als OBJ...")
    step_start = time.time()
    output_obj = "beamng.obj"
    print(f"  Schreibe: {output_obj}")

    # Exportiere Debug-Ebenen
    print(f"  Exportiere Debug-Ebenen...")

    # ebene1.obj: Roads (nur Straßen-Mesh)
    print(f"  Schreibe ebene1.obj (Roads-Mesh)...")
    save_ebene1_roads(all_vertices_combined, road_faces, road_face_to_idx)

    # ebene2.obj: Centerlines + Junction-Points
    print(f"  Schreibe ebene2.obj (Centerlines + Junction-Points)...")
    # Nutze die unkürzten Centerlines aus dem Backup für Debug-Layer
    # Koordinaten sind bereits in polygon.py transformiert (UTM -> Local, einmalig!)
    save_ebene2_centerlines_junctions(
        road_polygons_full,
        junctions,
        remesh_boundaries=remesh_boundaries,
        radius_circles=remesh_radius_circles,
    )

    export_start = time_module.time()
    save_unified_obj(
        output_obj,
        all_vertices_combined,
        road_faces_final,
        slope_faces_final,
        terrain_faces_final,
        junction_faces_final,  # WICHTIG: Übergebe Junction-Faces separat!
    )
    print(f"    -> save_unified_obj(): {time_module.time() - export_start:.2f}s")

    cleanup_start = time_module.time()
    del (
        all_vertices_combined,
        terrain_faces_final,
        slope_faces_final,
        road_faces_final,
    )
    gc.collect()
    print(f"    -> Cleanup + GC: {time.time() - cleanup_start:.2f}s")

    timings["14_Mesh_Export"] = time.time() - step_start

    # ===== ZUSAMMENFASSUNG =====
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print(f"[OK] GENERATOR BEENDET!")
    print(f"{'=' * 60}")
    print(f"  Output-Datei: {output_obj}")
    if config.LOCAL_OFFSET:
        print(
            f"  Lokaler Offset: X={config.LOCAL_OFFSET[0]:.2f}m, Y={config.LOCAL_OFFSET[1]:.2f}m, Z={config.LOCAL_OFFSET[2]:.2f}m"
        )

    # DETAILLIERTE TIMING-ÜBERSICHT
    print(f"\n{'=' * 60}")
    print(f"ZEITMESSUNG (Gesamtzeit: {elapsed_time:.2f}s / {elapsed_time/60:.1f} min)")
    print(f"{'=' * 60}")
    for step_name, step_time in timings.items():
        percentage = (step_time / elapsed_time) * 100
        step_display = step_name.replace("_", " ").replace("  ", " ")
        bar_length = int(percentage / 2)  # 50 chars = 100%
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {step_display:.<35} {step_time:>6.2f}s ({percentage:>5.1f}%) {bar}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
