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
from world_to_beamng.mesh.cleanup import (
    cleanup_duplicate_faces,
    enforce_ccw_up,
    report_boundary_edges,
)
from world_to_beamng.mesh.stitching import (
    stitch_terrain_gaps,
)
from world_to_beamng.io.obj import (
    save_unified_obj,
    save_roads_obj,
    save_ebene1_roads,
    save_ebene2_centerlines_junctions,
)
from world_to_beamng.utils.timing import StepTimer


def main():
    """Hauptfunktion der Anwendung - koordiniert alle Module."""

    import time as time_module  # Umbenennen um Konflikt zu vermeiden

    # === Command-Line Arguments ===
    import argparse

    parser = argparse.ArgumentParser(description="World-to-BeamNG Straßen-Generator")
    parser.add_argument(
        "--junction-id",
        type=int,
        default=None,
        help="Optional: Nur diese Junction-ID remeshen (für Debugging/Profiling)",
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
    timer = StepTimer()

    # ===== SCHRITT 1: Lade Höhendaten =====
    print("=" * 60)
    print("WORLD-TO-BEAMNG - OSM zu BeamNG Strassen-Generator")
    print("=" * 60)

    timer.begin("Lade Hoehendaten")
    height_points, height_elevations = load_height_data()

    # ===== SCHRITT 2: Berechne BBOX =====
    timer.begin("Berechne BBOX aus Hoehendaten")
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

    # ===== SCHRITT 3: Prüfe OSM-Daten-Cache =====
    timer.begin("Pruefe OSM-Daten-Cache")
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
    if not osm_elements:
        print("Keine Daten gefunden.")
        return

    # ===== SCHRITT 4: Extrahiere Straßen =====
    timer.begin("Extrahiere Strassen aus OSM")
    roads = extract_roads_from_osm(osm_elements)
    if not roads:
        print("Keine Strassen gefunden.")
        return

    # ===== SCHRITT 5: Erstelle Terrain-Grid =====
    timer.begin("Erstelle Terrain-Grid")
    grid_points, grid_elevations, nx, ny = create_terrain_grid(
        height_points, height_elevations, grid_spacing=config.GRID_SPACING
    )

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
    timer.begin(f"Extrahiere {len(roads)} Strassen-Polygone")
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

    # ===== SCHRITT 6a: Erkenne Straßen-Junctions in Centerlines =====
    timer.begin("Erkenne Junctions in Centerlines")
    junctions = detect_junctions_in_centerlines(road_polygons)
    road_polygons = mark_junction_endpoints(road_polygons, junctions)
    junction_stats(junctions, road_polygons)

    # Backup der unkürzten Centerlines vor Truncation (für Debug/Analyse)
    road_polygons_full = copy.deepcopy(road_polygons)
    # ===== SCHRITT 6b: Initialisiere VertexManager =====
    timer.begin("Initialisiere Vertex-Manager")
    vertex_manager = VertexManager(tolerance=0.01)  # 1cm Toleranz für präzises Snapping
    print(f"    [OK] VertexManager bereit (Toleranz: 1cm)")
    # ===== SCHRITT 7: Generiere Straßen-Mesh =====
    timer.begin("Generiere Strassen-Mesh")
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
    timer.begin("Klassifiziere Grid-Vertices")
    vertex_types, modified_heights = classify_grid_vertices(
        grid_points, grid_elevations, road_slope_polygons_2d
    )

    # ===== SCHRITT 9: Regeneriere Terrain-Mesh (mit Straßenausschnitten) =====
    timer.begin("Regeneriere Terrain-Mesh")
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

    # ===== SCHRITT 10: Junction Remeshing (mit sauberem Terrain) =====
    timer.begin("Junction Remesh (Delaunay)")

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
            print(
                f"  [WARN] Junction #{debug_junction_id} nicht gefunden oder hat <3 Verbindungen!"
            )
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
        if total_junctions > 0 and (
            processed_count % 100 == 0 or processed_count == total_junctions
        ):
            print(
                f"  [Fortschritt] Schritt 10: {processed_count}/{total_junctions} Junctions"
            )

    # Konvertiere zurück zu Listen für Kompatibilität mit nachfolgendem Code
    road_faces = road_faces_array.tolist()
    road_face_to_idx = road_face_to_idx_array.tolist()

    # Initialisiere Combined-Lists für spätere Verwendung
    combined_junction_faces = []

    # ===== SCHRITT 11: Normalisiere CCW-Orientierung =====
    timer.begin("CCW-Orientierung herstellen")
    all_vertices_combined = np.asarray(vertex_manager.get_array())

    terrain_faces_final = enforce_ccw_up(terrain_faces_final, all_vertices_combined)
    road_faces = enforce_ccw_up(road_faces, all_vertices_combined)
    slope_faces = enforce_ccw_up(slope_faces, all_vertices_combined)
    print(f"  [OK] CCW-Orientierung sichergestellt")

    # ===== SCHRITT 12: Stiching zwischen Terrain und Böschungen =====
    timer.begin("Stitching Terrain/Boeschungen")
    if config.HOLE_CHECK_ENABLED and len(slope_faces) > 0:
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
    else:
        reason = (
            "HOLE_CHECK_ENABLED=False"
            if not config.HOLE_CHECK_ENABLED
            else "Boeschungen deaktiviert"
        )
        print(f"\n[{timer.current_label}] Stitching SKIP ({reason})")
        timer.set_duration(0.0)

    # ===== SCHRITT 13: Hole finale Vertex-Daten =====
    timer.begin("Extrahiere finale Vertex-Daten")

    # Cleanup: Entferne doppelte Faces
    terrain_faces_final = cleanup_duplicate_faces(terrain_faces_final)
    road_faces = cleanup_duplicate_faces(road_faces)
    slope_faces = cleanup_duplicate_faces(slope_faces)

    all_vertices_combined = np.asarray(vertex_manager.get_array())
    total_vertex_count = len(all_vertices_combined)

    # WICHTIG: Junction+Connector-Faces sind bereits in road_faces integriert (Schritt 7a/7b)
    # Sie durchlaufen damit automatisch CCW-Normalisierung, cleanup, etc.
    print(f"  [Info] Road-Faces bereits komplett (inkl. Junctions+Connectors)...")
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

    # ===== SCHRITT 14: Terrain-Simplification (deaktiviert) =====
    timer.begin("Terrain-Simplification")
    if config.TERRAIN_REDUCTION > 0:
        print(
            "\n[14] Terrain-Simplification aktuell deaktiviert (zentraler VertexManager) -> bitte Terrain_REDUCTION=0 lassen."
        )
    else:
        print("\n[14] Überspringe Terrain-Simplification (TERRAIN_REDUCTION = 0)...")

    # ===== SCHRITT 15: Bereite Faces für Export vor =====
    timer.begin("Faces f. OBJ-Export vorbereiten")

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

    # ===== SCHRITT 16: Exportiere Meshes als OBJ =====
    timer.begin("Exportiere Meshes als OBJ")
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

    # ===== ZUSAMMENFASSUNG =====
    print(f"\n{'=' * 60}")
    print(f"[OK] GENERATOR BEENDET!")
    print(f"{'=' * 60}")
    print(f"  Output-Datei: {output_obj}")
    if config.LOCAL_OFFSET:
        print(
            f"  Lokaler Offset: X={config.LOCAL_OFFSET[0]:.2f}m, Y={config.LOCAL_OFFSET[1]:.2f}m, Z={config.LOCAL_OFFSET[2]:.2f}m"
        )

    timer.report()


if __name__ == "__main__":
    main()
