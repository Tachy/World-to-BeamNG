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
import json
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
    split_roads_at_mid_junctions,
)
from world_to_beamng.geometry.vertices import classify_grid_vertices
from world_to_beamng.mesh.road_mesh import generate_road_mesh_strips
from world_to_beamng.mesh.vertex_manager import VertexManager
from world_to_beamng.mesh.mesh import Mesh
from world_to_beamng.mesh.terrain_mesh import generate_full_grid_mesh

from world_to_beamng.mesh.cleanup import (
    report_boundary_edges,
)
from world_to_beamng.mesh.stitch_gaps import stitch_all_gaps
from world_to_beamng.mesh.tile_slicer import slice_mesh_into_tiles
from world_to_beamng.io.dae import export_merged_dae
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
    parser.add_argument(
        "--stitch-road-id",
        type=str,
        default=None,
        help="Optional: Nur diese road_id beim Stitching berücksichtigen (wie im dae_viewer sichtbar)",
    )
    parser.add_argument(
        "--stitch-junction-id",
        type=str,
        default=None,
        help="Optional: Nur diese Junction-ID beim Stitching berücksichtigen (wie im dae_viewer sichtbar)",
    )
    parser.add_argument(
        "--remesh-debug-dump",
        action="store_true",
        help="Remesh-Debugdaten (remesh_debug_data.json) schreiben",
    )
    args = parser.parse_args()

    debug_junction_id = args.junction_id
    if debug_junction_id is not None:
        print(f"[DEBUG] Junction-Remeshing nur für Junction #{debug_junction_id}")
    remesh_debug_dump = args.remesh_debug_dump

    stitch_filter_road_id = args.stitch_road_id
    stitch_filter_junction_id = args.stitch_junction_id

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
    height_points, height_elevations, needs_aerial_processing = load_height_data()

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

    # Berechne Grid-Bounds SOFORT (für Luftbild-Verarbeitung)
    config.GRID_BOUNDS_LOCAL = (
        height_points[:, 0].min(),
        height_points[:, 0].max(),
        height_points[:, 1].min(),
        height_points[:, 1].max(),
    )
    print(
        f"  Grid Bounds (lokal): X=[{config.GRID_BOUNDS_LOCAL[0]:.1f}, {config.GRID_BOUNDS_LOCAL[1]:.1f}], "
        f"Y=[{config.GRID_BOUNDS_LOCAL[2]:.1f}, {config.GRID_BOUNDS_LOCAL[3]:.1f}]"
    )

    # Verarbeite Luftbilder (nur wenn Höhendaten neu geladen wurden)
    if needs_aerial_processing:
        print(f"\n  [i] Verarbeite Luftbilder (da Hoehendaten neu geladen)...")
        from world_to_beamng.io.aerial import process_aerial_images

        tile_count = process_aerial_images(aerial_dir="aerial", output_dir=config.BEAMNG_DIR_TEXTURES, tile_size=2500)
        if tile_count > 0:
            print(f"  [OK] {tile_count} Luftbild-Kacheln exportiert")
        else:
            print("  [i] Keine Luftbilder verarbeitet")

    timer.begin("Pruefe OSM-Daten-Cache")
    height_hash = get_height_data_hash()
    if not height_hash:
        height_hash = "no_files"

    # Setze HEIGHT_HASH global (für konsistente Cache-Namensgebung)
    config.HEIGHT_HASH = height_hash

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

    timer.begin("Extrahiere Strassen aus OSM")
    roads = extract_roads_from_osm(osm_elements)
    if not roads:
        print("Keine Strassen gefunden.")
        return

    timer.begin("Erstelle Terrain-Grid")
    grid_points, grid_elevations, nx, ny = create_terrain_grid(
        height_points, height_elevations, grid_spacing=config.GRID_SPACING
    )

    # Grid-Bounds wurden bereits in Schritt 2 berechnet (aus height_points)

    timer.begin(f"Extrahiere {len(roads)} Strassen-Polygone")
    road_polygons = get_road_polygons(roads, config.BBOX, height_points, height_elevations)
    print(f"  [OK] {len(road_polygons)} Strassen-Polygone extrahiert")

    # Clippe Straßen-Polygone am Grid-Rand (vor Mesh-Generierung!)
    if config.ROAD_CLIP_MARGIN > 0:
        print(f"  Clippe Straßen-Polygone am Grid-Rand ({config.ROAD_CLIP_MARGIN}m Margin)...")
        road_polygons = clip_road_polygons(road_polygons, config.GRID_BOUNDS_LOCAL, margin=config.ROAD_CLIP_MARGIN)
        print(f"  [OK] {len(road_polygons)} Strassen nach Clipping")

    # ===== SCHRITT 6a: Erkenne Junctions in Centerlines (NUR mit Centerlines!) =====
    timer.begin("Erkenne Junctions in Centerlines")
    junctions = detect_junctions_in_centerlines(road_polygons, height_points, height_elevations)
    road_polygons, junctions = split_roads_at_mid_junctions(road_polygons, junctions)
    road_polygons = mark_junction_endpoints(road_polygons, junctions)
    junction_stats(junctions, road_polygons)

    # ===== SCHRITT 7: Initialisiere Vertex-Manager =====
    timer.begin("Initialisiere Vertex-Manager")
    vertex_manager = VertexManager(tolerance=0.01)  # 1cm Toleranz für präzises Snapping
    print(f"    [OK] VertexManager bereit (Toleranz: 1cm)")

    # Erstelle Mesh-Instanz
    mesh = Mesh(vertex_manager)

    # ===== SCHRITT 9: Generiere Road-Mesh (mit Junction-Informationen) =====
    timer.begin("Generiere Strassen-Mesh (7m Breite)")
    (
        road_faces,
        road_face_to_idx,
        _slope_faces,  # Unused - keine Slopes mehr
        road_slope_polygons_2d,
        original_to_mesh_idx,
        all_road_polygons_2d,
        _all_slope_polygons_2d,  # Unused - keine Slopes mehr
    ) = generate_road_mesh_strips(road_polygons, height_points, height_elevations, vertex_manager, junctions)

    # Füge Straßen-Faces zum Mesh hinzu
    print(f"  [OK] {len(road_faces)} Strassen-Faces generiert")
    for face in road_faces:
        mesh.add_face(face[0], face[1], face[2], material="road")

    # Konsistenter Debug-Dump: Junctions und getrimmte Straßen im selben Zustand
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    debug_network_path = os.path.join(config.CACHE_DIR, "debug_network.json")

    trimmed_roads_data = []
    junction_connections = {}  # junction_id -> {road_indices: [...], connection_types: {road_idx: ["start"|"end"]}}

    for road_meta in road_slope_polygons_2d:
        if "trimmed_centerline" in road_meta and road_meta["trimmed_centerline"]:
            coords = road_meta["trimmed_centerline"]
            dump_idx = len(trimmed_roads_data)

            start_jid = road_meta.get("junction_start_id")
            end_jid = road_meta.get("junction_end_id")

            trimmed_roads_data.append(
                {
                    "road_id": road_meta.get("road_id"),
                    "original_idx": road_meta.get("original_idx"),
                    "coords": coords if isinstance(coords, list) else coords.tolist(),
                    "num_points": len(coords),
                    "junction_start_id": start_jid,
                    "junction_end_id": end_jid,
                    "junction_buffer_start": road_meta.get("junction_buffer_start"),
                    "junction_buffer_end": road_meta.get("junction_buffer_end"),
                }
            )

            def _add_conn(jid, conn_type):
                if jid is None:
                    return
                data = junction_connections.setdefault(jid, {"road_indices": [], "connection_types": {}})
                if dump_idx not in data["road_indices"]:
                    data["road_indices"].append(dump_idx)
                data["connection_types"].setdefault(dump_idx, []).append(conn_type)

            _add_conn(start_jid, "start")
            _add_conn(end_jid, "end")

    junction_dump = []
    final_stats = {"two": 0, "three": 0, "four": 0, "five_plus": 0}

    for j_idx, j in enumerate(junctions):
        conn_data = junction_connections.get(j_idx, {})
        road_idxs = sorted(conn_data.get("road_indices", []))
        connection_types = conn_data.get("connection_types", {})

        junction_dump.append(
            {
                "position": list(j.get("position", (0, 0, 0))),
                "road_indices": road_idxs,
                "connection_types": connection_types,
                "num_connections": len(road_idxs),
            }
        )

        n = len(road_idxs)
        if n == 2:
            final_stats["two"] += 1
        elif n == 3:
            final_stats["three"] += 1
        elif n == 4:
            final_stats["four"] += 1
        elif n >= 5:
            final_stats["five_plus"] += 1

    if config.DEBUG_EXPORTS:
        with open(debug_network_path, "w", encoding="utf-8") as f:
            json.dump({"roads": trimmed_roads_data, "junctions": junction_dump}, f, indent=2)
        print(f"  [Debug] Netz-Daten exportiert: {debug_network_path}")
    if config.DEBUG_VERBOSE:
        print(
            "  [i] Finaler Junction-Status (nach Trimming): "
            f"2er: {final_stats['two']}, 3er: {final_stats['three']}, "
            f"4er: {final_stats['four']}, 5+: {final_stats['five_plus']}"
        )

    timer.end()

    # ===== SCHRITT 8: Klassifiziere Grid-Vertices =====
    timer.begin("Klassifiziere Grid-Vertices")

    # Bereite Eingabedaten für classify_grid_vertices vor:
    # 1. Regular Roads: Kombiniere road_slope_polygons_2d (enthält trimmed_centerline) mit XY-Polygonen
    # 2. Junction Fans: Nur XY-Polygone (keine Centerlines, da Fans keine Centerlines haben)
    road_data_for_classification = []

    # Regular Roads (haben Centerlines für KDTree-Sampling)
    for idx, meta in enumerate(road_slope_polygons_2d):
        # Hole XY-Polygon aus der Liste (gleicher Index wie road_slope_polygons_2d)
        road_polygon_xy = all_road_polygons_2d[idx] if idx < len(all_road_polygons_2d) else []

        # trimmed_centerline enthält finale Centerline (nach Splitting + Trimming)
        trimmed_centerline = meta.get("trimmed_centerline", [])

        road_data_for_classification.append(
            {
                "road_id": meta.get("road_id"),
                "road_polygon": np.array(road_polygon_xy) if road_polygon_xy else np.array([]),
                "slope_polygon": (
                    np.array(road_polygon_xy) if road_polygon_xy else np.array([])
                ),  # Identisch zu road_polygon
                "trimmed_centerline": trimmed_centerline,  # Finale Centerline für KDTree-Sampling
                "osm_tags": meta.get("osm_tags", {}),  # OSM-Tags für Stitching-Parameter-Skalierung
            }
        )

    # Junction Fans (am Ende von all_road_polygons_2d angehängt, haben KEINE Centerlines)
    num_regular_roads = len(road_slope_polygons_2d)
    for idx in range(num_regular_roads, len(all_road_polygons_2d)):
        fan_polygon_xy = all_road_polygons_2d[idx]

        road_data_for_classification.append(
            {
                "road_id": None,
                "road_polygon": np.array(fan_polygon_xy) if len(fan_polygon_xy) > 0 else np.array([]),
                "slope_polygon": np.array(fan_polygon_xy) if len(fan_polygon_xy) > 0 else np.array([]),
                "trimmed_centerline": [],  # Keine Centerline - Fan wird nur per Polygon-Test markiert
            }
        )

    vertex_types, modified_heights = classify_grid_vertices(
        grid_points,
        grid_elevations,
        road_data_for_classification,
    )

    # ===== SCHRITT 9a: Regeneriere Terrain-Mesh (mit Straßenausschnitten) =====
    timer.begin("Regeneriere Terrain-Mesh")
    terrain_faces_final, terrain_vertex_indices = generate_full_grid_mesh(
        grid_points,
        modified_heights,
        vertex_types,
        nx,
        ny,
        vertex_manager,
        dedup=False,
    )
    if config.DEBUG_VERBOSE:
        print(f"  [OK] {vertex_manager.get_count()} Vertices final (gesamt)")
        print(f"  [OK] {len(terrain_faces_final)} Terrain-Faces generiert")

    # ===== SCHRITT 9b: Füge Terrain-Faces zum Mesh hinzu =====
    for face in terrain_faces_final:
        mesh.add_face(face[0], face[1], face[2], material="terrain")

    # ===== SCHRITT 9c: Face-Klassifizierung abgeschlossen =====
    # (Road-Polygone wurden bereits in Schritt 8 verarbeitet)
    if config.DEBUG_VERBOSE:
        print(f"  [OK] {len(all_road_polygons_2d)} Road-Polygone klassifiziert")

    # Sammle Junction-Punkte mit Informationen über ankommende Straßen für dynamische Search-Radien
    junction_points = []
    for j_idx, j in enumerate(junctions):
        pos = j.get("position")
        if pos is None or len(pos) < 2:
            continue
        jid = j.get("id", j_idx)

        # Hole ankommende Straßen-Indices und sammle deren OSM-Tags
        road_indices = j.get("road_indices", [])
        connected_road_tags = []
        for r_idx in road_indices:
            # road_slope_polygons_2d enthält die osm_tags
            if r_idx < len(road_slope_polygons_2d):
                osm_tags = road_slope_polygons_2d[r_idx].get("osm_tags", {})
                connected_road_tags.append(osm_tags)

        junction_points.append(
            {
                "id": jid,
                "position": np.asarray(pos, dtype=float),
                "connected_road_tags": connected_road_tags,  # Liste von OSM-Tag-Dicts
            }
        )

    # ===== SCHRITT 9d: Stitching - Suche und Fülle Terrain-Lücken =====
    timer.begin("Stitching - Terrain-Lücken")
    stitch_faces = stitch_all_gaps(
        road_data_for_classification=road_data_for_classification,
        vertex_manager=vertex_manager,
        mesh=mesh,
        terrain_vertex_indices=terrain_vertex_indices,
        junction_points=junction_points,
        filter_road_id=stitch_filter_road_id,
        filter_junction_id=stitch_filter_junction_id,
    )
    timer.end()

    # Stitch-Faces werden nun direkt in mesh.add_face() eingefügt (in _triangulate_polygons)
    # Keine weitere Verarbeitung nötig

    # ===== SCHRITT 10: Slice Mesh in Tiles und exportiere als DAE =====
    timer.begin(f"Slice Mesh in {config.TILE_SIZE}×{config.TILE_SIZE}m Tiles")

    # Nur wenn Mesh generiert wurde
    if mesh.get_face_count() > 0:
        # Gebe Faces mit Materialien zurück
        all_faces_0based, materials_per_face = mesh.get_faces_with_materials()
        all_vertices_combined = mesh.get_vertices()

        # Slice in Tiles
        tiles_dict = slice_mesh_into_tiles(
            vertices=all_vertices_combined,
            faces=all_faces_0based.tolist(),
            materials_per_face=materials_per_face,
            tile_size=config.TILE_SIZE,
        )

        # Exportiere ALLE Tiles in EINE .dae Datei nach BEAMNG_DIR_SHAPES
        dae_output_path = os.path.join(config.BEAMNG_DIR_SHAPES, "terrain.dae")
        export_merged_dae(
            tiles_dict=tiles_dict,
            output_path=dae_output_path,
        )
    else:
        print(f"  [i] Kein Mesh generiert - überspringe DAE-Export")

    # ===== ZUSAMMENFASSUNG =====
    print(f"\n{'=' * 60}")
    print(f"[OK] GENERATOR BEENDET!")
    print(f"{'=' * 60}")
    print(f"  Output-Format: DAE-Tiles (tiles_dae/)")
    if config.LOCAL_OFFSET:
        print(
            f"  Lokaler Offset: X={config.LOCAL_OFFSET[0]:.2f}m, Y={config.LOCAL_OFFSET[1]:.2f}m, Z={config.LOCAL_OFFSET[2]:.2f}m"
        )

    timer.report()


if __name__ == "__main__":
    main()
