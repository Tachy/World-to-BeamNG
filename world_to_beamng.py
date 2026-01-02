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
from world_to_beamng.mesh.mesh import Mesh
from world_to_beamng.mesh.terrain_mesh import generate_full_grid_mesh
from world_to_beamng.mesh.terrain_mesh_conforming import (
    generate_full_grid_mesh_conforming,
)
from world_to_beamng.mesh.grid_conforming import (
    adjust_grid_to_roads_simple,
    classify_face_by_center,
)
from world_to_beamng.mesh.cleanup import (
    cleanup_duplicate_faces,
    enforce_ccw_up,
    report_boundary_edges,
)
from world_to_beamng.mesh.stitching import (
    stitch_terrain_gaps,
)
from world_to_beamng.io.aerial import process_aerial_images
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
        "--remesh-debug-dump",
        action="store_true",
        help="Remesh-Debugdaten (remesh_debug_data.json) schreiben",
    )
    args = parser.parse_args()

    debug_junction_id = args.junction_id
    if debug_junction_id is not None:
        print(f"[DEBUG] Junction-Remeshing nur für Junction #{debug_junction_id}")
    remesh_debug_dump = args.remesh_debug_dump

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

    timer.begin("Extrahiere Strassen aus OSM")
    roads = extract_roads_from_osm(osm_elements)
    if not roads:
        print("Keine Strassen gefunden.")
        return

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

    # ===== SCHRITT 6a: Erkenne Junctions in Centerlines (NUR mit Centerlines!) =====
    timer.begin("Erkenne Junctions in Centerlines")
    junctions = detect_junctions_in_centerlines(road_polygons)
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
        slope_faces,
        road_slope_polygons_2d,
        original_to_mesh_idx,
        all_road_polygons_2d,
        all_slope_polygons_2d,
    ) = generate_road_mesh_strips(
        road_polygons, height_points, height_elevations, vertex_manager
    )

    # Füge Straßen-Faces zum Mesh hinzu
    print(f"  [OK] {len(road_faces)} Strassen-Faces generiert")
    for face in road_faces:
        mesh.add_face(face[0], face[1], face[2], material="road")

    timer.end()

    # ===== SCHRITT 8: Klassifiziere Grid-Vertices =====
    timer.begin("Klassifiziere Grid-Vertices")
    vertex_types, modified_heights = classify_grid_vertices(
        grid_points, grid_elevations, road_slope_polygons_2d
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
    print(f"  [OK] {vertex_manager.get_count()} Vertices final (gesamt)")
    print(f"  [OK] {len(terrain_faces_final)} Terrain-Faces generiert")

    # ===== SCHRITT 9b: Füge Terrain-Faces zum Mesh hinzu =====
    for face in terrain_faces_final:
        mesh.add_face(face[0], face[1], face[2], material="terrain")

    # ===== SCHRITT 9c: Vorbereitung für Face-Klassifizierung =====
    # (Road-Polygone sind bereits vorhanden)
    all_road_and_slope_polygons = all_road_polygons_2d + all_slope_polygons_2d
    print(
        f"  [OK] {len(all_road_polygons_2d)} Road-Polygone + {len(all_slope_polygons_2d)} Slope-Polygone klassifiziert"
    )

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

    # HINWEIS: Luftbild-Verarbeitung NACH Schritt "Erkenne Junctions" (Schritt 6a)
    timer.begin("Verarbeite Luftbilder")
    # tile_size in Pixeln für 500m Tiles:
    # 5000 Pixel = 1000m → 0.2m/Pixel
    # 500m ÷ 0.2m = 2500 Pixel
    tile_count = process_aerial_images(
        aerial_dir="aerial", output_dir=config.BEAMNG_DIR_TEXTURES, tile_size=2500
    )
    if tile_count > 0:
        print(f"  [OK] {tile_count} Luftbild-Kacheln exportiert")
    else:
        print("  [i] Keine Luftbilder verarbeitet")

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
