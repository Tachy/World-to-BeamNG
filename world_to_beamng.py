"""
WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator

Refactored Version mit modularer Architektur.
Main Entry Point für die Anwendung.

Benötigte Pakete:
  pip install requests numpy scipy pyproj pyvista shapely rtree

Alle Abhängigkeiten sind ERFORDERLICH - kein Fallback!
"""

import time
import os
import glob
import gc
import numpy as np

# Importiere alle Module
from world_to_beamng import config
from world_to_beamng.terrain.elevation import load_height_data, get_height_data_hash
from world_to_beamng.terrain.grid import create_terrain_grid
from world_to_beamng.osm.parser import (
    calculate_bbox_from_height_data,
    extract_roads_from_osm,
)
from world_to_beamng.osm.downloader import get_osm_data
from world_to_beamng.geometry.polygon import get_road_polygons
from world_to_beamng.geometry.vertices import classify_grid_vertices
from world_to_beamng.mesh.road_mesh import generate_road_mesh_strips
from world_to_beamng.mesh.terrain_mesh import generate_full_grid_mesh
from world_to_beamng.mesh.overlap import check_face_overlaps
from world_to_beamng.io.obj import create_pyvista_mesh, save_unified_obj, save_roads_obj


def main():
    """Hauptfunktion der Anwendung - koordiniert alle Module."""

    # Reset globale Zustände
    config.LOCAL_OFFSET = None
    config.BBOX = None
    config.GRID_BOUNDS_UTM = None

    start_time = time.time()
    timings = {}

    # ===== SCHRITT 1: Lade Höhendaten =====
    print("=" * 60)
    print("WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator")
    print("=" * 60)

    print("\n[1] Lade Höhendaten...")
    step_start = time.time()
    height_points, height_elevations = load_height_data()
    timings["1_Höhendaten_laden"] = time.time() - step_start

    # ===== SCHRITT 2: Berechne BBOX =====
    print("\n[2] Berechne BBOX aus Höhendaten...")
    step_start = time.time()
    config.BBOX = calculate_bbox_from_height_data(height_points)
    timings["2_BBOX_berechnen"] = time.time() - step_start

    # ===== SCHRITT 3: Prüfe OSM-Daten-Cache =====
    print("\n[3] Prüfe OSM-Daten-Cache...")
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
            print("  ⚠ Höhendaten haben sich geändert - lade OSM-Daten neu")
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
                print(f"  Alter Cache gelöscht: {os.path.basename(cache)}")
            except:
                pass

    osm_elements = get_osm_data(config.BBOX)
    timings["3_OSM_Daten_holen"] = time.time() - step_start
    if not osm_elements:
        print("Keine Daten gefunden.")
        return

    # ===== SCHRITT 4: Extrahiere Straßen =====
    print("\n[4] Extrahiere Straßen aus OSM-Daten...")
    step_start = time.time()
    roads = extract_roads_from_osm(osm_elements)
    timings["4_Straßen_extrahieren"] = time.time() - step_start
    if not roads:
        print("Keine Straßen gefunden.")
        return

    # ===== SCHRITT 5: Erstelle Terrain-Grid =====
    print("\n[5] Erstelle Terrain-Grid...")
    step_start = time.time()
    grid_points, grid_elevations, nx, ny = create_terrain_grid(
        height_points, height_elevations, grid_spacing=config.GRID_SPACING
    )
    timings["5_Grid_erstellen"] = time.time() - step_start

    # ===== SCHRITT 6: Extrahiere Straßen-Polygone =====
    step_start = time.time()
    print(f"\n[6] Extrahiere {len(roads)} Straßen-Polygone...")
    road_polygons = get_road_polygons(
        roads, config.BBOX, height_points, height_elevations
    )
    print(f"  ✓ {len(road_polygons)} Straßen-Polygone extrahiert")
    timings["6_Straßen_Polygone"] = time.time() - step_start

    # ===== SCHRITT 7: Generiere Straßen-Mesh =====
    print("\n[7] Generiere Straßen-Mesh-Streifen...")
    step_start = time.time()
    (
        road_vertices,
        road_faces,
        road_face_to_idx,
        slope_vertices,
        slope_faces_strips,
        road_slope_polygons_2d,
    ) = generate_road_mesh_strips(road_polygons, height_points, height_elevations)
    print(
        f"  ✓ {len(road_slope_polygons_2d)} 2D-Polygone für Grid-Klassifizierung extrahiert"
    )
    timings["7_Straßen_Mesh"] = time.time() - step_start

    # ===== SCHRITT 8: Klassifiziere Grid-Vertices =====
    print("\n[8] Klassifiziere Grid-Vertices...")
    step_start = time.time()
    vertex_types, modified_heights = classify_grid_vertices(
        grid_points, grid_elevations, road_slope_polygons_2d
    )
    timings["8_Vertex_Klassifizierung"] = time.time() - step_start

    # ===== SCHRITT 9: Generiere Terrain-Grid-Mesh =====
    step_start = time.time()
    print("\n[9] Generiere Terrain-Grid-Mesh...")
    grid_vertices, _, _, terrain_faces = generate_full_grid_mesh(
        grid_points,
        modified_heights,
        vertex_types,
        nx,
        ny,
    )
    timings["9_Terrain_Grid_Generierung"] = time.time() - step_start

    # ===== SCHRITT 10: Kombiniere Meshes =====
    step_start = time.time()
    print("\n[10] Kombiniere Straßen-Mesh, Böschungs-Mesh und Terrain-Mesh...")

    terrain_vertex_count = len(grid_vertices)
    road_vertex_count = len(road_vertices)
    slope_vertex_count = len(slope_vertices)

    road_faces_offset = [
        [idx + terrain_vertex_count for idx in face] for face in road_faces
    ]
    slope_faces_offset = [
        [idx + terrain_vertex_count + road_vertex_count for idx in face]
        for face in slope_faces_strips
    ]

    all_vertices = grid_vertices + road_vertices + slope_vertices

    print(f"\n  Kombiniere Vertex-Daten...")
    print(f"    • Terrain: {terrain_vertex_count} Vertices")
    print(f"    • Straßen: {road_vertex_count} Vertices")
    print(f"    • Böschungen: {slope_vertex_count} Vertices")
    print(f"    ✓ Total: {len(all_vertices)} Vertices")

    combined_road_faces = road_faces_offset
    combined_slope_faces = slope_faces_offset
    combined_terrain_faces = terrain_faces

    print(f"  ✓ Kombiniert: {len(all_vertices)} Vertices total")
    print(f"    • Terrain: {terrain_vertex_count} Vertices, {len(terrain_faces)} Faces")
    print(
        f"    • Straßen: {road_vertex_count} Vertices, {len(combined_road_faces)} Faces"
    )
    print(
        f"    • Böschungen: {slope_vertex_count} Vertices, {len(combined_slope_faces)} Faces"
    )

    timings["10_Mesh_Kombination"] = time.time() - step_start

    # ===== SCHRITT 11: Vereinfache Terrain =====
    step_start = time.time()
    print("\n[11] Vereinfache Terrain...")

    terrain_mesh = create_pyvista_mesh(grid_vertices, combined_terrain_faces)

    terrain_vertices_original = grid_vertices
    terrain_faces_original = combined_terrain_faces
    road_vertices_original = road_vertices
    slope_vertices_original = slope_vertices
    road_faces_original = combined_road_faces
    slope_faces_original = combined_slope_faces
    original_terrain_vertex_count = len(grid_vertices)

    del all_vertices
    gc.collect()

    if config.TERRAIN_REDUCTION > 0:
        print(f"  ({config.TERRAIN_REDUCTION * 100:.0f}% Reduktion)...")
        original_points = terrain_mesh.n_points
        terrain_simplified = terrain_mesh.decimate_pro(
            reduction=config.TERRAIN_REDUCTION,
            feature_angle=25.0,
            preserve_topology=True,
            boundary_vertex_deletion=False,
        )
        print(f"    ✓ {original_points:,} → {terrain_simplified.n_points:,} Vertices")
        del terrain_mesh
    else:
        print("\n[11] Überspringe Terrain-Simplification (TERRAIN_REDUCTION = 0)...")
        terrain_simplified = None
        del terrain_mesh

    gc.collect()
    timings["11_PyVista_Simplification"] = time.time() - step_start

    # ===== SCHRITT 12: Kombiniere finale Vertices =====
    step_start = time.time()
    print("\n[12] Kombiniere Vertices manuell (VEKTORISIERT)...")

    if terrain_simplified is not None:
        terrain_vertices_decimated = terrain_simplified.points
    else:
        terrain_vertices_decimated = np.array(
            terrain_vertices_original, dtype=np.float32
        )

    terrain_vertex_count = len(terrain_vertices_decimated)
    road_vertex_count = len(road_vertices_original)
    slope_vertex_count = len(slope_vertices_original)

    terrain_array = np.array(terrain_vertices_decimated, dtype=np.float32)
    road_array = np.array(road_vertices_original, dtype=np.float32)
    slope_array = np.array(slope_vertices_original, dtype=np.float32)

    all_vertices_combined = np.vstack([terrain_array, road_array, slope_array])
    total_vertex_count = len(all_vertices_combined)

    print(f"    • Terrain: {terrain_vertex_count:,} Vertices")
    print(f"    • Straßen: {road_vertex_count:,} Vertices")
    print(f"    • Böschungen: {slope_vertex_count:,} Vertices")
    print(f"    ✓ Gesamt: {total_vertex_count:,} Vertices")

    terrain_vertices_decimated_count = len(terrain_vertices_decimated)

    del (
        road_vertices_original,
        slope_vertices_original,
        terrain_array,
        road_array,
        slope_array,
    )
    gc.collect()

    timings["12_Vertices_Kombinieren"] = time.time() - step_start

    # ===== SCHRITT 13: Extrahiere Terrain-Faces =====
    step_start = time.time()
    if terrain_simplified is not None:
        print("  Extrahiere Terrain-Faces aus PyVista (DIREKT)...")
        faces_raw = terrain_simplified.faces
        if len(faces_raw) > 0:
            terrain_faces_decimated = faces_raw.reshape(-1, 4)[:, 1:].astype(np.int32)
        else:
            terrain_faces_decimated = np.array([], dtype=np.int32).reshape(0, 3)
    else:
        print("  Verwende originale Terrain-Faces (TERRAIN_REDUCTION=0)...")
        terrain_faces_decimated = np.array(terrain_faces_original, dtype=np.int32) - 1

    timings["13_Terrain_Faces_Extrahieren"] = time.time() - step_start

    # ===== SCHRITT 14: Bereite Faces für Export vor =====
    step_start = time.time()
    print("  Bereite Faces für OBJ-Export vor (VEKTORISIERT)...")

    # Konvertiere zu NumPy Array wenn nötig (terrain_faces kann Liste oder 1D Array sein)
    if isinstance(terrain_faces_decimated, list):
        terrain_faces_decimated = np.array(terrain_faces_decimated, dtype=np.int32)
    elif terrain_faces_decimated.ndim == 1:
        # 1D Array zu 2D reshape (Triplets)
        terrain_faces_decimated = terrain_faces_decimated.reshape(-1, 3).astype(
            np.int32
        )

    terrain_faces_final = terrain_faces_decimated + 1

    # FILTER 1: Entferne Terrain-Faces mit gelöschten Vertices
    print(f"  Filtere Terrain-Faces mit gelöschten Vertices...")
    vertex_keep_mask = vertex_types == 0

    face_has_valid_vertices = (
        vertex_keep_mask[terrain_faces_decimated[:, 0]]
        & vertex_keep_mask[terrain_faces_decimated[:, 1]]
        & vertex_keep_mask[terrain_faces_decimated[:, 2]]
    )

    terrain_faces_final = terrain_faces_final[face_has_valid_vertices]
    deleted_vertex_faces = np.sum(~face_has_valid_vertices)
    print(f"    • {deleted_vertex_faces:,} Faces entfernt (haben gelöschte Vertices)")

    print(f"    • Terrain: {len(terrain_faces_final):,} Faces")
    road_faces_array = np.array(road_faces_original, dtype=np.int32)
    slope_faces_array = np.array(slope_faces_original, dtype=np.int32)

    road_faces_final = road_faces_array + 1
    slope_faces_final = slope_faces_array + 1

    print(f"    • Terrain: {len(terrain_faces_final):,} Faces")
    print(f"    • Straßen: {len(road_faces_final):,} Faces")
    print(f"    • Böschungen: {len(slope_faces_final):,} Faces")

    # DEBUG: Prüfe Böschungen
    if len(slope_faces_final) == 0:
        print("    ⚠ WARNUNG: KEINE Böschungs-Faces generiert!")
    else:
        print(
            f"    ✓ Böschungen OK: {slope_vertex_count:,} Vertices, {len(slope_faces_final):,} Faces"
        )

    timings["14_Faces_Vorbereiten"] = time.time() - step_start

    # ===== SCHRITT 15: Korrigiere Face-Indices =====
    step_start = time.time()
    offset_diff = terrain_vertices_decimated_count - original_terrain_vertex_count

    if offset_diff != 0:
        road_faces_final = road_faces_final - 1
        mask = road_faces_final >= original_terrain_vertex_count
        road_faces_final[mask] += offset_diff
        road_faces_final = road_faces_final + 1

        slope_offset_original = original_terrain_vertex_count + road_vertex_count
        slope_offset_final = terrain_vertices_decimated_count + road_vertex_count
        slope_offset_diff = slope_offset_final - slope_offset_original

        slope_faces_final = slope_faces_final - 1
        mask = slope_faces_final >= slope_offset_original
        slope_faces_final[mask] += slope_offset_diff
        slope_faces_final = slope_faces_final + 1

    del (
        terrain_simplified,
        terrain_faces_decimated,
        road_faces_original,
        slope_faces_original,
    )
    gc.collect()
    print("\n  ✓ PyVista-Mesh und temporäre Listen aus Speicher entfernt")

    timings["15_Face_Index_Korrektur"] = time.time() - step_start

    # ===== SCHRITT 16: Exportiere Meshes als OBJ =====
    print("\n[16] Exportiere Meshes als OBJ...")
    step_start = time.time()
    output_obj = "beamng.obj"
    print(f"  Schreibe: {output_obj}")

    # Extra: roads.obj mit pro-road Material für Debug-Viewer
    print(f"  Schreibe roads.obj (nur Straßen, pro road_idx Material)...")
    export_start = time.time()
    save_roads_obj("roads.obj", road_vertices, road_faces, road_face_to_idx)
    print(f"    → roads.obj: {time.time() - export_start:.2f}s")

    export_start = time.time()
    save_unified_obj(
        output_obj,
        all_vertices_combined,
        road_faces_final,
        slope_faces_final,
        terrain_faces_final,
    )
    print(f"    → save_unified_obj(): {time.time() - export_start:.2f}s")

    # DEBUG: Exportiere nur Terrain zum Debuggen
    print(f"  DEBUG: Schreibe nur Terrain zu beamng_terrain_only.obj...")
    export_start = time.time()
    save_unified_obj(
        "beamng_terrain_only.obj",
        all_vertices_combined,
        [],  # Keine Straßen
        [],  # Keine Böschungen
        terrain_faces_final,  # Nur Terrain
    )
    print(f"    → save_unified_obj() (nur Terrain): {time.time() - export_start:.2f}s")

    cleanup_start = time.time()
    del (
        all_vertices_combined,
        terrain_faces_final,
        slope_faces_final,
        road_faces_final,
    )
    gc.collect()
    print(f"    → Cleanup + GC: {time.time() - cleanup_start:.2f}s")

    timings["16_Mesh_Export"] = time.time() - step_start

    # ===== ZUSAMMENFASSUNG =====
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print(f"✓ GENERATOR BEENDET!")
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
