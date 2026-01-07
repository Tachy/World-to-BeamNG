"""
Multi-Tile Verarbeitungs-Funktionen.

Diese Funktionen implementieren die 3-Phasen-Architektur für Multi-Tile-Verarbeitung:
- Phase 1: Pre-Scan und Initialisierung
- Phase 2: Pro-Tile Verarbeitung (in Schleife)
- Phase 3: Post-Merge und Finalisierung
"""

import os
import json
import zipfile
import io
from pathlib import Path

import numpy as np

from world_to_beamng import config
from world_to_beamng.utils.tile_scanner import scan_lgl_tiles, compute_global_bbox, compute_global_center
from world_to_beamng.io.cache import load_height_hashes, save_height_hashes, calculate_file_hash
from world_to_beamng.io.materials_merge import merge_materials_json, merge_items_json, save_materials_json, save_items_json
from world_to_beamng.osm.parser import calculate_bbox_from_height_data, extract_roads_from_osm
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
from world_to_beamng.mesh.stitch_gaps import stitch_all_gaps
from world_to_beamng.mesh.tile_slicer import slice_mesh_into_tiles
from world_to_beamng.mesh.cleanup import remove_road_faces_outside_bounds
from world_to_beamng.io.dae import (
    export_merged_dae,
    create_terrain_materials_json,
    create_terrain_items_json,
)
from world_to_beamng.io.lod2 import (
    cache_lod2_buildings,
    export_buildings_to_dae,
    export_materials_json as export_lod2_materials_json,
    create_items_json_entry as create_lod2_items_entry,
)
from world_to_beamng.terrain.grid import create_terrain_grid
from world_to_beamng.utils.timing import StepTimer


def _expand_bbox(bbox, margin):
    """Erweitert eine BBox um einen gegebenen Rand in Metern."""
    if not bbox:
        return bbox
    min_x, max_x, min_y, max_y = bbox
    return (min_x - margin, max_x + margin, min_y - margin, max_y + margin)


def _load_tile_height_data(tile):
    """Laedt die Hoehendaten einer einzelnen DGM1-Kachel.
    
    WICHTIG: Ein LGL DGM1-ZIP enthält 4 XYZ-Dateien (2×2 Kacheln à 1000×1000m).
    Diese werden zu einem 2000×2000m Gebiet zusammengesetzt.
    """

    filepath = tile.get("filepath")
    if not filepath or not os.path.exists(filepath):
        print(f"  [!] DGM1-Datei fehlt: {filepath}")
        return None, None

    all_data = []
    
    if filepath.lower().endswith(".xyz"):
        # Einzelne XYZ-Datei
        data = np.loadtxt(filepath)
        all_data.append(data)
        
    elif filepath.lower().endswith(".zip"):
        # ZIP mit mehreren XYZ-Dateien (typisch: 4 Kacheln)
        with zipfile.ZipFile(filepath, "r") as zf:
            xyz_members = [name for name in zf.namelist() if name.lower().endswith(".xyz")]
            if not xyz_members:
                print(f"  [!] Keine .xyz in {os.path.basename(filepath)} gefunden")
                return None, None
            
            print(f"  [i] Lade {len(xyz_members)} XYZ-Dateien aus ZIP...")
            for xyz_name in sorted(xyz_members):
                with zf.open(xyz_name) as f:
                    data = np.loadtxt(io.TextIOWrapper(f, encoding="utf-8"))
                    all_data.append(data)
                    print(f"    - {xyz_name}: {len(data)} Punkte")
    else:
        print(f"  [!] Unbekanntes DGM1-Format: {filepath}")
        return None, None

    # Kombiniere alle Daten
    if len(all_data) == 0:
        return None, None
    
    combined_data = np.vstack(all_data)
    
    if combined_data.ndim != 2 or combined_data.shape[1] < 3:
        print(f"  [!] DGM1-Daten ungültig: {filepath}")
        return None, None

    print(f"  [OK] {len(combined_data)} Höhenpunkte geladen (aus {len(all_data)} Datei(en))")
    return combined_data[:, :2].copy(), combined_data[:, 2].copy()


def _ensure_local_offset(global_offset, height_points, height_elevations):
    """Setzt config.LOCAL_OFFSET falls noch nicht vorhanden."""

    if config.LOCAL_OFFSET is not None:
        return config.LOCAL_OFFSET

    cx, cy = (global_offset[:2] if global_offset else (height_points[0, 0], height_points[0, 1]))
    cz = None
    if global_offset and len(global_offset) >= 3 and global_offset[2] is not None:
        cz = global_offset[2]
    if cz is None:
        cz = float(height_elevations.min())

    config.LOCAL_OFFSET = (float(cx), float(cy), float(cz))
    print(f"  [OK] LOCAL_OFFSET gesetzt: {config.LOCAL_OFFSET}")
    return config.LOCAL_OFFSET


def phase1_multitile_init(dgm1_dir="data/DGM1"):
    """
    PHASE 1: Pre-Scan aller Tiles und Initialisierung.
    
    Returns:
        Tuple: (tiles, global_offset)
            - tiles: Liste von Tile-Metadaten
            - global_offset: (center_x, center_y, z_min) für LOCAL_OFFSET
    """
    print("\n" + "=" * 60)
    print("[PHASE 1] Multi-Tile Pre-Scan & Initialisierung")
    print("=" * 60)
    
    # Scanne DGM1-Verzeichnis
    print(f"\nScanne DGM1-Kacheln in {dgm1_dir}...")
    tiles = scan_lgl_tiles(dgm1_dir)
    
    if not tiles:
        print("[!] Keine DGM1-Kacheln gefunden - Fallback auf Single-Tile Mode")
        return None, None
    
    print(f"[OK] {len(tiles)} DGM1-Kachel(n) gefunden")
    
    # Berechne globale Bounding Box
    global_bbox = compute_global_bbox(tiles)
    global_center = compute_global_center(tiles)
    
    print(f"[OK] Globale BBox (UTM): min_x={global_bbox[0]:.1f}, max_x={global_bbox[1]:.1f}, "
          f"min_y={global_bbox[2]:.1f}, max_y={global_bbox[3]:.1f}")
    print(f"[OK] Globaler Center (UTM): ({global_center[0]:.1f}, {global_center[1]:.1f})")
    
    # Lade existierende Height-Hashes
    print("\nLade Height-Data-Hashes...")
    height_hashes = load_height_hashes()
    if height_hashes:
        print(f"  [OK] {len(height_hashes)} Hashes geladen")
    else:
        print("  [i] Keine bestehenden Hashes gefunden (Neustart)")
    
    # Prüfe welche Tiles geändert haben
    changed_tiles = []
    unchanged_tiles = []
    
    print("\nPrüfe Tile-Änderungen...")
    for tile in tiles:
        filepath = tile['filepath']
        filename = tile['filename']
        
        if not os.path.exists(filepath):
            print(f"  [!] Datei nicht gefunden: {filename}")
            continue
        
        # Berechne Hash
        file_hash = calculate_file_hash(filepath)
        if file_hash is None:
            changed_tiles.append(tile)
            print(f"  [?] {filename} - Hash-Fehler, verwende als geändert")
            continue
        
        # Vergleiche mit gespeichertem Hash
        if filename in height_hashes and height_hashes[filename] == file_hash:
            unchanged_tiles.append(tile)
            print(f"  [✓] {filename} - Unverändert (Cache gültig)")
        else:
            changed_tiles.append(tile)
            height_hashes[filename] = file_hash
            print(f"  [!] {filename} - Geändert (Cache invalidiert)")
    
    # Speichere aktualisierte Hashes
    if changed_tiles or not height_hashes:
        print("\nSpeichere Height-Data-Hashes...")
        save_height_hashes(height_hashes)
    
    # Löschen alte Materials/Items für Fresh Start
    materials_path = os.path.join(config.BEAMNG_DIR, "main.materials.json")
    items_path = os.path.join(config.BEAMNG_DIR, "main.items.json")
    
    print(f"\nInitialisiere Materials & Items...")
    for path in [materials_path, items_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  [OK] Gelöschte alte Datei: {os.path.basename(path)}")
    
    # Globaler Offset: Nutze den Center des Grids
    # Später wird dieser beim Laden der Höhendaten verfeinert
    global_offset = (global_center[0], global_center[1], 0.0)  # Z wird später gesetzt
    
    # WICHTIG: Keine globalen BBOXen in config speichern!
    # Diese werden pro Tile berechnet und als Parameter übergeben
    
    print(f"\n[OK] PHASE 1 abgeschlossen")
    print(f"[OK] {len(changed_tiles)} Tile(s) müssen verarbeitet werden")
    print(f"[OK] {len(unchanged_tiles)} Tile(s) sind unverändert (Cache nutzbar)")
    
    return tiles, global_offset


def phase2_process_tile(tile, global_offset=None, bbox_margin=50.0, buildings_data=None):
    """
    PHASE 2: Verarbeitet eine einzelne DGM1-Kachel end-to-end.

    Schritte:
    - Tile-Höhendaten laden und auf LOCAL_OFFSET normalisieren
    - OSM-Daten im Tile-BBox (mit Puffer) laden
    - Road-Mesh + Terrain-Mesh erzeugen
    - Slicing + DAE-Export
    - Materials/Items additiv mergen (add_new)
    """

    print(f"\n[PHASE 2] Verarbeite Tile: {tile.get('filename')}")

    timer = StepTimer()

    # === Höhendaten laden ===
    timer.begin("Lade Hoehendaten (Tile)")
    height_points_raw, height_elevations_raw = _load_tile_height_data(tile)
    if height_points_raw is None or height_elevations_raw is None:
        print("  [!] Abbruch: Keine Hoehendaten geladen")
        return None

    # === TILE-SPEZIFISCHER HASH (für alle Cache-Files: osm, lod2, elevations, grid) ===
    tile_hash = calculate_file_hash(tile.get("filepath")) or "no_hash"
    # Setze auch config.HEIGHT_HASH als Fallback für Code, der noch darauf zugreift
    config.HEIGHT_HASH = tile_hash

    # === BBox berechnen (Tile-spezifisch, NICHT in config!) ===
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    
    # UTM-BBox aus Höhendaten
    easting_min, easting_max = height_points_raw[:, 0].min(), height_points_raw[:, 0].max()
    northing_min, northing_max = height_points_raw[:, 1].min(), height_points_raw[:, 1].max()
    bbox_utm = (easting_min, easting_max, northing_min, northing_max)
    
    if bbox_margin > 0:
        bbox_utm = _expand_bbox(bbox_utm, bbox_margin)
    
    # Konvertiere zu lat/lon für OSM/LoD2
    lon_min, lat_min = transformer.transform(bbox_utm[0], bbox_utm[2])
    lon_max, lat_max = transformer.transform(bbox_utm[1], bbox_utm[3])
    bbox_latlon = (lat_min, lon_min, lat_max, lon_max)

    # LOCAL_OFFSET initialisieren (global fuer alle Tiles)
    local_offset = _ensure_local_offset(global_offset, height_points_raw, height_elevations_raw)
    ox, oy, oz = local_offset

    # Transformiere zu lokalen Koordinaten (global konsistent)
    height_points = height_points_raw.copy()
    height_points[:, 0] -= ox
    height_points[:, 1] -= oy
    height_elevations = height_elevations_raw - oz

    config.GRID_BOUNDS_LOCAL = (
        float(height_points[:, 0].min()),
        float(height_points[:, 0].max()),
        float(height_points[:, 1].min()),
        float(height_points[:, 1].max()),
    )
    timer.end()

    # === Terrain-Grid erzeugen ===
    timer.begin("Erstelle Terrain-Grid")
    grid_points, grid_elevations, nx, ny = create_terrain_grid(
        height_points,
        height_elevations,
        grid_spacing=config.GRID_SPACING,
        tile_hash=tile_hash,
    )
    timer.end()

    # === Luftbilder verarbeiten (falls vorhanden) ===
    timer.begin("Verarbeite Luftbilder")
    try:
        from world_to_beamng.io.aerial import process_aerial_images
        
        tile_count = process_aerial_images(
            aerial_dir="data/DOP20",
            output_dir=config.BEAMNG_DIR_TEXTURES,
            tile_size=2500
        )
        if tile_count > 0:
            print(f"  [OK] {tile_count} Luftbild-Kacheln exportiert")
        else:
            print("  [i] Keine Luftbilder verarbeitet")
    except Exception as e:
        print(f"  [i] Luftbild-Verarbeitung übersprungen: {e}")
    timer.end()

    # === LoD2-Gebäude laden (wenn aktiviert) ===
    if buildings_data is None and config.LOD2_ENABLED:
        timer.begin("Lade LoD2-Gebaeude")
        buildings_cache_path = cache_lod2_buildings(
            lod2_dir=config.LOD2_DATA_DIR,
            bbox=bbox_latlon,  # Tile-spezifische WGS84-BBox
            local_offset=(ox, oy),
            cache_dir=config.CACHE_DIR,
            height_hash=tile_hash,  # Tile-Hash
        )
        if buildings_cache_path:
            from world_to_beamng.io.lod2 import load_buildings_from_cache, snap_buildings_to_terrain_batch

            buildings_data = load_buildings_from_cache(buildings_cache_path)
            if buildings_data:
                print(f"  [OK] {len(buildings_data)} Gebaeude geladen")
                
                # Z-Koordinaten ans Terrain anpassen
                buildings_data = snap_buildings_to_terrain_batch(buildings_data, height_points, height_elevations)
                print(f"  [OK] Z-Koordinaten normalisiert ({len(buildings_data)} Gebaeude)")

        if not buildings_data:
            print("  [i] Keine LoD2-Gebaeude gefunden")
        timer.end()

    # === OSM-Daten laden ===
    timer.begin("Lade OSM-Daten")
    
    # Nutze Tile-spezifische lat/lon-BBox
    osm_elements = get_osm_data(bbox_latlon, height_hash=tile_hash)
    if not osm_elements:
        print("  [!] Keine OSM-Daten gefunden - Tile wird uebersprungen")
        return None

    roads = extract_roads_from_osm(osm_elements)
    if not roads:
        print("  [!] Keine Strassen gefunden - Tile wird uebersprungen")
        return None
    timer.end()

    # === Strassen-Polygone ===
    timer.begin("Erstelle Strassen-Polygone")
    road_polygons = get_road_polygons(roads, bbox_latlon, height_points, height_elevations, tile_hash=tile_hash)

    if config.ROAD_CLIP_MARGIN > 0:
        road_polygons = clip_road_polygons(road_polygons, config.GRID_BOUNDS_LOCAL, margin=config.ROAD_CLIP_MARGIN)

    timer.end()

    # === Junctions erkennen ===
    timer.begin("Erkenne Junctions")
    junctions = detect_junctions_in_centerlines(road_polygons, height_points, height_elevations)
    road_polygons, junctions = split_roads_at_mid_junctions(road_polygons, junctions)
    road_polygons = mark_junction_endpoints(road_polygons, junctions)
    junction_stats(junctions, road_polygons)
    timer.end()

    # === Vertex-Manager + Mesh ===
    vertex_manager = VertexManager(tolerance=0.01)
    mesh = Mesh(vertex_manager)

    # === Road-Mesh generieren ===
    timer.begin("Generiere Strassen-Mesh")
    (
        road_faces,
        road_face_to_idx,
        _slope_faces,
        road_slope_polygons_2d,
        original_to_mesh_idx,
        all_road_polygons_2d,
        _all_slope_polygons_2d,
    ) = generate_road_mesh_strips(road_polygons, height_points, height_elevations, vertex_manager, junctions)

    # Material pro Road bestimmen
    road_material_map = {}
    for poly in road_slope_polygons_2d:
        r_id = poly.get("road_id")
        if r_id is None:
            continue
        props = config.OSM_MAPPER.get_road_properties(poly.get("osm_tags", {}))
        mat_name = props.get("internal_name", "road_default")
        road_material_map[r_id] = (mat_name, props)

    default_props = config.OSM_MAPPER.get_road_properties({})
    default_mat = default_props.get("internal_name", "road_default")

    unique_materials = {}
    for idx, face in enumerate(road_faces):
        r_id = road_face_to_idx[idx] if idx < len(road_face_to_idx) else None
        if r_id is not None and r_id in road_material_map:
            mat_name, props = road_material_map[r_id]
        else:
            mat_name, props = default_mat, default_props

        unique_materials[mat_name] = props
        mesh.add_face(face[0], face[1], face[2], material=mat_name)

    # Materials fuer Roads vorbereiten
    road_material_entries = [config.OSM_MAPPER.generate_materials_json_entry(n, p) for n, p in unique_materials.items()]
    timer.end()

    # === Klassifiziere Grid-Vertices (Road/Terrain) ===
    timer.begin("Klassifiziere Grid-Vertices")
    road_data_for_classification = []
    for idx, meta in enumerate(road_slope_polygons_2d):
        road_polygon_xy = all_road_polygons_2d[idx] if idx < len(all_road_polygons_2d) else []
        road_data_for_classification.append(
            {
                "road_id": meta.get("road_id"),
                "road_polygon": np.array(road_polygon_xy) if road_polygon_xy else np.array([]),
                "slope_polygon": np.array(road_polygon_xy) if road_polygon_xy else np.array([]),
                "trimmed_centerline": meta.get("trimmed_centerline", []),
                "osm_tags": meta.get("osm_tags", {}),
            }
        )

    num_regular_roads = len(road_slope_polygons_2d)
    for idx in range(num_regular_roads, len(all_road_polygons_2d)):
        fan_polygon_xy = all_road_polygons_2d[idx]
        road_data_for_classification.append(
            {
                "road_id": None,
                "road_polygon": np.array(fan_polygon_xy) if len(fan_polygon_xy) > 0 else np.array([]),
                "slope_polygon": np.array(fan_polygon_xy) if len(fan_polygon_xy) > 0 else np.array([]),
                "trimmed_centerline": [],
            }
        )

    vertex_types, modified_heights = classify_grid_vertices(
        grid_points,
        grid_elevations,
        road_data_for_classification,
    )
    timer.end()

    # === Terrain-Mesh regenerieren ===
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

    for face in terrain_faces_final:
        mesh.add_face(face[0], face[1], face[2], material="terrain")
    timer.end()

    # === Cleanup + Stitching ===
    timer.begin("Cleanup & Stitching")
    remove_road_faces_outside_bounds(mesh, vertex_manager)

    junction_points = []
    for j_idx, j in enumerate(junctions):
        pos = j.get("position")
        if pos is None or len(pos) < 2:
            continue
        junction_points.append({"id": j.get("id", j_idx), "position": np.asarray(pos, dtype=float)})

    stitch_all_gaps(
        road_data_for_classification=road_data_for_classification,
        vertex_manager=vertex_manager,
        mesh=mesh,
        terrain_vertex_indices=terrain_vertex_indices,
        junction_points=junction_points,
        filter_road_id=None,
        filter_junction_id=None,
    )
    timer.end()

    # === Mesh slicen und exportieren ===
    timer.begin("Slice & Export DAE")
    dae_files = []
    materials_path = os.path.join(config.BEAMNG_DIR, "main.materials.json")
    items_path = os.path.join(config.BEAMNG_DIR, "main.items.json")

    if mesh.get_face_count() > 0:
        all_faces_0based, materials_per_face = mesh.get_faces_with_materials()
        all_vertices_combined = mesh.get_vertices()

        tiles_dict = slice_mesh_into_tiles(
            vertices=all_vertices_combined,
            faces=all_faces_0based.tolist(),
            materials_per_face=materials_per_face,
            tile_size=config.TILE_SIZE,
        )

        dae_output_path = os.path.join(config.BEAMNG_DIR_SHAPES, "terrain.dae")
        actual_dae_filename = export_merged_dae(
            tiles_dict=tiles_dict,
            output_path=dae_output_path,
            tile_size=config.TILE_SIZE,
        )
        dae_files.append(actual_dae_filename)

        # Materials (Terrain + Roads) additiv mergen
        terrain_materials = create_terrain_materials_json(
            tiles_dict=tiles_dict,
            level_name=config.LEVEL_NAME,
            tile_size=config.TILE_SIZE,
        )
        road_materials = {entry["name"]: entry for entry in road_material_entries}
        merged_materials = merge_materials_json(materials_path, {**terrain_materials, **road_materials}, mode="add_new")
        save_materials_json(materials_path, merged_materials)

        # Items (Terrain)
        terrain_item = create_terrain_items_json(dae_filename=actual_dae_filename)
        items_payload = {terrain_item["__name"]: terrain_item}

        # LoD2 Buildings (optional)
        if buildings_data and config.LOD2_ENABLED:
            from collections import defaultdict

            buildings_by_tile = defaultdict(list)
            for building in buildings_data:
                bounds = building.get("bounds")
                if not bounds:
                    continue
                center_x = (bounds[0] + bounds[3]) / 2
                center_y = (bounds[1] + bounds[4]) / 2
                tile_x = int((center_x // config.TILE_SIZE) * config.TILE_SIZE)
                tile_y = int((center_y // config.TILE_SIZE) * config.TILE_SIZE)
                buildings_by_tile[(tile_x, tile_y)].append(building)

            for (tile_x, tile_y), tile_buildings in buildings_by_tile.items():
                dae_path = export_buildings_to_dae(
                    buildings=tile_buildings,
                    output_dir=config.BEAMNG_DIR_BUILDINGS,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    wall_color=config.LOD2_WALL_COLOR,
                    roof_color=config.LOD2_ROOF_COLOR,
                )

                if dae_path:
                    dae_files.append(os.path.basename(dae_path))
                    item = create_lod2_items_entry(
                        dae_path=f"buildings/{os.path.basename(dae_path)}",
                        tile_x=tile_x,
                        tile_y=tile_y,
                    )
                    items_payload[item["__name"]] = item

            export_lod2_materials_json(output_dir=config.BEAMNG_DIR)

        merged_items = merge_items_json(items_path, items_payload, mode="add_new")
        save_items_json(items_path, merged_items)

    else:
        print("  [i] Kein Mesh generiert - nichts zu exportieren")

    timer.end()

    print(f"  [OK] Tile abgeschlossen: {tile.get('filename')}")

    return {
        "tile": tile,
        "materials": materials_path,
        "items": items_path,
        "dae_files": dae_files,
    }


def phase3_multitile_finalize(beamng_dir):
    """
    PHASE 3: Post-Merge und Finalisierung nach Tile-Verarbeitung.
    
    Merged alle Materials und Items aus den Tile-Ergebnissen.
    
    Args:
        beamng_dir: Zielverzeichnis für BeamNG-Dateien
    """
    print("\n" + "=" * 60)
    print("[PHASE 3] Multi-Tile Post-Merge & Finalisierung")
    print("=" * 60)
    
    materials_path = os.path.join(beamng_dir, "main.materials.json")
    items_path = os.path.join(beamng_dir, "main.items.json")
    
    # Lade alle Materials und Items
    if os.path.exists(materials_path):
        with open(materials_path, 'r', encoding='utf-8') as f:
            final_materials = json.load(f)
        print(f"[OK] {len(final_materials)} Materials final")
    else:
        print("[i] Keine Materials vorhanden")
        final_materials = {}
    
    if os.path.exists(items_path):
        with open(items_path, 'r', encoding='utf-8') as f:
            final_items = json.load(f)
        print(f"[OK] {len(final_items)} Items final")
    else:
        print("[i] Keine Items vorhanden")
        final_items = {}
    
    print(f"\n[OK] PHASE 3 abgeschlossen")
    return final_materials, final_items
