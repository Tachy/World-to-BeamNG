"""
Terrain-Export Workflow.

Orchestriert den kompletten Terrain-Export-Prozess.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from .tile_processor import TileProcessor


class TerrainWorkflow:
    """
    Orchestriert den Terrain-Export-Workflow.

    Verantwortlich für:
    - Mesh-Generierung
    - Straßen-Integration
    - DAE-Export
    - Material/Item-Management
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        material_manager: MaterialManager,
        item_manager: ItemManager,
        dae_exporter: DAEExporter,
    ):
        self.cache = cache_manager
        self.materials = material_manager
        self.items = item_manager
        self.dae = dae_exporter
        self.tile_processor = TileProcessor(cache_manager)

    def process_tile(
        self,
        tile: Dict,
        global_offset: Tuple[float, float],
        bbox_margin: float = 50.0,
        buildings_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Verarbeite einzelnes Tile.

        Args:
            tile: Tile-Metadaten
            global_offset: Globaler Offset (origin_x, origin_y)
            bbox_margin: BBox-Erweiterung in Metern
            buildings_data: Optional - LoD2 Gebäudedaten

        Returns:
            Dict mit Verarbeitungs-Ergebnissen
        """
        from ..osm.parser import calculate_bbox_from_height_data, extract_roads_from_osm
        from ..osm.downloader import get_osm_data
        from ..geometry.polygon import get_road_polygons, clip_road_polygons
        from ..geometry.junctions import (
            detect_junctions_in_centerlines,
            mark_junction_endpoints,
            split_roads_at_mid_junctions,
        )
        from ..geometry.vertices import classify_grid_vertices
        from ..mesh.road_mesh import generate_road_mesh_strips
        from ..mesh.vertex_manager import VertexManager
        from ..mesh.terrain_mesh import generate_full_grid_mesh
        from ..mesh.stitch_gaps import stitch_all_gaps
        from ..terrain.grid import create_terrain_grid

        print(f"\n{'='*60}")
        print(f"TILE: {tile.get('name', 'Unknown')}")
        print(f"{'='*60}")

        # 1. Lade Höhendaten
        height_points, height_elevations = self.tile_processor.load_height_data(tile)
        if height_points is None:
            return {"status": "failed", "reason": "no_height_data"}

        # Berechne tile_hash für Caching
        tile_hash = self.cache.hash_file(tile.get("filepath")) if tile.get("filepath") else "unknown"

        # 2. BBox berechnen (VOR der lokalen Transformation!)
        # Berechne BBox mit Margin direkt in UTM (Meter), dann Transformation zu WGS84
        osm_bbox = calculate_bbox_from_height_data(height_points, margin=bbox_margin)

        # 3. Transformiere zu lokalen Koordinaten
        local_points, elevations = self.tile_processor.ensure_local_offset(
            global_offset, height_points, height_elevations
        )

        # 4. OSM-Daten laden (mit tile_hash für tile-spezifischen Cache)
        osm_data = get_osm_data(osm_bbox, height_hash=tile_hash)

        if not osm_data:
            print("  [!] Keine OSM-Daten")
            return {"status": "failed", "reason": "no_osm_data"}

        # 5. Straßen extrahieren
        roads = extract_roads_from_osm(osm_data)

        # 6. Road Polygons (konvertiert OSM-Daten zu coords)
        # WICHTIG: Übergebe LOKALE Koordinaten! Alle internen Berechnungen in lokal!
        # Prüfe VORHER ob Elevation-Cache existiert
        import os
        from ..io.cache import get_cache_path

        elevation_cache_path = get_cache_path(osm_bbox, "elevations", tile_hash)
        elevation_was_cached = os.path.exists(elevation_cache_path)

        road_polygons = get_road_polygons(roads, osm_bbox, local_points, elevations, global_offset, tile_hash=tile_hash)

        # 6a. Luftbilder verarbeiten (nur wenn Elevation-Cache NEU erstellt wurde)
        if not elevation_was_cached:
            from pathlib import Path

            aerial_dir = Path("data/DOP20")

            if aerial_dir.exists() and any(aerial_dir.glob("*.zip")):
                print("  [i] Verarbeite Luftbilder für dieses Tile...")
                try:
                    from ..io.aerial import process_aerial_images
                    from .. import config as legacy_config

                    # Berechne Grid-Bounds schon hier (für Luftbilder)
                    grid_bounds_local = (
                        float(local_points[:, 0].min()),
                        float(local_points[:, 0].max()),
                        float(local_points[:, 1].min()),
                        float(local_points[:, 1].max()),
                    )

                    num_textures = process_aerial_images(
                        aerial_dir=str(aerial_dir),
                        output_dir=config.BEAMNG_DIR_TEXTURES,
                        grid_bounds=grid_bounds_local,
                        global_offset=global_offset,
                        tile_world_size=config.TILE_SIZE,
                        tile_size=2500,  # 2500 Pixel pro Texturkachel (→ 4096x4096 DDS)
                    )

                    if num_textures > 0:
                        print(f"  [OK] {num_textures} Luftbild-Texturen für dieses Tile exportiert")
                except Exception as e:
                    print(f"  [!] Fehler bei Luftbild-Verarbeitung: {e}")
        else:
            print(f"  [i] Elevation-Cache vorhanden - Luftbilder werden übersprungen")

        # 6b. LoD2-Gebäude laden (wenn aktiviert und noch nicht übergeben)
        if buildings_data is None and config.LOD2_ENABLED:
            from ..io.lod2 import cache_lod2_buildings, load_buildings_from_cache

            # Berechne Z-Min aus den Höhendaten für vollständige 3D-Normalisierung
            # WICHTIG: Gebäude-Z-Koordinaten NICHT normalisieren!
            # Das Terrain selbst hat absolute Höhen (263-580m), nicht normalisiert.
            # Die Gebäude in CityGML haben ebenfalls absolute Höhen über NN.
            # Daher: z_offset = 0 (keine Z-Normalisierung für Gebäude!)
            z_offset = 0.0
            # Erweitere global_offset zu 3D-Offset
            local_offset_3d = (global_offset[0], global_offset[1], z_offset)

            # Versuche normalisierte Gebäude direkt aus Cache zu laden
            # (sie wurden mit cache_lod2_buildings bereits normalisiert)
            buildings_cache_path = cache_lod2_buildings(
                lod2_dir=config.LOD2_DATA_DIR,
                bbox=osm_bbox,  # WGS84-BBox
                local_offset=local_offset_3d,  # 3D-Offset mit Z-Min!
                cache_dir=config.CACHE_DIR,
                height_hash=tile_hash,
            )
            if buildings_cache_path:
                buildings_data = load_buildings_from_cache(buildings_cache_path)
                if buildings_data:
                    print(f"  [OK] {len(buildings_data)} normalisierte Gebäude aus Cache geladen")

            if not buildings_data:
                print("  [i] Keine LoD2-Gebäude gefunden")

        # Berechne Grid-Bounds aus lokalen Punkten für Clipping
        grid_bounds_local = (
            float(local_points[:, 0].min()),
            float(local_points[:, 0].max()),
            float(local_points[:, 1].min()),
            float(local_points[:, 1].max()),
        )

        # Setze globales config.GRID_BOUNDS_LOCAL
        config.GRID_BOUNDS_LOCAL = grid_bounds_local

        # Verwende ROAD_CLIP_MARGIN aus Config (negativ = erweitern!)
        road_polygons = clip_road_polygons(road_polygons, grid_bounds_local, margin=config.ROAD_CLIP_MARGIN)

        # 7. Junction-Detection (benötigt road_polygons mit coords)
        # WICHTIG: Reihenfolge wie im alten Workflow: detect → split → mark
        junctions = detect_junctions_in_centerlines(road_polygons)
        road_polygons, junctions = split_roads_at_mid_junctions(road_polygons, junctions)  # ZUERST Split
        road_polygons = mark_junction_endpoints(road_polygons, junctions)  # DANN Mark

        # Wandle road_polygons in road_slope_polygons_2d um (für Klassifizierung)
        # WICHTIG: NACH Junction-Detection, damit die gesplitteten Straßen verwendet werden!
        # WICHTIG: Erzeuge tatsächliche Straßen-Polygone (Puffer um Centerline)
        from shapely.geometry import LineString
        from ..config import OSM_MAPPER

        road_slope_polygons_2d = []
        for road in road_polygons:
            coords = np.asarray(road.get("coords", []), dtype=float)
            if len(coords) < 2:
                continue

            # Berechne Straßenbreite aus OSM-Tags
            osm_tags = road.get("osm_tags", {})
            road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]

            # Erzeuge Polygon durch Pufferung der Centerline
            centerline_2d = coords[:, :2]
            try:
                line = LineString(centerline_2d)
                road_poly = line.buffer(road_width / 2.0, cap_style=2)  # cap_style=2 = flat
                road_polygon_2d = np.array(road_poly.exterior.coords[:-1])  # ohne Duplikat
            except Exception:
                # Fallback: verwende Centerline direkt
                road_polygon_2d = centerline_2d

            road_slope_polygons_2d.append(
                {
                    "road_id": road.get("id"),  # Wichtig für Material-Mapping
                    "road_polygon": road_polygon_2d,
                    "trimmed_centerline": coords,
                    "osm_tags": osm_tags,
                }
            )

        # 8. Grid erstellen (mit Builder)
        from ..builders import GridBuilder

        grid_builder = GridBuilder()
        grid = (
            grid_builder.with_points(local_points)
            .with_elevations(elevations)
            .with_spacing(config.GRID_SPACING)
            .with_cache(self.cache, f"grid_{tile_hash}")
            .build()
        )

        # 9. Vertex-Klassifizierung (nutzt road_slope_polygons_2d)
        grid_points, grid_elevations, nx, ny = grid
        vertex_states = classify_grid_vertices(grid_points, grid_elevations, road_slope_polygons_2d)

        # 10. Road Mesh (mit Builder)
        from ..builders import RoadMeshBuilder

        vertex_manager = VertexManager()
        road_mesh = (
            RoadMeshBuilder()
            .with_roads(road_polygons)
            .with_junctions(junctions)
            .with_grid(grid)
            .with_vertex_manager(vertex_manager)
            .build()
        )

        # 10a. Road-Face Cleanup: Clippe Road-Faces an Grid-Grenzen (messerscharf)
        from ..mesh.road_cleanup import clip_road_faces_at_bounds

        # Entpacke road_mesh Tupel (Slopes sind deaktiviert: config.GENERATE_SLOPES=False)
        all_road_faces = road_mesh[0]
        all_road_face_uvs = road_mesh[2]  # UV-Koordinaten (neuer Index 2!)

        # Clippe Road-Faces (TODO: auch UVs müssen gekl ippt werden!)
        clipped_road_faces = clip_road_faces_at_bounds(all_road_faces, vertex_manager, grid_bounds_local)

        # Packe Tupel neu zusammen (mit geclippten Faces)
        road_mesh = (
            clipped_road_faces,
            road_mesh[1],  # all_road_face_to_idx
            all_road_face_uvs,  # all_road_face_uvs (Index 2)
            road_mesh[3],  # road_slope_polygons_2d (alt Index 3)
            road_mesh[4],  # original_to_mesh_idx (alt Index 4)
            road_mesh[5],  # all_road_polygons_2d (alt Index 5)
            road_mesh[6],  # junction_fans (alt Index 7 → jetzt 6!)
        )

        # 11. Terrain Mesh (mit Builder)
        from ..builders import TerrainMeshBuilder

        terrain_mesh = (
            TerrainMeshBuilder()
            .with_grid(grid)
            .with_vertex_states(vertex_states)
            .with_vertex_manager(vertex_manager)
            .with_road_mesh(road_mesh, road_slope_polygons_2d)  # NEU: Road-Faces mit Material-Mapping
            .with_stitching(road_slope_polygons_2d, junctions)
            .build()
        )

        return {
            "status": "success",
            "road_mesh": road_mesh,
            "terrain_mesh": terrain_mesh,
            "grid": grid,
            "vertex_manager": vertex_manager,
            "road_polygons": road_polygons,
            "road_slope_polygons_2d": road_slope_polygons_2d,  # Für Material-Mapping
            "grid_bounds_local": grid_bounds_local,
            "global_offset": global_offset,
            "buildings_data": buildings_data,  # Übergebe Gebäude-Daten
            "height_points": local_points,  # Für Spawn-Punkt-Berechnung
            "height_elevations": elevations,  # Für Spawn-Punkt-Berechnung
            "road_face_uvs": all_road_face_uvs,  # Für UV-Zuordnung in tile_slicer
        }

    def export_tile(self, tile_x: int, tile_y: int, mesh_data: Dict) -> str:
        """
        Exportiere Tile als DAE.

        Args:
            tile_x, tile_y: Tile-Koordinaten
            mesh_data: Mesh-Daten aus process_tile()

        Returns:
            Pfad zur DAE-Datei
        """
        from ..io.dae import export_merged_dae
        from ..mesh.tile_slicer import slice_mesh_into_tiles
        from .. import config as legacy_config

        # Extrahiere Daten
        road_mesh_tuple = mesh_data["road_mesh"]
        terrain_mesh = mesh_data["terrain_mesh"]
        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        vertex_manager = mesh_data["vertex_manager"]

        # Entpacke road_mesh 7-Tupel
        all_road_faces = road_mesh_tuple[0]
        road_face_to_idx = road_mesh_tuple[1]  # Mapping Face-Index -> Road-ID
        # all_road_face_uvs = road_mesh_tuple[2]  # UVs sind jetzt zentral im Mesh.face_uvs!
        # Slopes sind deaktiviert (config.GENERATE_SLOPES=False)

        # Entpacke terrain_mesh
        terrain_faces = terrain_mesh["faces"]
        mesh_obj = terrain_mesh.get("mesh_obj")  # NEU: Hole Mesh-Objekt mit face_uvs
        vertex_normals = terrain_mesh.get("vertex_normals")

        # === Material-Mapping via OSM_MAPPER (wie im alten multitile.py) ===
        from ..config import OSM_MAPPER

        # Baue Material-Map: road_id -> (material_name, properties)
        # WICHTIG: Sammle Materials von ALLEN Roads, nicht nur den exportierten!
        road_material_map = {}
        unique_materials = {}  # ← Initialisiere hier schon, damit es auch leere Roads fängt

        for poly in road_slope_polygons_2d:
            r_id = poly.get("road_id")
            if r_id is None:
                continue
            props = OSM_MAPPER.get_road_properties(poly.get("osm_tags", {}))
            mat_name = props.get("internal_name", "road_default")
            road_material_map[r_id] = (mat_name, props)
            # ← Füge auch hier zu unique_materials hinzu, um sicherzustellen, dass alle Materials dabei sind
            unique_materials[mat_name] = props

        default_props = OSM_MAPPER.get_road_properties({})
        default_mat = default_props.get("internal_name", "road_default")

        # unique_materials wurde schon oben initialisiert
        if not unique_materials:
            unique_materials = {}

        # Extrahiere junction_fans aus road_mesh_tuple (falls vorhanden)
        junction_fans = road_mesh_tuple[7] if len(road_mesh_tuple) > 7 else {}

        # === Hilfsfunktion für Junction-Material-Selection ===
        def _get_junction_material(junction_id, junction_fans, road_material_map):
            """
            Bestimme das Material für eine Junction basierend auf angrenzenden Straßen.
            Logik:
            1. Zähle welche Materialien an der Junction ankommen
            2. Nutze das häufigste Material
            3. Bei Gleichstand: nutze das Material mit höherer Priorität
            """
            junction_data = junction_fans.get(junction_id, {})
            connected_road_ids = junction_data.get("connected_road_ids", [])

            if not connected_road_ids:
                # Keine angrenzenden Straßen: nutze Default
                return default_mat, default_props

            # Sammle alle Materialien der angrenzenden Straßen
            material_counts = {}  # {material_name: count}
            material_props = {}  # {material_name: properties}

            for road_id in connected_road_ids:
                if road_id in road_material_map:
                    mat_name, props = road_material_map[road_id]
                    material_counts[mat_name] = material_counts.get(mat_name, 0) + 1
                    material_props[mat_name] = props

            if not material_counts:
                # Keine Materialien gefunden: nutze Default
                return default_mat, default_props

            # Finde das häufigste Material
            max_count = max(material_counts.values())
            candidates = [mat for mat, count in material_counts.items() if count == max_count]

            if len(candidates) == 1:
                # Eindeutiger Gewinner
                mat_name = candidates[0]
                return mat_name, material_props[mat_name]

            # Bei Gleichstand: nutze das Material mit höherer Priorität
            # Priorität ist in der Properties gespeichert
            best_mat = candidates[0]
            best_priority = material_props[best_mat].get("priority", 0)

            for mat in candidates[1:]:
                mat_priority = material_props[mat].get("priority", 0)
                if mat_priority > best_priority:
                    best_mat = mat
                    best_priority = mat_priority

            return best_mat, material_props[best_mat]

        # Kombiniere alle Faces mit Materials
        all_faces = []
        materials_per_face = []

        # Road Faces (mit OSM-Mapper Materials)
        for idx, face in enumerate(all_road_faces):
            r_id = road_face_to_idx[idx] if idx < len(road_face_to_idx) else None

            # Prüfe ob es eine Junction-ID ist (negative Zahlen)
            if r_id is not None and r_id < 0:
                # Junction-ID: -(junction_id + 1)
                junction_id = -(r_id + 1)
                mat_name, props = _get_junction_material(junction_id, junction_fans, road_material_map)
            elif r_id is not None and r_id in road_material_map:
                mat_name, props = road_material_map[r_id]
            else:
                mat_name, props = default_mat, default_props

            unique_materials[mat_name] = props
            all_faces.append(face)
            materials_per_face.append(mat_name)

        # Terrain Faces
        for face in terrain_faces:
            all_faces.append(face)
            materials_per_face.append("terrain")

        # Hole alle Vertices vom VertexManager
        all_vertices = np.array(vertex_manager.get_array())

        # Slice in Tiles (übergebe UVs aus mesh_obj!)
        # terrain_mesh.face_uvs ist ein Dict: {face_idx: {vertex_idx: (u, v)}}
        # Das speichert die korrekten UVs für ALLE Faces (Road + Terrain + Stitched)
        tiles_dict = slice_mesh_into_tiles(
            vertices=all_vertices,
            faces=all_faces,
            materials_per_face=materials_per_face,
            tile_size=config.TILE_SIZE,
            vertex_normals=vertex_normals,
            face_uvs_dict=mesh_obj.face_uvs,  # Nutze die korrekten UVs vom Mesh-Objekt
        )

        # Export als DAE
        import os

        dae_output_path = os.path.join(config.BEAMNG_DIR_SHAPES, "terrain.dae")
        dae_path = export_merged_dae(
            tiles_dict=tiles_dict,
            output_path=dae_output_path,
            tile_size=config.TILE_SIZE,
            mesh_obj=mesh_obj,  # Übergebe Mesh für direkte UV-Zugriff
        )

        # Generiere und füge Materials hinzu
        from ..io.dae import create_terrain_materials_json, create_terrain_items_json

        # WICHTIG: Sammle auch alle Materials, die tatsächlich in materials_per_face sind
        # Manche Materials könnten in den Faces sein, aber nicht in unique_materials
        for mat in materials_per_face:
            if mat and mat not in unique_materials and mat != "terrain":
                # Material ist in den Faces aber nicht in unique_materials
                # Versuche es von OSM_MAPPER zu holen
                props = OSM_MAPPER.get_road_properties({"surface": mat})
                unique_materials[mat] = props

        # Road-Materials via OSM_MAPPER generieren (nach dem Sammeln aller Materials)
        road_material_entries = [
            OSM_MAPPER.generate_materials_json_entry(mat_name, props) for mat_name, props in unique_materials.items()
        ]

        # Terrain-Materials generieren
        terrain_materials = create_terrain_materials_json(
            tiles_dict=tiles_dict,
            level_name=config.LEVEL_NAME,
            tile_size=config.TILE_SIZE,
        )

        # Füge Road-Materials hinzu
        for mat_entry in road_material_entries:
            mat_name = mat_entry.pop("__name", None)
            if mat_name:
                self.materials.materials[mat_name] = mat_entry

        # Füge Terrain-Materials zum MaterialManager hinzu
        for mat_name, mat_data in terrain_materials.items():
            self.materials.materials[mat_name] = mat_data

        # Erstelle und füge Terrain-Item hinzu
        from ..io.dae import create_terrain_items_json
        import os

        dae_filename = os.path.basename(dae_path)
        terrain_item = create_terrain_items_json(dae_filename)
        item_name = terrain_item.get("__name", os.path.splitext(dae_filename)[0])
        self.items.items[item_name] = terrain_item

        return dae_path
