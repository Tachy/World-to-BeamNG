"""
Terrain-Export Workflow.

Orchestriert den kompletten Terrain-Export-Prozess.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import logging

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from .tile_processor import TileProcessor

logger = logging.getLogger(__name__)


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
        dae_exporter: DAEExporter,
    ):
        self.cache = cache_manager
        self.materials = MaterialManager.get_instance()  # Singleton
        self.items = ItemManager.get_instance()  # Singleton
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
        from ..mesh.road_mesh import generate_road_mesh_strips
        from ..mesh.vertex_manager import VertexManager
        from ..terrain.grid import create_terrain_grid

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
            logger.warning("  [!] Keine OSM-Daten")
            return {"status": "failed", "reason": "no_osm_data"}

        # 5. Straßen extrahieren
        roads = extract_roads_from_osm(osm_data)

        # 6. Road Polygons (konvertiert OSM-Daten zu coords)
        # WICHTIG: Übergebe LOKALE Koordinaten! Alle internen Berechnungen in lokal!
        from ..io.cache import get_cache_path

        elevation_cache_path = get_cache_path(osm_bbox, "elevations", tile_hash)
        elevation_was_cached = elevation_cache_path.exists()

        road_polygons = get_road_polygons(roads, osm_bbox, local_points, elevations, global_offset, tile_hash=tile_hash)

        # 6a. Luftbilder verarbeiten (nur wenn die Ziel-Texturen noch fehlen)
        # WICHTIG: Nicht an elevation_was_cached koppeln! Der Elevation-Cache bleibt über
        # Sessions hinweg bestehen, die Textur-Dateien im BeamNG-Level-Ordner aber nicht
        # (z.B. nach einem BeamNG-Update, das den Userdata-Ordner zurücksetzt) - sonst
        # werden die Texturen dauerhaft übersprungen, obwohl sie nie geschrieben wurden.
        from pathlib import Path

        aerial_dir = Path("data/DOP20")
        textures_dir = config.BEAMNG_DIR_TEXTURES
        textures_missing = not (textures_dir.exists() and any(textures_dir.glob("tile_*.dds")))

        if textures_missing:
            if aerial_dir.exists() and any(aerial_dir.glob("*.zip")):
                logger.info("  [i] Verarbeite Luftbilder für dieses Tile...")
                try:
                    from ..io.aerial import process_aerial_images

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
                        logger.info(f"  [OK] {num_textures} Luftbild-Texturen für dieses Tile exportiert")
                except Exception as e:
                    logger.error(f"  [!] Fehler bei Luftbild-Verarbeitung: {e}")
        else:
            logger.info(f"  [i] Luftbild-Texturen bereits vorhanden - werden übersprungen")

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
                    logger.info(f"  [OK] {len(buildings_data)} normalisierte Gebäude aus Cache geladen")

            if not buildings_data:
                logger.info("  [i] Keine LoD2-Gebäude gefunden")

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
        from ..utils.debug_exporter import DebugNetworkExporter

        road_slope_polygons_2d = []
        debug_exporter = DebugNetworkExporter.get_instance()

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

            road_id = road.get("id")
            road_slope_polygons_2d.append(
                {
                    "road_id": road_id,  # Wichtig für Material-Mapping
                    "road_polygon": road_polygon_2d,
                    "trimmed_centerline": coords,
                    "osm_tags": osm_tags,
                }
            )

            # Exportiere Road zur Debug-Visualisierung
            debug_exporter.add_line(
                coords,
                color=[0.0, 0.0, 1.0],
                width=2.0,
                label=f"Road_{road_id}",
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

        # 9. Grid-Dimensionen extrahieren (Vertex-Klassifizierung entfällt -
        # das Terrain wird nicht mehr trianguliert, siehe Task 9)
        grid_points, grid_elevations, nx, ny = grid

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
        from ..mesh.road_cleanup import clip_road_mesh_data

        # Road-Mesh ist jetzt strukturiert: [{'vertices': [...], 'road_id': ..., 'uvs': {...}}, ...]
        road_mesh_data = road_mesh[0]

        # Clippe Road-Faces (mit UVs zusammen!)
        clipped_road_mesh_data = clip_road_mesh_data(road_mesh_data, vertex_manager, grid_bounds_local)

        # Packe Tupel neu zusammen (mit geclippten Daten)
        road_mesh = (
            clipped_road_mesh_data,  # Strukturierte Road-Daten
            road_mesh[1],  # road_slope_polygons_2d (alt Index 2)
            road_mesh[2],  # original_to_mesh_idx (alt Index 3)
            road_mesh[3],  # all_road_polygons_2d (alt Index 4)
            road_mesh[4],  # junction_fans (alt Index 5)
        )

        # 10b. Junction-Material-Mapping
        # Baue road_material_map für Roads UND Junctions
        from ..config import OSM_MAPPER

        road_material_map = {}
        junction_fans = road_mesh[4] if len(road_mesh) > 4 else {}

        # Zuerst: Sammle Road-Materials
        for poly in road_slope_polygons_2d:
            r_id = poly.get("road_id")
            if r_id is not None:
                props = OSM_MAPPER.get_road_properties(poly.get("osm_tags", {}))
                mat_name = props.get("internal_name", "road_default")
                road_material_map[r_id] = (mat_name, props)

        # Dann: Füge Junction-Materials hinzu (negative road_id)
        default_props = OSM_MAPPER.get_road_properties({})
        default_mat = default_props.get("internal_name", "road_default")

        for junction_id, junction_data in junction_fans.items():
            connected_road_ids = junction_data.get("connected_road_ids", [])

            if not connected_road_ids:
                # Keine angrenzenden Straßen: nutze Default
                road_material_map[-(junction_id + 1)] = (default_mat, default_props)
                continue

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
                road_material_map[-(junction_id + 1)] = (default_mat, default_props)
                continue

            # Finde das häufigste Material
            max_count = max(material_counts.values())
            candidates = [mat for mat, count in material_counts.items() if count == max_count]

            if len(candidates) == 1:
                # Eindeutiger Gewinner
                mat_name = candidates[0]
                road_material_map[-(junction_id + 1)] = (mat_name, material_props[mat_name])
            else:
                # Bei Gleichstand: nutze das Material mit höherer Priorität
                best_mat = candidates[0]
                best_priority = material_props[best_mat].get("priority", 0)

                for mat in candidates[1:]:
                    mat_priority = material_props[mat].get("priority", 0)
                    if mat_priority > best_priority:
                        best_mat = mat
                        best_priority = mat_priority

                road_material_map[-(junction_id + 1)] = (best_mat, material_props[best_mat])

        # 11. Terrain-Heightmap statt Mesh-Triangulierung (siehe Spec:
        # docs/superpowers/specs/2026-09-17-terrain-heightmap-migration-design.md)
        from ..terrain.heightmap import build_heightmap
        from ..terrain.road_embedding import (
            embed_roads_into_heightmap,
            road_mesh_to_arrays,
            build_road_embankment_profiles,
            apply_embankment_blend,
        )
        from ..terrain.terrain_materials import build_photo_fallback_layer, paint_landuse_materials

        heightmap_result = build_heightmap(
            grid_points, grid_elevations, nx, ny, config.TERRAIN_SQUARE_SIZE
        )
        heights = heightmap_result["heights"]
        terrain_size = heightmap_result["size"]
        terrain_origin_x = heightmap_result["origin_x"]
        terrain_origin_y = heightmap_result["origin_y"]

        # Böschung: Übergang von Straßenkante zur natürlichen Umgebung direkt
        # im Heightmap erzeugen (GENERATE_SLOPES bleibt False, das Mesh
        # generiert keine Böschungs-Geometrie mehr - siehe Spec Abschnitt 4b).
        # WICHTIG: muss auf den noch UNVERÄNDERTEN heights laufen, damit
        # "natürliche Höhe" wirklich natürlich ist (vor embed_roads_into_heightmap).
        embankment_profiles = build_road_embankment_profiles(
            road_slope_polygons_2d,
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            OSM_MAPPER,
            config.SLOPE_ANGLE,
            config.MIN_SLOPE_WIDTH,
            max_slope_width=config.MAX_SLOPE_WIDTH,
        )
        heights = apply_embankment_blend(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, embankment_profiles)

        # Straßen-Einbettung: Terrain unter der (unveränderten) Straßenfläche
        # knapp absenken (Böschung ist bereits durch apply_embankment_blend
        # abgedeckt, hier geht es nur noch um die reine Fahrbahnfläche)
        all_vertices = np.array(vertex_manager.get_array())
        road_mesh_data_for_embedding = road_mesh[0]
        road_vertices, road_triangles = road_mesh_to_arrays(road_mesh_data_for_embedding, all_vertices)
        heights = embed_roads_into_heightmap(
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            road_vertices,
            road_triangles,
            config.ROAD_EMBED_MARGIN,
        )

        # Layer-Map: Foto-Fallback pro Tile, dann OSM-Landnutzung obenauf
        # real_max_x/real_max_y: Ende der ECHTEN (nicht gepaddeten) Höhendaten -
        # Zellen jenseits davon (Zweierpotenz-Padding, siehe heightmap.py) werden
        # auf die letzte echte Kachel geklemmt statt eine nicht-existente
        # Foto-Textur zu referenzieren.
        real_max_x = terrain_origin_x + (nx - 1) * config.TERRAIN_SQUARE_SIZE
        real_max_y = terrain_origin_y + (ny - 1) * config.TERRAIN_SQUARE_SIZE
        layer_map, photo_tile_names = build_photo_fallback_layer(
            terrain_size,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            config.TILE_SIZE,
            real_max_x,
            real_max_y,
        )

        from shapely.geometry import shape as shapely_shape
        from pyproj import Transformer
        from ..geometry.coordinates import transformer_to_wgs84

        # osm_data enthält an dieser Stelle noch RAW Overpass-Geometrie (lat/lon,
        # {"lat":.., "lon":..} pro Punkt) - dieselbe Situation, die
        # ForestWorkflow._transform_osm_to_local() für Wald-Polygone löst.
        # Für Landnutzungs-Polygone hier dieselbe WGS84->UTM->lokal-Transformation.
        transformer_utm = Transformer.from_proj(
            transformer_to_wgs84.target_crs,  # WGS84
            transformer_to_wgs84.source_crs,  # UTM
        )
        offset_x, offset_y = global_offset[0], global_offset[1]

        landuse_polygons = []
        for element in osm_data:
            tags = element.get("tags", {})
            if not tags:
                continue
            geometry = element.get("geometry")
            if not geometry or len(geometry) < 3:
                continue
            try:
                coords_2d = []
                for pt in geometry:
                    if not isinstance(pt, dict) or "lat" not in pt or "lon" not in pt:
                        continue
                    utm_x, utm_y = transformer_utm.transform(pt["lon"], pt["lat"])
                    coords_2d.append((utm_x - offset_x, utm_y - offset_y))
                if len(coords_2d) < 3:
                    continue
                polygon = shapely_shape({"type": "Polygon", "coordinates": [coords_2d]})
                if not polygon.is_valid or polygon.is_empty:
                    continue
            except Exception:
                continue
            landuse_polygons.append({"osm_tags": tags, "geometry": polygon})

        layer_map, terrain_material_names = paint_landuse_materials(
            layer_map,
            photo_tile_names,
            terrain_size,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            landuse_polygons,
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
        )

        z_min = float(heights.min())
        z_max = float(heights.max())
        max_height = (z_max - z_min) + config.TERRAIN_MAX_HEIGHT_BUFFER

        return {
            "status": "success",
            "road_mesh": road_mesh,
            "heightmap": heights,
            "terrain_size": terrain_size,
            "terrain_origin_x": terrain_origin_x,
            "terrain_origin_y": terrain_origin_y,
            "z_min": z_min,
            "max_height": max_height,
            "layer_map": layer_map,
            "terrain_material_names": terrain_material_names,
            "photo_tile_names": photo_tile_names,
            "grid": grid,
            "vertex_manager": vertex_manager,
            "road_polygons": road_polygons,
            "road_slope_polygons_2d": road_slope_polygons_2d,  # Für Material-Mapping
            "road_material_map": road_material_map,  # Material-Map inkl. Junction-Materials
            "grid_bounds_local": grid_bounds_local,
            "global_offset": global_offset,
            "buildings_data": buildings_data,  # Übergebe Gebäude-Daten
            "height_points": local_points,  # Für Spawn-Punkt-Berechnung
            "height_elevations": elevations,  # Für Spawn-Punkt-Berechnung
            "height_hash": tile_hash,  # Für Cache-Konsistenz in Forest-Workflow
        }

    def export_tile(self, tile_x: int, tile_y: int, mesh_data: Dict) -> str:
        """
        Exportiere Tile als DAE.

        Args:
            tile_x, tile_y: Tile-Koordinaten
            mesh_data: Mesh-Daten aus process_tile()

        Returns:
            Liste von exportierten DAE-Dateinamen
        """
        from ..io.dae import export_separate_tile_daes
        from ..mesh.tile_slicer import slice_mesh_into_tiles

        # Extrahiere Daten
        road_mesh_tuple = mesh_data["road_mesh"]
        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        vertex_manager = mesh_data["vertex_manager"]

        # Entpacke strukturierte Road-Daten
        # Format: [{'vertices': [v0,v1,v2], 'road_id': id, 'uvs': {...}}, ...]
        road_mesh_data = road_mesh_tuple[0]

        # === Material-Mapping via OSM_MAPPER (wie im alten multitile.py) ===
        from ..config import OSM_MAPPER
        from ..utils.debug_exporter import DebugNetworkExporter

        debug_exporter = DebugNetworkExporter.get_instance()

        # Hole vorgefertigte road_material_map (enthält Roads UND Junctions!)
        road_material_map = mesh_data.get("road_material_map", {})
        unique_materials = {}  # ← Initialisiere hier schon, damit es auch leere Roads fängt

        for poly in road_slope_polygons_2d:
            r_id = poly.get("road_id")
            if r_id is None:
                continue

            # Hole Material aus vorgefertigter Map (wurde in process_tile() erstellt)
            mat_tuple = road_material_map.get(r_id)
            if mat_tuple:
                mat_name = mat_tuple[0]
                props = mat_tuple[1]
            else:
                # Fallback: Berechne Material neu (sollte nicht passieren)
                props = OSM_MAPPER.get_road_properties(poly.get("osm_tags", {}))
                mat_name = props.get("internal_name", "road_default")

            # Schreibe Material direkt in die Road-Struktur
            poly["material_name"] = mat_name

            # Erstelle Road-Label an der Centerline-Mitte
            trimmed_centerline = poly.get("trimmed_centerline", [])
            if len(trimmed_centerline) >= 2:
                mid_idx = len(trimmed_centerline) // 2
                mid_point = trimmed_centerline[mid_idx]
                debug_exporter.add_label(
                    f"Road_{r_id} ({poly['material_name']})",
                    position=[mid_point[0], mid_point[1], mid_point[2] if len(mid_point) > 2 else 0.0],
                    color=[1.0, 0.5, 0.0],  # Orange
                    size=12.0,
                )

            # Füge zu unique_materials hinzu
            unique_materials[mat_name] = props

        # Default-Material für Junctions ohne angrenzende Straßen
        default_props = OSM_MAPPER.get_road_properties({})
        default_mat = default_props.get("internal_name", "road_default")

        # Extrahiere junction_fans aus road_mesh_tuple für Label-Erstellung
        # (Material-Mapping wurde bereits in process_tile() durchgeführt!)
        # Index: [0]=road_mesh_data, [1]=road_slope_polygons_2d, [2]=original_to_mesh_idx, [3]=all_road_polygons_2d, [4]=junction_fans
        junction_fans = road_mesh_tuple[4] if len(road_mesh_tuple) > 4 else {}

        # === Erstelle Junction-Labels (Material ist bereits in road_material_map!) ===
        for junction_id, junction_data in junction_fans.items():
            # Material wurde bereits in process_tile() zugewiesen
            junction_road_id = -(junction_id + 1)  # Negative road_id für Junction-Faces
            mat_tuple = road_material_map.get(junction_road_id, (default_mat, default_props))
            mat_name = mat_tuple[0]
            mat_props = mat_tuple[1]

            # Füge Material zu unique_materials hinzu
            unique_materials[mat_name] = mat_props

            # Hole Position aus vertex_manager via center_idx
            center_idx = junction_data.get("center_idx")
            if center_idx is not None:
                # Hole Vertex-Position aus VertexManager
                center_vertex = vertex_manager.vertices[center_idx]
                position = [center_vertex[0], center_vertex[1], center_vertex[2]]

                debug_exporter.add_label(
                    f"Junction_{junction_id} ({mat_name})",
                    position=position,
                    color=[0.0, 0.0, 1.0],  # Blau
                    size=14.0,
                )

        # Kombiniere alle Faces mit Materials
        # WICHTIG: mesh_obj.faces enthält BEREITS alle Road-Faces + Terrain-Faces + Stitch-Faces!
        # Wir müssen diese NICHT doppelt hinzufügen!
        all_faces = []
        materials_per_face = []

        for face_data in road_mesh_data:
            all_faces.append(face_data["vertices"])
            mat_name = None
            r_id = face_data.get("road_id")
            if r_id in road_material_map:
                mat_name = road_material_map[r_id][0]
            materials_per_face.append(mat_name or "road_default")

        # Hole alle Vertices vom VertexManager
        all_vertices = np.array(vertex_manager.get_array())

        # slice_mesh_into_tiles() erwartet für hochwertige Straßen-Texturierung
        # ein mesh_obj mit .uv_indices/.uvs (siehe mesh/tile_slicer.py:330-360:
        # "if mesh_obj and hasattr(mesh_obj, 'uv_indices') and original_face_idx
        # in mesh_obj.uv_indices"). Vorher kamen diese UVs aus dem kombinierten
        # Terrain+Road mesh_obj von TerrainMeshBuilder; die UV-Rohdaten selbst
        # stammen aber unverändert aus RoadMeshBuilder (road_mesh_data[i]["uvs"]),
        # nicht aus TerrainMeshBuilder. Ein minimaler Adapter reicht, um exakt
        # dieselbe Straßen-Textur-Qualität wie vor der Migration zu erhalten
        # (statt auf die gröbere Tile-planare Fallback-UV zurückzufallen, die
        # slice_mesh_into_tiles sonst für Faces ohne mesh_obj-Treffer nutzt -
        # siehe tile_slicer.py:361-369).
        class _RoadUVAdapter:
            """Minimaler mesh_obj-Ersatz: stellt nur die Road-UVs aus
            road_mesh_data bereit, im von slice_mesh_into_tiles erwarteten
            Format (uv_indices: {face_idx: [i0,i1,i2]}, uvs: [(u,v), ...])."""

            def __init__(self, road_mesh_data, faces):
                self.uvs = []
                self.uv_indices = {}
                uv_lookup = {}

                for face_idx, face_data in enumerate(road_mesh_data):
                    face_vertices = faces[face_idx]
                    per_vertex_uv = face_data.get("uvs", {})
                    indices = []
                    for vertex_idx in face_vertices:
                        uv = per_vertex_uv.get(vertex_idx, (0.0, 0.0))
                        if uv not in uv_lookup:
                            uv_lookup[uv] = len(self.uvs)
                            self.uvs.append(uv)
                        indices.append(uv_lookup[uv])
                    self.uv_indices[face_idx] = indices

        road_uv_adapter = _RoadUVAdapter(road_mesh_data, all_faces)

        # Slice in Tiles (nur noch Straßen-Faces - Terrain ist jetzt .ter, kein Mesh mehr)
        tiles_dict = slice_mesh_into_tiles(
            vertices=all_vertices,
            faces=all_faces,
            materials_per_face=materials_per_face,
            tile_size=config.TILE_SIZE,
            vertex_normals=None,
            mesh_obj=road_uv_adapter,
        )

        # Export als DAE (SEPARATE Dateien pro Tile!)
        import os

        shapes_dir = config.BEAMNG_DIR_SHAPES
        logger.info(f"  Exportiere {len(tiles_dict)} Tiles als separate DAE-Dateien...")
        dae_files = export_separate_tile_daes(
            tiles_dict=tiles_dict,
            output_dir=shapes_dir,
            material_manager=self.materials,  # Übergebe MaterialManager-Referenz
            tile_size=config.TILE_SIZE,
        )

        # Terrain als .ter exportieren (natives BeamNG-Heightmap statt Mesh)
        from ..terrain.ter_writer import write_ter, encode_heights_to_u16
        from ..terrain.terrain_materials import build_terrain_material_entries, build_terrain_material_texture_set

        heights = mesh_data["heightmap"]
        z_min = mesh_data["z_min"]
        max_height = mesh_data["max_height"]
        layer_map = mesh_data["layer_map"]
        terrain_material_names = mesh_data["terrain_material_names"]
        photo_tile_names = mesh_data["photo_tile_names"]

        heightmap_u16 = encode_heights_to_u16(heights, z_min, max_height)
        ter_filename = f"{config.LEVEL_NAME}.ter"
        ter_path = config.BEAMNG_DIR / ter_filename
        write_ter(ter_path, heightmap_u16, layer_map.astype("uint8"), terrain_material_names)
        logger.info(f"  [OK] Terrain exportiert: {ter_filename} ({mesh_data['terrain_size']}x{mesh_data['terrain_size']})")

        terrain_material_entries = build_terrain_material_entries(
            terrain_material_names,
            photo_tile_names,
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            config.LEVEL_NAME,
            config.TILE_SIZE,
        )
        texture_set_name = f"{config.LEVEL_NAME}TerrainMaterialTextureSet"
        terrain_material_entries.update(
            build_terrain_material_texture_set(texture_set_name, base_tex_size=int(config.TILE_SIZE) * 8)
        )
        self.materials.add_terrain_materials(terrain_material_entries)

        self.items.add_terrain_block(
            name="theTerrain",
            terrain_filename=ter_filename,
            material_texture_set=texture_set_name,
            max_height=max_height,
            z_min=z_min,
            origin_x=mesh_data["terrain_origin_x"],
            origin_y=mesh_data["terrain_origin_y"],
            square_size=config.TERRAIN_SQUARE_SIZE,
            overwrite=True,
        )

        # Generiere und füge Materials hinzu
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

        # Füge Road-Materials hinzu
        for mat_entry in road_material_entries:
            mat_name = mat_entry.pop("__name", None)
            if mat_name:
                self.materials.materials[mat_name] = mat_entry

        # Erstelle TSStatic-Items für JEDES Straßen-Tile (separate DAEs!)
        # add_terrain() bleibt die richtige Convenience-Methode (TSStatic +
        # "Visible Mesh Final"-Kollision) - dae_files enthält jetzt nur noch
        # Straßen-Geometrie, kein Terrain mehr (siehe Step 4).
        logger.info(f"  Erstelle {len(dae_files)} TSStatic-Items für Straßen-Tiles...")
        for dae_filename in dae_files:
            # Extrahiere Tile-Koordinaten: tile_-1000_-1000.dae → "-1000_-1000"
            tile_coords = Path(dae_filename).stem.replace("tile_", "")
            item_name = f"road_tile_{tile_coords}"  # z.B. "road_tile_-1000_-1000"
            self.items.add_terrain(
                name=item_name,
                dae_filename=dae_filename,
                position=(0, 0, 0),
                overwrite=True,
            )

        logger.info(f"  [OK] {len(dae_files)} Straßen-Tile-DAEs exportiert")
        return dae_files
