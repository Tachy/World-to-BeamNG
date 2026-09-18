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
        tiles: List[Dict],
        global_offset: Tuple[float, float],
        bbox_margin: float = 50.0,
        buildings_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Verarbeite alle übergebenen Kacheln als EINE zusammenhängende Fläche.

        Die Höhendaten aller Kacheln werden zu einer einzigen Punktwolke
        kombiniert (setzt voraus, dass sie einen lückenlosen, rechteckigen
        Bereich bilden - Nutzer-Verantwortung, siehe utils.tile_scanner).
        Ab hier läuft die komplette restliche Verarbeitung (BBox, OSM-Abfrage,
        Grid, Straßen-Mesh, Junction-Erkennung, Böschung, Heightmap) EINMAL
        über die Gesamtfläche statt einmal pro Kachel - Clipping findet nur
        noch am Außenrand der Gesamtfläche statt, nicht mehr an den früheren
        Kachelgrenzen. Ein einzelnes Tile ist einfach der Spezialfall
        len(tiles) == 1 desselben Codepfads.

        Args:
            tiles: Liste von Tile-Metadaten (typischerweise alle DGM1-Kacheln
                eines Exports)
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
        from ..terrain.grid import create_terrain_grid
        from ..io.cache import calculate_global_tiles_hash

        # 1. Höhendaten aller Kacheln zu einer Punktwolke kombinieren
        height_points, height_elevations = self.tile_processor.load_height_data_multi(tiles)
        if height_points is None:
            return {"status": "failed", "reason": "no_height_data"}

        # Kombinierter Hash über alle Kacheln - Cache-Identität für OSM/
        # Elevation/Grid (ersetzt den früheren Pro-Kachel-Dateihash)
        tile_hash = calculate_global_tiles_hash(tiles) if tiles else "unknown"

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

        # 6a. Luftbilder: werden NICHT mehr hier pro Kachel verarbeitet - bei
        # mehreren Kacheln würde die Datei-Existenz-Prüfung ("gibt es schon
        # irgendeine .dds?") den Export für alle Kacheln außer der ersten
        # überspringen. Stattdessen ruft BeamNGExporter.export_complete_level()
        # process_aerial_images() einmalig für die Gesamt-BBox aller Kacheln auf,
        # bevor die Tile-Schleife beginnt.

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

        # 10. Terrain-Heightmap statt Mesh-Triangulierung (siehe Spec:
        # docs/superpowers/specs/2026-09-17-terrain-heightmap-migration-design.md).
        # Straßen werden seit der DecalRoad-Umstellung nicht mehr als Mesh
        # gebaut (kein RoadMeshBuilder/Junction-Fan-Material-Mehrheitsvotum
        # mehr nötig) - siehe export_decal_roads().
        from ..config import OSM_MAPPER
        from ..terrain.heightmap import build_heightmap
        from ..terrain.road_embedding import (
            embed_roads_into_heightmap,
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

        # Straßen-Einbettung: Terrain auf der reinen Fahrbahnfläche exakt auf
        # Centerline-Höhe setzen (Böschung ist bereits durch
        # apply_embankment_blend abgedeckt) - siehe road_embedding.py-
        # Moduldocstring für die DecalRoad-Begründung.
        heights = embed_roads_into_heightmap(
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            road_slope_polygons_2d,
        )

        # Layer-Map: EIN Luftbild-Material für die gesamte Fläche, dann OSM-
        # Landnutzung obenauf (siehe build_photo_fallback_layer()).
        layer_map, photo_tile_names = build_photo_fallback_layer(terrain_size)

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
            "road_polygons": road_polygons,
            "road_slope_polygons_2d": road_slope_polygons_2d,  # Für DecalRoad-Export
            "grid_bounds_local": grid_bounds_local,
            "global_offset": global_offset,
            "buildings_data": buildings_data,  # Übergebe Gebäude-Daten
            "height_points": local_points,  # Für Spawn-Punkt-Berechnung
            "height_elevations": elevations,  # Für Spawn-Punkt-Berechnung
            "height_hash": tile_hash,  # Für Cache-Konsistenz in Forest-Workflow
        }

    def export_decal_roads(self, mesh_data: Dict) -> int:
        """
        Exportiert jede Straße als eigenes BeamNG `DecalRoad`-Item - ein
        Spline-Decal, das zur Laufzeit direkt auf die Terrain-Oberfläche
        projiziert wird (siehe road_embedding.py-Moduldocstring für die
        Begründung). Ersetzt das frühere Mesh-Bauwerk (prepare_road_export()/
        export_merged_roads()/RoadMeshBuilder/DAE-Export) komplett - kein
        Straßen-Mesh, keine Junction-Fan-Geometrie, keine
        Face-Material-Mehrheitsentscheidung mehr nötig: jede Straße bekommt
        ihr eigenes DecalRoad-Item mit ihrem eigenen (aus OSM abgeleiteten)
        Material, BeamNGs `autoJunction` verbindet angrenzende Straßen
        automatisch.

        Args:
            mesh_data: Ergebnis von process_tile() (braucht
                "road_slope_polygons_2d")

        Returns:
            Anzahl der erzeugten DecalRoad-Items
        """
        from ..config import OSM_MAPPER

        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        unique_materials: Dict[str, Dict] = {}
        count = 0

        for poly in road_slope_polygons_2d:
            road_id = poly.get("road_id")
            centerline = poly.get("trimmed_centerline")
            if road_id is None or centerline is None or len(centerline) < 2:
                continue

            # Entartete (Nulllängen-)Straßen überspringen: Clipping/Junction-
            # Split können vereinzelt einen "Rest" mit 2 identischen Punkten
            # hinterlassen. Ein DecalRoad mit Länge 0 ist ein degenerierter
            # Spline (im alten Mesh-Ansatz war das ein unsichtbares
            # Nulldreieck, hier würde es ein kaputtes Decal-Item erzeugen).
            xy_unique = {(round(float(x), 3), round(float(y), 3)) for x, y, _ in centerline}
            if len(xy_unique) < 2:
                continue

            props = OSM_MAPPER.get_road_properties(poly.get("osm_tags", {}))
            mat_name = props.get("internal_name", "road_default")
            unique_materials[mat_name] = props

            width = float(props.get("width", 4.0))
            nodes = [[float(x), float(y), float(z), width] for x, y, z in centerline]

            # renderPriority aus dem vorhandenen "priority"-Feld ableiten
            # (surface_types in data/osm_to_beamng.json): an Kreuzungen
            # überlappen sich die (immer volle Breite habenden) Enden
            # mehrerer DecalRoad-Objekte - ohne explizite, konsistente
            # Zeichenreihenfolge sortiert BeamNG das beliebig, was an
            # Kreuzungen wie ein "Flickenteppich" aussieht. Höherwertige
            # Straßen (Asphalt) werden so immer über niedrigerwertigen
            # (Dirt/Concrete) gezeichnet.
            render_priority = int(props.get("priority", 0))

            self.items.add_decal_road(
                name=f"road_{road_id}",
                nodes=nodes,
                material=mat_name,
                drivability=props.get("drivability", 1.0),
                overwrite=True,
                autoLanes=True,
                autoJunction=True,
                improvedSpline=True,
                renderPriority=render_priority,
            )
            count += 1

        road_material_entries = [
            OSM_MAPPER.generate_materials_json_entry(mat_name, props) for mat_name, props in unique_materials.items()
        ]
        for mat_entry in road_material_entries:
            mat_name = mat_entry.pop("__name", None)
            if mat_name:
                self.materials.materials[mat_name] = mat_entry

        logger.info(f"  [OK] {count} DecalRoad-Item(s) exportiert ({len(unique_materials)} Materialien)")
        return count

    def export_merged_terrain(
        self,
        heights: np.ndarray,
        layer_map: np.ndarray,
        terrain_material_names: List[str],
        terrain_origin_x: float,
        terrain_origin_y: float,
        terrain_size: int,
        z_min: float,
        max_height: float,
        photo_tile_names: List[str],
    ) -> None:
        """
        Schreibt EIN .ter und registriert TerrainBlock + TerrainMaterials.

        Args:
            heights, layer_map: fertige (bereits gepaddete) globale Arrays,
                shape (terrain_size, terrain_size)
            terrain_material_names: globale, deduplizierte Materialliste
                (Index entspricht layer_map-Werten)
            terrain_origin_x, terrain_origin_y: Welt-Koordinaten der Zelle [0, 0]
            z_min, max_height: siehe ter_writer.encode_heights_to_u16()
            photo_tile_names: Teilmenge von terrain_material_names, die auf
                das zusammengesetzte Luftbild verweisen (siehe
                terrain_materials.build_terrain_material_entries)
        """
        from ..terrain.ter_writer import write_ter, encode_heights_to_u16
        from ..terrain.terrain_materials import (
            build_terrain_material_entries,
            build_terrain_material_texture_set,
            ensure_flat_pbr_placeholders,
            ensure_landuse_base_textures_sized,
        )

        heightmap_u16 = encode_heights_to_u16(heights, z_min, max_height)
        ter_filename = f"{config.LEVEL_NAME}.ter"
        ter_path = config.BEAMNG_DIR / ter_filename
        write_ter(ter_path, heightmap_u16, layer_map.astype("uint8"), terrain_material_names)
        logger.info(f"  [OK] Terrain exportiert: {ter_filename} ({terrain_size}x{terrain_size})")

        placeholders = ensure_flat_pbr_placeholders(
            config.BEAMNG_DIR_TEXTURES, config.LEVEL_NAME, config.TERRAIN_BASE_TEX_PIXEL_SIZE
        )
        sized_landuse_mappings = ensure_landuse_base_textures_sized(
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            config.TERRAIN_BASE_TEX_PIXEL_SIZE,
            config.BEAMNG_DIR,
            config.BEAMNG_DIR_TEXTURES,
            config.LEVEL_NAME,
        )
        # photo_extent_size: der Wert, den wir BeamNG als baseColorBaseTexSize
        # für das Luftbild-Material mitteilen.
        #
        # DIAGNOSE 2026-09-18 (drei Messpunkte mit echtem Nutzer-Test, siehe
        # [[project_road_embed_slope_margin]]-Nachbar-Memory für den Kontext
        # dieser Session):
        #   - Grid @ 2m/Zelle: terrain_size=1024, TERRAIN_SQUARE_SIZE=2.0,
        #     physische Größe 2048m -> deklarierter Wert 1024 war korrekt.
        #   - Grid @ 1m/Zelle: terrain_size=2048, TERRAIN_SQUARE_SIZE=1.0,
        #     physische Größe UNVERÄNDERT 2048m -> derselbe Wert 1024 (aus der
        #     alten "physische_Größe * 0.5"-Formel) war jetzt FALSCH, korrekt
        #     ist 2048.
        # Die physische Kartengröße blieb in beiden Fällen identisch (2048m),
        # nur die Rasterauflösung hat sich geändert - trotzdem musste sich der
        # korrekte Wert mit der Rasterauflösung verdoppeln. Das heißt,
        # baseColorBaseTexSize wird von BeamNG offenbar in Heightmap-
        # Rasterzellen interpretiert, NICHT in Weltmetern (entgegen der
        # Doku-Formel "world_size = size * squareSize"): der korrekte Wert
        # ist schlicht terrain_size, die reine .ter-Rasterpunktzahl, komplett
        # unabhängig von TERRAIN_SQUARE_SIZE.
        photo_extent_size = float(terrain_size)
        terrain_material_entries = build_terrain_material_entries(
            terrain_material_names,
            photo_tile_names,
            sized_landuse_mappings,
            config.LEVEL_NAME,
            photo_extent_size,
            placeholders,
        )
        texture_set_name = f"{config.LEVEL_NAME}TerrainMaterialTextureSet"
        terrain_material_entries.update(
            build_terrain_material_texture_set(
                texture_set_name, base_tex_size=config.TERRAIN_BASE_TEX_PIXEL_SIZE
            )
        )
        self.materials.add_terrain_materials(terrain_material_entries)

        self.items.add_terrain_block(
            name="theTerrain",
            terrain_filename=ter_filename,
            material_texture_set=texture_set_name,
            max_height=max_height,
            z_min=z_min,
            origin_x=terrain_origin_x,
            origin_y=terrain_origin_y,
            square_size=config.TERRAIN_SQUARE_SIZE,
            overwrite=True,
        )

    def export_tile(self, tile_x: int, tile_y: int, mesh_data: Dict) -> int:
        """
        Exportiere EINE einzelne Kachel komplett (DecalRoad-Straßen + eigenes .ter).

        Convenience-Wrapper um export_decal_roads()/export_merged_terrain()
        für Aufrufer, die nur eine einzelne Kachel exportieren
        (export_single_tile()/export_terrain_only()).

        Args:
            tile_x, tile_y: unbenutzt, nur für Aufrufer-Kompatibilität
            mesh_data: Mesh-Daten aus process_tile()

        Returns:
            Anzahl der erzeugten DecalRoad-Items
        """
        road_count = self.export_decal_roads(mesh_data)
        self.export_merged_terrain(
            heights=mesh_data["heightmap"],
            layer_map=mesh_data["layer_map"],
            terrain_material_names=list(mesh_data["terrain_material_names"]),
            terrain_origin_x=mesh_data["terrain_origin_x"],
            terrain_origin_y=mesh_data["terrain_origin_y"],
            terrain_size=mesh_data["terrain_size"],
            z_min=mesh_data["z_min"],
            max_height=mesh_data["max_height"],
            photo_tile_names=mesh_data["photo_tile_names"],
        )
        return road_count
