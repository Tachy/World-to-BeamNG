"""
Zentrale BeamNG-Exporter-Fassade.

Bietet eine einheitliche API für den gesamten Export-Workflow.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import json

import numpy as np

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from ..io.beamng_install import get_beamng_install_dir
from ..io.vineyard_assets import ITEM_NAMES as VINEYARD_ITEM_NAMES, ensure_vineyard_assets
from ..workflow import TileProcessor, TerrainWorkflow, BuildingWorkflow, HorizonWorkflow, ForestWorkflow
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()


class BeamNGExporter:
    """
    Zentrale Fassade für BeamNG-Level-Export.

    Vereinfacht die API und orchestriert alle Sub-Workflows.

    Beispiel:
        >>> exporter = BeamNGExporter()
        >>> exporter.export_complete_level(tiles)
    """

    def __init__(self):
        """
        Initialisiere BeamNGExporter.
        """
        # Core Components
        self.cache = CacheManager(Path(config.CACHE_DIR))

        # Singleton-Manager (reset für neuen Export)
        MaterialManager.reset_instance()
        self.materials = MaterialManager.get_instance(config.BEAMNG_DIR)

        ItemManager.reset_instance()
        self.items = ItemManager.get_instance(config.BEAMNG_DIR)

        # Registriere zentrales Forest-Objekt in items.level.json
        self.items.add_item(
            name="the_forest",
            item_class="Forest",
            dataFile="levels/world_to_beamng/forest/forest.forest4.json",
            lodScale=1.0,
            overwrite=True,
        )
        logger.debug("✓ Forest-Objekt registriert in ItemManager")

        self.dae = DAEExporter(material_manager=self.materials)  # Übergebe MaterialManager-Referenz

        # Lade osm_to_beamng.json Config (Materials werden SPÄTER generiert!)
        osm_config_path = Path("data/osm_to_beamng.json")
        self.osm_config = {}
        self.forest_config = {"forest_type_templates": {}, "forest_mappings": {}}

        if osm_config_path.exists():
            with open(osm_config_path, "r", encoding="utf-8") as f:
                self.osm_config = json.load(f)
                # Lade forest_type_templates und forest_mappings
                self.forest_config["forest_type_templates"] = self.osm_config.get("forest_type_templates", {})
                self.forest_config["forest_mappings"] = self.osm_config.get("forest_mappings", {})

        # Workflows (nutzen MaterialManager/ItemManager.get_instance() intern)
        self.terrain = TerrainWorkflow(self.cache, self.dae)
        self.buildings = BuildingWorkflow(self.cache, self.dae)
        self.horizon = HorizonWorkflow(self.cache, self.dae)
        self.tile_processor = TileProcessor(self.cache)
        self.forests = ForestWorkflow(config)  # Nur config, kein Asset Scanning

        # Debug-Exporter für Visualisierung (Singleton - reset für neuen Export)
        from ..utils.debug_exporter import DebugNetworkExporter

        DebugNetworkExporter.reset_instance()
        self.debug_exporter = DebugNetworkExporter.get_instance()

        # Speichere Höhendaten für Spawn-Punkt-Berechnung
        self.height_points = None
        self.height_elevations = None
        self.global_offset = None

    def export_complete_level(
        self,
        tiles: List[Dict],
        global_offset: Tuple[float, float, float],
        include_buildings: bool = True,
        include_horizon: bool = True,
        include_forests: bool = True,  # NEU: Forest-Export
    ) -> Dict:
        """
        Exportiere komplettes BeamNG-Level.

        Args:
            tiles: Liste von Tile-Metadaten
            global_offset: (origin_x, origin_y, origin_z)
            include_buildings: LoD2-Gebäude exportieren
            include_horizon: Horizon-Layer exportieren
            include_forests: Wald-Vegetation exportieren (per Config überschreibbar)

        Returns:
            Dict mit Export-Statistiken
        """
        from ..utils.timing import StepTimer

        timer = StepTimer()
        stats = {
            "tiles_processed": 0,
            "tiles_failed": 0,
            "buildings_exported": 0,
            "horizon_exported": False,
            "forests_registered": 0,  # NEU
            "trees_generated": 0,  # NEU
            "vine_segments": 0,
        }

        forests_enabled = include_forests and config.FORESTS_ENABLED
        if include_forests and not config.FORESTS_ENABLED:
            logger.info("Forest-Export in Config deaktiviert (config.FORESTS_ENABLED=False)")

        logger.info(f"\n{'='*60}")
        logger.info(f"BEAMNG LEVEL EXPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Tiles: {len(tiles)}")
        logger.info(f"Global Offset: {global_offset}")
        logger.info(
            f"Forests: {'Yes' if forests_enabled else 'No'} | Config: {'on' if config.FORESTS_ENABLED else 'off'}"
        )  # NEU
        logger.info(f"{'='*60}\n")

        # Speichere global_offset für Spawn-Punkt-Berechnung
        self.global_offset = global_offset[:2]  # Nur (x, y)

        # Erstelle Verzeichnisse
        config.BEAMNG_DIR_SHAPES.mkdir(parents=True, exist_ok=True)
        config.BEAMNG_DIR_TEXTURES.mkdir(parents=True, exist_ok=True)
        config.BEAMNG_DIR_BUILDINGS.mkdir(parents=True, exist_ok=True)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Alle Texturen aus data/textures prüfen (prozedurale bei Bedarf einmalig erzeugen); fehlt eine Foto-Textur,
        # bricht der Export hier ab (MissingTexturesError) - vor dem rechenintensiven Teil
        from ..textures import registry

        registry.prepare_textures()

        # NEU: Phase 0 - Forest Asset Initialization (DIREKT VOR Tile-Loop)
        registered_trees = {}
        vineyard_assets_ready = False
        if forests_enabled:
            timer.begin("Forest Asset Initialization")

            # Reben-Assets für Weinberge sicherstellen (idempotent) - VOR dem Laden von
            # managedItemData.json, damit die Reben als Forest-Items registriert sind.
            if config.VINEYARDS_ENABLED:
                try:
                    ensure_vineyard_assets(config.BEAMNG_DIR, get_beamng_install_dir(), config.LEVEL_NAME)
                    vineyard_assets_ready = True
                except Exception as e:
                    logger.warning(f"Reben-Assets nicht verfügbar - Weinberge bleiben ohne Reben: {e}")

            # Lade managedItemData.json (wird von generate_forest_assets.py erzeugt)
            forest_item_data_path = config.BEAMNG_DIR / "art" / "forest" / "managedItemData.json"

            if forest_item_data_path.exists():
                try:
                    with open(forest_item_data_path, "r", encoding="utf-8") as f:
                        forest_item_data = json.load(f)

                    # Konvertiere zu registered_trees Format (Key MUSS der internalName sein,
                    # denn forest.forest4.json referenziert Bäume darüber im "type"-Feld)
                    for item_key, item_info in forest_item_data.items():
                        internal_name = item_info.get("internalName", item_key)
                        # Reben sind keine Waldbäume: sonst könnte der Fallback der
                        # Baumartenwahl ("erster verfügbarer Baum") sie in den Wald pflanzen.
                        if internal_name in VINEYARD_ITEM_NAMES:
                            continue
                        registered_trees[internal_name] = {
                            "name": internal_name,
                            "dae_path": item_info.get("shapeFile", ""),
                            "radius": item_info.get("radius", 1.5),
                        }

                    logger.info(f"✓ {len(registered_trees)} Tree-Items aus managedItemData.json geladen")

                except Exception as e:
                    logger.error(f"Fehler beim Laden von managedItemData.json: {e}")
                    registered_trees = {}
            else:
                logger.warning(f"managedItemData.json nicht gefunden: {forest_item_data_path}")
                logger.warning("  Bitte führen Sie zuerst aus: python tools/generate_forest_assets.py")

            stats["forests_registered"] = len(registered_trees)

            # Setze Forest-Konfiguration (initialisiert Normalizer, InstanceGenerator, JSONWriter)
            if registered_trees:
                self.forests.set_forest_config(
                    self.forest_config,
                    osm_mapper=config.OSM_MAPPER,
                    registered_trees=registered_trees,
                )

        # Sammle alle Gebäude über alle Tiles
        all_buildings = []
        tile_bounds_local = []  # Sammle Tile-Grenzen für Horizon-Clipping

        # Höhenabfrage der fertigen Terrain-Heightmap (Naht und Höhenübergang des Horizonts)
        terrain_height_at = None

        # EIN zusammengesetztes Luftbild für die Gesamtfläche generieren (nicht
        # mehr eine Textur pro 500m-Kachel - siehe io/aerial.py::process_aerial_images()
        # Docstring: BeamNGs Terrain-Atlas-Packer verdreht Kacheln sichtbar, wenn
        # ihm zu viele große, einzigartige Materialien übergeben werden).
        from ..utils.tile_scanner import compute_global_bbox
        from ..terrain.heightmap import next_power_of_two_size

        utm_min_x, utm_max_x, utm_min_y, utm_max_y = compute_global_bbox(tiles)

        # Das Luftbild MUSS exakt dieselbe Fläche abdecken, die das Terrain
        # später tatsächlich einnimmt - und das ist NICHT die rohe DGM-Kachel-
        # Bbox (z.B. 2000x2000m), sondern die auf Zweierpotenz aufgefüllte
        # Heightmap-Größe (z.B. 2048x2048m, siehe heightmap.py:build_heightmap()).
        # Ohne diesen Abgleich behauptet die TerrainMaterialTextureSet eine
        # andere Kantenlänge (baseColorBaseTexSize), als das Foto tatsächlich
        # zeigt -> das Bild wird falsch skaliert auf das Terrain projiziert.
        # Padding erweitert (wie bei den Höhendaten) NUR nach Osten/Norden.
        nx = len(np.arange(utm_min_x, utm_max_x + config.GRID_SPACING * 0.5, config.GRID_SPACING))
        ny = len(np.arange(utm_min_y, utm_max_y + config.GRID_SPACING * 0.5, config.GRID_SPACING))
        padded_size = next_power_of_two_size(max(nx, ny))
        padded_extent = padded_size * config.GRID_SPACING

        combined_grid_bounds_local = (
            utm_min_x - global_offset[0],
            utm_min_x - global_offset[0] + padded_extent,
            utm_min_y - global_offset[1],
            utm_min_y - global_offset[1] + padded_extent,
        )
        textures_dir = config.BEAMNG_DIR_TEXTURES
        aerial_dir = Path("data/DOP20")
        from ..io.aerial import ensure_aerial_photos, SINGLE_PHOTO_NAME
        from ..terrain.photo_tiles import photo_tile_specs

        # Vier-Bilder-Modus: bei mehreren DGM1-Kacheln ein eigenes Foto je Kachel (siehe terrain/photo_tiles.py),
        # sonst ein Gesamtfoto. process_tile() nimmt dieselbe Aufteilung (photo_tile_specs) für die Layer-Map.
        if config.AERIAL_PHOTO_PER_TILE and len(tiles) > 1:
            photos = photo_tile_specs(tiles, global_offset)
        else:
            photos = [{"name": SINGLE_PHOTO_NAME, "bounds": combined_grid_bounds_local}]

        # Die Fotos werden neu gebaut, sobald Fläche, Ursprung, Auflösung, Kachelaufteilung oder Quellbilder nicht
        # mehr zu den vorhandenen passen (z.B. Umstellung von einer auf vier DGM1-Kacheln) - nicht nur, wenn sie fehlen.
        try:
            status = ensure_aerial_photos(
                aerial_dir=aerial_dir, output_dir=textures_dir, photos=photos, global_offset=global_offset
            )
            if status == "current":
                logger.info(f"[i] Luftbild(er) passen zur Fläche ({len(photos)}) - werden übernommen")
            elif status == "built":
                logger.info(f"[OK] {len(photos)} Luftbild(er) neu gebaut und exportiert")
            elif status == "failed":
                logger.error("[!] Luftbild konnte nicht gebaut werden")
        except Exception as e:
            logger.error(f"[!] Fehler bei Luftbild-Verarbeitung: {e}")

        # Phase 1: Terrain + Straßen - ALLE Kacheln als EINE zusammenhängende
        # Fläche verarbeiten (ein Grid, ein Straßennetz, ein Junction-Pass).
        # Clipping findet nur noch am Außenrand der Gesamtfläche statt, nicht
        # mehr an den früheren DGM1-Kachelgrenzen (siehe process_tile()-Docstring).
        timer.begin("Terrain + Straßen (Gesamtfläche)")
        result = self.terrain.process_tile(tiles=tiles, global_offset=global_offset[:2], bbox_margin=50.0)

        if result["status"] != "success":
            stats["tiles_failed"] = len(tiles)
            logger.error(f"[!] Terrain-Verarbeitung fehlgeschlagen: {result.get('reason')}")
        else:
            stats["tiles_processed"] = len(tiles)

            # Höhendaten für Spawn-Punkt-Berechnung
            self.height_points = result.get("height_points")
            self.height_elevations = result.get("height_elevations")

            self.terrain.export_tile(0, 0, result)

            # Höhenabfrage der fertigen Heightmap: der Horizont bekommt daraus sein Terrain-Loch
            # samt Randhöhen (kein Terrain-Mesh mehr, das vernäht werden könnte)
            from ..terrain.road_embedding import sample_heightmap_bilinear

            heightmap = result["heightmap"]
            hm_origin = (result["terrain_origin_x"], result["terrain_origin_y"])
            terrain_height_at = lambda x, y: sample_heightmap_bilinear(
                heightmap, hm_origin[0], hm_origin[1], config.TERRAIN_SQUARE_SIZE,
                np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]),
            )

            from ..forest.vineyard_generator import make_height_sampler

            terrain_height_at_1d = make_height_sampler(heightmap, hm_origin[0], hm_origin[1], config.TERRAIN_SQUARE_SIZE)

            # Gesamt-BBox in lokalen Koordinaten für Horizon-Clipping
            x_min, x_max, y_min, y_max = result["grid_bounds_local"]
            tile_bounds_local.append((x_min, y_min, x_max, y_max))

            # Phase 1b: Forest Processing (für die Gesamtfläche, nicht mehr pro Kachel)
            if forests_enabled:
                forest_result = self.forests.process_tile(
                    tile_bounds=(x_min, y_min, x_max, y_max),
                    tile_name="combined_area",
                    elevation_data=result.get("height_points"),
                    height_grid_info={
                        "origin": (x_min, y_min),
                        "spacing": 1.0,
                        "elevations": result.get("height_elevations"),
                    },
                    height_hash=result.get("height_hash"),  # Für Cache-Konsistenz
                    global_offset=global_offset,  # NEU: Für WGS84-Transformation
                    # Bäume stehen auf der fertigen Heightmap (nach Straßen-Einbettung), nicht auf rohen DGM1-Punkten,
                    # und meiden die tatsächlich eingebetteten Straßenflächen
                    height_at=terrain_height_at_1d,
                    road_surfaces=result.get("road_surface_union"),
                )
                if forest_result["status"] == "success":
                    stats["trees_generated"] += forest_result.get("tree_count", 0)

                # Weinberg-Reben (Forest-Items) zusammen mit den Bäumen in forest.forest4.json
                if vineyard_assets_ready and result.get("vineyard_instances"):
                    stats["vine_segments"] += self.forests.add_instances(result["vineyard_instances"])

            # Sammle Gebäude-Daten (werden später gruppiert nach Tiles exportiert)
            if include_buildings and result.get("buildings_data"):
                all_buildings.extend(result["buildings_data"])

        # Phase 2: Buildings (nach Terrain-Export, wie im alten multitile.py)
        if include_buildings and all_buildings:

            timer.begin("Buildings Export")

            # Gebäude: EIN Objekt auf der Gesamtfläche (wie die Straßen) oder - wenn abgeschaltet - je 500-m-Kachel
            from ..workflow.building_workflow import plan_building_shapes, remove_stale_building_daes

            # Mehr als 2048 Nodes je Shape verwirft BeamNG -> Gesamtfläche in Teil-Shapes (buildings, buildings_part_N)
            shapes = plan_building_shapes(
                all_buildings,
                None if config.BUILDINGS_AS_ONE_OBJECT else config.TILE_SIZE,
                config.MAX_BUILDINGS_PER_SHAPE,
            )

            written = set()
            for tile_x, tile_y, name, tile_buildings in shapes:
                dae_path = self.buildings.export_buildings(tile_buildings, tile_x, tile_y, grid_bounds=None, name=name)
                if dae_path:
                    written.add(Path(dae_path).stem)
                    self.buildings.add_items(tile_buildings, tile_x, tile_y, name=name)
                    stats["buildings_exported"] += len(tile_buildings)

            # DAEs der jeweils anderen Aufteilung (frühere Kacheln bzw. das Gesamtobjekt) entfernen
            remove_stale_building_daes(config.BEAMNG_DIR_BUILDINGS, keep=written)

            # Materials exportieren
            # Füge LoD2-Materialien zu gemeinsamen Materials hinzu (NICHT separat exportieren!)
            self._add_lod2_materials()

        timer.begin("Horizon Export")

        # Phase 3: Horizon-Layer (optional)
        if include_horizon:
            # Tile-Grenzen (Terrain-Loch) und Höhenabfrage der Terrain-Heightmap (Naht, Höhenübergang)
            horizon_dae = self.horizon.generate_horizon(
                global_offset=global_offset,
                tile_bounds=tile_bounds_local,
                terrain_height_at=terrain_height_at,
            )
            stats["horizon_exported"] = horizon_dae is not None

        timer.begin("Finalisierung")

        # Phase 4: Finalisierung
        self._finalize_export(forests_enabled)

        timer.report()

        return stats

    def _add_lod2_materials(self):
        """
        Füge LoD2-Gebäude-Materialien hinzu.

        - Wände: je Putzfarbe ein Material (eigene Albedo-, gemeinsame Normal-/Roughness-Textur)
        - Fenster: Sprite-Atlas für Fenster, Türen und Kellerfenster
        - Dach: Biberschwanz aus osm_to_beamng.json (unverändert)
        - Flachdach: Kiesfläche (Textur aus data/textures); Blechrand und Dachüberstand-Trim: untexturiert

        Texturen der prozeduralen Materialien kommen aus ensure_building_textures(), die Kies-Textur aus der Textur-Registry
        (textures/registry.py); Farben und Faktoren der
        untexturierten aus osm_to_beamng.json (OSM_MAPPER), Template-Hinweise aus material_templates.json.
        """
        from ..config import OSM_MAPPER
        from ..facade.building_textures import ensure_building_textures
        from ..facade.facade_styles import PLASTER_COLORS
        from ..textures import registry
        from ..facade.material_names import (
            FLAT_ROOF_MATERIAL,
            ROOF_EDGE_MATERIAL,
            ROOF_MATERIAL,
            ROOF_TRIM_MATERIAL,
            WALL_MATERIALS,
            WINDOW_MATERIAL,
        )

        templates = self.materials.get_templates().get("buildings", {})

        def hints(kind: str) -> dict:
            material_hints = templates.get(kind, {}).get("material_hints", {})
            return {
                "groundType": material_hints.get("groundType", "concrete"),
                "materialTag0": material_hints.get("materialTag0", "beamng"),
                "materialTag1": material_hints.get("materialTag1", "Building"),
            }

        def textured(prefix: str) -> dict:
            return {
                "normalMap": generated[f"{prefix}_normal"],
                "roughnessMap": generated[f"{prefix}_roughness"],
                "useAnisotropic": True,
            }

        def untextured(props: dict) -> dict:
            return {
                "color": props["diffuseColor"],
                "stage_properties": {
                    "baseColorFactor": props["diffuseColor"],
                    "roughnessFactor": props["roughnessFactor"],
                    "metallicFactor": props["metallicFactor"],
                },
            }

        generated = ensure_building_textures()
        roof_props = OSM_MAPPER.get_building_properties("roof")

        for name, color in zip(WALL_MATERIALS, PLASTER_COLORS):
            textures = {"baseColorMap": generated[f"plaster_color_{color.name}"], **textured("plaster")}
            self.materials.add_building_material(name, textures=textures, **hints("wall"))
        self.materials.add_building_material(
            WINDOW_MATERIAL, textures={"baseColorMap": generated["windows_color"], **textured("windows")}, **hints("wall")
        )
        self.materials.add_building_material(
            ROOF_MATERIAL, color=roof_props.get("diffuseColor"), textures=roof_props.get("textures"), **hints("roof")
        )
        gravel = registry.prepared_textures()[config.FLAT_ROOF_GRAVEL_TEXTURE]
        self.materials.add_building_material(FLAT_ROOF_MATERIAL, textures={**gravel, "useAnisotropic": True}, **hints("roof"))
        self.materials.add_building_material(
            ROOF_EDGE_MATERIAL, **untextured(OSM_MAPPER.get_building_properties("roof_edge")), **hints("roof")
        )
        self.materials.add_building_material(
            ROOF_TRIM_MATERIAL, **untextured(OSM_MAPPER.get_building_properties("roof_trim")), **hints("roof")
        )

    def _finalize_export(self, include_forests: bool = False):
        """Finalisiere Export: Speichere Materials/Items/Forest JSON und Debug-Daten."""
        # Materials (nutze config.MATERIALS_JSON)
        self.materials.save()  # nutzt automatisch config.MATERIALS_JSON
        mat_path = config.BEAMNG_DIR / config.MATERIALS_JSON
        logger.info(f"\n[✓] Materials: {mat_path.name}")

        # Items mit Höhendaten für Spawn-Punkt-Berechnung
        self.items.save(
            height_points=self.height_points, height_elevations=self.height_elevations, global_offset=self.global_offset
        )
        items_path = config.BEAMNG_DIR / config.ITEMS_JSON
        logger.info(f"[✓] Items: {items_path.name}")

        # info.json ins Level-Root-Verzeichnis schreiben
        self.items.save_info_json()
        info_path = config.BEAMNG_DIR / "info.json"
        logger.debug(f"[✓] Info: {info_path.name}")

        # Forest.json (falls Forests aktiviert)
        if include_forests:
            forest_result = self.forests.finalize_forest_export()

            if forest_result["status"] == "success":
                logger.info(f"[✓] Forest: forest.forest4.json ({forest_result['total_trees']} Bäume)")

                # Detaillierte Statistiken
                stats = forest_result["statistics"]
                if stats.get("types"):
                    logger.info(f"    Baumarten:")
                    for tree_type, count in sorted(stats["types"].items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"      • {tree_type}: {count}")
            elif forest_result["status"] == "no_forests":
                logger.info("Keine Wälder generiert")
            else:
                logger.error(f"Forest-Export fehlgeschlagen: {forest_result.get('error')}")

        # main.level.json ist NICHT nötig - BeamNG lädt automatisch main/items.level.json

        # Debug-Netzwerk-Export (auskommentiert für Performance)
        if config.DEBUG_EXPORTS:
            self.debug_exporter.export(config.CACHE_DIR)
