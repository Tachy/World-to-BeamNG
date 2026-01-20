"""
Zentrale BeamNG-Exporter-Fassade.

Bietet eine einheitliche API für den gesamten Export-Workflow.
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from ..workflow import TileProcessor, TerrainWorkflow, BuildingWorkflow, HorizonWorkflow, ForestWorkflow
from ..forest_generator import ForestGenerator

logger = logging.getLogger(__name__)


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

        self.dae = DAEExporter(material_manager=self.materials)  # Übergebe MaterialManager-Referenz

        # Lade osm_to_beamng.json Config (Materials werden SPÄTER generiert!)
        osm_config_path = Path("data/osm_to_beamng.json")
        self.osm_config = {}
        self.forest_config = {"forest_types": {}, "forest_mappings": {}}

        if osm_config_path.exists():
            with open(osm_config_path, "r", encoding="utf-8") as f:
                self.osm_config = json.load(f)
                # Nur Mappings laden - forest_types wird LAZY generiert!
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
            include_forests: Wald-Vegetation exportieren (NEU)

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
        }

        print(f"\n{'='*60}")
        print(f"BEAMNG LEVEL EXPORT")
        print(f"{'='*60}")
        print(f"Tiles: {len(tiles)}")
        print(f"Global Offset: {global_offset}")
        print(f"Forests: {'Yes' if include_forests else 'No'}")  # NEU
        print(f"{'='*60}\n")

        # Speichere global_offset für Spawn-Punkt-Berechnung
        self.global_offset = global_offset[:2]  # Nur (x, y)

        # Erstelle Verzeichnisse
        config.BEAMNG_DIR_SHAPES.mkdir(parents=True, exist_ok=True)
        config.BEAMNG_DIR_TEXTURES.mkdir(parents=True, exist_ok=True)
        config.BEAMNG_DIR_BUILDINGS.mkdir(parents=True, exist_ok=True)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # NEU: Phase 0 - Forest Asset Initialization (DIREKT VOR Tile-Loop)
        registered_trees = {}
        if include_forests:
            timer.begin("Forest Asset Scanning")

            # Importiere und nutze Asset Scanner direkt
            from ..forest.forest_asset_scanner import ForestAssetScanner

            asset_scanner = ForestAssetScanner(
                base_dir=Path(config.BEAMNG_DIR), material_manager=self.materials, item_manager=self.items
            )
            registered_trees = asset_scanner.scan_and_register_trees()
            stats["forests_registered"] = len(registered_trees)

            print(f"\n✓ {stats['forests_registered']} Baumarten registriert\n")

            # LAZY: Generiere forest_types JETZT mit den registrierten Trees!
            print(f"  [→] Generiere dynamische Forest-Types aus {len(registered_trees)} Baumarten...")
            generator = ForestGenerator(
                registered_trees=registered_trees, template_config=self.osm_config.get("forest_type_templates", {})
            )
            self.forest_config["forest_types"] = generator.generate_forest_types()

            # Setze Forest-Konfiguration (initialisiert Normalizer, InstanceGenerator, JSONWriter)
            self.forests.set_forest_config(
                self.forest_config,
                osm_mapper=config.OSM_MAPPER,
                registered_trees=registered_trees,
            )

        # Sammle alle Gebäude über alle Tiles
        all_buildings = []
        tile_bounds_local = []  # Sammle Tile-Grenzen für Horizon-Clipping

        # Speichere Terrain-Daten für Horizon-Stitching
        terrain_mesh = None
        terrain_vertex_manager = None
        terrain_grid_bounds = None

        # Phase 1: Terrain-Tiles
        for tile_idx, tile in enumerate(tiles):

            timer.begin(f"[Tile {tile_idx + 1}/{len(tiles)}]")

            # Terrain benötigt nur (x, y)
            result = self.terrain.process_tile(tile=tile, global_offset=global_offset[:2], bbox_margin=50.0)

            if result["status"] != "success":
                stats["tiles_failed"] += 1
                continue

            # Speichere Höhendaten vom ersten erfolgreichen Tile für Spawn-Punkt-Berechnung
            if self.height_points is None and self.height_elevations is None:
                self.height_points = result.get("height_points")
                self.height_elevations = result.get("height_elevations")
                print(
                    f"  [i] Spawn-Höhendaten: {len(self.height_points) if self.height_points is not None else 0} Punkte"
                )

            # Speichere Terrain-Daten für Horizon-Stitching (vom letzten erfolgreichen Tile)
            if result.get("terrain_mesh") is not None:
                terrain_mesh = result.get("terrain_mesh")
                terrain_vertex_manager = result.get("vertex_manager")
                terrain_grid_bounds = result.get("grid_bounds_local")
                vm_count = terrain_vertex_manager.get_count() if terrain_vertex_manager else 0
                print(f"  [i] Terrain-Daten für Stitching: VM={vm_count} Vertices, Bounds={terrain_grid_bounds}")

            # Exportiere DAE
            tile_x = tile.get("tile_x", 0)
            tile_y = tile.get("tile_y", 0)

            self.terrain.export_tile(tile_x, tile_y, result)
            stats["tiles_processed"] += 1

            # Sammle Tile-Grenzen für Horizon-Clipping (in lokalen Koordinaten!)
            # tile ist der große DGM1-Tile (2×2 km), nicht der DAE-Export-Tile (500m)
            # tile_x und tile_y sind in UTM-Koordinaten, also erst zu lokal konvertieren
            ox, oy = global_offset[0], global_offset[1]
            tile_x_local = tile_x - ox
            tile_y_local = tile_y - oy

            # Verwende die tatsächliche Tile-Größe (2000m für DGM1-Tiles)
            large_tile_size = tile.get("tile_size", 2000)

            x_min = tile_x_local
            x_max = tile_x_local + large_tile_size
            y_min = tile_y_local
            y_max = tile_y_local + large_tile_size
            tile_bounds_local.append((x_min, y_min, x_max, y_max))

            # Phase 1b: Forest Processing (pro Tile)
            if include_forests:
                forest_result = self.forests.process_tile(
                    tile_bounds=(x_min, y_min, x_max, y_max),
                    tile_name=f"T{tile_x:.0f}_{tile_y:.0f}",
                    elevation_data=result.get("height_points"),
                    height_grid_info={
                        "origin": (x_min, y_min),
                        "spacing": 1.0,
                        "elevations": result.get("height_elevations"),
                    },
                    height_hash=result.get("height_hash"),  # Für Cache-Konsistenz
                    global_offset=global_offset,  # NEU: Für WGS84-Transformation
                )
                if forest_result["status"] == "success":
                    stats["trees_generated"] += forest_result.get("tree_count", 0)

            # Sammle Gebäude-Daten (werden später gruppiert nach Tiles exportiert)
            if include_buildings and result.get("buildings_data"):
                all_buildings.extend(result["buildings_data"])

        # Phase 2: Buildings (nach Terrain-Export, wie im alten multitile.py)
        if include_buildings and all_buildings:
            from collections import defaultdict

            timer.begin("Buildings Export")

            # Gruppiere Gebäude nach DAE-Tiles (500m x 500m)
            buildings_by_tile = defaultdict(list)
            for building in all_buildings:
                bounds = building.get("bounds")
                if not bounds:
                    continue
                center_x = (bounds[0] + bounds[3]) / 2
                center_y = (bounds[1] + bounds[4]) / 2
                tile_x = int((center_x // config.TILE_SIZE) * config.TILE_SIZE)
                tile_y = int((center_y // config.TILE_SIZE) * config.TILE_SIZE)
                buildings_by_tile[(tile_x, tile_y)].append(building)

            # Exportiere pro DAE-Tile
            for (tile_x, tile_y), tile_buildings in buildings_by_tile.items():
                dae_path = self.buildings.export_buildings(tile_buildings, tile_x, tile_y, grid_bounds=None)
                if dae_path:
                    self.buildings.add_items(tile_buildings, tile_x, tile_y)
                    stats["buildings_exported"] += len(tile_buildings)

            # Materials exportieren
            # Füge LoD2-Materialien zu gemeinsamen Materials hinzu (NICHT separat exportieren!)
            self._add_lod2_materials()

        timer.begin("Horizon Export")

        # Phase 3: Horizon-Layer (optional)
        stitching_faces = []
        if include_horizon:
            # Übergebe Tile-Grenzen und Terrain-Daten für Boundary-Stitching
            result = self.horizon.generate_horizon(
                global_offset=global_offset,
                tile_bounds=tile_bounds_local,
                terrain_mesh=terrain_mesh,
                terrain_vertex_manager=terrain_vertex_manager,
                terrain_grid_bounds=terrain_grid_bounds,
            )

            if result is not None:
                horizon_dae, stitching_faces = result
                stats["horizon_exported"] = horizon_dae is not None
                if stitching_faces:
                    print(f"  [i] {len(stitching_faces)} Stitching-Faces für Horizon-Terrain-Verbindung")
            else:
                stats["horizon_exported"] = False

        timer.begin("Finalisierung")

        # Phase 4: Finalisierung
        self._finalize_export(include_forests)

        timer.report()

        return stats

    def export_single_tile(
        self, tile: Dict, global_offset: Tuple[float, float], tile_x: int = 0, tile_y: int = 0
    ) -> Optional[str]:
        """
        Exportiere einzelnes Tile.

        Args:
            tile: Tile-Metadaten
            global_offset: (origin_x, origin_y)
            tile_x, tile_y: Tile-Koordinaten

        Returns:
            Pfad zur DAE-Datei oder None
        """
        result = self.terrain.process_tile(tile=tile, global_offset=global_offset)

        if result["status"] != "success":
            return None

        return self.terrain.export_tile(tile_x, tile_y, result)

    def export_terrain_only(self, tiles: List[Dict], global_offset: Tuple[float, float]) -> int:
        """
        Exportiere nur Terrain (keine Buildings/Horizon).

        Args:
            tiles: Liste von Tile-Metadaten
            global_offset: (origin_x, origin_y)

        Returns:
            Anzahl erfolgreich exportierter Tiles
        """
        count = 0

        for tile in tiles:
            result = self.terrain.process_tile(tile, global_offset)

            if result["status"] == "success":
                tile_x = tile.get("tile_x", 0)
                tile_y = tile.get("tile_y", 0)
                self.terrain.export_tile(tile_x, tile_y, result)
                count += 1

        self._finalize_export()
        return count

    def _add_lod2_materials(self):
        """
        Füge LoD2-Gebäude-Materialien hinzu (aus JSON-Templates und OSM_MAPPER Config).

        Diese Methode mergt:
        - buildings wall/roof Konfigurationen aus material_templates.json
        - Farben/Texturen aus osm_to_beamng.json (OSM_MAPPER)
        """
        from ..config import OSM_MAPPER

        # Hole Template-Konfigurationen
        config = self.materials.get_templates()
        buildings_config = config.get("buildings", {})

        # Wall-Material: Zusammenführung von Template + OSM-Properties
        wall_template = buildings_config.get("wall", {})
        wall_props = OSM_MAPPER.get_building_properties("wall")
        wall_name = wall_props.get("internal_name", "lod2_wall_white")

        self.materials.add_building_material(
            wall_name,
            color=wall_props.get("diffuseColor"),
            textures=wall_props.get("textures"),
            tiling_scale=wall_template.get("tiling_scale", wall_props.get("tiling_scale", 4.0)),
            groundType=wall_template.get("material_hints", {}).get("groundType", "concrete"),
            materialTag0=wall_template.get("material_hints", {}).get("materialTag0", "beamng"),
            materialTag1=wall_template.get("material_hints", {}).get("materialTag1", "Building"),
        )

        # Roof-Material: Zusammenführung von Template + OSM-Properties
        roof_template = buildings_config.get("roof", {})
        roof_props = OSM_MAPPER.get_building_properties("roof")
        roof_name = roof_props.get("internal_name", "lod2_roof_red")

        self.materials.add_building_material(
            roof_name,
            color=roof_props.get("diffuseColor"),
            textures=roof_props.get("textures"),
            tiling_scale=roof_template.get("tiling_scale", roof_props.get("tiling_scale", 2.0)),
            groundType=roof_template.get("material_hints", {}).get("groundType", "concrete"),
            materialTag0=roof_template.get("material_hints", {}).get("materialTag0", "beamng"),
            materialTag1=roof_template.get("material_hints", {}).get("materialTag1", "Building"),
        )

    def _finalize_export(self, include_forests: bool = False):
        """Finalisiere Export: Speichere Materials/Items/Forest JSON und Debug-Daten."""
        # Materials (nutze config.MATERIALS_JSON)
        self.materials.save()  # nutzt automatisch config.MATERIALS_JSON
        mat_path = config.BEAMNG_DIR / config.MATERIALS_JSON
        print(f"\n[✓] Materials: {mat_path.name}")

        # Items mit Höhendaten für Spawn-Punkt-Berechnung
        self.items.save(
            height_points=self.height_points, height_elevations=self.height_elevations, global_offset=self.global_offset
        )
        items_path = config.BEAMNG_DIR / config.ITEMS_JSON
        print(f"[✓] Items: {items_path.name}")

        # info.json ins Level-Root-Verzeichnis schreiben
        self.items.save_info_json()
        info_path = config.BEAMNG_DIR / "info.json"
        print(f"[✓] Info: {info_path.name}")

        # Forest.json (falls Forests aktiviert)
        if include_forests:
            forest_result = self.forests.finalize_forest_export()

            if forest_result["status"] == "success":
                print(f"[✓] Forest: forest.json ({forest_result['total_trees']} Bäume)")

                # Detaillierte Statistiken
                stats = forest_result["statistics"]
                if stats.get("types"):
                    print(f"    Baumarten:")
                    for tree_type, count in sorted(stats["types"].items(), key=lambda x: x[1], reverse=True):
                        print(f"      • {tree_type}: {count}")
            elif forest_result["status"] == "no_forests":
                logger.info("Keine Wälder generiert")
            else:
                logger.error(f"Forest-Export fehlgeschlagen: {forest_result.get('error')}")

        # main.level.json ist NICHT nötig - BeamNG lädt automatisch main/items.level.json

        # Debug-Netzwerk-Export (auskommentiert für Performance)
        if config.DEBUG_EXPORTS:
            self.debug_exporter.export(config.CACHE_DIR)

    def clear_cache(self):
        """Lösche gesamten Cache."""
        self.cache.clear_all()
        print("[✓] Cache gelöscht")

    def reset_export(self):
        """Reset: Lösche Materials/Items JSON."""
        mat_path = config.BEAMNG_DIR / config.MATERIALS_JSON
        items_path = config.BEAMNG_DIR / config.ITEMS_JSON

        if mat_path.exists():
            mat_path.unlink()

        if items_path.exists():
            items_path.unlink()

        # Reset Singleton-Instanzen
        MaterialManager.reset_instance()
        self.materials = MaterialManager.get_instance(config.BEAMNG_DIR)

        ItemManager.reset_instance()
        self.items = ItemManager.get_instance(config.BEAMNG_DIR)

        print("[✓] Export zurückgesetzt")
