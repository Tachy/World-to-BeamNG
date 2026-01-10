"""
Zentrale BeamNG-Exporter-Fassade.

Bietet eine einheitliche API für den gesamten Export-Workflow.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from ..workflow import TileProcessor, TerrainWorkflow, BuildingWorkflow, HorizonWorkflow


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
        self.materials = MaterialManager(config.BEAMNG_DIR)
        self.items = ItemManager(config.BEAMNG_DIR)
        self.dae = DAEExporter()

        # Workflows
        self.terrain = TerrainWorkflow(self.cache, self.materials, self.items, self.dae)
        self.buildings = BuildingWorkflow(self.cache, self.materials, self.items, self.dae)
        self.horizon = HorizonWorkflow(self.cache, self.materials, self.items, self.dae)
        self.tile_processor = TileProcessor(self.cache)

        # Debug-Exporter für Visualisierung (Singleton - reset für neuen Export)
        from ..utils.debug_exporter import DebugNetworkExporter

        DebugNetworkExporter.reset_instance()
        self.debug_exporter = DebugNetworkExporter.get_instance()

    def export_complete_level(
        self,
        tiles: List[Dict],
        global_offset: Tuple[float, float, float],
        include_buildings: bool = True,
        include_horizon: bool = True,
    ) -> Dict:
        """
        Exportiere komplettes BeamNG-Level.

        Args:
            tiles: Liste von Tile-Metadaten
            global_offset: (origin_x, origin_y, origin_z)
            include_buildings: LoD2-Gebäude exportieren
            include_horizon: Horizon-Layer exportieren

        Returns:
            Dict mit Export-Statistiken
        """
        from ..utils.timing import StepTimer

        timer = StepTimer()
        stats = {"tiles_processed": 0, "tiles_failed": 0, "buildings_exported": 0, "horizon_exported": False}

        print(f"\n{'='*60}")
        print(f"BEAMNG LEVEL EXPORT")
        print(f"{'='*60}")
        print(f"Tiles: {len(tiles)}")
        print(f"Global Offset: {global_offset}")
        print(f"{'='*60}\n")

        # Erstelle Verzeichnisse
        os.makedirs(config.BEAMNG_DIR_SHAPES, exist_ok=True)
        os.makedirs(config.BEAMNG_DIR_TEXTURES, exist_ok=True)
        os.makedirs(config.BEAMNG_DIR_BUILDINGS, exist_ok=True)
        os.makedirs(config.CACHE_DIR, exist_ok=True)

        # Sammle alle Gebäude über alle Tiles
        all_buildings = []

        # Phase 1: Terrain-Tiles
        for tile_idx, tile in enumerate(tiles):
            print(f"\n[Tile {tile_idx + 1}/{len(tiles)}]")

            # Terrain benötigt nur (x, y)
            result = self.terrain.process_tile(tile=tile, global_offset=global_offset[:2], bbox_margin=50.0)

            if result["status"] != "success":
                stats["tiles_failed"] += 1
                continue

            # Exportiere DAE
            tile_x = tile.get("tile_x", 0)
            tile_y = tile.get("tile_y", 0)

            self.terrain.export_tile(tile_x, tile_y, result)
            stats["tiles_processed"] += 1

            # Sammle Gebäude-Daten (werden später gruppiert nach Tiles exportiert)
            if include_buildings and result.get("buildings_data"):
                all_buildings.extend(result["buildings_data"])

        timer.begin("Terrain Export")

        # Phase 2: Buildings (nach Terrain-Export, wie im alten multitile.py)
        if include_buildings and all_buildings:
            from collections import defaultdict

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

        timer.begin("Buildings Export")

        # Phase 3: Horizon-Layer (optional)
        if include_horizon:
            horizon_dae = self.horizon.generate_horizon(global_offset=global_offset)
            stats["horizon_exported"] = horizon_dae is not None

        timer.begin("Horizon Export")

        # Phase 4: Finalisierung
        self._finalize_export()

        timer.begin("Finalisierung")
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
        """Füge LoD2-Gebäude-Materialien zu gemeinsamen Materials hinzu (aus osm_to_beamng.json)."""
        from ..config import OSM_MAPPER

        # Wall-Material aus OSM_MAPPER Config
        wall_props = OSM_MAPPER.get_building_properties("wall")
        wall_name = wall_props.get("internal_name", "lod2_wall_white")

        self.materials.add_building_material(
            wall_name,
            textures=wall_props.get("textures"),
            tiling_scale=wall_props.get("tiling_scale", 4.0),
            groundType="STONE",
            materialTag0="beamng",
            materialTag1="Building",
        )

        # Roof-Material aus OSM_MAPPER Config
        roof_props = OSM_MAPPER.get_building_properties("roof")
        roof_name = roof_props.get("internal_name", "lod2_roof_red")

        self.materials.add_building_material(
            roof_name,
            textures=roof_props.get("textures"),
            tiling_scale=roof_props.get("tiling_scale", 2.0),
            groundType="ROOF_TILES",
            materialTag0="beamng",
            materialTag1="Building",
        )

    def _finalize_export(self):
        """Finalisiere Export: Speichere Materials/Items JSON und Debug-Daten."""
        # Materials
        mat_path = os.path.join(config.BEAMNG_DIR, "main.materials.json")
        self.materials.save(mat_path)
        print(f"\n[✓] Materials: {os.path.basename(mat_path)}")

        # Items
        items_path = os.path.join(config.BEAMNG_DIR, "main.items.json")
        self.items.save(items_path)
        print(f"[✓] Items: {os.path.basename(items_path)}")

        # Debug-Netzwerk-Export (auskommentiert für Performance)
        # if config.DEBUG_EXPORTS:
        #     self.debug_exporter.export(config.CACHE_DIR)

    def clear_cache(self):
        """Lösche gesamten Cache."""
        self.cache.clear_all()
        print("[✓] Cache gelöscht")

    def reset_export(self):
        """Reset: Lösche Materials/Items JSON."""
        mat_path = os.path.join(config.BEAMNG_DIR, "main.materials.json")
        items_path = os.path.join(config.BEAMNG_DIR, "main.items.json")

        if os.path.exists(mat_path):
            os.remove(mat_path)

        if os.path.exists(items_path):
            os.remove(items_path)

        self.materials = MaterialManager(config.BEAMNG_DIR)
        self.items = ItemManager(config.BEAMNG_DIR)

        print("[✓] Export zurückgesetzt")
