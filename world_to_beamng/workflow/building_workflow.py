"""
Building (LoD2) Workflow.

Orchestriert den LoD2-Gebäude-Export.
"""

from typing import Dict, List, Optional
from pathlib import Path

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter


class BuildingWorkflow:
    """
    Orchestriert den LoD2-Gebäude-Workflow.

    Verantwortlich für:
    - LoD2-Daten cachen
    - Gebäude-Export
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

    def cache_buildings(self, bbox: tuple, global_offset: tuple) -> Optional[Dict]:
        """
        Lade und cache LoD2-Gebäude.

        Args:
            bbox: (min_x, max_x, min_y, max_y)
            global_offset: (origin_x, origin_y)

        Returns:
            Gebäude-Daten oder None
        """
        from ..io.lod2 import cache_lod2_buildings

        return cache_lod2_buildings(bbox=bbox, local_offset=global_offset, cache_manager=self.cache)

    def export_buildings(
        self, buildings: List[Dict], tile_x: int, tile_y: int, grid_bounds: Optional[tuple] = None
    ) -> Optional[str]:
        """
        Exportiere Gebäude als DAE.

        Args:
            buildings: Liste von Gebäude-Dicts
            tile_x, tile_y: Tile-Koordinaten
            grid_bounds: Optional - (min_x, max_x, min_y, max_y) für Filterung

        Returns:
            Pfad zur DAE-Datei oder None
        """
        from ..builders import BuildingMeshBuilder

        if not buildings:
            return None

        # Verwende Builder für Mesh-Generierung
        meshes = BuildingMeshBuilder().with_buildings(buildings).with_bounds_filter(grid_bounds).build()

        if not meshes:
            return None

        # Exportiere mit DAEExporter
        import os

        output_path = os.path.join(config.BEAMNG_DIR_BUILDINGS, f"buildings_tile_{tile_x}_{tile_y}.dae")

        self.dae.export_multi_mesh(output_path=output_path, meshes=meshes, with_uv=True)

        print(f"  [✓] Buildings DAE: {os.path.basename(output_path)} ({len(meshes)} Gebäude)")

        return output_path

    def export_materials(self) -> str:
        """
        Exportiere LoD2-Materialien.

        Returns:
            Pfad zur materials.json
        """
        from ..io.lod2 import export_materials_json

        return export_materials_json(output_dir=config.BEAMNG_DIR)

    def add_items(self, buildings: List[Dict], tile_x: int, tile_y: int):
        """
        Füge Gebäude-Items hinzu.

        Args:
            buildings: Liste von Gebäude-Dicts
            tile_x, tile_y: Tile-Koordinaten
        """
        from ..io.lod2 import create_items_json_entry

        if not buildings:
            return

        dae_filename = f"buildings/buildings_tile_{tile_x}_{tile_y}.dae"
        item_entry = create_items_json_entry(dae_filename, tile_x, tile_y)

        # Item-Name muss mit create_items_json_entry() übereinstimmen
        item_name = f"buildings_tile_{tile_x}_{tile_y}"

        # Nutze alle Felder aus item_entry
        self.items.add_item(
            name=item_name,
            item_class=item_entry.get("className", "TSStatic"),
            shape_name=item_entry.get("shapeName", ""),
            position=tuple(item_entry.get("position", (0, 0, 0))),
            rotation=tuple(item_entry.get("rotation", (0, 0, 1, 0))),
            scale=tuple(item_entry.get("scale", (1, 1, 1))),
            collisionType=item_entry.get("collisionType", "Visible Mesh"),
        )
