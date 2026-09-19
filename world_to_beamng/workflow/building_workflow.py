"""
Building (LoD2) Workflow.

Orchestriert den LoD2-Gebäude-Export.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import logging

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter

logger = logging.getLogger(__name__)

SINGLE_BUILDINGS_NAME = "buildings"  # DAE- und Item-Name, wenn alle Gebäude EIN Objekt sind


def group_buildings(buildings: List[Dict], tile_size: Optional[float]) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Gruppiert Gebäude für den DAE-Export.

    Args:
        buildings: Gebäude-Dicts; "bounds" = (min_x, min_y, min_z, max_x, max_y, max_z)
        tile_size: Kachelgröße in Metern. None/0 = KEINE Kacheln: alle Gebäude in einer Gruppe (0, 0), wie die
            Straßen (DecalRoads) auf der Gesamtfläche liegen.

    Returns:
        {(tile_x, tile_y): [Gebäude]}
    """
    if not tile_size:
        return {(0, 0): list(buildings)} if buildings else {}

    groups: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for building in buildings:
        bounds = building.get("bounds")
        if not bounds:
            continue
        center_x = (bounds[0] + bounds[3]) / 2
        center_y = (bounds[1] + bounds[4]) / 2
        groups[(int((center_x // tile_size) * tile_size), int((center_y // tile_size) * tile_size))].append(building)
    return dict(groups)


def remove_stale_building_daes(directory, keep: Set[str]) -> int:
    """
    Entfernt Gebäude-DAEs (und kompilierte .cdae) einer früheren Aufteilung: buildings_tile_*.dae bzw.
    buildings.dae, die nicht zu `keep` (Dateinamen ohne Endung) gehören.

    Returns:
        Anzahl entfernter Dateien
    """
    removed = 0
    directory = Path(directory)
    if not directory.exists():
        return 0
    for path in directory.iterdir():
        if path.suffix.lower() not in (".dae", ".cdae"):
            continue
        if re.fullmatch(r"buildings(_tile_-?\d+_-?\d+)?", path.stem) and path.stem not in keep:
            path.unlink()
            removed += 1
    return removed


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
        dae_exporter: DAEExporter,
    ):
        self.cache = cache_manager
        self.materials = MaterialManager.get_instance()  # Singleton
        self.items = ItemManager.get_instance()  # Singleton
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
        self,
        buildings: List[Dict],
        tile_x: int,
        tile_y: int,
        grid_bounds: Optional[tuple] = None,
        name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Exportiere Gebäude als DAE.

        Args:
            buildings: Liste von Gebäude-Dicts
            tile_x, tile_y: Tile-Koordinaten
            grid_bounds: Optional - (min_x, max_x, min_y, max_y) für Filterung
            name: Optional - Dateiname ohne Endung (z.B. "buildings" für EIN Objekt auf der Gesamtfläche);
                Standard: buildings_tile_<x>_<y>

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
        output_path = config.BEAMNG_DIR_BUILDINGS / f"{name or f'buildings_tile_{tile_x}_{tile_y}'}.dae"

        self.dae.export_multi_mesh(output_path=output_path, meshes=meshes, with_uv=True)

        logger.info(f"  [✓] Buildings DAE: {output_path.name} ({len(meshes)} Gebäude)")

        return output_path

    def export_materials(self) -> str:
        """
        Exportiere LoD2-Materialien.

        Returns:
            Pfad zur materials.json
        """
        from ..io.lod2 import export_materials_json

        return export_materials_json(output_dir=config.BEAMNG_DIR, material_manager=self.materials)

    def add_items(self, buildings: List[Dict], tile_x: int, tile_y: int, name: Optional[str] = None):
        """
        Füge Gebäude-Items hinzu.

        Args:
            buildings: Liste von Gebäude-Dicts
            tile_x, tile_y: Tile-Koordinaten
            name: Optional - Item-/Dateiname (z.B. "buildings" für EIN Objekt auf der Gesamtfläche)
        """
        from ..io.lod2 import create_items_json_entry

        if not buildings:
            return

        item_name = name or f"buildings_tile_{tile_x}_{tile_y}"
        dae_filename = f"buildings/{item_name}.dae"
        item_entry = create_items_json_entry(dae_filename, tile_x, tile_y, self.items, item_name=item_name)

        # Nutze alle Felder aus item_entry
        self.items.add_item(
            name=item_name,
            item_class=item_entry.get("className", "TSStatic"),
            shape_name=item_entry.get("shapeName", ""),
            position=tuple(item_entry.get("position", (0, 0, 0))),
            rotation=tuple(item_entry.get("rotation", (0, 0, 1, 0))),
            scale=tuple(item_entry.get("scale", (1, 1, 1))),
            collisionType=item_entry.get("collisionType", "Visible Mesh Final"),
        )
