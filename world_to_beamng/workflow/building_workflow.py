"""
Building (LoD2) Workflow.

Orchestrates the LoD2 building export.
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

SINGLE_BUILDINGS_NAME = "buildings"  # DAE and item name when all buildings are ONE object


def group_buildings(buildings: List[Dict], tile_size: Optional[float]) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Groups buildings for the DAE export.

    Args:
        buildings: building dicts; "bounds" = (min_x, min_y, min_z, max_x, max_y, max_z)
        tile_size: tile size in meters. None/0 = NO tiles: all buildings in one group (0, 0), just as the
            roads (DecalRoads) lie on the whole area.

    Returns:
        {(tile_x, tile_y): [buildings]}
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


def plan_building_shapes(
    buildings: List[Dict], tile_size: Optional[float], max_per_shape: int
) -> List[Tuple[int, int, Optional[str], List[Dict]]]:
    """
    Plans the DAE shapes: (tile_x, tile_y, name, buildings).

    BeamNG loads at most 2048 nodes per shape and ignores the rest ("Shape exceeds the maximum node count") -
    each building is one node. Without tiles the whole area is therefore split into spatially contiguous parts with
    at most `max_per_shape` buildings: "buildings", "buildings_part_2", ... (part 1 keeps the name of the
    single object). With tiles it stays at buildings_tile_<x>_<y> (name = None).
    """
    if tile_size:
        return [(x, y, None, group) for (x, y), group in group_buildings(buildings, tile_size).items()]

    if not buildings:
        return []

    def _cell(building: Dict) -> Tuple[int, float]:
        # Buildings without bounds go to the end; otherwise strips of 250 m (south->north), within them west->east
        bounds = building.get("bounds")
        if not bounds:
            return (1 << 30, 0.0)
        return (int(((bounds[1] + bounds[4]) / 2) // 250), (bounds[0] + bounds[3]) / 2)

    ordered = sorted(buildings, key=_cell) if len(buildings) > max_per_shape else list(buildings)
    shapes = []
    for index, start in enumerate(range(0, len(ordered), max_per_shape)):
        name = SINGLE_BUILDINGS_NAME if index == 0 else f"{SINGLE_BUILDINGS_NAME}_part_{index + 1}"
        shapes.append((0, 0, name, ordered[start : start + max_per_shape]))
    return shapes


def remove_stale_building_daes(directory, keep: Set[str]) -> int:
    """
    Removes building DAEs (and compiled .cdae) of an earlier split: buildings_tile_*.dae, buildings.dae
    or buildings_part_*.dae that do not belong to `keep` (file names without extension).

    Returns:
        Number of removed files
    """
    removed = 0
    directory = Path(directory)
    if not directory.exists():
        return 0
    for path in directory.iterdir():
        if path.suffix.lower() not in (".dae", ".cdae"):
            continue
        if re.fullmatch(r"buildings(_tile_-?\d+_-?\d+|_part_\d+)?", path.stem) and path.stem not in keep:
            path.unlink()
            removed += 1
    return removed


class BuildingWorkflow:
    """
    Orchestrates the LoD2 building workflow.

    Responsible for:
    - caching LoD2 data
    - building export
    - material/item management
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

    def export_buildings(
        self,
        buildings: List[Dict],
        tile_x: int,
        tile_y: int,
        grid_bounds: Optional[tuple] = None,
        name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Export buildings as DAE.

        Args:
            buildings: list of building dicts
            tile_x, tile_y: tile coordinates
            grid_bounds: optional - (min_x, max_x, min_y, max_y) for filtering
            name: optional - file name without extension (e.g. "buildings" for ONE object on the whole area);
                default: buildings_tile_<x>_<y>

        Returns:
            Path to the DAE file or None
        """
        from ..builders import BuildingMeshBuilder

        if not buildings:
            return None

        # Use the builder for mesh generation
        meshes = BuildingMeshBuilder().with_buildings(buildings).with_bounds_filter(grid_bounds).build()

        if not meshes:
            return None

        # Export with DAEExporter
        output_path = config.BEAMNG_DIR_BUILDINGS / f"{name or f'buildings_tile_{tile_x}_{tile_y}'}.dae"

        self.dae.export_multi_mesh(output_path=output_path, meshes=meshes, with_uv=True)

        logger.info(f"  [✓] Buildings DAE: {output_path.name} ({len(meshes)} buildings)")

        return output_path

    def add_items(self, buildings: List[Dict], tile_x: int, tile_y: int, name: Optional[str] = None):
        """
        Add building items.

        Args:
            buildings: list of building dicts
            tile_x, tile_y: tile coordinates
            name: optional - item/file name (e.g. "buildings" for ONE object on the whole area)
        """
        from ..io.lod2 import create_items_json_entry

        if not buildings:
            return

        item_name = name or f"buildings_tile_{tile_x}_{tile_y}"
        dae_filename = f"buildings/{item_name}.dae"
        item_entry = create_items_json_entry(dae_filename, tile_x, tile_y, self.items, item_name=item_name)

        # Use all fields from item_entry
        self.items.add_item(
            name=item_name,
            item_class=item_entry.get("className", "TSStatic"),
            shape_name=item_entry.get("shapeName", ""),
            position=tuple(item_entry.get("position", (0, 0, 0))),
            scale=tuple(item_entry.get("scale", (1, 1, 1))),
            collisionType=item_entry.get("collisionType", "Visible Mesh Final"),
        )
