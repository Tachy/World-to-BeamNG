"""
Central BeamNG exporter facade.

Provides a unified API for the entire export workflow.
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
from ..progress import Pipeline, optional_subtask

logger = LoggerConfig.get_logger()


class BeamNGExporter:
    """
    Central facade for the BeamNG level export.

    Simplifies the API and orchestrates all sub-workflows.

    Example:
        >>> exporter = BeamNGExporter(pipeline)
        >>> exporter.export_complete_level(tiles)
    """

    def __init__(self, pipeline: Pipeline):
        """
        Initialize the BeamNGExporter.

        Args:
            pipeline: Pipeline instance for the main task display (see progress.py)
        """
        self.pipeline = pipeline

        # Core Components
        self.cache = CacheManager(Path(config.CACHE_DIR))

        # Singleton managers (reset for a new export)
        MaterialManager.reset_instance()
        self.materials = MaterialManager.get_instance(config.BEAMNG_DIR)

        ItemManager.reset_instance()
        self.items = ItemManager.get_instance(config.BEAMNG_DIR)

        # Register the central Forest object in items.level.json
        self.items.add_item(
            name="the_forest",
            item_class="Forest",
            dataFile="levels/world_to_beamng/forest/forest.forest4.json",
            lodScale=1.0,
            overwrite=True,
        )
        logger.debug("✓ Forest object registered in ItemManager")

        self.dae = DAEExporter(material_manager=self.materials)  # pass the MaterialManager reference

        # Load the osm_to_beamng.json config (materials are generated LATER!)
        osm_config_path = Path("data/osm_to_beamng.json")
        self.osm_config = {}
        self.forest_config = {"forest_type_templates": {}, "forest_mappings": {}}

        if osm_config_path.exists():
            with open(osm_config_path, "r", encoding="utf-8") as f:
                self.osm_config = json.load(f)
                # Load forest_type_templates and forest_mappings
                self.forest_config["forest_type_templates"] = self.osm_config.get("forest_type_templates", {})
                self.forest_config["forest_mappings"] = self.osm_config.get("forest_mappings", {})

        # Workflows (use MaterialManager/ItemManager.get_instance() internally)
        self.terrain = TerrainWorkflow(self.cache, self.dae)
        self.buildings = BuildingWorkflow(self.cache, self.dae)
        self.horizon = HorizonWorkflow(self.cache, self.dae)
        self.tile_processor = TileProcessor(self.cache)
        self.forests = ForestWorkflow(config)  # config only, no asset scanning

        # Debug exporter for visualization (singleton - reset for a new export)
        from ..utils.debug_exporter import DebugNetworkExporter

        DebugNetworkExporter.reset_instance()
        self.debug_exporter = DebugNetworkExporter.get_instance()

        # Road centerlines for the automatic vehicle spawn position (see
        # managers/item_manager.py::_compute_vehicle_spawn())
        self.road_polygons = None

        # POI candidates (villages/towns, large parking lots) for additional spawn points, see
        # managers/item_manager.py::_compute_poi_spawn_points()
        self.poi_points = None
        self.tunnel_spawns = None

        # Photo tiles (+ status) for POI preview images, see _finalize_export()/io/aerial.py::
        # build_poi_preview_image() - None/"none" until export_complete_level() has built them.
        self.aerial_photos = None
        self.aerial_photo_status = "none"

    def export_complete_level(
        self,
        tiles: List[Dict],
        global_offset: Tuple[float, float, float],
        include_buildings: bool = True,
        include_horizon: bool = True,
        include_forests: bool = True,  # NEW: forest export
    ) -> Dict:
        """
        Export the complete BeamNG level.

        Args:
            tiles: List of tile metadata
            global_offset: (origin_x, origin_y, origin_z)
            include_buildings: Export LoD2 buildings
            include_horizon: Export the horizon layer
            include_forests: Export forest vegetation (can be overridden via config)

        Returns:
            Dict with export statistics
        """
        stats = {
            "tiles_processed": 0,
            "tiles_failed": 0,
            "buildings_exported": 0,
            "horizon_exported": False,
            "forests_registered": 0,  # NEW
            "trees_generated": 0,  # NEW
            "vine_segments": 0,
        }

        forests_enabled = include_forests and config.FORESTS_ENABLED
        if include_forests and not config.FORESTS_ENABLED:
            logger.info("Forest export disabled in config (config.FORESTS_ENABLED=False)")

        # Combined hash over all tiles - same cache identity as in
        # terrain_workflow.py::process_tile() (OSM/Elevation/Grid), additionally needed here for the
        # DGM30 horizon cache (see horizon.py::_dgm30_cache_file()).
        from ..io.cache import calculate_global_tiles_hash

        tile_hash = calculate_global_tiles_hash(tiles) if tiles else "unknown"

        self.pipeline.banner(
            f"BeamNG Level Export - {len(tiles)} Tiles, Offset {global_offset}, "
            f"Forests: {'on' if forests_enabled else 'off'}"
        )

        # Create directories
        config.BEAMNG_DIR_SHAPES.mkdir(parents=True, exist_ok=True)
        config.BEAMNG_DIR_TEXTURES.mkdir(parents=True, exist_ok=True)
        config.BEAMNG_DIR_BUILDINGS.mkdir(parents=True, exist_ok=True)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Check all textures from data/textures (generate procedural ones once if needed); if a photo texture is missing,
        # the export aborts here (MissingTexturesError) - before the compute-intensive part
        from ..textures import registry

        with self.pipeline.task("Textures") as task:
            registry.prepare_textures()
            task.done()

        # NEW: Phase 0 - forest asset initialization (DIRECTLY BEFORE the tile loop)
        registered_trees = {}
        vineyard_assets_ready = False
        if forests_enabled:
            with self.pipeline.task("Forest assets") as task:
                # Ensure vine assets for vineyards (idempotent) - BEFORE loading
                # managedItemData.json, so that the vines are registered as forest items.
                if config.VINEYARDS_ENABLED:
                    try:
                        ensure_vineyard_assets(config.BEAMNG_DIR, get_beamng_install_dir(), config.LEVEL_NAME)
                        vineyard_assets_ready = True
                    except Exception as e:
                        logger.warning(f"Vine assets not available - vineyards stay without vines: {e}")

                # Load managedItemData.json (created by generate_forest_assets.py)
                forest_item_data_path = config.BEAMNG_DIR / "art" / "forest" / "managedItemData.json"

                if forest_item_data_path.exists():
                    try:
                        with open(forest_item_data_path, "r", encoding="utf-8") as f:
                            forest_item_data = json.load(f)

                        # Convert to registered_trees format (key MUST be the internalName,
                        # because forest.forest4.json references trees via the "type" field)
                        for item_key, item_info in forest_item_data.items():
                            internal_name = item_info.get("internalName", item_key)
                            # Vines are not forest trees: otherwise the fallback of the
                            # tree species selection ("first available tree") could plant them in the forest.
                            if internal_name in VINEYARD_ITEM_NAMES:
                                continue
                            registered_trees[internal_name] = {
                                "name": internal_name,
                                "dae_path": item_info.get("shapeFile", ""),
                                "radius": item_info.get("radius", 1.5),
                            }

                    except Exception as e:
                        logger.error(f"Error loading managedItemData.json: {e}")
                        registered_trees = {}
                else:
                    logger.warning(f"managedItemData.json not found: {forest_item_data_path}")
                    logger.warning("  Please run first: python tools/generate_forest_assets.py")

                stats["forests_registered"] = len(registered_trees)

                # Set the forest configuration (initializes Normalizer, InstanceGenerator, JSONWriter)
                if registered_trees:
                    self.forests.set_forest_config(
                        self.forest_config,
                        osm_mapper=config.OSM_MAPPER,
                        registered_trees=registered_trees,
                    )
                task.done(f"{len(registered_trees)} tree items")
        else:
            self.pipeline.skip("Forest assets", "FORESTS_ENABLED=False")

        # Collect all buildings across all tiles
        all_buildings = []
        tile_bounds_local = []  # collect tile bounds for horizon clipping

        # Height lookup on the finished terrain heightmap (seam and height transition of the horizon)
        terrain_height_at = None

        # Generate ONE combined aerial photo for the whole area (no longer
        # one texture per 500 m tile - see io/aerial.py::process_aerial_images()
        # docstring: BeamNG's terrain atlas packer visibly rotates tiles when
        # it is given too many large, unique materials).
        from ..utils.tile_scanner import compute_global_bbox
        from ..terrain.heightmap import next_power_of_two_size

        utm_min_x, utm_max_x, utm_min_y, utm_max_y = compute_global_bbox(tiles)

        # The aerial photo MUST cover exactly the same area that the terrain
        # will actually occupy later - and that is NOT the raw DGM tile
        # bbox (e.g. 2000x2000 m), but the heightmap size padded to a power
        # of two (e.g. 2048x2048 m, see heightmap.py:build_heightmap()).
        # Without this alignment the TerrainMaterialTextureSet claims a
        # different edge length (baseColorBaseTexSize) than the photo actually
        # shows -> the image is projected onto the terrain with the wrong scale.
        # Padding extends (as with the elevation data) ONLY to the east/north.
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
        aerial_dir = config.AERIAL_DATA_DIR
        from ..io.aerial import ensure_aerial_photos, SINGLE_PHOTO_NAME
        from ..terrain.photo_tiles import build_processing_tile_grid, photo_tile_specs

        # Four-image mode: fixed tile grid over the whole area (config.PHOTO_TILE_SIZE_M),
        # independent of the size/number of the raw elevation data tiles (see terrain/photo_tiles.py).
        # process_tile() (terrain_workflow.py) builds the same grid from the same inputs
        # (deterministically, without having to share data) for the layer map split.
        processing_tiles = build_processing_tile_grid((utm_min_x, utm_max_x, utm_min_y, utm_max_y), config.PHOTO_TILE_SIZE_M)
        if config.AERIAL_PHOTO_PER_TILE and len(processing_tiles) > 1:
            photos = photo_tile_specs(processing_tiles, global_offset)
        else:
            photos = [{"name": SINGLE_PHOTO_NAME, "bounds": combined_grid_bounds_local}]

        # The photos are rebuilt as soon as area, origin, resolution, tile layout or source images no
        # longer match the existing ones (e.g. switching from one to four DGM1 tiles) - not only when they are missing.
        status = "none"  # fallback in case ensure_aerial_photos() below raises an exception (see minimap step further below)
        with self.pipeline.task("Aerial photo") as task:
            try:
                status = ensure_aerial_photos(
                    aerial_dir=aerial_dir, output_dir=textures_dir, photos=photos, global_offset=global_offset
                )
                if status == "current":
                    task.done(f"{len(photos)} aerial photo(s) match the area - reused")
                elif status == "built":
                    task.done(f"{len(photos)} aerial photo(s) rebuilt")
                elif status == "failed":
                    task.fail("Aerial photo could not be built")
            except Exception as e:
                task.fail(str(e))

        # For POI preview images in _finalize_export() (build_poi_preview_image() crops from the
        # already built aerial photo PNGs, see there) - only useful if a photo actually exists.
        self.aerial_photos = photos
        self.aerial_photo_status = status

        # Phase 1: terrain + roads - process ALL tiles as ONE contiguous
        # area (one grid, one road network, one junction pass).
        # Clipping now only happens at the outer edge of the whole area, no
        # longer at the former DGM1 tile borders (see process_tile() docstring).
        with self.pipeline.task("Terrain + roads") as task:
            result = self.terrain.process_tile(tiles=tiles, global_offset=global_offset[:2], bbox_margin=50.0, task=task)

            if result["status"] != "success":
                stats["tiles_failed"] = len(tiles)
                task.fail(f"Terrain processing failed: {result.get('reason')}")
            else:
                stats["tiles_processed"] = len(tiles)

                # Road centerlines for the automatic vehicle spawn position (already with the
                # later terrain embedding height, see road_embedding.py)
                self.road_polygons = result.get("road_slope_polygons_2d")
                self.poi_points = result.get("poi_points")
                self.tunnel_spawns = result.get("tunnel_spawns")

                self.terrain.export_tile(0, 0, result, task=task)

                # BigMap preview image from the already built aerial photo PNGs (only if any were built/are current -
                # without an aerial photo a minimap image makes no sense, see io/aerial.py::build_minimap_image()).
                if config.MINIMAP_ENABLED and status in ("current", "built"):
                    from ..io.aerial import MINIMAP_FILENAME, MINIMAP_SUBDIR, ensure_minimap_image, minimap_info_json_fields

                    with task.subtask("Minimap") as sub:
                        x_min, x_max, y_min, y_max = combined_grid_bounds_local
                        minimap_path = config.BEAMNG_DIR / MINIMAP_SUBDIR / MINIMAP_FILENAME
                        minimap_status = ensure_minimap_image(textures_dir, minimap_path, photos, combined_grid_bounds_local)
                        if minimap_status != "missing":
                            self.items.set_info_json_fields(**minimap_info_json_fields(x_min, y_max, x_max - x_min))
                            if minimap_status == "current":
                                logger.info(f"[OK] Minimap matches the aerial photo - reused: {minimap_path}")
                                sub.finish(f"{minimap_path.name} reused")
                            else:
                                logger.info(f"[OK] Minimap saved: {minimap_path}")
                                sub.finish(minimap_path.name)
                        else:
                            logger.info("[i] Minimap skipped (source photo missing)")
                            sub.warn("source photo missing")

                # Height lookup on the finished heightmap: the horizon derives its terrain hole
                # including the edge heights from it (no terrain mesh anymore that could be stitched)
                from ..terrain.road_embedding import sample_heightmap_bilinear

                heightmap = result["heightmap"]
                hm_origin = (result["terrain_origin_x"], result["terrain_origin_y"])
                terrain_height_at = lambda x, y: sample_heightmap_bilinear(
                    heightmap, hm_origin[0], hm_origin[1], config.TERRAIN_SQUARE_SIZE,
                    np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]),
                )

                from ..forest.vineyard_generator import make_height_sampler

                terrain_height_at_1d = make_height_sampler(heightmap, hm_origin[0], hm_origin[1], config.TERRAIN_SQUARE_SIZE)

                # Overall bbox in local coordinates for horizon clipping
                x_min, x_max, y_min, y_max = result["grid_bounds_local"]
                tile_bounds_local.append((x_min, y_min, x_max, y_max))

                # Phase 1b: forest processing (for the whole area, no longer per tile)
                if forests_enabled:
                    with task.subtask("Forest placement") as sub:
                        forest_result = self.forests.process_tile(
                            tile_bounds=(x_min, y_min, x_max, y_max),
                            tile_name="combined_area",
                            elevation_data=result.get("height_points"),
                            height_grid_info={
                                "origin": (x_min, y_min),
                                "spacing": 1.0,
                                "elevations": result.get("height_elevations"),
                            },
                            height_hash=result.get("height_hash"),  # for cache consistency
                            global_offset=global_offset,  # NEW: for the WGS84 transformation
                            # Trees stand on the finished heightmap (after road embedding), not on raw DGM1 points,
                            # and avoid the road surfaces that were actually embedded
                            height_at=terrain_height_at_1d,
                            road_surfaces=result.get("road_surface_union"),
                        )
                        if forest_result["status"] == "success":
                            stats["trees_generated"] += forest_result.get("tree_count", 0)

                        # Vineyard vines (forest items) together with the trees in forest.forest4.json
                        vine_segments = 0
                        if vineyard_assets_ready and result.get("vineyard_instances"):
                            vine_segments = self.forests.add_instances(result["vineyard_instances"])
                            stats["vine_segments"] += vine_segments

                        sub.finish(f"{forest_result.get('tree_count', 0)} trees, {vine_segments} vine row segments")

                # Collect building data (exported later, grouped by tiles)
                if include_buildings and result.get("buildings_data"):
                    all_buildings.extend(result["buildings_data"])

        # Phase 2: buildings (after the terrain export, as in the old multitile.py)
        if include_buildings and all_buildings:
            with self.pipeline.task("Export buildings") as task:
                # Buildings: ONE object over the whole area (like the roads) or - if disabled - one per 500 m tile
                from ..workflow.building_workflow import plan_building_shapes, remove_stale_building_daes

                # BeamNG discards shapes with more than 2048 nodes -> split the whole area into partial shapes
                # (buildings, buildings_part_N)
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

                # Remove the DAEs of the respective other layout (earlier tiles or the single overall object)
                remove_stale_building_daes(config.BEAMNG_DIR_BUILDINGS, keep=written)

                # Export materials
                # Add LoD2 materials to the shared materials (do NOT export separately!)
                self._add_lod2_materials()
                task.done(f"{stats['buildings_exported']} buildings")
        elif not include_buildings:
            self.pipeline.skip("Export buildings", "LOD2_ENABLED=False")
        else:
            self.pipeline.skip("Export buildings", "no building data found")

        # Phase 3: horizon layer (optional)
        if include_horizon:
            with self.pipeline.task("Export horizon") as task:
                horizon_dae = self.horizon.generate_horizon(
                    global_offset=global_offset,
                    tile_hash=tile_hash,
                    tile_bounds=tile_bounds_local,
                    terrain_height_at=terrain_height_at,
                    task=task,
                )
                stats["horizon_exported"] = horizon_dae is not None
                if horizon_dae:
                    task.done(Path(horizon_dae).name)
                else:
                    # identical to the warning in horizon_workflow.py::generate_horizon()
                    task.warn("DGM30 data not found - no horizon created")
        else:
            self.pipeline.skip("Export horizon", "PHASE5_ENABLED=False")

        # Phase 4: finalization
        with self.pipeline.task("Finalization") as task:
            self._finalize_export(forests_enabled, task=task)
            task.done()

        return stats

    def _add_lod2_materials(self):
        """
        Add the LoD2 building materials.

        - Walls: one material per plaster color (own albedo texture, shared normal/roughness texture)
        - Windows: sprite atlas for windows, doors and basement windows
        - Roof: beaver-tail tiles from osm_to_beamng.json (unchanged)
        - Flat roof: gravel surface (texture from data/textures); sheet-metal rim and roof overhang trim: untextured

        Textures of the procedural materials come from ensure_building_textures(), the gravel texture from the texture
        registry (textures/registry.py); colors and factors of the
        untextured ones from osm_to_beamng.json (OSM_MAPPER), template hints from material_templates.json.
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

    def _finalize_export(self, include_forests: bool = False, task=None):
        """Finalize the export: save materials/items/forest JSON and debug data (one subtask per step)."""
        # Materials (use config.MATERIALS_JSON)
        with optional_subtask(task, "Materials"):
            self.materials.save()  # automatically uses config.MATERIALS_JSON
            mat_path = config.BEAMNG_DIR / config.MATERIALS_JSON
            logger.info(f"\n[✓] Materials: {mat_path.name}")

        # Items incl. automatic vehicle spawn position (nearest road to the area center) and POI
        # spawn points (villages/towns, large parking lots) with a preview image from the already built aerial photo.
        with optional_subtask(task, "Items + spawn points"):
            # Preview images: top-down crop from the already built aerial photo, POI centered - unchanged
            # ones are reused (see io/aerial.py::PoiPreviewBuilder)
            preview_builder = None
            if self.aerial_photo_status in ("current", "built"):
                from ..io.aerial import PoiPreviewBuilder

                preview_builder = PoiPreviewBuilder(config.BEAMNG_DIR_TEXTURES, config.BEAMNG_DIR, self.aerial_photos)
            try:
                self.items.save(
                    road_polygons=self.road_polygons,
                    poi_points=self.poi_points,
                    preview_builder=preview_builder,
                    fixed_spawns=self.tunnel_spawns,
                )
            finally:
                if preview_builder is not None:
                    preview_builder.close()
            if preview_builder is not None:
                logger.info(f"[✓] Spawn previews: {preview_builder.built} new, {preview_builder.reused} reused")
            items_path = config.BEAMNG_DIR / config.ITEMS_JSON
            logger.info(f"[✓] Items: {items_path.name}")

            # Write info.json to the level root directory
            self.items.save_info_json()
            info_path = config.BEAMNG_DIR / "info.json"
            logger.debug(f"[✓] Info: {info_path.name}")

        # Forest.json (if forests are enabled)
        if include_forests:
            with optional_subtask(task, "Forest JSON"):
                forest_result = self.forests.finalize_forest_export()

                if forest_result["status"] == "success":
                    # Detailed statistics (total trees, tree species list, scale, height range) are already logged
                    # by forests.finalize_forest_export() itself (forest_workflow.py) - no
                    # second, redundant summary here.
                    pass
                elif forest_result["status"] == "no_forests":
                    logger.info("No forests generated")
                else:
                    logger.error(f"Forest export failed: {forest_result.get('error')}")

        # main.level.json is NOT needed - BeamNG automatically loads main/items.level.json

        # Debug network export (commented out for performance)
        if config.DEBUG_EXPORTS:
            with optional_subtask(task, "Debug export"):
                self.debug_exporter.export(config.CACHE_DIR)
