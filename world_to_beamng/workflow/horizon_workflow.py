"""
Horizon layer workflow.

Orchestrates the horizon layer generation.
"""

from typing import Tuple, Optional
import logging

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from ..terrain.horizon_image import horizon_area, horizon_area_wgs84
from ..progress import optional_subtask

logger = logging.getLogger(__name__)


class HorizonWorkflow:
    """
    Orchestrates the horizon layer workflow.

    Responsible for:
    - Horizon mesh generation
    - Texture management
    - DAE export
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

    def generate_horizon(
        self,
        global_offset: Tuple[float, float, float],
        tile_hash: Optional[str] = None,
        tile_bounds: Optional[list] = None,
        terrain_height_at=None,
        task=None,
    ) -> Optional[str]:
        """
        Generate the horizon layer (as in multitile.py phase5_generate_horizon_layer).

        Args:
            global_offset: (origin_x, origin_y, origin_z) - UTM offset
            tile_hash: Optional - hash for the cache
            tile_bounds: Optional - list of (x_min, y_min, x_max, y_max) tuples
            terrain_height_at: Optional - elevation lookup of the terrain heightmap (x, y) -> z. This gives
                the horizon an exactly fitting hole including a border ring and elevation transition
                (terrain/horizon_seam.py) - without any terrain mesh stitching.
            task: Optional - PipelineTask; each step is then shown as a subtask

        Returns:
            Path of the horizon DAE or None (horizon disabled or DGM30 missing)
        """
        from ..terrain.horizon import (
            load_dgm30_tiles,
            load_sentinel2_geotiff,
            generate_horizon_mesh,
            texture_horizon_mesh,
            export_horizon_dae,
        )
        from ..terrain.dgm30_fetch import ensure_dgm30_coverage
        from ..terrain.sentinel2_fetch import ensure_horizon_texture

        # Check whether phase 5 is enabled
        if not config.PHASE5_ENABLED:
            logger.info("  [i] Phase 5 is disabled")
            return None

        # Compute the horizon BBOX (config.HORIZON_HALF_SIZE_M around the core area)
        ox, oy, oz = global_offset
        horizon_bbox = horizon_area(global_offset)
        x_min, x_max, y_min, y_max = horizon_bbox

        logger.debug(f"  [i] Horizon bbox: ±{config.HORIZON_HALF_SIZE_M / 1000:.0f}km around ({ox:.0f}, {oy:.0f})")
        logger.debug(f"      UTM (EPSG:25832): X=[{x_min:.0f}..{x_max:.0f}], Y=[{y_min:.0f}..{y_max:.0f}]")
        logger.debug(f"      Width: {x_max - x_min:.0f}m, height: {y_max - y_min:.0f}m")

        # === Load DGM30 ===
        with optional_subtask(task, "Load DGM30") as sub:
            dgm30_dir = config.DGM30_CACHE_DIR
            if config.DGM30_AUTO_DOWNLOAD:
                logger.debug("  [i] Checking DGM30 coverage (loads missing tiles if needed)...")
                ensure_dgm30_coverage(horizon_area_wgs84(global_offset), dgm30_dir)

            logger.debug("  [i] Loading DGM30 data (30m)...")
            height_points, height_elevations = load_dgm30_tiles(
                dgm30_dir, horizon_bbox, local_offset=global_offset, tile_hash=tile_hash
            )

            if height_points is None:
                logger.warning("  [!] DGM30 data not found - phase 5 skipped")
                sub.warn("not found")
                return None
            sub.finish(f"{len(height_points)} points")

        # === STEP 1: Generate the horizon mesh (separate VM, WITHOUT UVs yet) ===
        logger.debug("  [i] Generating horizon mesh...")

        # IMPORTANT: ALWAYS a separate VM
        with optional_subtask(task, "Horizon mesh"):
            horizon_mesh, nx, ny = generate_horizon_mesh(
                height_points,
                height_elevations,
                global_offset,
                tile_bounds=tile_bounds,
                terrain_height_at=terrain_height_at,
            )

        with optional_subtask(task, "Sentinel-2 texture"):
            # === Load Sentinel-2 (optional) ===
            # Fully automatic - ensure_horizon_texture() returns an area-dependent cache path
            # under config.EOX_TEXTURE_CACHE_DIR or None (auto-download disabled/failed).
            sentinel2_file = None
            if config.EOX_AUTO_DOWNLOAD:
                logger.debug("  [i] Checking Sentinel-2 texture (loads automatically if needed)...")
                sentinel2_file = ensure_horizon_texture(horizon_bbox)

            if sentinel2_file is not None:
                logger.debug("  [i] Loading Sentinel-2 satellite images...")
                sentinel2_data = load_sentinel2_geotiff(sentinel2_file, horizon_bbox, tile_hash=tile_hash)
            else:
                sentinel2_data = None

            texture_info = None
            if sentinel2_data is None:
                logger.info("  [i] Sentinel-2 not available - horizon without texture")
            else:
                horizon_image, bounds_utm, transform = sentinel2_data

                # Show the coordinate match with the mesh
                vertices = horizon_mesh.vertex_manager.vertices
                mesh_x_min, mesh_x_max = vertices[:, 0].min(), vertices[:, 0].max()
                mesh_y_min, mesh_y_max = vertices[:, 1].min(), vertices[:, 1].max()

                logger.debug(
                    f"      Mesh bounds (local): X=[{mesh_x_min:.0f}..{mesh_x_max:.0f}], Y=[{mesh_y_min:.0f}..{mesh_y_max:.0f}]"
                )
                logger.debug(
                    f"      Texture bounds (UTM): X=[{bounds_utm[0]:.0f}..{bounds_utm[2]:.0f}], Y=[{bounds_utm[1]:.0f}..{bounds_utm[3]:.0f}]"
                )

                # === Texturing ===
                logger.debug("  [i] Texturing horizon mesh...")
                texture_info = texture_horizon_mesh(vertices, horizon_image, nx, ny, bounds_utm, transform, global_offset)

        with optional_subtask(task, "UVs + DAE export"):
            # === STEP 2: Generate UVs (for all vertices) ===
            logger.debug("  [i] Generating UVs for the horizon mesh...")
            horizon_vertices = horizon_mesh.vertex_manager.vertices
            mesh_x_min = horizon_vertices[:, 0].min()
            mesh_x_max = horizon_vertices[:, 0].max()
            mesh_y_min = horizon_vertices[:, 1].min()
            mesh_y_max = horizon_vertices[:, 1].max()

            mesh_width = mesh_x_max - mesh_x_min
            mesh_height = mesh_y_max - mesh_y_min

            # Generate UVs for ALL vertices
            horizon_mesh.uvs = []
            for vertex in horizon_vertices:
                u = (vertex[0] - mesh_x_min) / max(mesh_width, 1e-10)
                v = (vertex[1] - mesh_y_min) / max(mesh_height, 1e-10)
                horizon_mesh.uvs.append((u, v))

            logger.debug(f"  [✓] {len(horizon_mesh.uvs)} UVs generated for {len(horizon_vertices)} vertices")

            # === Export ===
            logger.debug("  [i] Exporting horizon DAE...")

            dae_filename = export_horizon_dae(
                horizon_mesh,
                texture_info,
                config.BEAMNG_DIR,
                level_name=config.LEVEL_NAME,
                global_offset=global_offset,
            )

            logger.info(f"  [✓] Horizon DAE: {dae_filename}")

            # === Materials & Items ===
            logger.debug("  [i] Registering materials & items...")
            texture_path = str(config.RELATIVE_DIR_TEXTURES / "horizon_sentinel2.dds")
            self.materials.add_horizon_material(texture_path)
            self.items.add_horizon(
                dae_filename=dae_filename,
            )

        return str(config.BEAMNG_DIR_SHAPES / dae_filename)
