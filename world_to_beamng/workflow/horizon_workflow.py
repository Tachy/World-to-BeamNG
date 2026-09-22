"""
Horizon-Layer Workflow.

Orchestriert die Horizon-Layer-Generierung.
"""

from typing import Tuple, Optional
import logging

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from ..terrain.horizon_image import horizon_area, horizon_area_wgs84

logger = logging.getLogger(__name__)


class HorizonWorkflow:
    """
    Orchestriert den Horizon-Layer-Workflow.

    Verantwortlich für:
    - Horizon-Mesh-Generierung
    - Textur-Verwaltung
    - DAE-Export
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
    ) -> Optional[str]:
        """
        Generiere Horizon-Layer (wie in multitile.py phase5_generate_horizon_layer).

        Args:
            global_offset: (origin_x, origin_y, origin_z) - UTM Offset
            tile_hash: Optional - Hash für Cache
            tile_bounds: Optional - Liste von (x_min, y_min, x_max, y_max) Tuples
            terrain_height_at: Optional - Höhenabfrage der Terrain-Heightmap (x, y) -> z. Damit
                bekommt der Horizont ein exakt passendes Loch samt Randring und Höhenübergang
                (terrain/horizon_seam.py) - ganz ohne Terrain-Mesh-Stitching.

        Returns:
            Pfad der Horizont-DAE oder None (Horizont deaktiviert oder DGM30 fehlt)
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

        # Prüfe ob Phase 5 aktiviert ist
        if not config.PHASE5_ENABLED:
            logger.info("  [i] Phase 5 ist deaktiviert")
            return None

        # Berechne Horizont-BBOX (config.HORIZON_HALF_SIZE_M um das Kerngebiet)
        ox, oy, oz = global_offset
        horizon_bbox = horizon_area(global_offset)
        x_min, x_max, y_min, y_max = horizon_bbox

        logger.info(f"  [i] Horizont-BBOX: ±{config.HORIZON_HALF_SIZE_M / 1000:.0f}km um ({ox:.0f}, {oy:.0f})")
        logger.info(f"      UTM (EPSG:25832): X=[{x_min:.0f}..{x_max:.0f}], Y=[{y_min:.0f}..{y_max:.0f}]")
        logger.info(f"      Breite: {x_max - x_min:.0f}m, Höhe: {y_max - y_min:.0f}m")

        # === DGM30 laden ===
        dgm30_dir = config.DGM30_DATA_DIR
        if config.DGM30_AUTO_DOWNLOAD:
            logger.info("  [i] Prüfe DGM30-Abdeckung (lädt fehlende Kacheln bei Bedarf)...")
            ensure_dgm30_coverage(horizon_area_wgs84(global_offset), dgm30_dir)

        logger.info("  [i] Lade DGM30-Daten (30m)...")
        height_points, height_elevations = load_dgm30_tiles(
            dgm30_dir, horizon_bbox, local_offset=global_offset, tile_hash=tile_hash
        )

        if height_points is None:
            logger.warning("  [!] DGM30-Daten nicht gefunden - Phase 5 übersprungen")
            return None

        # === Mesh generieren ===
        logger.info("  [i] Generiere Horizont-Mesh...")

        # === STEP 1: Generiere Horizont-Mesh (separater VM, OHNE UVs noch) ===
        logger.info("  [i] Generiere Horizont-Mesh...")

        # WICHTIG: IMMER separater VM
        horizon_mesh, nx, ny = generate_horizon_mesh(
            height_points,
            height_elevations,
            global_offset,
            tile_bounds=tile_bounds,
            terrain_height_at=terrain_height_at,
        )

        # === Sentinel-2 laden (optional) ===
        sentinel2_file = config.DOP300_DATA_DIR / config.SENTINEL2_FILE
        if config.EOX_AUTO_DOWNLOAD:
            logger.info("  [i] Prüfe Sentinel-2-Textur (lädt bei Bedarf automatisch)...")
            ensure_horizon_texture(horizon_bbox, dest=sentinel2_file)

        logger.info("  [i] Lade Sentinel-2 Satellitenbilder...")
        sentinel2_data = load_sentinel2_geotiff(sentinel2_file, horizon_bbox, tile_hash=tile_hash)

        texture_info = None
        if sentinel2_data is None:
            logger.info("  [i] Sentinel-2 nicht vorhanden - Horizont ohne Textur")
        else:
            horizon_image, bounds_utm, transform = sentinel2_data

            # Zeige Koordinaten-Übereinstimmung mit Mesh
            vertices = horizon_mesh.vertex_manager.vertices
            mesh_x_min, mesh_x_max = vertices[:, 0].min(), vertices[:, 0].max()
            mesh_y_min, mesh_y_max = vertices[:, 1].min(), vertices[:, 1].max()

            logger.info(
                f"      Mesh Bounds (lokal): X=[{mesh_x_min:.0f}..{mesh_x_max:.0f}], Y=[{mesh_y_min:.0f}..{mesh_y_max:.0f}]"
            )
            logger.info(
                f"      Texture Bounds (UTM): X=[{bounds_utm[0]:.0f}..{bounds_utm[2]:.0f}], Y=[{bounds_utm[1]:.0f}..{bounds_utm[3]:.0f}]"
            )

            # === Texturierung ===
            logger.info("  [i] Texturiere Horizont-Mesh...")
            texture_info = texture_horizon_mesh(vertices, horizon_image, nx, ny, bounds_utm, transform, global_offset)

        # === STEP 2: UVs generieren (für alle Vertices) ===
        logger.info("  [i] Generiere UVs für Horizont-Mesh...")
        horizon_vertices = horizon_mesh.vertex_manager.vertices
        mesh_x_min = horizon_vertices[:, 0].min()
        mesh_x_max = horizon_vertices[:, 0].max()
        mesh_y_min = horizon_vertices[:, 1].min()
        mesh_y_max = horizon_vertices[:, 1].max()

        mesh_width = mesh_x_max - mesh_x_min
        mesh_height = mesh_y_max - mesh_y_min

        # Generiere UVs für ALLE Vertices
        horizon_mesh.uvs = []
        for vertex in horizon_vertices:
            u = (vertex[0] - mesh_x_min) / max(mesh_width, 1e-10)
            v = (vertex[1] - mesh_y_min) / max(mesh_height, 1e-10)
            horizon_mesh.uvs.append((u, v))

        logger.info(f"  [✓] {len(horizon_mesh.uvs)} UVs generiert für {len(horizon_vertices)} Vertices")

        # === Export ===
        logger.info("  [i] Exportiere Horizont DAE...")

        dae_filename = export_horizon_dae(
            horizon_mesh,
            texture_info,
            config.BEAMNG_DIR,
            level_name=config.LEVEL_NAME,
            global_offset=global_offset,
        )

        logger.info(f"  [✓] Horizon DAE: {dae_filename}")

        # === Materials & Items ===
        logger.info("  [i] Registriere Materials & Items...")
        texture_path = str(config.RELATIVE_DIR_TEXTURES / "horizon_sentinel2.dds")
        self.materials.add_horizon_material(texture_path)
        self.items.add_horizon(
            dae_filename=dae_filename,
        )

        return str(config.BEAMNG_DIR_SHAPES / dae_filename)
