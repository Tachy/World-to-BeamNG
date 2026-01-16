"""
Horizon-Layer Workflow.

Orchestriert die Horizon-Layer-Generierung.
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter


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
        material_manager: MaterialManager,
        item_manager: ItemManager,
        dae_exporter: DAEExporter,
    ):
        self.cache = cache_manager
        self.materials = material_manager
        self.items = item_manager
        self.dae = dae_exporter

    def generate_horizon(
        self,
        global_offset: Tuple[float, float, float],
        tile_hash: Optional[str] = None,
        tile_bounds: Optional[list] = None,
    ) -> Optional[str]:
        """
        Generiere Horizon-Layer (wie in multitile.py phase5_generate_horizon_layer).

        Args:
            global_offset: (origin_x, origin_y, origin_z) - UTM Offset
            tile_hash: Optional - Hash für Cache
            tile_bounds: Optional - Liste von (x_min, y_min, x_max, y_max) Tuples in lokalen Koordinaten
                         zum Filtern von Quads die über Terrain liegen

        Returns:
            Pfad zur DAE-Datei oder None
        """
        from ..terrain.horizon import (
            load_dgm30_tiles,
            load_sentinel2_geotiff,
            generate_horizon_mesh,
            texture_horizon_mesh,
            export_horizon_dae,
        )

        # Prüfe ob Phase 5 aktiviert ist
        if not config.PHASE5_ENABLED:
            print("  [i] Phase 5 ist deaktiviert")
            return None

        # Berechne Horizont-BBOX (±50km um Kerngebiet)
        ox, oy, oz = global_offset
        horizon_bbox = (ox - 50000, ox + 50000, oy - 50000, oy + 50000)
        x_min, x_max, y_min, y_max = horizon_bbox

        print(f"  [i] Horizont-BBOX: ±50km um ({ox:.0f}, {oy:.0f})")
        print(f"      UTM (EPSG:25832): X=[{x_min:.0f}..{x_max:.0f}], Y=[{y_min:.0f}..{y_max:.0f}]")
        print(f"      Breite: {x_max - x_min:.0f}m, Höhe: {y_max - y_min:.0f}m")

        # === DGM30 laden ===
        print("  [i] Lade DGM30-Daten (30m)...")
        dgm30_dir = config.DGM30_DATA_DIR
        height_points, height_elevations = load_dgm30_tiles(
            dgm30_dir, horizon_bbox, local_offset=global_offset, tile_hash=tile_hash
        )

        if height_points is None:
            print("  [!] DGM30-Daten nicht gefunden - Phase 5 übersprungen")
            return None

        # === Mesh generieren ===
        print("  [i] Generiere Horizont-Mesh...")
        mesh, nx, ny = generate_horizon_mesh(height_points, height_elevations, global_offset, tile_bounds=tile_bounds)

        # === Sentinel-2 laden (optional) ===
        print("  [i] Lade Sentinel-2 Satellitenbilder...")
        sentinel2_dir = config.DOP300_DATA_DIR
        sentinel2_data = load_sentinel2_geotiff(sentinel2_dir, horizon_bbox, tile_hash=tile_hash)

        texture_info = None
        if sentinel2_data is None:
            print("  [i] Sentinel-2 nicht vorhanden - Horizont ohne Textur")
        else:
            horizon_image, bounds_utm, transform = sentinel2_data

            # Zeige Koordinaten-Übereinstimmung mit Mesh
            vertices = mesh.vertex_manager.vertices
            mesh_x_min, mesh_x_max = vertices[:, 0].min(), vertices[:, 0].max()
            mesh_y_min, mesh_y_max = vertices[:, 1].min(), vertices[:, 1].max()

            print(
                f"      Mesh Bounds (lokal): X=[{mesh_x_min:.0f}..{mesh_x_max:.0f}], Y=[{mesh_y_min:.0f}..{mesh_y_max:.0f}]"
            )
            print(
                f"      Texture Bounds (UTM): X=[{bounds_utm[0]:.0f}..{bounds_utm[2]:.0f}], Y=[{bounds_utm[1]:.0f}..{bounds_utm[3]:.0f}]"
            )

            # === Texturierung ===
            print("  [i] Texturiere Horizont-Mesh...")
            texture_info = texture_horizon_mesh(vertices, horizon_image, nx, ny, bounds_utm, transform, global_offset)

        # === Export ===
        print("  [i] Exportiere Horizont DAE...")
        import os

        dae_filename = export_horizon_dae(
            mesh,
            texture_info,
            config.BEAMNG_DIR,
            level_name=config.LEVEL_NAME,
            global_offset=global_offset,
        )

        print(f"  [✓] Horizon DAE: {dae_filename}")

        # === Materials & Items ===
        print("  [i] Registriere Materials & Items...")

        # Material hinzufügen
        if texture_info:
            texture_path = texture_info.get("texture_path", "shapes/textures/white.png")
        else:
            texture_path = "shapes/textures/white.png"

        self.materials.add_horizon_material(texture_path)

        # Item hinzufügen
        self.items.add_horizon(name="Horizon", dae_filename=dae_filename)

        print(f"  [OK] Phase 5 abgeschlossen")

        import os

        return os.path.join(config.BEAMNG_DIR_SHAPES, dae_filename)
