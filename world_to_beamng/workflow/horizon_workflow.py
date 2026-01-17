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
        terrain_mesh=None,
        terrain_vertex_manager=None,
        terrain_grid_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Tuple[str, list]]:
        """
        Generiere Horizon-Layer (wie in multitile.py phase5_generate_horizon_layer).

        Args:
            global_offset: (origin_x, origin_y, origin_z) - UTM Offset
            tile_hash: Optional - Hash für Cache
            tile_bounds: Optional - Liste von (x_min, y_min, x_max, y_max) Tuples
            terrain_mesh: Optional - Terrain-Mesh für Boundary-Stitching
            terrain_vertex_manager: Optional - Terrain VertexManager (für gemeinsamen VM)
            terrain_grid_bounds: Optional - (x_min, x_max, y_min, y_max) der Terrain-Tiles

        Returns:
            Tuple (dae_path, stitching_faces) oder (None, [])
        """
        from ..terrain.horizon import (
            load_dgm30_tiles,
            load_sentinel2_geotiff,
            generate_horizon_mesh,
            texture_horizon_mesh,
            export_horizon_dae,
        )
        from ..mesh.stitch_boundary import stitch_terrain_horizon_boundary

        # Prüfe ob Phase 5 aktiviert ist
        if not config.PHASE5_ENABLED:
            print("  [i] Phase 5 ist deaktiviert")
            return None, []

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
            return None, []

        # === Mesh generieren (optional mit bestehendem VertexManager) ===
        print("  [i] Generiere Horizont-Mesh...")
        horizon_mesh, nx, ny, horizon_vertex_indices = generate_horizon_mesh(
            height_points,
            height_elevations,
            global_offset,
            tile_bounds=tile_bounds,
            vertex_manager=terrain_vertex_manager,  # Gemeinsamer VM wenn vorhanden!
        )

        # === Boundary-Stitching (falls Terrain-Mesh übergeben) ===
        stitching_faces = []
        if terrain_mesh is not None and terrain_vertex_manager is not None:
            print(f"  [i] Generiere Boundary-Stitching zwischen Terrain und Horizon...")

            stitching_faces = stitch_terrain_horizon_boundary(
                terrain_mesh,
                terrain_vertex_manager,
                horizon_mesh,  # Horizon-Mesh statt Vertex-Indices
                None,  # grid_bounds wird ignoriert - wird aus Boundary-Vertices berechnet!
                grid_spacing=200.0,
            )
            print(f"  [i] {len(stitching_faces)} Stitching-Faces generiert")

            # Noch NICHT ins Mesh integrieren - erst nach Texturierung!

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
            vertices = horizon_mesh.vertex_manager.vertices
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

        # === Integriere Stitching-Faces mit korrekten UVs ===
        if stitching_faces:
            print(f"  [i] Füge {len(stitching_faces)} Stitching-Faces mit UVs hinzu...")

            # Berechne UV-Koordinaten basierend auf Mesh-Bounds (wie Horizon-UVs)
            # Diese UVs werden später in export_horizon_dae() mit uv_offset/uv_scale transformiert
            vertices = horizon_mesh.vertex_manager.vertices

            # Mesh-Bounds (lokal)
            mesh_x_min = vertices[:, 0].min()
            mesh_x_max = vertices[:, 0].max()
            mesh_y_min = vertices[:, 1].min()
            mesh_y_max = vertices[:, 1].max()

            mesh_width = mesh_x_max - mesh_x_min
            mesh_height = mesh_y_max - mesh_y_min

            # === WICHTIG: Initialisiere UV-Liste für ALL Vertices ===
            # Nicht nur die Stitching-Faces haben Vertices - auch Terrain-Vertices
            # können im gemeinsamen VertexManager sein!
            print(f"  [i] Initialisiere UV-Liste für alle {len(vertices)} Vertices...")
            while len(horizon_mesh.uvs) < len(vertices):
                v_idx = len(horizon_mesh.uvs)
                vx_local = vertices[v_idx, 0]
                vy_local = vertices[v_idx, 1]

                u = (vx_local - mesh_x_min) / mesh_width if mesh_width > 0 else 0.5
                v = (vy_local - mesh_y_min) / mesh_height if mesh_height > 0 else 0.5
                horizon_mesh.uvs.append((u, v))

            # Jetzt füge die Stitching-Faces hinzu (UVs existieren bereits)
            for face_tuple in stitching_faces:
                horizon_mesh.faces.append(face_tuple)

            print(f"  [✓] {len(stitching_faces)} Stitching-Faces integriert")
            print(f"  [i] UV-Liste: {len(horizon_mesh.uvs)} UVs für {len(vertices)} Vertices")

        # === Export ===
        print("  [i] Exportiere Horizont DAE...")
        import os

        dae_filename = export_horizon_dae(
            horizon_mesh,
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

        return os.path.join(config.BEAMNG_DIR_SHAPES, dae_filename), stitching_faces
