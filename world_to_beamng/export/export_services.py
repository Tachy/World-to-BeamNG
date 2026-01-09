"""
DAE-Export-Logik.

Verschoben von io/dae.py - trennt Export-Logik von I/O.
"""

from typing import Dict, List, Optional
from pathlib import Path

from ..managers import DAEExporter, MaterialManager, ItemManager


class DAEExportService:
    """
    Service für DAE-Export-Operationen.

    Trennt Export-Logik von reinen I/O-Operationen.
    """

    def __init__(self, dae_exporter: DAEExporter, material_manager: MaterialManager, item_manager: ItemManager):
        self.dae = dae_exporter
        self.materials = material_manager
        self.items = item_manager

    def export_merged_terrain(
        self, road_mesh, terrain_mesh, vertex_manager, output_dir: str, tile_x: int, tile_y: int
    ) -> str:
        """
        Exportiere kombiniertes Terrain-/Road-Mesh.

        Args:
            road_mesh: Road-Mesh
            terrain_mesh: Terrain-Mesh
            vertex_manager: VertexManager
            output_dir: Ausgabe-Verzeichnis
            tile_x, tile_y: Tile-Koordinaten

        Returns:
            Pfad zur DAE-Datei
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dae_file = output_path / f"terrain_{tile_x}_{tile_y}.dae"

        # Sammle alle Vertices
        all_vertices = vertex_manager.get_all_vertices()

        # Sammle Faces
        combined_faces = {}

        # Road Faces
        for mat_name, faces in road_mesh.faces.items():
            combined_faces[mat_name] = faces

        # Terrain Faces
        if terrain_mesh and hasattr(terrain_mesh, "faces"):
            for mat_name, faces in terrain_mesh.faces.items():
                if mat_name in combined_faces:
                    combined_faces[mat_name].extend(faces)
                else:
                    combined_faces[mat_name] = faces

        # Export mit DAEExporter
        self.dae.export_single_mesh(
            output_path=str(dae_file),
            mesh_id=f"terrain_{tile_x}_{tile_y}",
            vertices=all_vertices,
            faces=combined_faces,
            material_name=None,  # Multi-Material
            with_uv=True,
        )

        return str(dae_file)

    def create_terrain_materials(self, roads: List[Dict], texture_path: str) -> Dict:
        """
        Erstelle Terrain-Materialien.

        Args:
            roads: Liste von Roads
            texture_path: Pfad zur Terrain-Textur

        Returns:
            Material-Dict
        """
        # Terrain-Material
        self.materials.add_terrain_material(tile_x=0, tile_y=0, texture_path=texture_path)

        # Road-Materialien
        for road in roads:
            properties = road.get("properties", {})
            road_type = road.get("highway", "road")

            self.materials.add_road_material(road_type=road_type, properties=properties)

        return self.materials.materials

    def create_terrain_items(self, dae_filename: str, tile_x: int, tile_y: int) -> Dict:
        """
        Erstelle Terrain-Item-Eintrag.

        Args:
            dae_filename: DAE-Dateiname
            tile_x, tile_y: Tile-Koordinaten

        Returns:
            Item-Dict
        """
        item_name = f"terrain_{tile_x}_{tile_y}"

        self.items.add_terrain(name=item_name, dae_filename=dae_filename, position=(0, 0, 0))

        return self.items.get_item(item_name)


class BuildingExportService:
    """
    Service für LoD2-Building-Export.

    Trennt Export-Logik von I/O.
    """

    def __init__(self, dae_exporter: DAEExporter, material_manager: MaterialManager, item_manager: ItemManager):
        self.dae = dae_exporter
        self.materials = material_manager
        self.items = item_manager

    def export_buildings(self, meshes: List[Dict], output_dir: str, tile_x: int, tile_y: int) -> Optional[str]:
        """
        Exportiere Gebäude-Meshes.

        Args:
            meshes: Liste von Mesh-Dicts
            output_dir: Ausgabe-Verzeichnis
            tile_x, tile_y: Tile-Koordinaten

        Returns:
            Pfad zur DAE-Datei oder None
        """
        if not meshes:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dae_file = output_path / f"buildings_tile_{tile_x}_{tile_y}.dae"

        self.dae.export_multi_mesh(output_path=str(dae_file), meshes=meshes, with_uv=False)

        print(f"  [✓] Buildings DAE: {dae_file.name} ({len(meshes)} Gebäude)")

        return str(dae_file)

    def create_building_materials(self) -> Dict:
        """
        Erstelle LoD2-Materialien.

        Returns:
            Material-Dict
        """
        # Wall-Material
        self.materials.add_building_material(
            "lod2_wall_white",
            color=[0.95, 0.95, 0.95, 1.0],
            groundType="STONE",
            materialTag0="beamng",
            materialTag1="Building",
        )

        # Roof-Material
        self.materials.add_building_material(
            "lod2_roof_red",
            color=[0.6, 0.2, 0.1, 1.0],
            groundType="ROOF_TILES",
            materialTag0="beamng",
            materialTag1="Building",
        )

        return self.materials.materials

    def create_building_item(self, dae_filename: str, tile_x: int, tile_y: int):
        """
        Erstelle Building-Item.

        Args:
            dae_filename: DAE-Dateiname
            tile_x, tile_y: Tile-Koordinaten
        """
        from .. import config

        item_name = f"buildings_{tile_x}_{tile_y}"
        shape_name = config.RELATIVE_DIR_BUILDINGS + dae_filename

        self.items.add_item(
            item_name, item_class="TSStatic", shape_name=shape_name, position=(0, 0, 0), collisionType="Visible Mesh"
        )


class HorizonExportService:
    """
    Service für Horizon-Layer-Export.
    """

    def __init__(self, dae_exporter: DAEExporter, material_manager: MaterialManager, item_manager: ItemManager):
        self.dae = dae_exporter
        self.materials = material_manager
        self.items = item_manager

    def export_horizon(self, mesh_data: Dict, output_path: str) -> str:
        """
        Exportiere Horizon-Mesh.

        Args:
            mesh_data: Dict mit vertices, faces, uvs
            output_path: Ausgabe-Pfad

        Returns:
            Pfad zur DAE-Datei
        """
        self.dae.export_single_mesh(
            output_path=output_path,
            mesh_id="horizon",
            vertices=mesh_data["vertices"],
            faces=mesh_data["faces"],
            material_name="horizon_terrain",
            with_uv=True,
        )

        return output_path

    def create_horizon_material(self, texture_path: str):
        """
        Erstelle Horizon-Material.

        Args:
            texture_path: Relativer Pfad zur Textur
        """
        self.materials.add_horizon_material(texture_path)

    def create_horizon_item(self, dae_filename: str):
        """
        Erstelle Horizon-Item.

        Args:
            dae_filename: DAE-Dateiname
        """
        self.items.add_horizon(name="Horizon", dae_filename=dae_filename)
