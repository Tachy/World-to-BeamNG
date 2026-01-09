"""
Beispiele für Builder-Pattern-Verwendung.

Zeigt wie die Builder-Klassen die Mesh-Generierung vereinfachen.
"""

from world_to_beamng.core import Config
from world_to_beamng.builders import TerrainMeshBuilder, RoadMeshBuilder, GridBuilder, BuildingMeshBuilder
from world_to_beamng.mesh.vertex_manager import VertexManager
import numpy as np


def example_grid_builder():
    """Beispiel: Grid-Generierung mit Builder."""

    # Mock-Daten
    height_points = np.random.rand(1000, 2) * 1000  # 1000 Punkte
    height_elevations = np.random.rand(1000) * 100  # Höhen 0-100m

    # Alte Methode (viele Parameter)
    # grid = create_terrain_grid(height_points, height_elevations,
    #                            spacing=2.0, cache_manager=cache)

    # Neue Methode (Builder)
    grid = GridBuilder().with_points(height_points).with_elevations(height_elevations).with_spacing(2.0).build()

    print(f"Grid Shape: {grid.shape}")


def example_terrain_mesh_builder():
    """Beispiel: Terrain-Mesh mit Builder."""

    # Mock-Daten
    grid = np.random.rand(50, 50, 3)
    vertex_states = np.ones((50, 50), dtype=int)
    vertex_manager = VertexManager()

    # Mit Builder (fluent API)
    terrain_mesh = (
        TerrainMeshBuilder()
        .with_grid(grid)
        .with_vertex_states(vertex_states)
        .with_vertex_manager(vertex_manager)
        .build()
    )

    print(f"Terrain Faces: {len(terrain_mesh.faces)}")


def example_road_mesh_builder():
    """Beispiel: Road-Mesh mit Builder."""

    config = Config(beamng_dir="C:/BeamNG/levels/Test")

    # Mock-Daten
    roads = [{"centerline": [[0, 0], [100, 0]], "highway": "primary"}]
    junctions = []
    grid = np.random.rand(50, 50, 3)
    vertex_manager = VertexManager()

    # Mit Builder (sehr übersichtlich)
    road_mesh = (
        RoadMeshBuilder(config)
        .with_roads(roads)
        .with_junctions(junctions)
        .with_grid(grid)
        .with_vertex_manager(vertex_manager)
        .build()
    )

    print(f"Road Mesh: {road_mesh}")


def example_terrain_with_stitching():
    """Beispiel: Terrain mit automatischem Stitching."""

    config = Config(beamng_dir="C:/BeamNG/levels/Test")

    grid = np.random.rand(50, 50, 3)
    vertex_states = np.ones((50, 50), dtype=int)
    vertex_manager = VertexManager()

    # 1. Road Mesh
    road_mesh = (
        RoadMeshBuilder(config)
        .with_roads([])
        .with_junctions([])
        .with_grid(grid)
        .with_vertex_manager(vertex_manager)
        .build()
    )

    # 2. Terrain Mesh mit Stitching (in einem Schritt!)
    terrain_mesh = (
        TerrainMeshBuilder()
        .with_grid(grid)
        .with_vertex_states(vertex_states)
        .with_vertex_manager(vertex_manager)
        .with_stitching(road_mesh)  # Aktiviert automatisches Stitching
        .build()
    )

    print("Terrain mit Stitching generiert!")


def example_building_mesh_builder():
    """Beispiel: Building-Meshes mit Builder."""

    # Mock-Buildings
    buildings = [
        {
            "bounds": [0, 0, 0, 10, 10, 5],
            "walls": [(np.array([[0, 0, 0], [10, 0, 0]]), [[0, 1, 2]])],
            "roofs": [(np.array([[0, 0, 5], [10, 10, 5]]), [[0, 1, 2]])],
        },
        {
            "bounds": [50, 50, 0, 60, 60, 8],
            "walls": [(np.array([[50, 50, 0], [60, 50, 0]]), [[0, 1, 2]])],
            "roofs": [(np.array([[50, 50, 8], [60, 60, 8]]), [[0, 1, 2]])],
        },
    ]

    # Mit Bounds-Filterung
    grid_bounds = (0, 100, 0, 100)  # Nur Gebäude innerhalb dieser Bounds

    meshes = BuildingMeshBuilder().with_buildings(buildings).with_bounds_filter(grid_bounds).build()

    print(f"Generiert: {len(meshes)} Building-Meshes")

    # Ohne Filterung
    all_meshes = BuildingMeshBuilder().with_buildings(buildings).build()

    print(f"Ohne Filter: {len(all_meshes)} Building-Meshes")


def example_complete_workflow():
    """Beispiel: Kompletter Workflow mit allen Buildern."""

    config = Config(beamng_dir="C:/BeamNG/levels/Test")

    # 1. Grid
    print("1. Erstelle Grid...")
    grid = (
        GridBuilder()
        .with_points(np.random.rand(1000, 2) * 1000)
        .with_elevations(np.random.rand(1000) * 100)
        .with_spacing(2.0)
        .build()
    )

    # 2. Road Mesh
    print("2. Erstelle Road Mesh...")
    vertex_manager = VertexManager()
    road_mesh = (
        RoadMeshBuilder(config)
        .with_roads([])
        .with_junctions([])
        .with_grid(grid)
        .with_vertex_manager(vertex_manager)
        .build()
    )

    # 3. Terrain Mesh mit Stitching
    print("3. Erstelle Terrain Mesh...")
    vertex_states = np.ones(grid.shape[:2], dtype=int)
    terrain_mesh = (
        TerrainMeshBuilder()
        .with_grid(grid)
        .with_vertex_states(vertex_states)
        .with_vertex_manager(vertex_manager)
        .with_stitching(road_mesh)
        .build()
    )

    # 4. Buildings
    print("4. Erstelle Buildings...")
    building_meshes = BuildingMeshBuilder().with_buildings([]).build()

    print("✅ Kompletter Workflow abgeschlossen!")
    print(f"   Grid: {grid.shape}")
    print(f"   Vertices: {len(vertex_manager.get_all_vertices())}")
    print(f"   Buildings: {len(building_meshes)}")


if __name__ == "__main__":
    print("=== Builder-Pattern Beispiele ===\n")

    print("1. GridBuilder")
    example_grid_builder()

    print("\n2. TerrainMeshBuilder")
    example_terrain_mesh_builder()

    print("\n3. RoadMeshBuilder")
    example_road_mesh_builder()

    print("\n4. Terrain mit Stitching")
    example_terrain_with_stitching()

    print("\n5. BuildingMeshBuilder")
    example_building_mesh_builder()

    print("\n6. Kompletter Workflow")
    example_complete_workflow()
