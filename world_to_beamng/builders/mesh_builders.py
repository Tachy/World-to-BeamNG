"""
Builder für Terrain-Meshes.

Vereinfacht die komplexe Terrain-Mesh-Generierung mit einem klaren Builder-Pattern.
"""

from typing import Optional, List, Dict, Tuple
import numpy as np

from .. import config
from ..mesh.mesh import Mesh
from ..mesh.vertex_manager import VertexManager


class TerrainMeshBuilder:
    """
    Builder für Terrain-Meshes.

    Beispiel:
        >>> builder = TerrainMeshBuilder()
        >>> mesh = (builder
        ...     .with_grid(grid)
        ...     .with_vertex_states(vertex_states)
        ...     .with_vertex_manager(vertex_manager)
        ...     .with_stitching(road_mesh)
        ...     .build())
    """

    def __init__(self):
        self._grid = None
        self._vertex_states = None
        self._vertex_manager = None
        self._road_data = None
        self._junctions = None
        self._enable_stitching = False
        self._road_mesh_data = (
            None  # Strukturierte Road-Daten: [{'vertices': [...], 'road_id': ..., 'uvs': {...}}, ...]
        )

    def with_grid(self, grid: np.ndarray) -> "TerrainMeshBuilder":
        """
        Setze Grid.

        Args:
            grid: N×M×3 Grid mit Koordinaten

        Returns:
            Self für Method-Chaining
        """
        self._grid = grid
        return self

    def with_vertex_states(self, vertex_states: np.ndarray) -> "TerrainMeshBuilder":
        """
        Setze Vertex-States.

        Args:
            vertex_states: N×M Array mit States (0=outside, 1=terrain, 2=road)

        Returns:
            Self für Method-Chaining
        """
        self._vertex_states = vertex_states
        return self

    def with_vertex_manager(self, vertex_manager: VertexManager) -> "TerrainMeshBuilder":
        """
        Setze VertexManager.

        Args:
            vertex_manager: VertexManager-Instanz

        Returns:
            Self für Method-Chaining
        """
        self._vertex_manager = vertex_manager
        return self

    def with_stitching(self, road_data, junctions=None) -> "TerrainMeshBuilder":
        """
        Aktiviere Stitching mit Road-Daten.

        Args:
            road_data: road_slope_polygons_2d für Stitching
            junctions: Optional junction_points

        Returns:
            Self für Method-Chaining
        """
        self._road_data = road_data
        self._junctions = junctions
        self._enable_stitching = True
        return self

    def with_road_mesh(self, road_mesh_tuple, road_polygons_2d) -> "TerrainMeshBuilder":
        """
        Füge Road-Mesh hinzu (für Material-Mapping).

        Args:
            road_mesh_tuple: (road_mesh_data, road_slope_polygons_2d, ...) von RoadMeshBuilder
                            road_mesh_data = [{'vertices': [v0,v1,v2], 'road_id': id, 'uvs': {...}}, ...]
            road_polygons_2d: road_slope_polygons_2d für OSM-Mapping

        Returns:
            Self für Method-Chaining
        """
        if road_mesh_tuple:
            self._road_mesh_data = road_mesh_tuple[0]  # Strukturierte Road-Daten
        self._road_data = road_polygons_2d  # Für OSM-Tag-Lookup
        return self

    def build(self) -> Mesh:
        """
        Baue Terrain-Mesh.

        Returns:
            Generiertes Mesh

        Raises:
            ValueError: Wenn erforderliche Parameter fehlen
        """
        if self._grid is None:
            raise ValueError("Grid required")
        if self._vertex_states is None:
            raise ValueError("Vertex states required")
        if self._vertex_manager is None:
            raise ValueError("VertexManager required")

        from ..mesh.terrain_mesh import generate_full_grid_mesh
        from ..mesh.stitch_gaps import stitch_all_gaps

        # Entpacke Grid
        grid_points, grid_elevations, nx, ny = self._grid
        vertex_types, modified_heights = self._vertex_states

        # Generiere Basis-Mesh (OHNE Dedup - Grid-Vertices sind einzigartig)
        terrain_faces, terrain_vertex_indices = generate_full_grid_mesh(
            grid_points, modified_heights, vertex_types, nx, ny, self._vertex_manager, dedup=False
        )

        # Optional: Stitching
        if self._enable_stitching and self._road_data is not None:
            # Konvertiere junctions zu junction_points Format
            junction_points = []
            if self._junctions:
                for j_idx, j in enumerate(self._junctions):
                    pos = j.get("position")
                    if pos is not None and len(pos) >= 2:
                        junction_points.append({"id": j.get("id", j_idx), "position": np.asarray(pos, dtype=float)})

            from ..mesh.mesh import Mesh

            mesh_obj = Mesh(self._vertex_manager)

            print(f"  [DEBUG] Füge {len(terrain_faces)} Terrain-Faces zu mesh_obj hinzu...")
            # Füge Terrain-Faces MIT Material hinzu (UVs werden später via compute_terrain_uvs_batch() berechnet)
            for face_idx, face in enumerate(terrain_faces):
                mesh_obj.add_face(face[0], face[1], face[2], material="terrain")

            print(f"  [DEBUG] mesh_obj hat jetzt {len(mesh_obj.faces)} Faces")

            # NEU: Füge Road-Faces MIT OSM-mapped Material hinzu
            if self._road_mesh_data is not None:
                from ..config import OSM_MAPPER

                print(f"  [DEBUG] Füge {len(self._road_mesh_data)} Road-Faces zu mesh_obj hinzu...")

                # Baue Material-Map: road_id → material_name
                road_material_map = {}
                for poly in self._road_data:
                    r_id = poly.get("road_id")
                    if r_id is not None:
                        osm_tags = poly.get("osm_tags", {})
                        props = OSM_MAPPER.get_road_properties(osm_tags)
                        mat_name = props.get("internal_name", "road_default")
                        road_material_map[r_id] = mat_name

                # Füge Road-Faces mit Material UND UV-Koordinaten hinzu
                for road_face_data in self._road_mesh_data:
                    v0, v1, v2 = road_face_data["vertices"]
                    road_id = road_face_data["road_id"]
                    uv_coords = road_face_data["uvs"]  # {v0: (u,v), v1: (u,v), v2: (u,v)}

                    mat_name = road_material_map.get(road_id, "road_default")
                    mesh_obj.add_face(v0, v1, v2, material=mat_name, uv_coords=uv_coords)

                print(f"  [DEBUG] mesh_obj hat jetzt {len(mesh_obj.faces)} Faces (nach Road-Faces)")

            stitch_all_gaps(
                road_data_for_classification=self._road_data,
                vertex_manager=self._vertex_manager,
                mesh=mesh_obj,
                terrain_vertex_indices=terrain_vertex_indices,
                junction_points=junction_points if junction_points else None,
            )

            # Update terrain_faces mit neuen Faces
            print(f"  [DEBUG] mesh_obj hat nach Stitching {len(mesh_obj.faces)} Faces")

            # KRITISCH: Entferne doppelte Faces (gleiches Set von Vertices)!
            # Stitching kann Faces erstellen die bereits als Road-Faces existieren!
            print(f"  [DEBUG] Entferne doppelte Faces...")
            removed = mesh_obj.cleanup_duplicate_faces()
            print(f"  [DEBUG] {removed} doppelte Faces entfernt, {len(mesh_obj.faces)} Faces verbleiben")

            terrain_faces = mesh_obj.faces

            # ZENTRALE UV-VERWALTUNG:
            # 1. Speichere Road-UV-Daten BEVOR Terrain-UVs berechnet werden
            print("  Speichere Road-UV-Daten...")
            road_uv_data = mesh_obj.preserve_road_uvs()

            # 2. Berechne Terrain-UVs (respektiert Road-UVs)
            # UVs werden NACH dem Slicing PRO TILE berechnet (siehe tile_slicer.py)
            # Hier speichern wir nur die Road-Daten für später.

            # Berechne geglättete Normalen (gesamtes Mesh, inkl. Roads)
            mesh_obj.compute_smooth_normals()
            vertex_normals = mesh_obj.vertex_normals

            # Speichere road_uv_data im Mesh-Objekt für tile_slicer
            mesh_obj.road_uv_data = road_uv_data
        else:
            # Kein Stitching → terrain_faces liegt als Liste vor, berechne Normals direkt
            vertices_array = np.asarray(self._vertex_manager.get_array(), dtype=np.float32)
            normals = np.zeros_like(vertices_array, dtype=np.float32)
            for v0, v1, v2 in terrain_faces:
                p0 = vertices_array[v0]
                p1 = vertices_array[v1]
                p2 = vertices_array[v2]
                fn = np.cross(p1 - p0, p2 - p0)
                nlen = np.linalg.norm(fn)
                if nlen <= 1e-12:
                    continue
                fn /= nlen
                normals[v0] += fn
                normals[v1] += fn
                normals[v2] += fn
            lengths = np.linalg.norm(normals, axis=1)
            mask = lengths > 1e-12
            normals[mask] /= lengths[mask][:, np.newaxis]
            vertex_normals = normals

        return {
            "faces": terrain_faces,
            "vertex_indices": terrain_vertex_indices,
            "mesh_obj": mesh_obj if self._enable_stitching and self._road_data is not None else None,
            "vertex_normals": vertex_normals,
        }


class RoadMeshBuilder:
    """
    Builder für Road-Meshes.

    Beispiel:
        >>> builder = RoadMeshBuilder()
        >>> mesh = (builder
        ...     .with_roads(roads)
        ...     .with_junctions(junctions)
        ...     .with_grid(grid)
        ...     .with_vertex_manager(vertex_manager)
        ...     .build())
    """

    def __init__(self):
        self._roads = None
        self._junctions = None
        self._grid = None
        self._vertex_manager = None

    def with_roads(self, roads: List[Dict]) -> "RoadMeshBuilder":
        """
        Setze Straßen.

        Args:
            roads: Liste von Road-Dicts

        Returns:
            Self für Method-Chaining
        """
        self._roads = roads
        return self

    def with_junctions(self, junctions: List[Dict]) -> "RoadMeshBuilder":
        """
        Setze Junctions.

        Args:
            junctions: Liste von Junction-Dicts

        Returns:
            Self für Method-Chaining
        """
        self._junctions = junctions
        return self

    def with_grid(self, grid: np.ndarray) -> "RoadMeshBuilder":
        """
        Setze Grid.

        Args:
            grid: N×M×3 Grid

        Returns:
            Self für Method-Chaining
        """
        self._grid = grid
        return self

    def with_vertex_manager(self, vertex_manager: VertexManager) -> "RoadMeshBuilder":
        """
        Setze VertexManager.

        Args:
            vertex_manager: VertexManager-Instanz

        Returns:
            Self für Method-Chaining
        """
        self._vertex_manager = vertex_manager
        return self

    def build(self) -> Mesh:
        """
        Baue Road-Mesh.

        Returns:
            Generiertes Road-Mesh

        Raises:
            ValueError: Wenn erforderliche Parameter fehlen
        """
        if self._roads is None:
            raise ValueError("Roads required")
        if self._junctions is None:
            raise ValueError("Junctions required")
        if self._grid is None:
            raise ValueError("Grid required")
        if self._vertex_manager is None:
            raise ValueError("VertexManager required")

        from ..mesh.road_mesh import generate_road_mesh_strips

        grid_points, grid_elevations, _, _ = self._grid

        return generate_road_mesh_strips(
            self._roads,
            grid_points,
            grid_elevations,
            self._vertex_manager,
            self._junctions,
        )


class GridBuilder:
    """
    Builder für Terrain-Grids.

    Beispiel:
        >>> builder = GridBuilder()
        >>> grid = (builder
        ...     .with_points(height_points)
        ...     .with_elevations(height_elevations)
        ...     .with_spacing(2.0)
        ...     .with_cache(cache_manager, cache_key)
        ...     .build())
    """

    def __init__(self):
        self._points = None
        self._elevations = None
        self._spacing = 2.0
        self._cache_manager = None
        self._cache_key = None
        self._was_cached = False

    def with_points(self, points: np.ndarray) -> "GridBuilder":
        """
        Setze Höhen-Punkte.

        Args:
            points: N×2 Array mit (x, y) Koordinaten

        Returns:
            Self für Method-Chaining
        """
        self._points = points
        return self

    def with_elevations(self, elevations: np.ndarray) -> "GridBuilder":
        """
        Setze Höhen-Werte.

        Args:
            elevations: N Array mit Z-Werten

        Returns:
            Self für Method-Chaining
        """
        self._elevations = elevations
        return self

    def with_spacing(self, spacing: float) -> "GridBuilder":
        """
        Setze Grid-Spacing.

        Args:
            spacing: Abstand zwischen Grid-Punkten in Metern

        Returns:
            Self für Method-Chaining
        """
        self._spacing = spacing
        return self

    def with_cache(self, cache_manager, cache_key: str) -> "GridBuilder":
        """
        Aktiviere Caching.

        Args:
            cache_manager: CacheManager-Instanz
            cache_key: Cache-Key

        Returns:
            Self für Method-Chaining
        """
        self._cache_manager = cache_manager
        self._cache_key = cache_key
        return self

    def build(self) -> np.ndarray:
        """
        Baue Grid.

        Returns:
            N×M×3 Grid-Array

        Raises:
            ValueError: Wenn erforderliche Parameter fehlen
        """
        if self._points is None:
            raise ValueError("Points required")
        if self._elevations is None:
            raise ValueError("Elevations required")

        from ..terrain.grid import create_terrain_grid

        spacing = self._spacing if self._spacing is not None else 10.0
        tile_hash = self._cache_key

        # Prüfe Cache BEVOR create_terrain_grid aufgerufen wird
        if tile_hash and self._cache_manager:
            import os

            cache_file = os.path.join(self._cache_manager.cache_dir, f"grid_v3_{tile_hash}_spacing{spacing:.1f}m.npz")
            self._was_cached = os.path.exists(cache_file)

        return create_terrain_grid(
            self._points,
            self._elevations,
            grid_spacing=spacing,
            tile_hash=tile_hash,
        )

    def was_cached(self) -> bool:
        """
        Prüfe ob Grid aus Cache geladen wurde.

        Returns:
            True wenn Grid aus Cache kam, False wenn neu generiert
        """
        return self._was_cached


class BuildingMeshBuilder:
    """
    Builder für LoD2-Gebäude-Meshes.

    Beispiel:
        >>> builder = BuildingMeshBuilder()
        >>> meshes = (builder
        ...     .with_buildings(buildings)
        ...     .with_bounds_filter(grid_bounds)
        ...     .build())
    """

    def __init__(self):
        self._buildings = None
        self._grid_bounds = None

    def with_buildings(self, buildings: List[Dict]) -> "BuildingMeshBuilder":
        """
        Setze Gebäude-Liste.

        Args:
            buildings: Liste von Gebäude-Dicts

        Returns:
            Self für Method-Chaining
        """
        self._buildings = buildings
        return self

    def with_bounds_filter(self, bounds: Optional[Tuple[float, float, float, float]]) -> "BuildingMeshBuilder":
        """
        Aktiviere Bounds-Filterung.

        Args:
            bounds: (min_x, max_x, min_y, max_y)

        Returns:
            Self für Method-Chaining
        """
        self._grid_bounds = bounds
        return self

    def build(self) -> List[Dict]:
        """
        Baue Gebäude-Meshes.

        Returns:
            Liste von Mesh-Dicts für DAEExporter

        Raises:
            ValueError: Wenn Buildings fehlen
        """
        if self._buildings is None:
            raise ValueError("Buildings required")

        # Filtere nach Bounds
        buildings = self._buildings
        if self._grid_bounds is not None:
            min_x, max_x, min_y, max_y = self._grid_bounds
            buildings = [b for b in buildings if self._is_in_bounds(b, min_x, max_x, min_y, max_y)]

        # Konvertiere zu Mesh-Format
        meshes = []
        for bldg_idx, building in enumerate(buildings):
            mesh = self._building_to_mesh(building, bldg_idx)
            if mesh:
                meshes.append(mesh)

        return meshes

    def _is_in_bounds(self, building: Dict, min_x: float, max_x: float, min_y: float, max_y: float) -> bool:
        """Prüfe ob Gebäude innerhalb Bounds liegt."""
        b = building.get("bounds")
        if not b:
            return False

        centroid_x = (b[0] + b[3]) / 2.0
        centroid_y = (b[1] + b[4]) / 2.0

        return min_x <= centroid_x <= max_x and min_y <= centroid_y <= max_y

    def _building_to_mesh(self, building: Dict, idx: int) -> Optional[Dict]:
        """Konvertiere Building zu Mesh-Dict."""
        from ..io.lod2 import _compute_wall_uvs, _compute_roof_uvs

        all_vertices = []
        all_uvs = []
        vertex_offset = 0
        wall_faces = []
        roof_faces = []

        # Wände
        for verts, faces in building.get("walls", []):
            for face in faces:
                wall_faces.append([f + vertex_offset for f in face])
            all_vertices.append(verts)
            # UV für Wände: 3D-basiert (horizontale Distanz + Höhe), 4m Tiling
            wall_uvs = _compute_wall_uvs(verts, tiling_scale=4.0)
            all_uvs.append(wall_uvs)
            vertex_offset += len(verts)

        # Dächer
        for verts, faces in building.get("roofs", []):
            for face in faces:
                roof_faces.append([f + vertex_offset for f in face])
            all_vertices.append(verts)
            # UV für Dächer: planare XY-Projektion, 2m Tiling
            roof_uvs = _compute_roof_uvs(verts, tiling_scale=2.0)
            all_uvs.append(roof_uvs)
            vertex_offset += len(verts)

        if not all_vertices:
            return None

        vertices_combined = np.vstack(all_vertices)
        uvs_combined = np.vstack(all_uvs)

        return {
            "id": f"building_{idx}",
            "vertices": vertices_combined,
            "uvs": uvs_combined,
            "faces": {"lod2_wall_white": wall_faces, "lod2_roof_red": roof_faces},
        }
