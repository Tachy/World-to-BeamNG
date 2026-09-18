"""
Builder für Grid- und Building-Meshes.

Vereinfacht komplexe Mesh-Generierung mit einem klaren Builder-Pattern.
"""

from typing import Optional, List, Dict, Tuple
import numpy as np


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
            cache_file = self._cache_manager.cache_dir / f"grid_v3_{tile_hash}_spacing{spacing:.1f}m.npz"
            self._was_cached = cache_file.exists()

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
