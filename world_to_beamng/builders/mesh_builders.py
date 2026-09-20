"""
Builder für Grid- und Building-Meshes.

Vereinfacht komplexe Mesh-Generierung mit einem klaren Builder-Pattern.
"""

from typing import Optional, List, Dict, Tuple
import numpy as np

from ..facade.facade_mapper import FacadeMapper
from ..facade.flat_roof import FlatRoofRimBuilder
from ..facade.material_names import (
    FLAT_ROOF_MATERIAL,
    ROOF_EDGE_MATERIAL,
    ROOF_MATERIAL,
    ROOF_TRIM_MATERIAL,
    WALL_MATERIALS,
    WINDOW_MATERIAL,
)
from ..facade.roof_mesh import RoofMeshBuilder


class GridBuilder:
    """
    Builder für Terrain-Grids.

    Beispiel:
        >>> builder = GridBuilder()
        >>> grid = (builder
        ...     .with_points(height_points)
        ...     .with_elevations(height_elevations)
        ...     .with_spacing(2.0)
        ...     .with_cache_key(cache_key)
        ...     .build())
    """

    def __init__(self):
        self._points = None
        self._elevations = None
        self._spacing = 2.0
        self._cache_key = None

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

    def with_cache_key(self, cache_key: str) -> "GridBuilder":
        """
        Setze den Cache-Schlüssel (Tile-Hash) für das Terrain-Grid.

        Args:
            cache_key: Cache-Key

        Returns:
            Self für Method-Chaining
        """
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

        return create_terrain_grid(
            self._points,
            self._elevations,
            grid_spacing=spacing,
            tile_hash=tile_hash,
        )


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
        self._facade_mapper = FacadeMapper()
        self._roof_builder = RoofMeshBuilder()
        self._rim_builder = FlatRoofRimBuilder()

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
        """
        Konvertiere Building zu Mesh-Dict.

        Wände: fugenloser Putz (Farbe je Gebäude) plus Fenster/Türen als eigene Flächen. Schrägdächer: Biberschwanz mit
        Überstand (Stirnbrett/Untersicht als Trim). Flachdächer: Kies plus Blechrand.
        """
        facade = self._facade_mapper.map_building(building)
        roof = self._roof_builder.build(building)
        rim = self._rim_builder.build(building)

        pieces = [
            (facade.vertices, facade.uvs, {WALL_MATERIALS[facade.plaster]: facade.wall_faces, WINDOW_MATERIAL: facade.window_faces}),
            (roof.sloped.vertices, roof.sloped.uvs, {ROOF_MATERIAL: roof.sloped.faces}),
            (roof.flat.vertices, roof.flat.uvs, {FLAT_ROOF_MATERIAL: roof.flat.faces}),
            (roof.trim.vertices, roof.trim.uvs, {ROOF_TRIM_MATERIAL: roof.trim.faces}),
            (rim.vertices, rim.uvs, {ROOF_EDGE_MATERIAL: rim.faces}),
        ]

        vertices, uvs, faces = [], [], {}
        offset = 0
        for piece_vertices, piece_uvs, faces_by_material in pieces:
            if not len(piece_vertices):
                continue
            vertices.append(piece_vertices)
            uvs.append(piece_uvs)
            for material, material_faces in faces_by_material.items():
                if material_faces:
                    faces.setdefault(material, []).extend([[i + offset for i in face] for face in material_faces])
            offset += len(piece_vertices)

        if not vertices:
            return None
        return {"id": f"building_{idx}", "vertices": np.vstack(vertices), "uvs": np.vstack(uvs), "faces": faces}
