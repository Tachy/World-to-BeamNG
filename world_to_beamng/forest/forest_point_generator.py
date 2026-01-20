"""
Forest Point Generator: Poisson-Disk-Sampling für Baumpositionen.

Generiert gleichmäßig verteilte Punkte innerhalb von Waldpolygonen unter
Berücksichtigung der Baumdichte (tree_density aus forest_types).
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.prepared import prep

logger = logging.getLogger(__name__)


class ForestPointGenerator:
    """
    Generiert Baumpositionen mit Poisson-Disk-Sampling.

    Poisson-Disk-Sampling erzeugt eine gleichmäßige, natürlich wirkende
    Verteilung von Punkten mit einem Mindestabstand.
    """

    def __init__(self, min_distance: float = 1.5, max_attempts: int = 30):
        """
        Args:
            min_distance: Mindestabstand zwischen Bäumen in Metern (default: 1.5m)
            max_attempts: Maximale Versuche pro Punkt (default: 30)
        """
        self.min_distance = min_distance
        self.max_attempts = max_attempts

    def generate_points(
        self, polygon: Polygon, tree_density: float, min_distance_override: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
        Generiere Baumpositionen innerhalb eines Polygons.

        Nutzt Poisson-Disk-Sampling für natürliche Verteilung.

        Args:
            polygon: Shapely Polygon (Waldgebiet)
            tree_density: Dichte-Faktor (0.0 - 1.0) aus forest_types
            min_distance_override: Optional - überschreibt self.min_distance

        Returns:
            Liste von (x, y) Koordinaten
        """
        if tree_density <= 0.0:
            return []

        min_dist = min_distance_override if min_distance_override is not None else self.min_distance

        # Passe Mindestabstand an Dichte an (quadratisch, weil Poisson-Disk mit Fläche skaliert)
        # Höhere Dichte → kleinerer Abstand
        # tree_density 1.0 → min_distance
        # tree_density 0.25 → 2× Abstand (400 → 100 Bäume/ha)
        import math

        adjusted_distance = min_dist / math.sqrt(tree_density) if tree_density > 0 else min_dist

        # Bounding Box des Polygons
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        if width <= 0 or height <= 0:
            logger.warning(f"Polygon mit ungültiger Bounding Box: {polygon.bounds}")
            return []

        # Poisson-Disk-Sampling
        points = self._poisson_disk_sampling(
            polygon=polygon, min_distance=adjusted_distance, bounds=(minx, miny, maxx, maxy)
        )

        logger.debug(
            f"  Generiert: {len(points)} Punkte "
            f"(Dichte={tree_density:.2f}, Abstand={adjusted_distance:.1f}m, Fläche={polygon.area:.0f}m²)"
        )

        return points

    def _poisson_disk_sampling(
        self, polygon: Polygon, min_distance: float, bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[float, float]]:
        """
        Poisson-Disk-Sampling-Algorithmus (Bridson's Algorithm - OPTIMIERT).

        Optimierungen:
        - Prepared geometry für schnelle contains()-Abfragen
        - Effizientere Active List Management
        - Begrenzte Nachbarschaftssuche mit frühem Exit

        Args:
            polygon: Shapely Polygon
            min_distance: Mindestabstand zwischen Punkten
            bounds: (minx, miny, maxx, maxy)

        Returns:
            Liste von (x, y) Punkten
        """
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny

        # OPTIMIERUNG: Prepared geometry für schnelle contains()-Abfragen
        prep_polygon = prep(polygon)

        # Grid-Zellgröße (für schnelle Nachbarschaftssuche)
        cell_size = min_distance / np.sqrt(2)
        grid_width = int(np.ceil(width / cell_size))
        grid_height = int(np.ceil(height / cell_size))

        # Grid für schnelle Nachbarschaftssuche (None = leer)
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]

        # Resultat
        points = []

        # Active list für Kandidaten (verwendet Index für O(1) Removal)
        active_indices = []

        # Startpunkt (zufällig innerhalb des Polygons)
        start_point = self._random_point_in_polygon(polygon, prep_polygon)
        if start_point is None:
            logger.warning("Konnte keinen Startpunkt im Polygon finden")
            return []

        points.append(start_point)
        active_indices.append(0)

        # Grid-Position
        gx = int((start_point[0] - minx) / cell_size)
        gy = int((start_point[1] - miny) / cell_size)
        if 0 <= gx < grid_width and 0 <= gy < grid_height:
            grid[gx][gy] = start_point

        # Hauptschleife
        while active_indices:
            # OPTIMIERUNG: Wähle zufälligen Punkt aus Active List
            idx_in_active = np.random.randint(len(active_indices))
            point_idx = active_indices[idx_in_active]
            point = points[point_idx]

            found = False

            # Versuche max_attempts neue Punkte um diesen Punkt
            for _ in range(self.max_attempts):
                # Zufälliger Winkel und Abstand
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(min_distance, 2 * min_distance)

                new_x = point[0] + distance * np.cos(angle)
                new_y = point[1] + distance * np.sin(angle)
                new_point = (new_x, new_y)

                # Prüfe ob innerhalb Bounds
                if not (minx <= new_x <= maxx and miny <= new_y <= maxy):
                    continue

                # OPTIMIERUNG: Nutze prepared geometry für schnellere contains()
                if not prep_polygon.contains(Point(new_x, new_y)):
                    continue

                # Grid-Position
                gx = int((new_x - minx) / cell_size)
                gy = int((new_y - miny) / cell_size)

                if not (0 <= gx < grid_width and 0 <= gy < grid_height):
                    continue

                # Prüfe Nachbarschaft im Grid
                if self._is_valid_point(new_point, grid, gx, gy, cell_size, min_distance, (minx, miny)):
                    points.append(new_point)
                    active_indices.append(len(points) - 1)
                    grid[gx][gy] = new_point
                    found = True
                    break

            # Wenn kein neuer Punkt gefunden wurde, entferne aus Active List
            if not found:
                active_indices.pop(idx_in_active)

        return points

    def _is_valid_point(
        self,
        point: Tuple[float, float],
        grid: List[List],
        gx: int,
        gy: int,
        cell_size: float,
        min_distance: float,
        grid_origin: Tuple[float, float],
    ) -> bool:
        """
        Prüfe ob Punkt gültigen Abstand zu allen Nachbarn hat.

        Args:
            point: (x, y) zu prüfender Punkt
            grid: Grid mit existierenden Punkten
            gx, gy: Grid-Koordinaten
            cell_size: Größe einer Grid-Zelle
            min_distance: Mindestabstand
            grid_origin: (minx, miny) des Grids

        Returns:
            True wenn Punkt gültigen Abstand hat
        """
        grid_width = len(grid)
        grid_height = len(grid[0]) if grid else 0

        # Prüfe 5×5 Nachbarschaft (2 Zellen in jede Richtung)
        for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
            for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                neighbor = grid[i][j]
                if neighbor is not None:
                    dx = point[0] - neighbor[0]
                    dy = point[1] - neighbor[1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < min_distance:
                        return False

        return True

    def _random_point_in_polygon(self, polygon: Polygon, prep_polygon=None, max_tries: int = 100) -> Optional[Tuple[float, float]]:
        """
        Generiere zufälligen Punkt innerhalb eines Polygons.

        Nutzt Rejection Sampling (Monte Carlo).

        Args:
            polygon: Shapely Polygon
            prep_polygon: Optional - prepared geometry für schnellere contains()
            max_tries: Maximale Versuche

        Returns:
            (x, y) oder None wenn kein Punkt gefunden
        """
        minx, miny, maxx, maxy = polygon.bounds

        # Nutze prepared geometry wenn vorhanden
        if prep_polygon is None:
            prep_polygon = prep(polygon)

        for _ in range(max_tries):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)

            if prep_polygon.contains(Point(x, y)):
                return (x, y)

        logger.warning(f"Konnte keinen Punkt in Polygon finden nach {max_tries} Versuchen")
        return None

    def generate_points_for_forests(
        self, forests: List[Dict], forest_properties: Dict[str, Dict]
    ) -> Dict[int, List[Tuple[float, float]]]:
        """
        Generiere Punkte für mehrere Waldpolygone.

        OPTIMIERUNG: Schneidet Wald-Polygon mit tile_box BEVOR Punkte generiert werden.
        Dadurch wird nur die relevante Fläche bearbeitet, nicht die ganze Wald-Geometrie.

        Args:
            forests: Liste von Forest-Dicts aus ForestNormalizer
                     (mit "type", "geometry", "bounds", "tile_box", ...)
            forest_properties: Dict von forest_type → properties
                              (mit "tree_density", ...)

        Returns:
            Dict: forest_index → Liste von (x, y) Punkten (nur innerhalb tile_box!)
        """
        result = {}

        for idx, forest in enumerate(forests):
            forest_type = forest.get("type")
            geometry = forest.get("geometry")
            tile_box = forest.get("tile_box")  # Für Filterung

            if not forest_type or not geometry:
                logger.warning(f"Waldpolygon {idx} ohne type/geometry, überspringe")
                continue

            # OPTIMIERUNG: Schneide Wald mit tile_box BEVOR Punkte generiert werden
            if tile_box:
                # Intersection mit tile_box - verwende das Ergebnis für Punkt-Generierung
                clipped_geometry = geometry.intersection(tile_box)
                
                if clipped_geometry.is_empty:
                    # Wald ist außerhalb der Tile
                    result[idx] = []
                    logger.debug(f"  Wald {idx}: Vollständig außerhalb Tile-Box, keine Punkte")
                    continue
                    
                geometry_to_use = clipped_geometry
                original_area = geometry.area if hasattr(geometry, 'area') else 0
                clipped_area = clipped_geometry.area if hasattr(clipped_geometry, 'area') else 0
            else:
                # Keine tile_box - verwende Wald wie er ist
                geometry_to_use = geometry
                original_area = geometry.area if hasattr(geometry, 'area') else 0
                clipped_area = original_area

            # Hole Properties
            props = forest_properties.get(forest_type, {})
            tree_density = props.get("tree_density", 0.5)

            # Generiere Punkte NUR auf der relevanten Geometrie
            if isinstance(geometry_to_use, Polygon):
                points = self.generate_points(geometry_to_use, tree_density)
            elif isinstance(geometry_to_use, MultiPolygon):
                # Für MultiPolygon: Generiere für jedes Teil-Polygon
                points = []
                for poly in geometry_to_use.geoms:
                    points.extend(self.generate_points(poly, tree_density))
            else:
                # Kann passieren wenn intersection ein Point/LineString zurückgibt
                logger.debug(f"  Wald {idx}: Nach Tile-Schnitt kein Polygon ({type(geometry_to_use).__name__}), keine Punkte")
                points = []

            # Debug-Info
            if tile_box:
                clipped_pct = (clipped_area / original_area * 100) if original_area > 0 else 0
                logger.debug(
                    f"  Wald {idx}: {len(points)} Punkte "
                    f"({clipped_pct:.0f}% im Tile, Fläche {clipped_area:.0f}m² von {original_area:.0f}m²)"
                )
            else:
                logger.debug(f"  Wald {idx} ({forest_type}): {len(points)} Punkte (keine Tile-Box)")
            
            result[idx] = points

        total_points = sum(len(pts) for pts in result.values())
        logger.info(f"✓ {total_points} Baumpositionen für {len(forests)} Wälder generiert")

        return result
