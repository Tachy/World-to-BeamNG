"""
Forest Point Generator: Poisson-Disk-Sampling für Baumpositionen.

Generiert gleichmäßig verteilte Punkte innerhalb von Waldpolygonen unter
Berücksichtigung der Baumdichte (tree_density aus forest_types).
"""

import logging
import numpy as np
import shapely
from typing import List, Tuple, Dict, Optional
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, MultiPolygon
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

POISSON_CANDIDATES_PER_CELL = 3.0  # Kandidaten je Runde und min_distance² Polygonfläche
POISSON_MIN_CANDIDATES = 64  # Untergrenze je Runde (kleine Polygone)
POISSON_MAX_DRAWS = 3_000_000  # Obergrenze der Zufallspunkte je Runde (Speicher bei sehr dünnen Polygonen)
POISSON_MAX_ROUNDS = 40
POISSON_STOP_FRACTION = 0.004  # Runden, die weniger als diesen Anteil neuer Punkte bringen, gelten als gesättigt


class ForestPointGenerator:
    """
    Generiert Baumpositionen mit Poisson-Disk-Sampling.

    Poisson-Disk-Sampling erzeugt eine gleichmäßige, natürlich wirkende
    Verteilung von Punkten mit einem Mindestabstand.

    Optional können Punkte auf Straßen gefiltert werden durch
    Übergabe einer road_buffer Geometrie.
    """

    def __init__(self, min_distance: float = 1.5, max_attempts: int = 30, road_buffer: Optional[Polygon] = None):
        """
        Args:
            min_distance: Mindestabstand zwischen Bäumen in Metern (default: 1.5m)
            max_attempts: Maximale Versuche pro Punkt (default: 30)
            road_buffer: Optional - Shapely Polygon mit bufferten Straßen (zum Filtern von Bäumen)
        """
        self.min_distance = min_distance
        self.max_attempts = max_attempts

        # Vorbereitete Geometrie für schnelle Straßen-Abfragen (shapely.prepare wirkt in place)
        self.road_buffer = self._prepared(road_buffer)
        self.has_roads = road_buffer is not None
        self.row_exclusion = None  # nur für Baumreihen, siehe set_row_exclusion()

    def set_road_buffer(self, road_buffer: Optional[Polygon]) -> None:
        """
        Setzt oder aktualisiert den Road Buffer.

        Kann jederzeit aufgerufen werden (z.B. vor jedem Tile).

        Args:
            road_buffer: Shapely Polygon mit bufferten Straßen oder None
        """
        self.road_buffer = self._prepared(road_buffer)
        self.has_roads = road_buffer is not None

    @staticmethod
    def _prepared(geometry):
        if geometry is not None:
            shapely.prepare(geometry)
        return geometry

    def set_row_exclusion(self, exclusion) -> None:
        """
        Ausschlussbereich nur für Baumreihen (Gebäude, Straßen mit kleinem Puffer).

        Baumreihen (Alleen) stehen näher an Straßen als Wald; der breite Wald-Straßenpuffer würde sie löschen.
        """
        self.row_exclusion = self._prepared(exclusion)

    def generate_points_along_line(self, line, spacing: float, jitter: float = 0.12) -> List[Tuple[float, float]]:
        """
        Baumpositionen im Abstand `spacing` entlang einer (Multi-)Linie, mit leichtem Versatz entlang der Linie
        (±jitter*spacing), damit die Reihe nicht maschinell wirkt. Eine Linie kürzer als der Abstand bekommt
        einen Baum in der Mitte. Punkte im Ausschlussbereich für Reihen entfallen.
        """
        import random

        parts = list(line.geoms) if hasattr(line, "geoms") else [line]
        points = []
        for part in parts:
            if part.geom_type != "LineString" or part.length <= 0:
                continue
            if part.length < spacing:
                distances = [part.length / 2.0]
            else:
                count = int(part.length // spacing)
                start = (part.length - (count - 1) * spacing) / 2.0
                distances = [start + i * spacing for i in range(count)]
            for d in distances:
                d = min(max(d + random.uniform(-jitter, jitter) * spacing, 0.0), part.length)
                x, y = part.interpolate(d).coords[0]
                if self.row_exclusion is not None and shapely.intersects_xy(self.row_exclusion, x, y):
                    continue
                points.append((x, y))
        return points

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

        # DEBUG: Prüfe Polygon-Validität
        if polygon.is_empty:
            logger.warning(f"Polygon ist leer (area={polygon.area:.2f}m²)")
            return []

        if polygon.area < 1.0:
            logger.debug(f"Polygon zu klein für Bäume (area={polygon.area:.2f}m²)")
            return []

        # Poisson-Disk-Sampling
        points = self._poisson_disk_sampling(
            polygon=polygon, min_distance=adjusted_distance, bounds=(minx, miny, maxx, maxy)
        )

        # OPTIMIERUNG: Filtere Punkte auf Straßen (wenn road_buffer gesetzt)
        if self.has_roads:
            points_before = len(points)
            points = self._filter_points_on_roads(points)
            points_after = len(points)
            logger.debug(
                f"      [Road Filter] {points_before} → {points_after} Punkte ({points_before - points_after} gefiltert)"
            )
            if points_before > points_after:
                logger.debug(
                    f"  Gefiltert: {points_before - points_after} Bäume auf Straßen entfernt "
                    f"({points_after}/{points_before} übrig)"
                )

        logger.debug(
            f"  Generiert: {len(points)} Punkte "
            f"(Dichte={tree_density:.2f}, Abstand={adjusted_distance:.1f}m, Fläche={polygon.area:.0f}m²)"
        )

        return points

    def _filter_points_on_roads(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Filtert Punkte, die auf Straßen liegen (mit Puffer) - vektorisiert über alle Punkte auf einmal.

        Args:
            points: Liste von (x, y) Punkt-Koordinaten

        Returns:
            Gefilterte Liste ohne Punkte auf Straßen
        """
        if not self.has_roads or self.road_buffer is None or not points:
            return points

        xy = np.asarray(points, dtype=float)
        on_roads = shapely.intersects_xy(self.road_buffer, xy[:, 0], xy[:, 1])
        if on_roads.any():
            logger.debug(f"        [Road Filter] {int(on_roads.sum())} Punkte auf Straßen gefunden")
        return [pt for pt, blocked in zip(points, on_roads.tolist()) if not blocked]

    def _poisson_disk_sampling(
        self, polygon: Polygon, min_distance: float, bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[float, float]]:
        """
        Poisson-Disk-Sampling in vektorisierten Runden (statt Bridson mit einer Python-Schleife je Kandidat).

        Jede Runde: Kandidaten gleichverteilt in der Bounding Box, alle außerhalb des Polygons oder näher als
        `min_distance` an bereits gesetzten Punkten verworfen, aus dem Rest eine zufällige unabhängige Menge
        (kein Paar näher als `min_distance`) gewählt. Das ist zufälliges sequentielles Setzen (Dart-Throwing) in
        wenigen NumPy-Schritten; die Dichte liegt wie bei Bridson bei etwa 0,7 Punkten je min_distance².
        Alle Geometrieabfragen laufen vektorisiert (shapely.contains_xy, cKDTree).

        Args:
            polygon: Shapely Polygon
            min_distance: Mindestabstand zwischen Punkten
            bounds: (minx, miny, maxx, maxy)

        Returns:
            Liste von (x, y) Punkten
        """
        minx, miny, maxx, maxy = bounds
        area = polygon.area
        box_area = (maxx - minx) * (maxy - miny)
        shapely.prepare(polygon)

        # Kandidaten je Runde: POISSON_CANDIDATES_PER_CELL je min_distance² Polygonfläche; die Bounding Box wird
        # entsprechend der Füllung überzogen (dünne Polygone in großer Box), gedeckelt gegen Speicherspitzen.
        inside_target = max(POISSON_MIN_CANDIDATES, int(POISSON_CANDIDATES_PER_CELL * area / min_distance**2))
        draws = int(min(POISSON_MAX_DRAWS, inside_target * box_area / max(area, 1e-9)))

        accepted = np.empty((0, 2))
        idle_rounds = 0
        for _ in range(POISSON_MAX_ROUNDS):
            x = np.random.uniform(minx, maxx, draws)
            y = np.random.uniform(miny, maxy, draws)
            keep = shapely.contains_xy(polygon, x, y)
            candidates = np.column_stack([x[keep], y[keep]])
            if len(accepted) and len(candidates):
                free = cKDTree(accepted).query(candidates, distance_upper_bound=min_distance)[0] >= min_distance
                candidates = candidates[free]
            if len(candidates) == 0:
                idle_rounds += 1
            else:
                fresh = candidates[self._independent_subset(candidates, min_distance)]
                accepted = np.vstack([accepted, fresh])
                # Sättigung: die Runde bringt kaum noch etwas
                idle_rounds = idle_rounds + 1 if len(fresh) < POISSON_STOP_FRACTION * len(accepted) else 0
            if idle_rounds >= 2:
                break

        if len(accepted) == 0:
            logger.warning(
                f"Konnte keinen Punkt im Polygon finden "
                f"(bounds={bounds}, area={area:.2f}m², is_valid={polygon.is_valid})"
            )
            return []
        return [tuple(p) for p in accepted.tolist()]

    @staticmethod
    def _independent_subset(points: np.ndarray, min_distance: float) -> np.ndarray:
        """
        Indizes einer zufälligen Teilmenge, in der kein Punktepaar näher als min_distance liegt (maximal: jeder
        nicht gewählte Punkt hat einen gewählten Nachbarn). Zufällige Rangfolge, in jeder Runde gewinnt der
        ranghöchste Punkt seiner verbleibenden Nachbarschaft (Luby) - vollständig vektorisiert.
        """
        count = len(points)
        if count < 2:
            return np.arange(count)
        rank = np.random.permutation(count)
        pairs = cKDTree(points).query_pairs(min_distance, output_type="ndarray")
        first, second = pairs[:, 0], pairs[:, 1]
        alive = np.ones(count, dtype=bool)
        chosen = np.zeros(count, dtype=bool)
        while True:
            both = alive[first] & alive[second]
            first, second = first[both], second[both]
            if len(first) == 0:
                chosen |= alive
                break
            blocked = np.zeros(count, dtype=bool)
            blocked[np.where(rank[first] > rank[second], second, first)] = True
            winners = alive & ~blocked
            chosen |= winners
            dead = winners.copy()
            dead[second[winners[first]]] = True
            dead[first[winners[second]]] = True
            alive &= ~dead
        return np.flatnonzero(chosen)

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
                original_area = geometry.area if hasattr(geometry, "area") else 0
                clipped_area = clipped_geometry.area if hasattr(clipped_geometry, "area") else 0
            else:
                # Keine tile_box - verwende Wald wie er ist
                geometry_to_use = geometry
                original_area = geometry.area if hasattr(geometry, "area") else 0
                clipped_area = original_area

            # Hole Properties
            props = forest_properties.get(forest_type, {})
            tree_density = props.get("tree_density", 0.5)

            # Baumreihe: Bäume entlang der Linie statt Poisson-Verteilung in einer Fläche
            if props.get("row_spacing") and geometry_to_use.geom_type in ("LineString", "MultiLineString"):
                points = self.generate_points_along_line(geometry_to_use, float(props["row_spacing"]))
                logger.debug(f"  Baumreihe {idx}: {len(points)} Punkte")
                result[idx] = points
                continue

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
                logger.debug(
                    f"  Wald {idx}: Nach Tile-Schnitt kein Polygon ({type(geometry_to_use).__name__}), keine Punkte"
                )
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
