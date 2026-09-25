"""
Forest Point Generator: Poisson-disk sampling for tree positions.

Generates evenly distributed points inside forest polygons, taking
the tree density (tree_density from forest_types) into account.
"""

import numpy as np
import shapely
from typing import List, Tuple, Dict, Optional
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, MultiPolygon
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

POISSON_CANDIDATES_PER_CELL = 3.0  # candidates per round and min_distance² of polygon area
POISSON_MIN_CANDIDATES = 64  # lower bound per round (small polygons)
POISSON_MAX_DRAWS = 3_000_000  # upper bound of random points per round (memory for very thin polygons)
POISSON_MAX_ROUNDS = 40
POISSON_STOP_FRACTION = 0.004  # rounds that yield less than this fraction of new points count as saturated


class ForestPointGenerator:
    """
    Generates tree positions with Poisson-disk sampling.

    Poisson-disk sampling produces an even, natural-looking
    distribution of points with a minimum distance.

    Optionally, points on roads can be filtered by
    passing a road_buffer geometry.
    """

    def __init__(self, min_distance: float = 1.5, max_attempts: int = 30, road_buffer: Optional[Polygon] = None):
        """
        Args:
            min_distance: minimum distance between trees in meters (default: 1.5m)
            max_attempts: maximum attempts per point (default: 30)
            road_buffer: optional - Shapely polygon with buffered roads (for filtering trees)
        """
        self.min_distance = min_distance
        self.max_attempts = max_attempts

        # Prepared geometry for fast road queries (shapely.prepare works in place)
        self.road_buffer = self._prepared(road_buffer)
        self.has_roads = road_buffer is not None
        self.row_exclusion = None  # only for tree rows, see set_row_exclusion()

    def set_road_buffer(self, road_buffer: Optional[Polygon]) -> None:
        """
        Sets or updates the road buffer.

        Can be called at any time (e.g. before each tile).

        Args:
            road_buffer: Shapely polygon with buffered roads or None
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
        Exclusion area for tree rows only (buildings, roads with a small buffer).

        Tree rows (avenues) stand closer to roads than forest; the wide forest road buffer would delete them.
        """
        self.row_exclusion = self._prepared(exclusion)

    def generate_points_along_line(self, line, spacing: float, jitter: float = 0.12) -> List[Tuple[float, float]]:
        """
        Tree positions at `spacing` intervals along a (multi-)line, with a slight offset along the line
        (±jitter*spacing) so that the row does not look mechanical. A line shorter than the spacing gets
        one tree in the middle. Points in the row exclusion area are dropped.
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
        Generate tree positions inside a polygon.

        Uses Poisson-disk sampling for a natural distribution.

        Args:
            polygon: Shapely polygon (forest area)
            tree_density: density factor (0.0 - 1.0) from forest_types
            min_distance_override: optional - overrides self.min_distance

        Returns:
            List of (x, y) coordinates
        """
        if tree_density <= 0.0:
            return []

        min_dist = min_distance_override if min_distance_override is not None else self.min_distance

        # Adjust the minimum distance to the density (quadratic, because Poisson-disk scales with area)
        # Higher density → smaller distance
        # tree_density 1.0 → min_distance
        # tree_density 0.25 → 2× distance (400 → 100 trees/ha)
        import math

        adjusted_distance = min_dist / math.sqrt(tree_density) if tree_density > 0 else min_dist

        # Bounding box of the polygon
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        if width <= 0 or height <= 0:
            logger.warning(f"Polygon with invalid bounding box: {polygon.bounds}")
            return []

        # DEBUG: check polygon validity
        if polygon.is_empty:
            logger.warning(f"Polygon is empty (area={polygon.area:.2f}m²)")
            return []

        if polygon.area < 1.0:
            logger.debug(f"Polygon too small for trees (area={polygon.area:.2f}m²)")
            return []

        # Poisson-disk sampling
        points = self._poisson_disk_sampling(
            polygon=polygon, min_distance=adjusted_distance, bounds=(minx, miny, maxx, maxy)
        )

        # OPTIMIZATION: filter points on roads (if road_buffer is set)
        if self.has_roads:
            points_before = len(points)
            points = self._filter_points_on_roads(points)
            points_after = len(points)
            logger.debug(
                f"      [Road Filter] {points_before} → {points_after} points ({points_before - points_after} filtered)"
            )
            if points_before > points_after:
                logger.debug(
                    f"  Filtered: {points_before - points_after} trees on roads removed "
                    f"({points_after}/{points_before} left)"
                )

        logger.debug(
            f"  Generated: {len(points)} points "
            f"(density={tree_density:.2f}, spacing={adjusted_distance:.1f}m, area={polygon.area:.0f}m²)"
        )

        return points

    def _filter_points_on_roads(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Filters points that lie on roads (with buffer) - vectorized over all points at once.

        Args:
            points: list of (x, y) point coordinates

        Returns:
            Filtered list without points on roads
        """
        if not self.has_roads or self.road_buffer is None or not points:
            return points

        xy = np.asarray(points, dtype=float)
        on_roads = shapely.intersects_xy(self.road_buffer, xy[:, 0], xy[:, 1])
        if on_roads.any():
            logger.debug(f"        [Road Filter] {int(on_roads.sum())} points found on roads")
        return [pt for pt, blocked in zip(points, on_roads.tolist()) if not blocked]

    def _poisson_disk_sampling(
        self, polygon: Polygon, min_distance: float, bounds: Tuple[float, float, float, float]
    ) -> List[Tuple[float, float]]:
        """
        Poisson-disk sampling in vectorized rounds (instead of Bridson with a Python loop per candidate).

        Each round: candidates uniformly distributed in the bounding box, all outside the polygon or closer than
        `min_distance` to already placed points discarded, and a random independent set (no pair closer than
        `min_distance`) chosen from the rest. This is random sequential placement (dart throwing) in
        a few NumPy steps; the density is about 0.7 points per min_distance², as with Bridson.
        All geometry queries run vectorized (shapely.contains_xy, cKDTree).

        Args:
            polygon: Shapely polygon
            min_distance: minimum distance between points
            bounds: (minx, miny, maxx, maxy)

        Returns:
            List of (x, y) points
        """
        minx, miny, maxx, maxy = bounds
        area = polygon.area
        box_area = (maxx - minx) * (maxy - miny)
        shapely.prepare(polygon)

        # Candidates per round: POISSON_CANDIDATES_PER_CELL per min_distance² of polygon area; the bounding box is
        # oversampled according to its fill (thin polygons in a large box), capped against memory spikes.
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
                # Saturation: the round hardly yields anything anymore
                idle_rounds = idle_rounds + 1 if len(fresh) < POISSON_STOP_FRACTION * len(accepted) else 0
            if idle_rounds >= 2:
                break

        if len(accepted) == 0:
            logger.warning(
                f"Could not find a point inside the polygon "
                f"(bounds={bounds}, area={area:.2f}m², is_valid={polygon.is_valid})"
            )
            return []
        return [tuple(p) for p in accepted.tolist()]

    @staticmethod
    def _independent_subset(points: np.ndarray, min_distance: float) -> np.ndarray:
        """
        Indices of a random subset in which no pair of points is closer than min_distance (maximal: every
        unchosen point has a chosen neighbor). Random ranking; in each round the highest-ranked point of its
        remaining neighborhood wins (Luby) - fully vectorized.
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
        Generate points for multiple forest polygons.

        OPTIMIZATION: clips the forest polygon with tile_box BEFORE points are generated.
        This way only the relevant area is processed, not the whole forest geometry.

        Args:
            forests: list of forest dicts from ForestNormalizer
                     (with "type", "geometry", "bounds", "tile_box", ...)
            forest_properties: dict of forest_type → properties
                              (with "tree_density", ...)

        Returns:
            Dict: forest_index → list of (x, y) points (only inside tile_box!)
        """
        result = {}

        for idx, forest in enumerate(forests):
            forest_type = forest.get("type")
            geometry = forest.get("geometry")
            tile_box = forest.get("tile_box")  # For filtering

            if not forest_type or not geometry:
                logger.warning(f"Forest polygon {idx} without type/geometry, skipping")
                continue

            # OPTIMIZATION: clip the forest with tile_box BEFORE points are generated
            if tile_box:
                # Intersection with tile_box - use the result for point generation
                clipped_geometry = geometry.intersection(tile_box)

                if clipped_geometry.is_empty:
                    # Forest is outside the tile
                    result[idx] = []
                    logger.debug(f"  Forest {idx}: completely outside the tile box, no points")
                    continue

                geometry_to_use = clipped_geometry
                original_area = geometry.area if hasattr(geometry, "area") else 0
                clipped_area = clipped_geometry.area if hasattr(clipped_geometry, "area") else 0
            else:
                # No tile_box - use the forest as it is
                geometry_to_use = geometry
                original_area = geometry.area if hasattr(geometry, "area") else 0
                clipped_area = original_area

            # Get properties
            props = forest_properties.get(forest_type, {})
            tree_density = props.get("tree_density", 0.5)

            # Tree row: trees along the line instead of a Poisson distribution in an area
            if props.get("row_spacing") and geometry_to_use.geom_type in ("LineString", "MultiLineString"):
                points = self.generate_points_along_line(geometry_to_use, float(props["row_spacing"]))
                logger.debug(f"  Tree row {idx}: {len(points)} points")
                result[idx] = points
                continue

            # Generate points ONLY on the relevant geometry
            if isinstance(geometry_to_use, Polygon):
                points = self.generate_points(geometry_to_use, tree_density)
            elif isinstance(geometry_to_use, MultiPolygon):
                # For MultiPolygon: generate for each sub-polygon
                points = []
                for poly in geometry_to_use.geoms:
                    points.extend(self.generate_points(poly, tree_density))
            else:
                # Can happen if intersection returns a Point/LineString
                logger.debug(
                    f"  Forest {idx}: no polygon after tile clipping ({type(geometry_to_use).__name__}), no points"
                )
                points = []

            # Debug info
            if tile_box:
                clipped_pct = (clipped_area / original_area * 100) if original_area > 0 else 0
                logger.debug(
                    f"  Forest {idx}: {len(points)} points "
                    f"({clipped_pct:.0f}% inside the tile, area {clipped_area:.0f}m² of {original_area:.0f}m²)"
                )
            else:
                logger.debug(f"  Forest {idx} ({forest_type}): {len(points)} points (no tile box)")

            result[idx] = points

        total_points = sum(len(pts) for pts in result.values())
        logger.info(f"✓ {total_points} tree positions generated for {len(forests)} forests")

        return result
