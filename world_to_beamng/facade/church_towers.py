"""
Detects church towers in the LOD2 buildings.

The LOD2 data has no building function, and nave and tower are ONE building. The church therefore comes from
OSM (polygons with building=church/cathedral/chapel or amenity=place_of_worship), the tower walls from the geometry:
walls that end well above the rest of the building. In addition, walls inside OSM bell tower polygons count
(man_made=tower + tower:type=bell_tower/church), and a building that lies mostly inside such a polygon is a
free-standing tower (all walls).

The result is stored as `building["tower_walls"]` (list of wall indices) in the building dict; the FacadeMapper places
no windows there, but a tower clock.
"""

import logging
from typing import Dict, List, Sequence

import numpy as np
from shapely.geometry import MultiPoint, Point

from .. import config
from .ring_geometry import open_ring

logger = logging.getLogger(__name__)

CHURCH_BUILDINGS = ("church", "cathedral", "chapel")
BELL_TOWER_TYPES = ("bell_tower", "church")
_TOWER_POLYGON_BUFFER_M = 1.0  # walls at the edge of the OSM tower polygon (wall thickness, position error) count too


def is_church(tags: Dict) -> bool:
    return tags.get("building") in CHURCH_BUILDINGS or tags.get("amenity") == "place_of_worship"


def is_bell_tower(tags: Dict) -> bool:
    return (tags.get("man_made") == "tower" and tags.get("tower:type") in BELL_TOWER_TYPES) or tags.get("building") in (
        "bell_tower",
        "church_tower",
    )


class ChurchTowerFinder:
    """Marks the walls of church towers (`tower_walls`) in building dicts."""

    def __init__(self, church_polygons: Sequence, tower_polygons: Sequence):
        self._churches = list(church_polygons)
        self._towers = list(tower_polygons)

    @classmethod
    def from_osm(cls, osm_data: Sequence[Dict], to_local) -> "ChurchTowerFinder":
        """
        Args:
            osm_data: raw Overpass elements (lat/lon)
            to_local: point list -> local coordinates (osm.landuse_polygons.make_local_transform)
        """
        from ..osm.landuse_polygons import build_landuse_polygons

        relevant = [e for e in osm_data if e.get("tags") and (is_church(e["tags"]) or is_bell_tower(e["tags"]))]
        keys = ("building", "amenity", "man_made")
        return cls.from_osm_polygons(build_landuse_polygons(relevant, to_local, tag_keys=keys))

    @classmethod
    def from_osm_polygons(cls, polygons: Sequence[Dict]) -> "ChurchTowerFinder":
        """
        Args:
            polygons: [{"osm_tags": Dict, "geometry": shapely}] in local coordinates (see
                osm.landuse_polygons.build_landuse_polygons)
        """
        churches = [p["geometry"] for p in polygons if is_church(p["osm_tags"])]
        towers = [p["geometry"] for p in polygons if is_bell_tower(p["osm_tags"])]
        return cls(churches, towers)

    def mark(self, buildings: Sequence[Dict]) -> int:
        """
        Sets `tower_walls` in all buildings with a church tower.

        Returns:
            Number of marked buildings
        """
        if not self._churches and not self._towers:
            return 0

        marked = 0
        for building in buildings:
            walls = self._tower_walls(building)
            if walls:
                building["tower_walls"] = walls
                marked += 1
        return marked

    # ------------------------------------------------------------------ Building

    def _tower_walls(self, building: Dict) -> List[int]:
        rings = [open_ring(verts) for verts, _ in building.get("walls", [])]
        if not rings:
            return []
        hull = MultiPoint(np.vstack(rings)[:, :2]).convex_hull
        if hull.geom_type != "Polygon" or hull.area <= 0:
            return []

        if self._covered(hull, self._towers):  # the whole building is a bell tower
            return list(range(len(rings)))

        walls = set()
        for tower in self._towers:  # walls inside the bell tower polygon (tower belongs to the church)
            if tower.intersects(hull):
                area = tower.buffer(_TOWER_POLYGON_BUFFER_M)
                walls.update(i for i, ring in enumerate(rings) if area.contains(Point(ring[:, :2].mean(axis=0))))

        if self._covered(hull, self._churches):
            walls.update(self._tall_walls(rings))
        return sorted(walls)

    @staticmethod
    def _covered(hull, polygons: Sequence) -> bool:
        """At least CHURCH_OVERLAP_MIN of the building footprint lies inside one of the polygons."""
        return any(hull.intersection(polygon).area >= config.CHURCH_OVERLAP_MIN * hull.area for polygon in polygons)

    @staticmethod
    def _tall_walls(rings: Sequence[np.ndarray]) -> List[int]:
        """Walls that end far above the median of the wall top edges (the tower rises out of the nave)."""
        tops = np.array([float(ring[:, 2].max()) for ring in rings])
        median, highest = float(np.median(tops)), float(tops.max())
        if highest - median < config.CHURCH_TOWER_MIN_RISE_M:
            return []
        threshold = median + config.CHURCH_TOWER_HEIGHT_FRACTION * (highest - median)
        return [int(i) for i in np.flatnonzero(tops >= threshold)]
