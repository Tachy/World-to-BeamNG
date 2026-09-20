"""
Kirchtürme in den LOD2-Gebäuden erkennen.

Die LOD2-Daten kennen keine Gebäudefunktion, und Kirchenschiff und Turm sind EIN Gebäude. Die Kirche kommt deshalb aus
OSM (Polygone mit building=church/cathedral/chapel oder amenity=place_of_worship), die Turmwände aus der Geometrie:
Wände, die deutlich über dem Rest des Gebäudes enden. Zusätzlich zählen Wände in OSM-Glockenturm-Polygonen
(man_made=tower + tower:type=bell_tower/church), und ein Gebäude, das überwiegend in so einem Polygon liegt, ist ein
alleinstehender Turm (alle Wände).

Das Ergebnis steht als `building["tower_walls"]` (Liste von Wand-Indizes) im Gebäude-Dict; der FacadeMapper setzt dort
keine Fenster, sondern eine Turmuhr.
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
_TOWER_POLYGON_BUFFER_M = 1.0  # Wände am Rand des OSM-Turmpolygons (Mauerdicke, Lagefehler) zählen dazu


def is_church(tags: Dict) -> bool:
    return tags.get("building") in CHURCH_BUILDINGS or tags.get("amenity") == "place_of_worship"


def is_bell_tower(tags: Dict) -> bool:
    return (tags.get("man_made") == "tower" and tags.get("tower:type") in BELL_TOWER_TYPES) or tags.get("building") in (
        "bell_tower",
        "church_tower",
    )


class ChurchTowerFinder:
    """Markiert in Gebäude-Dicts die Wände von Kirchtürmen (`tower_walls`)."""

    def __init__(self, church_polygons: Sequence, tower_polygons: Sequence):
        self._churches = list(church_polygons)
        self._towers = list(tower_polygons)

    @classmethod
    def from_osm(cls, osm_data: Sequence[Dict], to_local) -> "ChurchTowerFinder":
        """
        Args:
            osm_data: rohe Overpass-Elemente (lat/lon)
            to_local: Punktliste -> lokale Koordinaten (osm.landuse_polygons.make_local_transform)
        """
        from ..osm.landuse_polygons import build_landuse_polygons

        relevant = [e for e in osm_data if e.get("tags") and (is_church(e["tags"]) or is_bell_tower(e["tags"]))]
        keys = ("building", "amenity", "man_made")
        return cls.from_osm_polygons(build_landuse_polygons(relevant, to_local, tag_keys=keys))

    @classmethod
    def from_osm_polygons(cls, polygons: Sequence[Dict]) -> "ChurchTowerFinder":
        """
        Args:
            polygons: [{"osm_tags": Dict, "geometry": shapely}] in lokalen Koordinaten (siehe
                osm.landuse_polygons.build_landuse_polygons)
        """
        churches = [p["geometry"] for p in polygons if is_church(p["osm_tags"])]
        towers = [p["geometry"] for p in polygons if is_bell_tower(p["osm_tags"])]
        return cls(churches, towers)

    def mark(self, buildings: Sequence[Dict]) -> int:
        """
        Setzt `tower_walls` in allen Gebäuden mit Kirchturm.

        Returns:
            Anzahl markierter Gebäude
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

    # ------------------------------------------------------------------ Gebäude

    def _tower_walls(self, building: Dict) -> List[int]:
        rings = [open_ring(verts) for verts, _ in building.get("walls", [])]
        if not rings:
            return []
        hull = MultiPoint(np.vstack(rings)[:, :2]).convex_hull
        if hull.geom_type != "Polygon" or hull.area <= 0:
            return []

        if self._covered(hull, self._towers):  # das ganze Gebäude ist ein Glockenturm
            return list(range(len(rings)))

        walls = set()
        for tower in self._towers:  # Wände im Glockenturm-Polygon (Turm gehört zur Kirche)
            if tower.intersects(hull):
                area = tower.buffer(_TOWER_POLYGON_BUFFER_M)
                walls.update(i for i, ring in enumerate(rings) if area.contains(Point(ring[:, :2].mean(axis=0))))

        if self._covered(hull, self._churches):
            walls.update(self._tall_walls(rings))
        return sorted(walls)

    @staticmethod
    def _covered(hull, polygons: Sequence) -> bool:
        """Mindestens CHURCH_OVERLAP_MIN der Grundfläche des Gebäudes liegt in einem der Polygone."""
        return any(hull.intersection(polygon).area >= config.CHURCH_OVERLAP_MIN * hull.area for polygon in polygons)

    @staticmethod
    def _tall_walls(rings: Sequence[np.ndarray]) -> List[int]:
        """Wände, die weit über dem Median der Wandoberkanten enden (der Turm ragt aus dem Kirchenschiff)."""
        tops = np.array([float(ring[:, 2].max()) for ring in rings])
        median, highest = float(np.median(tops)), float(tops.max())
        if highest - median < config.CHURCH_TOWER_MIN_RISE_M:
            return []
        threshold = median + config.CHURCH_TOWER_HEIGHT_FRACTION * (highest - median)
        return [int(i) for i in np.flatnonzero(tops >= threshold)]
