"""Vereinigte Straßenflächen für Ausschlusszonen (Bäume, Reben, Bodenbewuchs)."""

from typing import Dict, Iterable, Optional

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Die Straßenpolygone sind feiner aufgelöst, als für Ausschlusszonen nötig (Mittellinie im Meterabstand, ca.
# 200 Eckpunkte je Straße). Mit 10 cm Toleranz schrumpft die Eckpunktzahl auf ein Zehntel; Vereinigung und
# Pufferung laufen dadurch ~5x schneller, die Zonen (Meter-Abstände) ändern sich nicht messbar.
ROAD_SURFACE_SIMPLIFY_TOLERANCE = 0.1


def union_road_surfaces(
    road_slope_polygons_2d: Optional[Iterable[Dict]], tolerance: float = ROAD_SURFACE_SIMPLIFY_TOLERANCE
) -> Optional[BaseGeometry]:
    """
    Vereinigung aller eingebetteten Straßenflächen (Polygone in lokalen Koordinaten) oder None ohne Straßen.

    Args:
        road_slope_polygons_2d: Liste von Dicts mit "road_polygon" ((M, 2) Array)
        tolerance: Vereinfachungstoleranz in Metern (0 = keine Vereinfachung)
    """
    polygons = []
    for road in road_slope_polygons_2d or []:
        coords = road.get("road_polygon")
        if coords is None or len(coords) < 3:
            continue
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if tolerance > 0:
            polygon = polygon.simplify(tolerance)
        if not polygon.is_empty:
            polygons.append(polygon)
    return unary_union(polygons) if polygons else None
