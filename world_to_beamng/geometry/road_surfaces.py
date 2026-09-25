"""Unioned road surfaces for exclusion zones (trees, vines, ground cover)."""

from typing import Dict, Iterable, Optional

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# The road polygons are finer than needed for exclusion zones (centerline at 1 m spacing, approx.
# 200 vertices per road). With a 10 cm tolerance the vertex count shrinks to a tenth; union and
# buffering therefore run ~5x faster, and the zones (distances in meters) do not change measurably.
ROAD_SURFACE_SIMPLIFY_TOLERANCE = 0.1


def union_road_surfaces(
    road_slope_polygons_2d: Optional[Iterable[Dict]], tolerance: float = ROAD_SURFACE_SIMPLIFY_TOLERANCE
) -> Optional[BaseGeometry]:
    """
    Union of all embedded road surfaces (polygons in local coordinates), or None without roads.

    Args:
        road_slope_polygons_2d: List of dicts with "road_polygon" ((M, 2) array)
        tolerance: Simplification tolerance in meters (0 = no simplification)
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
