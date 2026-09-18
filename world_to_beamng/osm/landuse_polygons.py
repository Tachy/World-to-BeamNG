"""
Baut Landnutzungs-Polygone (landuse/natural/leisure) aus rohen Overpass-Elementen.

Unterstützt sowohl geschlossene Ways als auch Multipolygon-Relationen (äußere
Ringe aus mehreren Ways zusammengesetzt, innere Ringe als Löcher). Relationen
sind wichtig: große Wald-, Weinberg- und Wohngebietsflächen sind in OSM fast
immer Relationen, nicht Ways.
"""

from typing import Callable, Dict, List, Sequence, Tuple

from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from ..geometry.coordinates import transformer_to_utm
from ..logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

AREA_TAG_KEYS = ("landuse", "natural", "leisure")

# to_local(points) -> [(x, y), ...]; points sind Overpass-Punkte {"lat":.., "lon":..}
ToLocal = Callable[[Sequence[Dict]], List[Tuple[float, float]]]


def make_local_transform(global_offset: Tuple[float, float]) -> ToLocal:
    """WGS84 (Overpass) -> UTM -> lokale Koordinaten (UTM minus global_offset)."""
    offset_x, offset_y = global_offset[0], global_offset[1]

    def to_local(points: Sequence[Dict]) -> List[Tuple[float, float]]:
        coords = []
        for pt in points:
            if not isinstance(pt, dict) or "lat" not in pt or "lon" not in pt:
                continue
            x, y = transformer_to_utm.transform(pt["lon"], pt["lat"])
            coords.append((x - offset_x, y - offset_y))
        return coords

    return to_local


def _repair(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty or geometry.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    return geometry


def _way_polygon(element: Dict, to_local: ToLocal):
    geometry = element.get("geometry")
    if not geometry or len(geometry) < 4:
        return None
    first, last = geometry[0], geometry[-1]
    if (first.get("lat"), first.get("lon")) != (last.get("lat"), last.get("lon")):
        return None  # offener Linienzug ist keine Fläche
    coords = to_local(geometry)
    if len(coords) < 4:
        return None
    return _repair(Polygon(coords))


def _member_lines(members: Sequence[Dict], roles: Tuple[str, ...], to_local: ToLocal) -> List[LineString]:
    lines = []
    for member in members:
        if member.get("role", "") not in roles:
            continue
        geometry = member.get("geometry")
        if not geometry or len(geometry) < 2:
            continue
        coords = to_local(geometry)
        if len(coords) >= 2:
            lines.append(LineString(coords))
    return lines


def _rings_to_polygons(lines: List[LineString]):
    if not lines:
        return []
    return list(polygonize(unary_union(lines)))


def _relation_polygon(element: Dict, to_local: ToLocal):
    if element.get("tags", {}).get("type") != "multipolygon":
        return None
    members = element.get("members", [])
    outer = _rings_to_polygons(_member_lines(members, ("outer", ""), to_local))
    if not outer:
        return None
    geometry = unary_union(outer)
    inner = _rings_to_polygons(_member_lines(members, ("inner",), to_local))
    if inner:
        geometry = geometry.difference(unary_union(inner))
    return _repair(geometry)


def build_landuse_polygons(
    osm_data: Sequence[Dict], to_local: ToLocal, tag_keys: Sequence[str] = AREA_TAG_KEYS
) -> List[Dict]:
    """
    Args:
        osm_data: rohe Overpass-Elemente (mit "geometry" bzw. Members mit "geometry")
        to_local: Punktliste -> lokale (x, y)-Koordinaten (siehe make_local_transform())
        tag_keys: Tag-Schlüssel, die ein Element zur Fläche machen (Standard:
            landuse/natural/leisure; z.B. ("building",) für Gebäudegrundrisse)

    Returns:
        Liste von {"osm_tags": Dict, "geometry": shapely (Multi)Polygon} für alle
        Elemente mit einem der tag_keys.
    """
    result = []
    for element in osm_data:
        tags = element.get("tags")
        if not tags or not any(key in tags for key in tag_keys):
            continue
        try:
            if element.get("type") == "way":
                geometry = _way_polygon(element, to_local)
            elif element.get("type") == "relation":
                geometry = _relation_polygon(element, to_local)
            else:
                geometry = None
        except Exception as exc:  # defekte Einzelgeometrie darf den Export nicht stoppen
            logger.debug(f"  [Landuse] Element {element.get('id')} übersprungen: {exc}")
            continue
        if geometry is not None:
            result.append({"osm_tags": tags, "geometry": geometry})
    return result
