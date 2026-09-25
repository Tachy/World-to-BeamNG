"""
POI candidates for additional spawn points selectable in the BeamNG vehicle selection: places (place=*) and
large parking lots (amenity=parking) from the raw Overpass data.

Road names (earlier variant, see the item_manager._compute_named_spawn_points() history) are often not very
meaningful as labels ("Nuova strada del Passo del San Gottardo") - a place name or "Parkplatz Xyz"
is easier for players to recognize.
"""

from typing import Callable, Dict, List, Sequence, Tuple

from .landuse_polygons import build_landuse_polygons

ToLocal = Callable[[Sequence[Dict]], List[Tuple[float, float]]]

# Larger values = more "meaningful"/better-known place - determines the ranking when more named places
# than config.MAX_POI_SPAWN_POINTS exist. Only values with a name reference (no amenity/office/etc.).
PLACE_RANK = {
    "city": 5,
    "town": 4,
    "village": 3,
    "suburb": 3,
    "hamlet": 2,
    "quarter": 2,
    "isolated_dwelling": 1,
    "farm": 1,
    "neighbourhood": 1,
    "locality": 1,
}


def extract_place_points(osm_data: Sequence[Dict], to_local: ToLocal) -> List[Dict]:
    """
    List of {"name", "position_xy", "kind": "place", "rank"} for named OSM `place=*` nodes
    (city/village/hamlet/...). Unnamed places are skipped - without a name there is no meaningful
    spawn label.
    """
    result = []
    for element in osm_data:
        if element.get("type") != "node":
            continue
        tags = element.get("tags") or {}
        place = tags.get("place")
        name = tags.get("name")
        rank = PLACE_RANK.get(place)
        if rank is None or not name or "lat" not in element or "lon" not in element:
            continue
        xy = to_local([{"lat": element["lat"], "lon": element["lon"]}])
        if not xy:
            continue
        result.append({"name": name, "position_xy": xy[0], "kind": "place", "rank": rank})
    return result


def extract_parking_points(osm_data: Sequence[Dict], to_local: ToLocal, min_area_m2: float) -> List[Dict]:
    """
    List of {"name", "position_xy", "kind": "parking", "rank": area in m²} for `amenity=parking`
    areas (ways/multipolygon relations) from min_area_m2 upward - small parking bays/single garages are not
    very meaningful as a spawn location. Position = area centroid, name = OSM name or "Parkplatz".
    """
    result = []
    for entry in build_landuse_polygons(osm_data, to_local, tag_keys=("amenity",)):
        tags = entry["osm_tags"]
        if tags.get("amenity") != "parking":
            continue
        geometry = entry["geometry"]
        area = geometry.area
        if area < min_area_m2:
            continue
        centroid = geometry.centroid
        name = tags.get("name") or "Parkplatz"
        result.append({"name": name, "position_xy": (centroid.x, centroid.y), "kind": "parking", "rank": area})
    return result
