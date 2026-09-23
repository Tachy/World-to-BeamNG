"""
POI-Kandidaten für zusätzliche, in der BeamNG-Fahrzeugauswahl wählbare Spawn-Punkte: Orte (place=*) und
große Parkplätze (amenity=parking) aus den rohen Overpass-Daten.

Straßennamen (frühere Variante, siehe item_manager._compute_named_spawn_points()-Historie) sind als Label
oft wenig aussagekräftig ("Nuova strada del Passo del San Gottardo") - ein Ortsname oder "Parkplatz Xyz"
ist für Spieler leichter wiederzuerkennen.
"""

from typing import Callable, Dict, List, Sequence, Tuple

from .landuse_polygons import build_landuse_polygons

ToLocal = Callable[[Sequence[Dict]], List[Tuple[float, float]]]

# Größere Werte = "aussagekräftigerer"/bekannterer Ort - bestimmt die Rangfolge, wenn mehr benannte Orte
# als config.MAX_POI_SPAWN_POINTS existieren. Nur Werte mit Namensbezug (kein amenity/office/etc.).
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
    Liste von {"name", "position_xy", "kind": "place", "rank"} für benannte OSM `place=*`-Nodes
    (Stadt/Dorf/Weiler/...). Unbenannte Places werden übersprungen - ohne Namen kein aussagekräftiges
    Spawn-Label.
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
    Liste von {"name", "position_xy", "kind": "parking", "rank": Fläche in m²} für `amenity=parking`-
    Flächen (Ways/Multipolygon-Relationen) ab min_area_m2 - kleine Parkbuchten/Einzelgaragen sind als
    Spawn-Ort wenig aussagekräftig. Position = Flächen-Zentroid, Name = OSM-Name oder "Parkplatz".
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
