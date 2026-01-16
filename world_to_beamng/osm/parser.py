"""
OSM Daten Parser und Datenextraktion.
"""

import numpy as np

from ..geometry.coordinates import transformer_to_wgs84, transformer_to_utm


def calculate_bbox_from_height_data(points, margin=0.0):
    """Berechnet die BBOX (WGS84) aus UTM-Hoehendaten.

    Args:
        points: UTM-Koordinaten (N x 2)
        margin: Erweiterung in Metern (in UTM)

    Returns:
        BBox im Format [lat_min, lon_min, lat_max, lon_max]
    """
    # Finde Min/Max in UTM
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)

    # Erweitere um Margin (in UTM, also in Metern)
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    # Konvertiere zu WGS84
    min_lon, min_lat = transformer_to_wgs84.transform(min_x, min_y)
    max_lon, max_lat = transformer_to_wgs84.transform(max_x, max_y)

    bbox = [min_lat, min_lon, max_lat, max_lon]
    print(f"  BBOX ermittelt: {bbox}")

    return bbox


def extract_roads_from_osm(osm_elements):
    """Extrahiert nur Strassen-Ways aus allen OSM-Daten."""
    roads = [
        element
        for element in osm_elements
        if element.get("type") == "way" 
        and "tags" in element 
        and "highway" in element["tags"]
        and element["tags"].get("area") != "yes"  # Filtere Flächen-Features (area=yes)
    ]
    print(f"  [->] {len(roads)} Strassensegmente aus {len(osm_elements)} OSM-Elementen extrahiert")
    return roads
