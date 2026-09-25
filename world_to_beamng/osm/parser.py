"""
OSM data parser and data extraction.
"""


from ..geometry.coordinates import transformer_to_wgs84
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def calculate_bbox_from_height_data(points, margin=0.0):
    """Computes the BBOX (WGS84) from UTM elevation data.

    Args:
        points: UTM coordinates (N x 2)
        margin: Expansion in meters (in UTM)

    Returns:
        BBox in the format [lat_min, lon_min, lat_max, lon_max]
    """
    # Find min/max in UTM
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)

    # Expand by margin (in UTM, i.e. in meters)
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin

    # Convert to WGS84
    min_lon, min_lat = transformer_to_wgs84.transform(min_x, min_y)
    max_lon, max_lat = transformer_to_wgs84.transform(max_x, max_y)

    bbox = [min_lat, min_lon, max_lat, max_lon]
    logger.info(f"  BBOX determined: {bbox}")

    return bbox


# Lifecycle values of highway=*: roads that do not (yet/anymore) exist as drivable - e.g. the 2nd Gotthard tube
# under construction (highway=construction + tunnel=yes), which would otherwise be built as a finished tunnel.
NON_EXISTING_HIGHWAY_VALUES = {"construction", "proposed", "planned", "abandoned", "disused", "razed", "demolished"}


def extract_roads_from_osm(osm_elements):
    """Extracts only road ways from all OSM data."""
    roads = [
        element
        for element in osm_elements
        if element.get("type") == "way"
        and "tags" in element
        and "highway" in element["tags"]
        and element["tags"]["highway"] not in NON_EXISTING_HIGHWAY_VALUES
        and element["tags"].get("area") != "yes"  # Filter out area features (area=yes)
    ]
    logger.info(f"  [->] {len(roads)} road segments extracted from {len(osm_elements)} OSM elements")
    return roads
