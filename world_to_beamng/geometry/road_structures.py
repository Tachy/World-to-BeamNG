"""
Classification of road ways as bridge, tunnel, gallery or regular carriageway based on their OSM tags
(see design spec docs/superpowers/specs/2026-09-22-bridges-tunnels-design.md section 1).
"""

from typing import Dict, List, Tuple


def _below_ground(osm_tags: Dict) -> bool:
    """`layer` is a negative integer (unparsable values like "-1;0" do not count)."""
    try:
        return int(str(osm_tags.get("layer", "0")).strip()) < 0
    except ValueError:
        return False


def classify_structure(osm_tags: Dict) -> str:
    """
    "bridge" | "tunnel" | "gallery" | "surface", based on the `bridge`/`tunnel`/`covered`/`layer` tags.

    Order: bridge=* (except "no") -> "bridge"; tunnel=avalanche_protector -> "gallery"; covered=yes with a
    negative layer and without a tunnel tag (or tunnel=no) -> "gallery" (covered road below terrain level, e.g.
    the galleries of the Nuova strada del San Gottardo, see
    docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md - a canopy over a service road
    without a negative layer stays surface); any other tunnel=* (except "no") -> "tunnel"; otherwise "surface".
    """
    osm_tags = osm_tags or {}
    bridge = str(osm_tags.get("bridge", "")).strip().lower()
    if bridge and bridge != "no":
        return "bridge"
    tunnel = str(osm_tags.get("tunnel", "")).strip().lower()
    if tunnel == "avalanche_protector":
        return "gallery"
    covered = str(osm_tags.get("covered", "")).strip().lower()
    if covered == "yes" and tunnel in ("", "no") and _below_ground(osm_tags):
        return "gallery"
    if tunnel and tunnel != "no":
        return "tunnel"
    return "surface"


def split_by_structure_type(road_slope_polygons_2d: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    (surface_roads, structure_roads) - `structure_roads` are bridges/tunnels/galleries
    (road["structure_type"] != "surface"; if the field is missing, the road counts as "surface").
    """
    surface, structures = [], []
    for road in road_slope_polygons_2d:
        target = surface if road.get("structure_type", "surface") == "surface" else structures
        target.append(road)
    return surface, structures
