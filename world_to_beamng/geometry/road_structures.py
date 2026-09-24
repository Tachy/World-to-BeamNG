"""
Klassifizierung von Straßen-Ways als Brücke, Tunnel, Galerie oder normale Fahrbahn anhand ihrer OSM-Tags
(siehe Design-Spec docs/superpowers/specs/2026-09-22-bridges-tunnels-design.md Abschnitt 1).
"""

from typing import Dict, List, Tuple


def _below_ground(osm_tags: Dict) -> bool:
    """`layer` ist eine negative Ganzzahl (unlesbare Werte wie "-1;0" zählen nicht)."""
    try:
        return int(str(osm_tags.get("layer", "0")).strip()) < 0
    except ValueError:
        return False


def classify_structure(osm_tags: Dict) -> str:
    """
    "bridge" | "tunnel" | "gallery" | "surface", anhand von `bridge`/`tunnel`/`covered`/`layer`-Tags.

    Reihenfolge: bridge=* (außer "no") -> "bridge"; tunnel=avalanche_protector -> "gallery"; covered=yes mit
    negativem layer und ohne tunnel-Tag (oder tunnel=no) -> "gallery" (überdachte Straße unter Geländeniveau, z.B.
    die Galerien der Nuova strada del San Gottardo, siehe
    docs/superpowers/specs/2026-09-24-tunnel-gallery-transition-design.md - ein Vordach über einer Service-Straße
    ohne negativen layer bleibt Oberfläche); jedes andere tunnel=* (außer "no") -> "tunnel"; sonst "surface".
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
    (surface_roads, structure_roads) - `structure_roads` sind Brücken/Tunnel/Galerien
    (road["structure_type"] != "surface"; fehlt das Feld, gilt die Straße als "surface").
    """
    surface, structures = [], []
    for road in road_slope_polygons_2d:
        target = surface if road.get("structure_type", "surface") == "surface" else structures
        target.append(road)
    return surface, structures
