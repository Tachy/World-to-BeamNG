"""
Klassifizierung von Straßen-Ways als Brücke, Tunnel, Galerie oder normale Fahrbahn anhand ihrer OSM-Tags
(siehe Design-Spec docs/superpowers/specs/2026-09-22-bridges-tunnels-design.md Abschnitt 1).
"""

from typing import Dict, List, Tuple

import numpy as np


def classify_structure(osm_tags: Dict) -> str:
    """
    "bridge" | "tunnel" | "gallery" | "surface", anhand von `bridge`/`tunnel`-Tags.

    Reihenfolge: bridge=* (außer "no") -> "bridge"; tunnel=avalanche_protector -> "gallery"; jedes andere
    tunnel=* (außer "no") -> "tunnel"; sonst "surface".
    """
    osm_tags = osm_tags or {}
    bridge = str(osm_tags.get("bridge", "")).strip().lower()
    if bridge and bridge != "no":
        return "bridge"
    tunnel = str(osm_tags.get("tunnel", "")).strip().lower()
    if tunnel == "avalanche_protector":
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


def _extrapolate_point(end_point: np.ndarray, next_point: np.ndarray, extension_m: float) -> np.ndarray:
    """end_point um extension_m (horizontal) über end_point hinaus verschoben, in Richtung von next_point
    nach end_point (Z linear mit extrapoliert - selbe Steigung wie das äußerste Segment)."""
    direction = end_point - next_point
    xy_len = float(np.hypot(direction[0], direction[1]))
    if xy_len < 1e-9:
        return end_point
    return end_point + direction * (extension_m / xy_len)


def extend_gallery_centerline_ends(road_slope_polygons_2d: List[Dict], extension_m: float, osm_mapper=None) -> List[Dict]:
    """
    Verschiebt den Grenzpunkt zum nächsten Streckenabschnitt an BEIDEN Enden jeder Galerie-Centerline um
    extension_m (Meter, horizontal) nach außen entlang der Centerline-Richtung (lineare Extrapolation,
    Höhe folgt derselben Steigung wie das jeweils äußerste Segment).

    Wirkt auf Galerie-Mesh UND Terrain-Einbettung gleichermaßen, da beide dieselbe "trimmed_centerline"
    verwenden (siehe tunnels/gallery_mesh.py, terrain/road_embedding.py) - deshalb hier zentral direkt
    nach split_by_structure_type() angewendet, statt an jeder Verwendungsstelle einzeln. Nur
    "gallery"-Einträge werden verändert, Tunnel/Brücken bleiben unverändert.

    WICHTIG: "road_polygon" (das 2D-Straßenpolygon, das embed_roads_into_heightmap()/
    apply_embankment_blend() für den Punkt-in-Polygon-Test benutzen) wird mit derselben Verlängerung
    neu gebuffert, sonst bleibt der Einbettungsbereich an den alten, unverlängerten Enden stehen -
    natürliches Gelände (Peaks, sichtbar bergseits durch die Wand oder vor den Stützen) bliebe dort
    unbehandelt, obwohl das Galerie-Mesh (aus derselben verlängerten Centerline gebaut) bereits darüber
    sitzt. Braucht dafür osm_mapper (für dieselbe Breite wie beim ursprünglichen road_polygon-Aufbau,
    siehe workflow/terrain_workflow.py::process_tile()) - ohne osm_mapper bleibt "road_polygon"
    unverändert (nur für Tests/Aufrufer, die dieses Feld nicht brauchen).

    Args:
        road_slope_polygons_2d: Liste von Straßen-Dicts (wie split_by_structure_type())
        extension_m: Verlängerung je Ende, in Metern (0 = keine Änderung)
        osm_mapper: OSMMapper-Instanz (für get_road_properties()["width"]) - siehe oben

    Returns:
        Neue Liste (Eingabe-Dicts bleiben unverändert, nur Galerie-Einträge werden per Kopie ersetzt)
    """
    if extension_m <= 0:
        return road_slope_polygons_2d

    result = []
    for road in road_slope_polygons_2d:
        centerline = road.get("trimmed_centerline")
        if road.get("structure_type") != "gallery" or centerline is None or len(centerline) < 2:
            result.append(road)
            continue
        coords = np.asarray(centerline, dtype=float).copy()
        coords[0] = _extrapolate_point(coords[0], coords[1], extension_m)
        coords[-1] = _extrapolate_point(coords[-1], coords[-2], extension_m)

        updated = {**road, "trimmed_centerline": coords}
        if osm_mapper is not None:
            from shapely.geometry import LineString

            width = osm_mapper.get_road_properties(road.get("osm_tags", {}))["width"]
            line = LineString(coords[:, :2])
            updated["road_polygon"] = np.array(line.buffer(width / 2.0, cap_style=2).exterior.coords[:-1])
        result.append(updated)
    return result
