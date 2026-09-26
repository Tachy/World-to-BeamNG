"""
Classification of road ways as bridge, tunnel, gallery or regular carriageway based on their OSM tags, and the height
correction of roads that pass under a bridge.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np


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
    the galleries of the Nuova strada del San Gottardo - a canopy over a service road
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


def _surface_chains(roads: List[np.ndarray], endpoint_tol: float = 0.5, max_angle_deg: float = 60.0) -> List[List[Tuple[int, bool]]]:
    """Surface road pieces joined along straight continuations (junction detection splits a road wherever another one
    crosses it): [[(piece index, reversed), ...], ...] in driving order."""
    from .road_width_transitions import find_continuations

    partner = {}
    for a, b in find_continuations([r.tolist() for r in roads], endpoint_tol, max_angle_deg):
        partner[a], partner[b] = b, a
    other = {"start": "end", "end": "start"}
    seen, chains = set(), []
    for first in range(len(roads)):
        if first in seen:
            continue
        head, entry, guard = first, "start", {first}
        while True:  # walk back to the start of the chain
            nxt = partner.get((head, entry))
            if nxt is None or nxt[0] in guard:
                break
            guard.add(nxt[0])
            head, entry = nxt[0], other[nxt[1]]
        chain, current = [], (head, entry)
        while current is not None and current[0] not in seen:
            index, entry = current
            seen.add(index)
            chain.append((index, entry == "end"))
            nxt = partner.get((index, other[entry]))
            current = None if nxt is None else (nxt[0], nxt[1])
        chains.append(chain)
    return chains


def fix_underpass_elevations(
    road_polygons: List[Dict], half_width_of: Callable[[Dict], float], margin: float, min_rise: float, min_length: float = 1.0
) -> int:
    """
    Roads that pass UNDER a bridge: the terrain model does not resolve the underpass and shows the bridge deck there, so a
    road sampled from it climbs to the deck. Its height is interpolated linearly (by arc length) from `margin` meters
    before the deck to `margin` meters behind it; the normal road embedding then cuts it into the terrain with its slopes
    on both sides. Only surface roads whose centerline crosses the bridge footprint (`half_width_of(road)` around the
    bridge centerline, flat ends) over at least `min_length` meters, that continue `margin` meters on both sides and whose
    sampled heights rise by at least `min_rise` above the interpolation - approach roads that merely touch the bridge end,
    roads ending under it and correctly sampled crossings stay untouched. Junction detection splits the road at the
    crossing, so the pieces are chained along straight continuations first. Modifies road["coords"] in place.

    Returns:
        Number of corrected road pieces
    """
    from shapely.geometry import LineString, Point

    footprints = []
    for road in road_polygons:
        coords = np.asarray(road.get("coords", []), dtype=float)
        if len(coords) >= 2 and classify_structure(road.get("osm_tags", {})) == "bridge":
            footprints.append(LineString(coords[:, :2]).buffer(half_width_of(road), cap_style=2))
    if not footprints:
        return 0

    surface = [
        i for i, road in enumerate(road_polygons)
        if len(road.get("coords", [])) >= 2 and classify_structure(road.get("osm_tags", {})) == "surface"
    ]
    arrays = [np.array(road_polygons[i]["coords"], dtype=float) for i in surface]
    changed = set()
    for chain in _surface_chains(arrays):
        pieces = [arrays[k][::-1] if reverse else arrays[k] for k, reverse in chain]
        owners = [(k, reverse) for k, reverse in chain]
        xyz = np.vstack([pieces[0]] + [p[1:] for p in pieces[1:]])  # the shared joint node once
        line = LineString(xyz[:, :2])
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))])
        z = xyz[:, 2].copy()
        modified = False
        for footprint in footprints:
            if not line.intersects(footprint):
                continue
            crossing = line.intersection(footprint)
            for piece in getattr(crossing, "geoms", [crossing]):
                if piece.geom_type != "LineString" or piece.length < min_length:
                    continue
                enter, leave = sorted([line.project(Point(piece.coords[0])), line.project(Point(piece.coords[-1]))])
                start, end = enter - margin, leave + margin
                if start <= 0.0 or end >= arc[-1]:
                    continue  # the road ends within the margin: nothing to interpolate from
                z_start, z_end = np.interp([start, end], arc, z)
                inside = (arc > start) & (arc < end)
                target = z_start + (z_end - z_start) * (arc[inside] - start) / (end - start)
                if float(np.max(z[inside] - target)) < min_rise:
                    continue
                z[inside] = target
                modified = True
        if not modified:
            continue
        # write the heights back to the pieces (in their own digitization direction)
        position = 0
        for n, ((k, reverse), piece) in enumerate(zip(owners, pieces)):
            count = len(piece)
            values = z[position : position + count]
            arrays[k][:, 2] = values[::-1] if reverse else values
            changed.add(k)
            position += count - 1
    for k in changed:
        road_polygons[surface[k]]["coords"] = arrays[k]
    return len(changed)
