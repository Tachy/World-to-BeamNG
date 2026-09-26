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


def _stable_reference(arc: np.ndarray, grade_ok: np.ndarray, start: float, direction: int, max_search: float, stable_length: float):
    """
    Arc position of the reference height on one side of an underpass: from `start` (the deck edge) outward
    (direction -1 = toward smaller arc, +1 = toward larger) the nearest position whose road height has been stable - grade
    below the limit (`grade_ok`) - over `stable_length` meters further out (at least 2 m, less only where the road ends).
    The terrain model raises the road before the deck and lets it fall behind it, so the flanks are skipped. None if
    there is none within `max_search` meters.
    """
    count = len(arc)
    order = range(count - 1, -1, -1) if direction < 0 else range(count)
    for i in order:
        if (direction < 0 and arc[i] > start) or (direction > 0 and arc[i] < start):
            continue
        if abs(arc[i] - start) > max_search:
            return None
        lo, hi = (max(arc[0], arc[i] - stable_length), arc[i]) if direction < 0 else (arc[i], min(arc[-1], arc[i] + stable_length))
        if hi - lo < 2.0:
            continue
        window = (arc >= lo - 1e-9) & (arc <= hi + 1e-9)
        if grade_ok[window].all():
            return float(arc[i])
    return None


def fix_underpass_elevations(
    road_polygons: List[Dict], half_width_of: Callable[[Dict], float], max_search: float, stable_length: float,
    max_grade: float, min_rise: float, min_length: float = 1.0,
) -> int:
    """
    Roads that pass UNDER a bridge: the terrain model does not resolve the underpass and shows the bridge deck there, so a
    road sampled from it climbs to the deck - and already before it and until behind it, on flanks up to several meters
    long. The reference heights are therefore searched outward from both deck edges, up to `max_search` meters, until
    the height is stable (grade below `max_grade` over `stable_length` meters); between them the height is interpolated
    linearly (by arc length), and the normal road embedding then cuts the road into the terrain with its slopes on both
    sides. Only surface roads whose centerline crosses the bridge footprint (`half_width_of(road)` around the bridge
    centerline, flat ends) over at least `min_length` meters, that have a stable reference on both sides and whose sampled
    heights rise by at least `min_rise` above the interpolation - approach roads that merely touch the bridge end, roads
    ending under it and correctly sampled crossings stay untouched. Junction detection splits the road at the
    crossing, so the pieces are chained along straight continuations first. Modifies road["coords"] in place and marks
    the corrected pieces with road["underpass"] = True.

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
        grade_ok = np.abs(np.gradient(z, np.maximum(arc, np.arange(len(arc)) * 1e-9))) < max_grade if len(z) > 2 else np.ones(len(z), bool)
        modified = False
        for footprint in footprints:
            if not line.intersects(footprint):
                continue
            crossing = line.intersection(footprint)
            for piece in getattr(crossing, "geoms", [crossing]):
                if piece.geom_type != "LineString" or piece.length < min_length:
                    continue
                enter, leave = sorted([line.project(Point(piece.coords[0])), line.project(Point(piece.coords[-1]))])
                start = _stable_reference(arc, grade_ok, enter, -1, max_search, stable_length)
                end = _stable_reference(arc, grade_ok, leave, 1, max_search, stable_length)
                if start is None or end is None or end <= start:
                    continue  # no stable height on one side: nothing reliable to interpolate from
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
        road_polygons[surface[k]]["underpass"] = True  # the embankment then cuts 45 degree slopes up to the terrain
    return len(changed)
