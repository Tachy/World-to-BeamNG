"""
Selectable spawn points in front of the entrances of tunnel chains: at every end of a chain of tunnels/galleries (with
at least one named tunnel for cars) to which a normal road connects, a spawn point facing into the tunnel is placed
`distance` meters in front of it on the access road. The name comes from `tunnel:name` of the structure at that end.
"""

from typing import Collection, Dict, List, Sequence

import numpy as np

from ..geometry.polygon import structure_chains
from ..geometry.road_structures import classify_structure

ENDPOINT_TOL = 0.5  # how close the chain end and the road end must be to each other, in meters


def _approach_point(coords: np.ndarray, from_start: bool, distance: float):
    """(point, viewing direction) `distance` meters away from the `from_start` end along the road, looking back toward that end."""
    ordered = coords if from_start else coords[::-1]
    seg = np.diff(ordered[:, :2], axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    target = min(distance, float(cum[-1]))
    i = int(np.clip(np.searchsorted(cum, target) - 1, 0, len(seg) - 1))
    t = (target - cum[i]) / seg_len[i] if seg_len[i] > 1e-9 else 0.0
    point = ordered[i] + t * (ordered[i + 1] - ordered[i])
    heading = -seg[i] / seg_len[i]
    return point, (float(heading[0]), float(heading[1]))


def plan_entrance_spawns(roads: Sequence[Dict], distance: float, excluded_highways: Collection[str]) -> List[Dict]:
    """
    Spawn points in front of tunnel entrances.

    Args:
        roads: road_slope_polygons_2d dicts ("road_id", "trimmed_centerline" (N, 3), "osm_tags", "structure_type")
        excluded_highways: Road types without car traffic (adits/access roads of these types do not count)

    Returns:
        [{"name", "position" (x, y, z) on the carriageway, "heading" (dx, dy) toward the tunnel}, ...]
    """
    light = [
        {"id": r["road_id"], "coords": [tuple(p) for p in np.asarray(r["trimmed_centerline"], dtype=float)], "osm_tags": r.get("osm_tags", {})}
        for r in roads
        if r.get("trimmed_centerline") is not None and len(r["trimmed_centerline"]) >= 2
    ]
    approaches = [
        r for r in light
        if classify_structure(r["osm_tags"]) == "surface" and r["osm_tags"].get("highway") not in excluded_highways
    ]

    spawns = []
    for chain in structure_chains(light):
        tags = [road["osm_tags"] for road, _ in chain]
        # Only named tunnels for cars - unnamed field-path underpasses are not a chain one wants to drive to
        if not any(
            classify_structure(t) == "tunnel" and t.get("highway") not in excluded_highways and (t.get("tunnel:name") or t.get("name"))
            for t in tags
        ):
            continue
        first, first_rev = chain[0]
        last, last_rev = chain[-1]
        ends = (
            (first["coords"][-1] if first_rev else first["coords"][0], first["osm_tags"]),
            (last["coords"][0] if last_rev else last["coords"][-1], last["osm_tags"]),
        )
        for end_point, end_tags in ends:
            for road in approaches:
                coords = np.asarray(road["coords"], dtype=float)
                if np.hypot(*(coords[0, :2] - end_point[:2])) <= ENDPOINT_TOL:
                    point, heading = _approach_point(coords, True, distance)
                elif np.hypot(*(coords[-1, :2] - end_point[:2])) <= ENDPOINT_TOL:
                    point, heading = _approach_point(coords, False, distance)
                else:
                    continue
                name = end_tags.get("tunnel:name") or end_tags.get("name") or "Tunnel"
                spawns.append({"name": f"{name} (Einfahrt)", "position": tuple(float(v) for v in point), "heading": heading})
                break
    return spawns
