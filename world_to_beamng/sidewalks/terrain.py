"""
Road dicts of roads with a sidewalk: the sides and a road outline widened by the sidewalk, so that the terrain embedding
(flat at road height), the embankment start and the vegetation exclusion all reach behind the sidewalk.
"""

from typing import Callable, Dict, FrozenSet, List, Mapping

import numpy as np

from ..geometry.road_width_transitions import variable_width_polygon
from .selection import select_sidewalk_sides


def attach_sidewalks(
    roads: List[Dict],
    road_width: Callable[[Dict], float],
    mapping: Mapping,
    excluded_highways: FrozenSet[str],
    extra: float,
) -> int:
    """
    Sets "sidewalk_sides" ({side: surface type}), "sidewalk_extra" ({side: extra}) and a widened "road_polygon" on every
    surface road with a sidewalk; structures (bridges, tunnels, galleries) are skipped. Returns the number of roads.

    Args:
        road_width: carriageway width of a road dict (used where the road has no blended "width_nodes")
        extra: width added beyond the carriageway edge on a sidewalk side (kerb + sidewalk), in meters
    """
    count = 0
    for poly in roads:
        if poly.get("structure_type", "surface") != "surface":
            continue
        sides = select_sidewalk_sides(poly.get("osm_tags") or {}, excluded_highways, mapping)
        if not sides:
            continue
        nodes = poly.get("width_nodes")
        if nodes is None:
            centerline = np.asarray(poly["trimmed_centerline"], dtype=float)
            nodes = np.column_stack([centerline[:, :3], np.full(len(centerline), road_width(poly))])
        poly["sidewalk_sides"] = sides
        poly["sidewalk_extra"] = {side: extra for side in sides}
        poly["road_polygon"] = variable_width_polygon(
            nodes, extra_left=extra if "left" in sides else 0.0, extra_right=extra if "right" in sides else 0.0
        )
        count += 1
    return count
