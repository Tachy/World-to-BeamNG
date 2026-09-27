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
    road_props: Callable[[Dict], Dict],
    mapping: Mapping,
    excluded_highways: FrozenSet[str],
    excluded_surfaces: FrozenSet[str],
    extra: float,
) -> int:
    """
    Sets "sidewalk_sides" ({side: surface type}), "sidewalk_extra" ({side: extra}) and a widened "road_polygon" on every
    surface road with a sidewalk; structures (bridges, tunnels, galleries) and roads that get no DecalRoad (only the aerial
    photo shows them, so a kerb could not follow their edge) are skipped. Returns the number of roads.

    Args:
        road_props: OSM mapper properties of a road dict - "internal_name" (surface type) and "width" (carriageway width,
            used where the road has no blended "width_nodes")
        excluded_surfaces: surface types without a DecalRoad
        extra: width added beyond the carriageway edge on a sidewalk side (kerb + sidewalk), in meters
    """
    count = 0
    for poly in roads:
        if poly.get("structure_type", "surface") != "surface":
            continue
        sides = select_sidewalk_sides(poly.get("osm_tags") or {}, excluded_highways, mapping)
        props = road_props(poly) if sides else {}
        if not sides or props.get("internal_name") in excluded_surfaces:
            continue
        nodes = poly.get("width_nodes")
        if nodes is None:
            centerline = np.asarray(poly["trimmed_centerline"], dtype=float)
            nodes = np.column_stack([centerline[:, :3], np.full(len(centerline), props["width"])])
        poly["sidewalk_sides"] = sides
        poly["sidewalk_extra"] = {side: extra for side in sides}
        poly["road_polygon"] = variable_width_polygon(
            nodes, extra_left=extra if "left" in sides else 0.0, extra_right=extra if "right" in sides else 0.0
        )
        count += 1
    return count
