"""
Roadblock in front of tunnel entrances whose tunnel extends beyond the map border: the world ends in the middle of
the tube there, so the entrance is blocked - a row of red-and-white plastic barriers (BeamNG default asset
config.ROADBLOCK_SHAPE) across the approach, just before the portal plane. The tube itself then lies flat at
entrance elevation (geometry/polygon.py::apply_structure_elevation_profiles).
"""

import math
from typing import Dict, List, Sequence, Tuple

from ..geometry.polygon import outside_map


def plan_roadblocks(
    plans: Sequence[Dict],
    bounds: Tuple[float, float, float, float],
    edge_margin: float,
    width_margin: float,
    distance: float,
    side_margin: float,
    spacing: float,
    entrances: Sequence[Tuple[float, float]] = (),
    entrance_tol: float = 0.5,
) -> List[Dict]:
    """
    Barrier elements for all tunnel plans (tunnel_portal.plan_tunnels()) that extend beyond the map border with
    exactly one end: in front of the other portal a row across the road surface (road width = tube width -
    width_margin, side_margin more on each side), `distance` before the portal plane, elements spaced `spacing`
    apart. Only at real entrances: the portal is at most entrance_tol from a point in `entrances` (end points of
    surface roads) - a chain end at a branch inside the mountain (e.g. a fortress adit) gets no roadblock.

    Returns:
        [{"name", "xy", "rotation_matrix"}, ...] - the caller sets the height from the terrain. rotation_matrix
        turns the local forward axis (+y) onto the tunnel axis, so the longitudinal axis of the barrier (local +x)
        lies across it.
    """
    blocks = []
    for plan in plans:
        coords = plan["coords"]
        outside = (outside_map(coords[0], bounds, edge_margin), outside_map(coords[-1], bounds, edge_margin))
        if outside[0] == outside[1]:
            continue
        portal = plan["portals"][1] if outside[0] else plan["portals"][0]
        px, py = portal["xy"]
        if not any(math.hypot(ex - px, ey - py) <= entrance_tol for ex, ey in entrances):
            continue
        ux, uy = portal["axis"]  # into the tunnel interior
        across = (uy, -ux)  # right when looking into the tunnel interior
        road_width = plan["tube_width"] - width_margin
        count = max(2, math.ceil((road_width + 2.0 * side_margin) / spacing))
        cx, cy = px - ux * distance, py - uy * distance
        # BeamNG: rows = images of the local axes (see ItemManager._heading_rotation_matrix()) - row 0 = across
        # (longitudinal axis of the barrier), row 1 = tunnel axis, row 2 = up
        rotation = [across[0], across[1], 0.0, ux, uy, 0.0, 0.0, 0.0, 1.0]
        for i in range(count):
            offset = (i - (count - 1) / 2.0) * spacing
            blocks.append(
                {
                    "name": f"roadblock_{plan['id']}_{portal['label']}_{i}",
                    "xy": (cx + across[0] * offset, cy + across[1] * offset),
                    "rotation_matrix": list(rotation),
                }
            )
    return blocks
