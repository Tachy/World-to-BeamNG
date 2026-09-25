"""
Edge adjacency of roof polygons and walls of a building.

Shared by the sheet-metal rim of flat roofs and the roof overhang of sloped roofs: both apply only to roof edges
that lie on the outside above a wall (not to ridges, valleys, roof steps or fire walls).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .ring_geometry import open_ring

EDGE_MATCH_TOLERANCE_M = 0.05  # edges of two roof polygons that are this close in plan view are "shared"
WALL_LINE_TOLERANCE_M = 0.10  # roof edge lies "above" a wall if both end points are this close to its plan-view line
WALL_ABOVE_MARGIN_M = 0.3  # wall must extend this far above the roof edge to count as fire wall/parapet


@dataclass(frozen=True)
class WallLine:
    """Plan-view line of a (vertical) wall including the height of its top edge."""

    origin: np.ndarray  # (2,) one of the two end points
    direction: np.ndarray  # (2,) unit vector
    length: float
    z_max: float

    def contains(self, point_xy: np.ndarray, tolerance: float = WALL_LINE_TOLERANCE_M) -> bool:
        offset = point_xy - self.origin
        along = float(offset @ self.direction)
        across = abs(float(offset[0] * self.direction[1] - offset[1] * self.direction[0]))
        return across <= tolerance and -tolerance <= along <= self.length + tolerance


def wall_lines(building: dict) -> List[WallLine]:
    """Plan-view lines of all walls (segment between the two points of the wall that are farthest apart)."""
    lines = []
    for verts, _ in building.get("walls", []):
        ring = open_ring(verts)
        xy = ring[:, :2]
        span = xy[:, None, :] - xy[None, :, :]
        i, j = np.unravel_index(np.argmax(np.einsum("ijk,ijk->ij", span, span)), (len(xy), len(xy)))
        vector = xy[j] - xy[i]
        length = float(np.linalg.norm(vector))
        if length < 1e-6:
            continue
        lines.append(WallLine(xy[i], vector / length, length, float(ring[:, 2].max())))
    return lines


def roof_rings(building: dict) -> List[np.ndarray]:
    """Roof rings without closing point."""
    return [open_ring(verts) for verts, _ in building.get("roofs", [])]


def is_shared(a: np.ndarray, b: np.ndarray, roofs: Sequence[np.ndarray], own_index: int) -> bool:
    """Edge a->b coincides in plan view with an edge of ANOTHER roof polygon (roof step, junction)."""
    tolerance = EDGE_MATCH_TOLERANCE_M
    for index, other in enumerate(roofs):
        if index == own_index:
            continue
        for i in range(len(other)):
            p, q = other[i], other[(i + 1) % len(other)]
            same = np.linalg.norm(a[:2] - p[:2]) <= tolerance and np.linalg.norm(b[:2] - q[:2]) <= tolerance
            swapped = np.linalg.norm(a[:2] - q[:2]) <= tolerance and np.linalg.norm(b[:2] - p[:2]) <= tolerance
            if same or swapped:
                return True
    return False


def walls_under_edge(a: np.ndarray, b: np.ndarray, lines: Sequence[WallLine]) -> List[WallLine]:
    """Walls whose plan-view line carries the roof edge a->b (both end points and the midpoint lie on it)."""
    middle = (a[:2] + b[:2]) / 2
    return [line for line in lines if all(line.contains(point) for point in (a[:2], middle, b[:2]))]


def is_covered_by_higher_wall(a: np.ndarray, b: np.ndarray, lines: Sequence[WallLine]) -> bool:
    """A wall extends well above the roof along this edge (fire wall, parapet)."""
    top = max(float(a[2]), float(b[2])) + WALL_ABOVE_MARGIN_M
    return any(line.z_max > top for line in walls_under_edge(a, b, lines))


def edge_is_exterior_over_wall(
    a: np.ndarray, b: np.ndarray, roofs: Sequence[np.ndarray], own_index: int, lines: Sequence[WallLine]
) -> bool:
    """Roof edge lies on the outside above a wall: no neighboring face, a wall underneath, no higher wall."""
    if is_shared(a, b, roofs, own_index):
        return False
    under = walls_under_edge(a, b, lines)
    if not under:
        return False
    return not is_covered_by_higher_wall(a, b, lines)


def orient_counter_clockwise(ring: np.ndarray) -> Optional[np.ndarray]:
    """Ring counterclockwise seen from above (plan-view area > 0); None for a degenerate ring."""
    xy = ring[:, :2]
    signed_area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    if abs(signed_area) < 1e-9:
        return None
    return ring if signed_area > 0 else ring[::-1]
