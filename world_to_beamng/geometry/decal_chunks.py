"""
Split long carriageway DecalRoads into chunks.

BeamNG draws only a limited amount of geometry per DecalRoad: the decal is clipped to the terrain triangles under its
area, and whatever exceeds the budget is missing without an error message (found in game 2026-09-24:
road_33264943009, 6.5 m wide, nodes every 0.81 m, stopped after 97 segments / ~510 m^2 - the narrow
marking lines on the same stretch stayed visible). Splitting is therefore done by area (length x width), not
by length or node count.

On top of that BeamNG groups DecalRoads by material and render priority and has a budget per group: in game
(2026-09-27, Baden-Wuerttemberg) one asphalt group of 302 000 m^2 was drawn only in patches, the same roads spread over
three render priorities of ~100 000 m^2 each were complete (the Swiss area's 178 000 m^2 in one group were fine too).
assign_render_priorities() therefore spreads every material over several neighbouring priorities.
"""

import logging
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def split_decal_nodes(
    nodes: Sequence[Sequence[float]], max_area: float, min_tail_length: float
) -> List[List[List[float]]]:
    """
    DecalRoad nodes [x, y, z, width] into consecutive chunks with at most `max_area` area. Adjacent chunks
    share their joint node (same position and width, so that they connect seamlessly). A final remainder
    shorter than `min_tail_length` is added to the previous chunk (which may then slightly exceed the budget).
    Each chunk has at least one segment, even if a single segment is already larger than the budget.
    """
    nodes = [list(n) for n in nodes]
    if len(nodes) < 3:
        return [nodes]
    arr = np.asarray(nodes, dtype=float)
    seg_len = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    seg_area = seg_len * (arr[:-1, 3] + arr[1:, 3]) / 2.0

    cuts = [0]  # node indices at which a chunk begins
    area = 0.0
    for seg in range(len(seg_area)):
        if area > 0.0 and area + seg_area[seg] > max_area:
            cuts.append(seg)
            area = 0.0
        area += seg_area[seg]
    if len(cuts) > 1 and seg_len[cuts[-1]:].sum() < min_tail_length:
        cuts.pop()

    bounds = cuts + [len(nodes) - 1]
    return [nodes[start : end + 1] for start, end in zip(bounds, bounds[1:])]


def decal_area(nodes: Sequence[Sequence[float]]) -> float:
    """Area of a DecalRoad [x, y, z, width] in m^2 (segment length x mean width)."""
    arr = np.asarray(nodes, dtype=float)
    if len(arr) < 2:
        return 0.0
    seg_len = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    return float(np.sum(seg_len * (arr[:-1, 3] + arr[1:, 3]) / 2.0))


def assign_render_priorities(entries: Sequence[Tuple[str, int, float]], max_group_area: float, step: int) -> List[int]:
    """
    renderPriority per DecalRoad so that no (material, priority) group exceeds `max_group_area` m^2.

    `entries`: (material, level, area) per DecalRoad in export order; `level` is the old single priority (asphalt 12,
    gravel 16, dirt 18, ...). A level owns the priorities level * step ... level * step + step - 1, so the order between
    surfaces stays (smaller value = drawn later = on top); within a level a material gets as many groups as its area needs
    (at most `step`, more is reported) and every road piece goes to the group that has the least area so far.
    """
    totals: Dict[str, float] = {}
    for material, _, area in entries:
        totals[material] = totals.get(material, 0.0) + area
    group_count = {}
    for material, total in totals.items():
        needed = max(1, math.ceil(total / max_group_area))
        if needed > step:
            logger.warning(
                f"  [!] DecalRoads '{material}': {total / 1000:.0f} k m^2 need {needed} render priority groups, only {step} "
                f"available - some of these roads may not be drawn"
            )
        group_count[material] = min(needed, step)

    group_area: Dict[str, List[float]] = {m: [0.0] * n for m, n in group_count.items()}
    priorities = []
    for material, level, area in entries:
        areas = group_area[material]
        group = int(np.argmin(areas))
        areas[group] += area
        priorities.append(level * step + group)
    return priorities
