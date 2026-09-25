"""
Split long carriageway DecalRoads into chunks.

BeamNG draws only a limited amount of geometry per DecalRoad: the decal is clipped to the terrain triangles under its
area, and whatever exceeds the budget is missing without an error message (found in game 2026-09-24:
road_33264943009, 6.5 m wide, nodes every 0.81 m, stopped after 97 segments / ~510 m^2 - the narrow
marking lines on the same stretch stayed visible). Splitting is therefore done by area (length x width), not
by length or node count.
"""

from typing import List, Sequence

import numpy as np


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
