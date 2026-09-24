"""
Lange Fahrbahn-DecalRoads in Stücke teilen.

BeamNG zeichnet pro DecalRoad nur eine begrenzte Menge Geometrie: das Decal wird auf die Terrain-Dreiecke unter seiner
Fläche zugeschnitten, und was über das Budget hinausgeht, fehlt ohne Fehlermeldung (im Spiel gefunden 2026-09-24:
road_33264943009, 6,5 m breit, Knoten alle 0,81 m, brach nach 97 Segmenten / ~510 m^2 ab - die schmalen
Markierungslinien auf derselben Strecke blieben sichtbar). Geteilt wird deshalb nach Fläche (Länge x Breite), nicht
nach Länge oder Knotenzahl.
"""

from typing import List, Sequence

import numpy as np


def split_decal_nodes(
    nodes: Sequence[Sequence[float]], max_area: float, min_tail_length: float
) -> List[List[List[float]]]:
    """
    DecalRoad-Knoten [x, y, z, width] in aufeinanderfolgende Stücke mit höchstens `max_area` Fläche. Benachbarte Stücke
    teilen sich ihren Stoßknoten (gleiche Position und Breite, damit sie nahtlos aneinanderschließen). Ein letzter Rest
    kürzer als `min_tail_length` wird dem vorherigen Stück zugeschlagen (das dann etwas über dem Budget liegen darf).
    Jedes Stück hat mindestens ein Segment, auch wenn ein einzelnes Segment schon größer als das Budget ist.
    """
    nodes = [list(n) for n in nodes]
    if len(nodes) < 3:
        return [nodes]
    arr = np.asarray(nodes, dtype=float)
    seg_len = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    seg_area = seg_len * (arr[:-1, 3] + arr[1:, 3]) / 2.0

    cuts = [0]  # Knotenindizes, an denen ein Stück beginnt
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
