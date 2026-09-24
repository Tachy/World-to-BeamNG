"""Tests für split_decal_nodes(): lange Fahrbahn-DecalRoads in Stücke teilen.

Hintergrund (im Spiel gefunden, 2026-09-24): BeamNG zeichnet pro DecalRoad nur eine begrenzte Menge Geometrie - das
Decal wird auf die Terrain-Dreiecke unter seiner Fläche zugeschnitten, und road_33264943009 (6,5 m breit, Knoten alle
0,81 m) brach nach 97 Segmenten (~78 m, ~510 m^2) ab; der Rest fehlte, die schmalen Linien darauf blieben sichtbar.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.decal_chunks import split_decal_nodes


def _road(length, width=6.5, spacing=1.0):
    xs = np.arange(0.0, length + 1e-9, spacing)
    return [[float(x), 0.0, 100.0 + 0.01 * x, width] for x in xs]


def _area(nodes):
    a = np.asarray(nodes, dtype=float)
    seg = np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1)
    return float((seg * (a[:-1, 3] + a[1:, 3]) / 2.0).sum())


def test_short_road_stays_one_decal():
    nodes = _road(30.0)  # 195 m^2
    assert split_decal_nodes(nodes, max_area=250.0, min_tail_length=5.0) == [nodes]


def test_long_road_is_split_into_chunks_within_the_budget():
    nodes = _road(200.0)  # 1300 m^2
    chunks = split_decal_nodes(nodes, max_area=250.0, min_tail_length=5.0)

    assert len(chunks) >= 6
    assert all(_area(c) <= 250.0 + 1e-6 for c in chunks)


def test_chunks_share_their_boundary_node_and_cover_all_nodes_in_order():
    nodes = _road(200.0)
    chunks = split_decal_nodes(nodes, max_area=250.0, min_tail_length=5.0)

    for first, second in zip(chunks, chunks[1:]):
        assert first[-1] == second[0]  # gleicher Stoßknoten (Position, Höhe, Breite) - nahtlos
    rebuilt = chunks[0] + [n for c in chunks[1:] for n in c[1:]]
    assert rebuilt == nodes


def test_wider_roads_get_shorter_chunks():
    narrow = split_decal_nodes(_road(200.0, width=4.0), max_area=250.0, min_tail_length=5.0)
    wide = split_decal_nodes(_road(200.0, width=13.0), max_area=250.0, min_tail_length=5.0)

    assert len(wide) > len(narrow)


def test_short_tail_is_merged_into_the_previous_chunk():
    # 250 m^2 / 6,5 m = 38,46 m je Stück: 40 m ergäben einen 1,5-m-Rest - der hängt am ersten Stück
    chunks = split_decal_nodes(_road(40.0), max_area=250.0, min_tail_length=5.0)

    assert len(chunks) == 1


def test_every_chunk_has_at_least_two_nodes_even_with_sparse_nodes():
    nodes = _road(200.0, spacing=50.0)  # ein Segment allein ist schon größer als das Budget
    chunks = split_decal_nodes(nodes, max_area=250.0, min_tail_length=5.0)

    assert all(len(c) >= 2 for c in chunks)
    assert chunks[0][0] == nodes[0] and chunks[-1][-1] == nodes[-1]
