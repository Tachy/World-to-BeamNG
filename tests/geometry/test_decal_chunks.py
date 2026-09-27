"""Tests for split_decal_nodes(): splitting long carriageway DecalRoads into pieces.

Background (found in game, 2026-09-24): BeamNG draws only a limited amount of geometry per DecalRoad - the
decal is clipped to the terrain triangles under its area, and road_33264943009 (6.5 m wide, nodes every
0.81 m) broke off after 97 segments (~78 m, ~510 m^2); the rest was missing, the narrow lines on it stayed visible.
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
        assert first[-1] == second[0]  # same joint node (position, height, width) - seamless
    rebuilt = chunks[0] + [n for c in chunks[1:] for n in c[1:]]
    assert rebuilt == nodes


def test_wider_roads_get_shorter_chunks():
    narrow = split_decal_nodes(_road(200.0, width=4.0), max_area=250.0, min_tail_length=5.0)
    wide = split_decal_nodes(_road(200.0, width=13.0), max_area=250.0, min_tail_length=5.0)

    assert len(wide) > len(narrow)


def test_short_tail_is_merged_into_the_previous_chunk():
    # 250 m^2 / 6.5 m = 38.46 m per piece: 40 m would leave a 1.5 m remainder - it attaches to the first piece
    chunks = split_decal_nodes(_road(40.0), max_area=250.0, min_tail_length=5.0)

    assert len(chunks) == 1


def test_every_chunk_has_at_least_two_nodes_even_with_sparse_nodes():
    nodes = _road(200.0, spacing=50.0)  # a single segment alone is already larger than the budget
    chunks = split_decal_nodes(nodes, max_area=250.0, min_tail_length=5.0)

    assert all(len(c) >= 2 for c in chunks)
    assert chunks[0][0] == nodes[0] and chunks[-1][-1] == nodes[-1]


# ---------------------------------------------------------------- render priority groups

from world_to_beamng.geometry.decal_chunks import assign_render_priorities


def test_one_small_group_keeps_the_first_priority_of_its_level():
    priorities = assign_render_priorities([("asphalt", 12, 5000.0)] * 4, max_group_area=100_000.0, step=6)

    assert priorities == [72] * 4  # level 12 x step 6


def test_a_material_is_spread_so_that_no_group_exceeds_the_budget():
    # 3000 asphalt pieces of 100 m^2 = 300 000 m^2 -> 3 groups (like the in-game test: 3 x 100 000 m^2 all visible)
    entries = [("asphalt", 12, 100.0)] * 3000

    priorities = assign_render_priorities(entries, max_group_area=100_000.0, step=6)

    groups = {p: sum(a for (_, _, a), q in zip(entries, priorities) if q == p) for p in set(priorities)}
    assert sorted(groups) == [72, 73, 74]
    assert max(groups.values()) <= 100_000.0


def test_the_surface_order_is_kept_between_levels():
    entries = [("asphalt", 12, 60_000.0)] * 5 + [("gravel", 16, 60_000.0)] * 5 + [("dirt", 18, 1000.0)]

    priorities = assign_render_priorities(entries, max_group_area=100_000.0, step=6)

    asphalt, gravel, dirt = priorities[:5], priorities[5:10], priorities[10]
    assert max(asphalt) < min(gravel) < dirt  # smaller value = drawn later = on top: asphalt above gravel above dirt
    assert max(priorities) <= 127


def test_more_groups_than_the_step_allows_are_capped_and_reported(caplog):
    entries = [("dirt", 18, 100_000.0)] * 8  # needs 8 groups, the level has room for 6

    priorities = assign_render_priorities(entries, max_group_area=100_000.0, step=6)

    assert set(priorities) == set(range(108, 114))
    assert "dirt" in caplog.text
