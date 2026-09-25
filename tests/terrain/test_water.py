"""Tests for world_to_beamng.terrain.water: streams (River) and ponds/lakes (WaterBlock).

Real water in BeamNG consists of dedicated objects. Streams are built as a `River` spline along the OSM line
with heights from the DGM1, water areas as tiled `WaterBlock` boxes that cover the polygon
including its edge (the surface is the mean of the three lowest edge points). Streams end at the
pond bank. The DGM1 itself stays unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from world_to_beamng.terrain.water import (
    build_pond_blocks,
    build_river_nodes,
    carve_pond_basins,
    clip_line_to_bounds,
    cut_line_by_area,
    is_pond_area,
    pond_level,
    select_pond_areas,
    select_waterways,
    split_nodes,
)

WIDTHS = {"stream": 2.5, "river": 8.0}


def _way(way_id, waterway, points, **tags):
    return {
        "type": "way",
        "id": way_id,
        "tags": {"waterway": waterway, **tags},
        "geometry": [{"lat": y, "lon": x} for x, y in points],
    }


def _to_local(points):
    return [(p["lon"], p["lat"]) for p in points]


def _terrain_with_channel(bottom=100.0, bank=0.5, half_width=1.5, slope=0.0):
    """Terrain: channel along the x axis (y=0), banks `bank` m higher, optional slope in +x."""

    def height_at(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        channel = np.where(np.abs(y) <= half_width, 0.0, bank)
        return bottom + slope * x + channel

    return height_at


# --- Selection of streams ---------------------------------------------------------------------


def test_streams_are_selected_with_default_width_and_local_coordinates():
    ways = [_way(1, "stream", [(0, 0), (10, 0), (20, 5)])]

    result = select_waterways(ways, _to_local, WIDTHS)

    assert len(result) == 1
    assert result[0]["waterway"] == "stream"
    assert result[0]["width"] == 2.5
    assert result[0]["coords"] == [(0, 0), (10, 0), (20, 5)]


def test_width_tag_overrides_the_default_and_parses_units():
    ways = [_way(1, "stream", [(0, 0), (10, 0)], width="4"), _way(2, "stream", [(0, 0), (10, 0)], width="3.5 m")]

    result = select_waterways(ways, _to_local, WIDTHS)

    assert [w["width"] for w in result] == [4.0, 3.5]


def test_tunnels_culverts_and_unlisted_waterways_are_skipped():
    ways = [
        _way(1, "stream", [(0, 0), (10, 0)], tunnel="culvert"),
        _way(2, "stream", [(0, 0), (10, 0)], tunnel="yes"),
        _way(3, "stream", [(0, 0), (10, 0)], culvert="yes"),
        _way(4, "ditch", [(0, 0), (10, 0)]),  # not in WIDTHS
        _way(5, "stream", [(0, 0)]),  # too short
        {"type": "node", "id": 6, "tags": {"waterway": "stream"}},
        _way(7, "river", [(0, 0), (10, 0)]),
    ]

    result = select_waterways(ways, _to_local, WIDTHS)

    assert [w["waterway"] for w in result] == ["river"]


def test_clip_line_splits_where_the_stream_leaves_and_reenters_the_terrain():
    line = [(-50.0, 0.0), (50.0, 0.0), (50.0, 100.0), (-50.0, 100.0)]

    parts = clip_line_to_bounds(line, (-10.0, -10.0, 10.0, 110.0))

    assert len(parts) == 2
    assert all(-10.0 <= x <= 10.0 for part in parts for x, _ in part)
    assert parts[0][0] == pytest.approx((-10.0, 0.0)) and parts[0][-1] == pytest.approx((10.0, 0.0))


def test_clip_line_fully_outside_gives_nothing():
    assert clip_line_to_bounds([(100.0, 0.0), (200.0, 0.0)], (-10.0, -10.0, 10.0, 10.0)) == []


# --- River nodes ------------------------------------------------------------------------------


def _line(x0, x1, step=4.0):
    return [(x, 0.0) for x in np.arange(x0, x1 + 1e-9, step)]


def test_river_nodes_have_the_river_format_and_the_requested_spacing():
    nodes = build_river_nodes(_line(0, 100), _terrain_with_channel(slope=-0.02), width=2.5, depth=1.0, spacing=10.0)

    assert all(len(n) == 8 for n in nodes)
    assert all(n[3] == 2.5 and n[4] == 1.0 and n[5:] == [0.0, 0.0, 1.0] for n in nodes)
    xs = [n[0] for n in nodes]
    assert xs[0] == pytest.approx(0.0) and xs[-1] == pytest.approx(100.0)
    assert np.diff(xs) == pytest.approx(np.full(len(xs) - 1, 10.0), abs=0.01)


def test_water_sits_in_the_channel_just_above_the_bed_and_never_floats_above_the_banks():
    lift = 0.2
    nodes = build_river_nodes(_line(0, 100), _terrain_with_channel(bottom=100.0, bank=0.5), width=2.5, depth=1.0, spacing=10.0, lift=lift)

    for n in nodes:
        assert n[2] == pytest.approx(100.0 + lift, abs=1e-6)  # channel bed + water level
        assert n[2] < 100.0 + 0.5  # below the bank edge: the water lies in the channel


def test_water_level_only_falls_downstream_and_hides_under_a_rise_like_a_culvert():
    # Slope in +x, with a fill in between (road/culvert), behind which the terrain drops again
    base = _terrain_with_channel(slope=-0.02)

    def with_dam(x, y):
        x = np.asarray(x, float)
        return base(x, y) + np.where((x > 40) & (x < 60), 3.0, 0.0)

    nodes = build_river_nodes(_line(0, 100), with_dam, width=2.5, depth=1.0, spacing=5.0)

    z = np.array([n[2] for n in nodes])
    assert (np.diff(z) <= 1e-9).all()  # never uphill
    dam = [n for n in nodes if 42 < n[0] < 58]
    assert all(n[2] < with_dam(n[0], 0.0) for n in dam)  # below the fill, invisible there


def test_line_drawn_uphill_is_reversed_so_the_water_runs_downhill():
    nodes = build_river_nodes(_line(0, 100), _terrain_with_channel(slope=+0.03), width=2.5, depth=1.0, spacing=10.0)

    assert nodes[0][0] > nodes[-1][0]  # starts at the top (large x), ends at the bottom
    assert nodes[0][2] >= nodes[-1][2]


def test_short_line_still_yields_two_nodes():
    nodes = build_river_nodes([(0.0, 0.0), (4.0, 0.0)], _terrain_with_channel(), width=2.5, depth=1.0, spacing=10.0)

    assert len(nodes) == 2


def test_split_nodes_chunks_overlap_by_one_node_and_respect_the_limit():
    nodes = [[float(i), 0.0, 0.0, 2.5, 1.0, 0.0, 0.0, 1.0] for i in range(25)]

    chunks = split_nodes(nodes, max_nodes=10)

    assert all(len(c) <= 10 for c in chunks)
    assert chunks[0][-1] == chunks[1][0]  # gapless: the last node pair is shared
    assert [n for c in chunks for n in c[(0 if c is chunks[0] else 1):]] == nodes


# --- Ponds / lakes ----------------------------------------------------------------------------


def _flat(z=100.0):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _rect(block):
    (cx, cy, _), (w, h, _) = block["position"], block["scale"]
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _union(blocks):
    from shapely.ops import unary_union

    return unary_union([_rect(b).buffer(1e-6) for b in blocks])  # tolerance against rounding gaps when reconstructing


def test_only_water_and_wet_basins_are_ponds_dry_detention_basins_are_not():
    assert is_pond_area({"natural": "water"})
    assert is_pond_area({"natural": "water", "water": "reservoir"})
    assert is_pond_area({"landuse": "basin", "name": "Rückhaltebecken Laufen"})
    assert is_pond_area({"landuse": "reservoir"})
    assert not is_pond_area({"landuse": "basin", "basin": "detention"})  # dry flood retention basin
    assert not is_pond_area({"landuse": "forest"})
    assert not is_pond_area({"natural": "water", "basin": "detention"})  # detention is always dry (meadow)


def test_pond_level_is_the_mean_of_the_three_lowest_rim_points():
    # Height = 100 + |y - 5|: on the edge, (0,5) and (10,5) are at 100, four points at y=4/6 at 101
    height = lambda x, y: 100.0 + np.abs(np.asarray(y, float) - 5.0)

    level = pond_level(box(0, 0, 10, 10), height, rim_step=1.0)

    assert level == pytest.approx((100.0 + 100.0 + 101.0) / 3.0)


def test_pond_level_ignores_the_terrain_inside_the_polygon():
    # deep pit in the middle of the polygon: only the edge counts
    pit = lambda x, y: np.where((np.asarray(x) > 3) & (np.asarray(x) < 7) & (np.asarray(y) > 3) & (np.asarray(y) < 7), 90.0, 100.0)

    assert pond_level(box(0, 0, 10, 10), pit, rim_step=1.0) == pytest.approx(100.0)


def test_pond_blocks_cover_the_whole_polygon_and_reach_a_little_beyond_it():
    pond = Polygon([(0, 0), (30, 4), (26, 28), (5, 20)])

    blocks = build_pond_blocks(pond, _flat(), depth=3.0, cell=6.0, margin=2.0)

    assert _union(blocks).contains(pond)
    assert all(_rect(b).intersects(pond.buffer(2.0)) for b in blocks)
    assert not _union(blocks).contains(pond.buffer(8.0))


def test_pond_blocks_do_not_overlap_each_other():
    blocks = build_pond_blocks(box(0, 0, 40, 30), _flat(), depth=3.0, cell=6.0, margin=2.0)

    total = sum(_rect(b).area for b in blocks)
    from shapely.ops import unary_union

    assert unary_union([_rect(b) for b in blocks]).area == pytest.approx(total)


def test_pond_uses_one_flat_level_with_the_configured_depth():
    blocks = build_pond_blocks(box(0, 0, 20, 20), _flat(281.0), depth=3.0, cell=5.0, margin=2.0)

    assert all(b["position"][2] == pytest.approx(281.0) for b in blocks)  # surface = position.z = edge height
    assert all(b["scale"][2] == 3.0 for b in blocks)
    assert all(b["rotationMatrix"] == [1, 0, 0, 0, 1, 0, 0, 0, 1] for b in blocks)


def test_pond_on_a_slope_takes_the_low_rim_so_it_does_not_flood_the_downhill_side():
    sloped = lambda x, y: 100.0 - 0.1 * np.asarray(x, float)  # 96 (x=40) .. 100 (x=0)

    blocks = build_pond_blocks(box(0, 0, 40, 20), sloped, depth=3.0, cell=5.0, margin=2.0)

    assert blocks[0]["position"][2] == pytest.approx(96.0, abs=0.2)


def test_tiny_pond_smaller_than_a_cell_is_still_covered():
    pond = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])

    blocks = build_pond_blocks(pond, _flat(), depth=3.0, cell=6.0, margin=2.0)

    assert blocks
    assert _union(blocks).contains(pond)


def test_l_shaped_pond_leaves_the_far_notch_empty():
    pond = Polygon([(0, 0), (40, 0), (40, 10), (10, 10), (10, 40), (0, 40)])

    blocks = build_pond_blocks(pond, _flat(), depth=3.0, cell=5.0, margin=2.0)

    assert _union(blocks).contains(pond)
    assert not any(_rect(b).intersects(box(25, 25, 40, 40)) for b in blocks)


# --- Streams end at the pond bank -------------------------------------------------------------


def test_stream_running_through_a_pond_is_cut_at_both_shores():
    pond = box(0, 0, 40, 10)

    parts = cut_line_by_area([(-20.0, 5.0), (60.0, 5.0)], pond)

    assert len(parts) == 2
    assert parts[0][0] == pytest.approx((-20.0, 5.0)) and parts[0][-1] == pytest.approx((0.0, 5.0))
    assert parts[1][0] == pytest.approx((40.0, 5.0)) and parts[1][-1] == pytest.approx((60.0, 5.0))


def test_stream_starting_in_a_pond_begins_at_the_shore():
    parts = cut_line_by_area([(20.0, 5.0), (60.0, 5.0)], box(0, 0, 40, 10))

    assert len(parts) == 1
    assert parts[0][0] == pytest.approx((40.0, 5.0)) and parts[0][-1] == pytest.approx((60.0, 5.0))


def test_stream_entirely_inside_a_pond_and_tiny_stubs_disappear():
    pond = box(0, 0, 40, 10)

    assert cut_line_by_area([(5.0, 5.0), (30.0, 5.0)], pond) == []
    assert cut_line_by_area([(-0.5, 5.0), (30.0, 5.0)], pond, min_length=1.0) == []  # only 0.5 m outside


def test_stream_without_a_pond_is_unchanged():
    line = [(0.0, 0.0), (10.0, 0.0), (20.0, 5.0)]

    assert cut_line_by_area(line, None) == [line]
    assert cut_line_by_area(line, box(100, 100, 110, 110)) == [line]


def test_nodes_snap_sideways_onto_the_real_channel_when_the_osm_line_is_a_bit_off():
    # Channel in the DGM1 lies at y=+1.2, the OSM line at y=0 (typical offset of 1-2 m)
    def offset_channel(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return 100.0 - 0.02 * x + np.where(np.abs(y - 1.2) <= 0.6, 0.0, 0.5)

    nodes = build_river_nodes(_line(0, 100), offset_channel, width=2.5, depth=1.0, spacing=5.0, lift=0.2, search=2.0)

    ys = np.array([n[1] for n in nodes])
    assert np.abs(ys - 1.2).max() < 0.4  # on the channel, no longer on the OSM line
    for n in nodes:  # the water thus visibly lies in the channel (above the bed, below the bank)
        bed = offset_channel(n[0], n[1])
        assert bed <= n[2] <= bed + 0.3


def test_snapping_does_not_make_the_stream_zigzag():
    rng = np.random.RandomState(0)
    noise = rng.rand(200) * 0.05

    def noisy_flat(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return 100.0 + noise[np.clip((np.abs(y) * 20 + x).astype(int) % 200, 0, 199)]

    nodes = build_river_nodes(_line(0, 100), noisy_flat, width=2.5, depth=1.0, spacing=5.0, search=2.0)

    ys = np.array([n[1] for n in nodes])
    assert np.abs(np.diff(ys)).max() < 1.5  # smooth, no jumps from edge to edge


# --- Lowering the terrain under the ponds ------------------------------------------------------


def _carve(heights, polygons, square_size=0.25, **kwargs):
    return carve_pond_basins(heights, 0.0, 0.0, square_size, polygons, **kwargs)


def test_carving_lowers_the_inside_by_the_depth_and_leaves_the_outside_untouched():
    heights = np.full((200, 200), 100.0)  # 0.25 m grid, 50 x 50 m

    carved = _carve(heights, [box(10, 10, 30, 30)], depth=0.5, slope_deg=45.0)

    assert carved[80, 80] == pytest.approx(99.5)  # (20, 20) in the middle of the pond
    assert carved[40, 40] == pytest.approx(100.0)  # (10, 10) on the edge
    assert carved[10, 10] == 100.0 and carved[190, 190] == 100.0  # outside
    assert carved.min() == pytest.approx(99.5)


def test_bank_slopes_at_45_degrees_inward():
    heights = np.full((200, 200), 100.0)

    carved = _carve(heights, [box(10, 10, 30, 30)], depth=0.5, slope_deg=45.0)

    # Distance to the edge d (x = 10 + d, y in the middle): lowering = d up to the full depth at 0.5 m
    assert 100.0 - carved[80, 41] == pytest.approx(0.25)  # x = 10.25
    assert 100.0 - carved[80, 42] == pytest.approx(0.5)  # x = 10.5
    assert 100.0 - carved[80, 44] == pytest.approx(0.5)  # x = 11 (bottom of the hollow)


def test_flatter_slope_makes_the_bank_wider():
    heights = np.full((200, 200), 100.0)

    carved = _carve(heights, [box(10, 10, 30, 30)], depth=0.5, slope_deg=30.0)

    assert 100.0 - carved[80, 42] == pytest.approx(0.5 * np.tan(np.radians(30.0)))  # d = 0.5 m -> 0.29 m deep
    assert 100.0 - carved[80, 48] == pytest.approx(0.5)  # full depth only from d = 0.87 m


def test_carving_is_relative_to_the_terrain_and_does_not_touch_the_input():
    heights = 100.0 + 0.1 * np.arange(200)[None, :] * np.ones((200, 1))  # slope in x
    before = heights.copy()

    carved = _carve(heights, [box(10, 10, 30, 30)])

    assert (heights == before).all()
    assert before[80, 80] - carved[80, 80] == pytest.approx(0.5)


def test_overlapping_ponds_are_lowered_once_and_islands_are_not_lowered():
    heights = np.full((200, 200), 100.0)
    island_pond = box(10, 10, 30, 30).difference(box(18, 18, 22, 22))

    both = _carve(heights, [island_pond, box(15, 15, 35, 35)], depth=0.5)
    only_island = _carve(heights, [island_pond], depth=0.5)

    assert both.min() == pytest.approx(99.5)  # overlapping ponds: not 99.0
    assert only_island[80, 80] == 100.0  # (20, 20) lies in the hole (island): untouched
    assert both[80, 80] == pytest.approx(99.5)  # ... unless a second pond covers the spot


def test_select_pond_areas_keeps_only_water_clipped_to_the_terrain():
    polygons = [
        {"osm_tags": {"natural": "water"}, "geometry": box(-20, 0, 20, 10)},
        {"osm_tags": {"landuse": "basin", "basin": "detention"}, "geometry": box(0, 0, 10, 10)},
        {"osm_tags": {"landuse": "meadow"}, "geometry": box(0, 0, 10, 10)},
        {"osm_tags": {"natural": "water"}, "geometry": box(100, 100, 110, 110)},  # outside the terrain
    ]

    areas = select_pond_areas(polygons, (-10.0, -10.0, 10.0, 20.0))

    assert len(areas) == 1
    assert areas[0].bounds == pytest.approx((-10.0, 0.0, 10.0, 10.0))
