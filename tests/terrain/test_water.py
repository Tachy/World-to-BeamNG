"""Tests für world_to_beamng.terrain.water: Bäche (River) und Teiche/Seen (WaterBlock).

Echtes Wasser sind in BeamNG eigene Objekte. Bäche werden als `River`-Spline entlang der OSM-Linie
mit Höhen aus dem DGM1 gebaut, Wasserflächen als gekachelte `WaterBlock`-Quader innerhalb des
Polygons. Das DGM1 selbst bleibt unverändert.
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
    clip_line_to_bounds,
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
    """Terrain: Rinne entlang der x-Achse (y=0), Ufer `bank` m höher, optional Gefälle in +x."""

    def height_at(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        channel = np.where(np.abs(y) <= half_width, 0.0, bank)
        return bottom + slope * x + channel

    return height_at


# --- Auswahl der Bäche ------------------------------------------------------------------------


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
        _way(4, "ditch", [(0, 0), (10, 0)]),  # nicht in WIDTHS
        _way(5, "stream", [(0, 0)]),  # zu kurz
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


# --- River-Knoten -----------------------------------------------------------------------------


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
        assert n[2] == pytest.approx(100.0 + lift, abs=1e-6)  # Rinnenboden + Wasserstand
        assert n[2] < 100.0 + 0.5  # unter der Uferkante: das Wasser liegt in der Rinne


def test_water_level_only_falls_downstream_and_hides_under_a_rise_like_a_culvert():
    # Gefälle in +x, dazwischen ein Damm (Straße/Durchlass), hinter dem das Gelände wieder abfällt
    base = _terrain_with_channel(slope=-0.02)

    def with_dam(x, y):
        x = np.asarray(x, float)
        return base(x, y) + np.where((x > 40) & (x < 60), 3.0, 0.0)

    nodes = build_river_nodes(_line(0, 100), with_dam, width=2.5, depth=1.0, spacing=5.0)

    z = np.array([n[2] for n in nodes])
    assert (np.diff(z) <= 1e-9).all()  # nie bergauf
    dam = [n for n in nodes if 42 < n[0] < 58]
    assert all(n[2] < with_dam(n[0], 0.0) for n in dam)  # unter dem Damm, dort unsichtbar


def test_line_drawn_uphill_is_reversed_so_the_water_runs_downhill():
    nodes = build_river_nodes(_line(0, 100), _terrain_with_channel(slope=+0.03), width=2.5, depth=1.0, spacing=10.0)

    assert nodes[0][0] > nodes[-1][0]  # beginnt oben (großes x), endet unten
    assert nodes[0][2] >= nodes[-1][2]


def test_short_line_still_yields_two_nodes():
    nodes = build_river_nodes([(0.0, 0.0), (4.0, 0.0)], _terrain_with_channel(), width=2.5, depth=1.0, spacing=10.0)

    assert len(nodes) == 2


def test_split_nodes_chunks_overlap_by_one_node_and_respect_the_limit():
    nodes = [[float(i), 0.0, 0.0, 2.5, 1.0, 0.0, 0.0, 1.0] for i in range(25)]

    chunks = split_nodes(nodes, max_nodes=10)

    assert all(len(c) <= 10 for c in chunks)
    assert chunks[0][-1] == chunks[1][0]  # lückenlos: das letzte Knotenpaar wird geteilt
    assert [n for c in chunks for n in c[(0 if c is chunks[0] else 1):]] == nodes


# --- Teiche / Seen ----------------------------------------------------------------------------


def _flat(z=100.0):
    return lambda x, y: np.full_like(np.asarray(x, float), z)


def _rect(block):
    (cx, cy, _), (w, h, _) = block["position"], block["scale"]
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_pond_blocks_stay_inside_the_polygon_and_cover_most_of_it():
    pond = box(0, 0, 40, 30)

    blocks = build_pond_blocks(pond, _flat(), lift=0.15, depth=3.0, cell=6.0)

    assert blocks
    assert all(pond.buffer(1e-6).contains(_rect(b)) for b in blocks)
    assert sum(_rect(b).area for b in blocks) > 0.7 * pond.area


def test_pond_blocks_do_not_overlap_each_other():
    blocks = build_pond_blocks(box(0, 0, 40, 30), _flat(), lift=0.15, depth=3.0, cell=6.0)

    total = sum(_rect(b).area for b in blocks)
    from shapely.ops import unary_union

    assert unary_union([_rect(b) for b in blocks]).area == pytest.approx(total)


def test_pond_water_level_sits_just_above_the_terrain_with_the_configured_depth():
    blocks = build_pond_blocks(box(0, 0, 20, 20), _flat(281.0), lift=0.15, depth=3.0, cell=5.0)

    assert all(b["position"][2] == pytest.approx(281.15) for b in blocks)  # Oberfläche = Position.z
    assert all(b["scale"][2] == 3.0 for b in blocks)
    assert all(b["rotationMatrix"] == [1, 0, 0, 0, 1, 0, 0, 0, 1] for b in blocks)


def test_pond_on_a_slope_uses_a_low_level_so_it_does_not_flood_the_downhill_side():
    # Gelände fällt in +x um 4 m; ein Teichniveau am Maximum würde nichts füllen, am Minimum alles
    sloped = lambda x, y: 100.0 - 0.1 * np.asarray(x, float)
    blocks = build_pond_blocks(box(0, 0, 40, 20), sloped, lift=0.0, depth=3.0, cell=5.0)

    level = blocks[0]["position"][2]
    assert 96.0 <= level <= 98.0  # unteres Viertel der Geländehöhen (96..100), nicht das Maximum


def test_tiny_pond_smaller_than_a_cell_still_gets_blocks_inside():
    # Die Zellgröße passt sich der Teichgröße an (mindestens 1 m), auch ein 3x3-m-Teich wird gefüllt
    pond = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])

    blocks = build_pond_blocks(pond, _flat(), lift=0.15, depth=3.0, cell=6.0)

    assert blocks
    assert all(pond.buffer(1e-6).contains(_rect(b)) for b in blocks)
    assert sum(_rect(b).area for b in blocks) > 0.7 * pond.area


def test_l_shaped_pond_leaves_the_notch_empty():
    pond = Polygon([(0, 0), (40, 0), (40, 10), (10, 10), (10, 40), (0, 40)])

    blocks = build_pond_blocks(pond, _flat(), lift=0.15, depth=3.0, cell=5.0)

    assert all(pond.buffer(1e-6).contains(_rect(b)) for b in blocks)
    assert not any(_rect(b).intersects(box(20, 20, 40, 40)) for b in blocks)


def test_nodes_snap_sideways_onto_the_real_channel_when_the_osm_line_is_a_bit_off():
    # Rinne im DGM1 liegt bei y=+1.2, die OSM-Linie bei y=0 (typischer Versatz von 1-2 m)
    def offset_channel(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return 100.0 - 0.02 * x + np.where(np.abs(y - 1.2) <= 0.6, 0.0, 0.5)

    nodes = build_river_nodes(_line(0, 100), offset_channel, width=2.5, depth=1.0, spacing=5.0, lift=0.2, search=2.0)

    ys = np.array([n[1] for n in nodes])
    assert np.abs(ys - 1.2).max() < 0.4  # auf der Rinne, nicht mehr auf der OSM-Linie
    for n in nodes:  # das Wasser liegt dadurch sichtbar in der Rinne (über dem Boden, unter dem Ufer)
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
    assert np.abs(np.diff(ys)).max() < 1.5  # glatt, keine Sprünge von Rand zu Rand
