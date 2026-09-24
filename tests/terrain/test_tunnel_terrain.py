"""Tests für world_to_beamng.terrain.tunnel_terrain: Überdeckung über der Tunnelröhre, Portal-Zone und die
Loch-Zellen, die der Portalblock verdecken muss."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from shapely.geometry import box

from world_to_beamng.terrain.tunnel_terrain import shape_terrain_for_tunnels
from world_to_beamng.tunnels.tunnel_portal import plan_tunnels, portal_local_coords

FLOOR = 100.0
COVER = 1.0


def _setup(natural=105.0, size=120):
    # Tunnel entlang y=60 von x=30 bis x=90, Boden auf 100 m; dazwischen Berg auf `natural` (Röhre steckt zu
    # mehr als halber Kronenhöhe im Gelände), vor beiden Portalen Straßenniveau.
    tunnel = {"id": 1, "coords": [(30.0, 60.0, FLOOR), (90.0, 60.0, FLOOR)], "width": 7.0, "floor_material": "f"}
    plans = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=COVER)
    heights = np.full((size, size), natural)
    heights[:, :30] = FLOOR
    heights[:, 91:] = FLOOR
    return plans, heights


def _shape(plans, heights, protected=None):
    return shape_terrain_for_tunnels(heights, 0.0, 0.0, 1.0, plans, cover=COVER, cover_slope=1.5, protected=protected)


def test_cover_raises_low_terrain_over_the_tube_and_leaves_high_terrain():
    plans, heights = _setup(natural=105.0)
    crown = plans[0]["crown"]

    result, _ = _shape(plans, heights)

    assert result[60, 60] == pytest.approx(FLOOR + crown + COVER)  # mitten über der Röhre
    assert result[60 + 30, 60] == pytest.approx(105.0)  # weit seitlich: natürliches Gelände

    plans, heights = _setup(natural=150.0)
    result, _ = _shape(plans, heights)
    assert result[60, 60] == pytest.approx(150.0)  # Berg schon hoch genug: nichts anheben


def test_cover_slopes_down_to_the_natural_terrain_on_the_side():
    plans, heights = _setup(natural=105.0)
    top = FLOOR + plans[0]["crown"] + COVER
    half_width = plans[0]["portals"][0]["half_width"]

    result, _ = _shape(plans, heights)

    row = int(round(60 + half_width + 3.0))
    distance = row - 60
    assert result[row, 60] == pytest.approx(top - (distance - half_width) / 1.5)


def test_portal_zone_is_at_floor_level_then_covered_and_the_step_becomes_holes_inside_the_block():
    plans, heights = _setup(natural=105.0)
    portal = plans[0]["portals"][0]  # Start bei x=30, Achse +x
    assert portal["axis"] == pytest.approx((1.0, 0.0))

    result, holes = _shape(plans, heights)

    assert result[60, 31] == pytest.approx(FLOOR - 0.05)  # 1 m hinter der Portalebene: unter dem Röhrenboden
    assert result[60, 32] >= FLOOR + portal["crown"] + COVER - 1e-9  # hinter der Portal-Zone: überdeckt
    assert holes[60, 31]  # Quadrat x=31..32 überspannt die Stufe bei 1,5 m
    assert not holes[60, 29]  # vor der Portalebene bleibt die Zufahrt geschlossen

    rows, cols = np.nonzero(holes)
    # Jede Loch-Zelle liegt komplett im Portalblock (beide Portale)
    for r, c in zip(rows, cols):
        corners_x = np.array([c, c + 1, c, c + 1], dtype=float)
        corners_y = np.array([r, r, r + 1, r + 1], dtype=float)
        inside_any = False
        for p in plans[0]["portals"]:
            along, across = portal_local_coords(p, corners_x, corners_y)
            if np.all(along >= 0.0) and np.all(along <= p["length"]) and np.all(np.abs(across) <= p["half_width"]):
                inside_any = True
        assert inside_any


def test_portal_block_top_reaches_over_the_hole_corners():
    plans, heights = _setup(natural=130.0)  # steiler Hang: Gelände weit über der Krone
    result, holes = _shape(plans, heights)
    portal = plans[0]["portals"][0]

    rows, cols = np.nonzero(holes)
    assert len(rows) > 0
    corner_max = max(result[r : r + 2, c : c + 2].max() for r, c in zip(rows, cols))
    assert portal["top_z"] >= corner_max


def test_surface_roads_are_protected():
    plans, heights = _setup(natural=105.0)
    road_over_tunnel = box(55.0, 0.0, 65.0, 120.0)  # Weg quer über die Röhre

    result, _ = _shape(plans, heights, protected=road_over_tunnel)

    assert result[60, 60] == pytest.approx(105.0)
    assert result[60, 50] > 105.0


def test_tunnel_end_inside_the_mountain_is_no_portal_and_leaves_the_terrain_alone():
    plans, heights = _setup(natural=105.0)
    heights[:, :35] = 900.0  # vor dem Start-Ende: Berg statt offenem Gelände
    before = heights.copy()

    result, holes = _shape(plans, heights)

    start, end = plans[0]["portals"]
    assert not start["open"] and end["open"]
    assert np.array_equal(result[:, :30], before[:, :30])  # kein Graben vor/an der geschlossenen Seite
    assert not holes[:, :40].any()
    assert holes[:, 80:].any()  # das offene Ende bekommt sein Portal

    from world_to_beamng.tunnels.tunnel_mesh import build_tunnels

    ids = [m["id"] for m in build_tunnels(plans, "w", "p")]
    assert ids == ["tunnel_1", "tunnel_1_portal_end"]


def test_no_cover_dam_where_the_tube_is_not_at_least_half_in_the_ground():
    plans, heights = _setup(natural=105.0)
    heights[:, 50:60] = 80.0  # Senke unter dem Tunnelboden (100 m): Röhre hinge hier in der Luft
    heights[:, 60:70] = 101.0  # "Tunnel" im DGM auf Straßenniveau (z.B. covered=yes-Galerie)

    result, _ = _shape(plans, heights)

    assert result[60, 55] == pytest.approx(80.0)  # kein Damm in der Senke
    assert result[60, 65] == pytest.approx(101.0)  # keiner über der offenen Straße
    assert result[60, 40] > 105.0  # daneben (Röhre im Gelände) weiterhin überdeckt


def _setup_gap(gap, size_x=160):
    # Tunnel entlang y=60 von x=30 bis x=130; hinter dem Start-Portal `gap` Meter flach (Röhre steckt dort laut
    # Höhenmodell nicht im Berg), danach Berg auf 105 m; vor beiden Portalen Straßenniveau.
    tunnel = {"id": 1, "coords": [(30.0, 60.0, FLOOR), (130.0, 60.0, FLOOR)], "width": 7.0, "floor_material": "f"}
    plans = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=COVER)
    heights = np.full((120, size_x), 105.0)
    heights[:, : 30 + gap + 1] = FLOOR
    heights[:, 131:] = FLOOR
    return plans, heights


def test_short_cover_gap_behind_the_portal_is_covered_without_a_step():
    plans, heights = _setup_gap(8)
    top = FLOOR + plans[0]["crown"] + COVER

    result, _ = shape_terrain_for_tunnels(heights, 0.0, 0.0, 1.0, plans, cover=COVER, cover_slope=1.5, cover_gap_max=25.0)

    # hinter der Portal-Zone bis in den Berg durchgehend überdeckt: keine Geländekante durch die Röhre
    assert all(result[60, x] >= top - 1e-9 for x in range(33, 50))


def test_cover_gap_longer_than_the_limit_stays_open():
    plans, heights = _setup_gap(40)

    result, _ = shape_terrain_for_tunnels(heights, 0.0, 0.0, 1.0, plans, cover=COVER, cover_slope=1.5, cover_gap_max=25.0)

    assert result[60, 50] == pytest.approx(FLOOR)  # 20 m hinter dem Portal, mitten in der 40-m-Lücke


def test_without_gap_limit_the_old_behaviour_stays():
    plans, heights = _setup_gap(8)

    result, _ = shape_terrain_for_tunnels(heights, 0.0, 0.0, 1.0, plans, cover=COVER, cover_slope=1.5)

    assert result[60, 36] == pytest.approx(FLOOR)


def test_gallery_transition_is_a_portal_even_with_mountain_in_front():
    tunnel = {"id": 1, "coords": [(30.0, 60.0, FLOOR), (90.0, 60.0, FLOOR)], "width": 7.0, "floor_material": "f"}
    gallery = {"id": 2, "coords": [(0.0, 60.0, FLOOR), (30.0, 60.0, FLOOR)], "width": 6.5, "floor_material": "f", "osm_tags": {}}
    plans = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=COVER, galleries=[gallery])
    heights = np.full((120, 120), 150.0)  # auch vor dem Portal Berg - ein offenes Portal gäbe es hier nicht

    result, holes = _shape(plans, heights)

    assert plans[0]["portals"][0]["open"] is True
    assert result[60, 31] == pytest.approx(FLOOR - 0.05)
    assert holes[60, 31]
