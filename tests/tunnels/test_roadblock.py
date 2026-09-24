"""Straßensperre vor Tunneleinfahrten, deren Tunnel über die Kartengrenze reicht (tunnels/roadblock.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.tunnels.roadblock import plan_roadblocks
from world_to_beamng.tunnels.tunnel_portal import plan_tunnels

BOUNDS = (-1000.0, 1000.0, -1000.0, 1000.0)
KW = dict(bounds=BOUNDS, edge_margin=25.0, width_margin=1.5, distance=5.0, side_margin=0.5, spacing=1.5, entrance_tol=0.5)
ENTRANCES = [(800.0, 0.0), (0.0, 0.0)]  # Endpunkte von Oberflächenstraßen (Zufahrten)


def _plans(coords, width=2.0):
    tunnel = {"id": 7, "coords": coords, "width": width, "floor_material": "f"}
    return plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, wing=2.0, flat_depth=1.5, length=3.5, cover=1.0)


def test_entrance_of_a_tunnel_leaving_the_map_gets_a_barrier_row_across_the_road():
    blocks = plan_roadblocks(_plans([(800.0, 0.0, 100.0), (1200.0, 0.0, 100.0)]), entrances=ENTRANCES, **KW)

    # 2 m Weg + 2 x 0,5 m = 3 m -> 2 Elemente à 1,5 m, 5 m vor der Portalebene (x = 800), quer zur Fahrtrichtung
    assert sorted(b["xy"] for b in blocks) == [pytest.approx((795.0, -0.75)), pytest.approx((795.0, 0.75))]
    # Zeilen = Bilder der lokalen Achsen (BeamNG): x (Längsachse der Barriere) quer = (0, -1), y = Tunnelachse (+x Welt)
    assert all(b["rotation_matrix"] == pytest.approx([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for b in blocks)
    assert all(b["name"].startswith("roadblock_7_start_") for b in blocks)


def test_barrier_row_spans_a_wide_road():
    blocks = plan_roadblocks(_plans([(800.0, 0.0, 100.0), (1200.0, 0.0, 100.0)], width=6.5), entrances=ENTRANCES, **KW)

    assert sorted(b["xy"][1] for b in blocks) == pytest.approx([-3.0, -1.5, 0.0, 1.5, 3.0])  # 7,5 m Sperrbreite


def test_entrance_at_the_end_of_the_way_is_handled_too():
    blocks = plan_roadblocks(_plans([(1200.0, 0.0, 100.0), (800.0, 0.0, 100.0)]), entrances=ENTRANCES, **KW)

    assert {round(b["xy"][0], 6) for b in blocks} == {795.0}
    assert all(b["name"].startswith("roadblock_7_end_") for b in blocks)


def test_tunnel_leaving_the_map_on_both_sides_gets_no_barrier():
    assert plan_roadblocks(_plans([(-1200.0, 0.0, 100.0), (1200.0, 0.0, 100.0)]), entrances=ENTRANCES, **KW) == []


def test_tunnel_inside_the_map_gets_no_barrier():
    assert plan_roadblocks(_plans([(0.0, 0.0, 100.0), (300.0, 0.0, 100.0)]), entrances=ENTRANCES, **KW) == []


def test_tunnel_end_without_an_approach_road_gets_no_barrier():
    # Verzweigung im Berg (z.B. Festungsstollen): keine Straße schließt an -> keine Einfahrt, keine Sperre
    assert plan_roadblocks(_plans([(800.0, 0.0, 100.0), (1200.0, 0.0, 100.0)]), entrances=[], **KW) == []
