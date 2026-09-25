"""Tests for world_to_beamng.geometry.polygon.smooth_structure_transitions: smooth_roads_xy_only smooths
each road independently and keeps its endpoints fixed, which can leave a visible kink at the transition to a
bridge/tunnel/gallery (different direction/gradient on both sides, among other reasons because the structure has a
linear instead of the natural DGM height profile). This step smooths the few points on both sides of an
unambiguous structure transition together, so that the shared point stays identical in both roads."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.polygon import smooth_structure_transitions


def _road(road_id, coords, **tags):
    return {"id": road_id, "coords": coords, "name": "r", "osm_tags": tags}


def test_blends_the_kink_at_a_unique_bridge_transition_and_keeps_the_boundary_shared():
    # A (surface road) runs flat into the shared point P=(10,0,0); B (bridge) continues steeply from P
    # (10 m of elevation over 10m) - a clear kink in the height profile exactly at P.
    road_a = _road(1, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], highway="residential")
    road_b = _road(2, [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)], highway="primary", bridge="yes")

    result = smooth_structure_transitions([road_a, road_b], window=1, iterations=1, weight_center=0.5)

    a = next(r for r in result if r["id"] == 1)
    b = next(r for r in result if r["id"] == 2)

    # Chaikin step by hand: temp[1] = 0.5*(10,0,0) + 0.25*(0,0,0) + 0.25*(20,0,10) = (10,0,2.5)
    assert a["coords"][-1] == (10.0, 0.0, 2.5)
    assert b["coords"][0] == (10.0, 0.0, 2.5)
    assert a["coords"][-1] == b["coords"][0]  # no gap at the transition

    # far ends (anchors) stay unchanged
    assert a["coords"][0] == (0.0, 0.0, 0.0)
    assert b["coords"][-1] == (20.0, 0.0, 10.0)


def test_blends_a_structure_to_structure_chain_too():
    # Tunnel directly followed by a bridge (no surface neighbor involved) - smoothing should happen here too.
    road_a = _road(1, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], highway="trunk", tunnel="yes")
    road_b = _road(2, [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)], highway="primary", bridge="yes")

    result = smooth_structure_transitions([road_a, road_b], window=1, iterations=1, weight_center=0.5)

    a = next(r for r in result if r["id"] == 1)
    b = next(r for r in result if r["id"] == 2)

    assert a["coords"][-1] == b["coords"][0]
    assert a["coords"][-1] == (10.0, 0.0, 2.5)


def test_real_multiway_junction_at_a_bridge_end_is_left_untouched():
    # Two surface roads A and C both touch the same bridge endpoint P -> ambiguous, no smoothing.
    road_a = _road(1, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], highway="residential")
    road_c = _road(3, [(10.0, 0.0, 0.0), (5.0, 5.0, 0.0)], highway="residential")
    road_b = _road(2, [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)], highway="primary", bridge="yes")

    result = smooth_structure_transitions([road_a, road_b, road_c], window=1, iterations=1, weight_center=0.5)

    a = next(r for r in result if r["id"] == 1)
    b = next(r for r in result if r["id"] == 2)
    c = next(r for r in result if r["id"] == 3)

    assert a["coords"] == [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    assert b["coords"] == [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)]
    assert c["coords"] == [(10.0, 0.0, 0.0), (5.0, 5.0, 0.0)]


def test_plain_surface_to_surface_junction_is_out_of_scope_and_untouched():
    road_a = _road(1, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)], highway="residential")
    road_b = _road(2, [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)], highway="residential")

    result = smooth_structure_transitions([road_a, road_b], window=1, iterations=1, weight_center=0.5)

    a = next(r for r in result if r["id"] == 1)
    b = next(r for r in result if r["id"] == 2)

    assert a["coords"] == [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    assert b["coords"] == [(10.0, 0.0, 0.0), (20.0, 0.0, 10.0)]
