"""Tests for geometry.road_structures.fix_underpass_elevations: roads that pass under a bridge get their height interpolated
from before to behind the bridge (the terrain model shows the bridge deck there)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.geometry.road_structures import fix_underpass_elevations

KW = dict(max_search=60.0, stable_length=6.0, max_grade=0.15, min_rise=1.0)
HALF = 4.875  # 3-lane bridge


def _half(road):
    return HALF


def _bridge(x0=-40.0, x1=40.0, z=100.0):
    xs = np.arange(x0, x1 + 0.1, 1.0)
    return {"id": 1, "coords": np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, z)]),
            "osm_tags": {"highway": "primary", "bridge": "yes"}}


def _road_across(z_of_y, road_id=2, y0=-100.0, y1=100.0, tags=None):
    ys = np.arange(y0, y1 + 0.1, 1.0)
    return {"id": road_id, "coords": np.column_stack([np.zeros_like(ys), ys, [z_of_y(y) for y in ys]]),
            "osm_tags": tags or {"highway": "primary_link"}}


def test_road_under_a_bridge_is_interpolated_from_before_to_behind_it():
    # The terrain model shows the deck (98 m) where the road really runs at 90 m
    road = _road_across(lambda y: 98.0 if abs(y) <= HALF else 90.0)
    bridge = _bridge()

    fixed = fix_underpass_elevations([bridge, road], _half, **KW)

    z = {round(y): zz for (_, y, zz) in road["coords"]}
    assert fixed == 1
    assert z[0] == pytest.approx(90.0) and z[3] == pytest.approx(90.0)  # under the deck
    assert z[-50] == pytest.approx(90.0) and z[50] == pytest.approx(90.0)  # untouched


def test_the_grade_of_the_road_carries_through_the_underpass():
    road = _road_across(lambda y: 90.0 + 0.04 * y + (8.0 if abs(y) <= HALF else 0.0))

    fix_underpass_elevations([_bridge(), road], _half, **KW)

    z = {round(y): zz for (_, y, zz) in road["coords"]}
    assert z[0] == pytest.approx(90.0, abs=1e-6) and z[5] == pytest.approx(90.2, abs=1e-6)


def test_no_change_where_the_terrain_model_does_not_show_the_deck():
    road = _road_across(lambda y: 90.0 + 0.3 * np.sin(y / 20.0))
    before = road["coords"].copy()

    assert fix_underpass_elevations([_bridge(), road], _half, **KW) == 0
    assert np.array_equal(road["coords"], before)


def test_approach_roads_that_only_touch_the_bridge_end_are_left_alone():
    ys = np.arange(40.0, 120.0, 1.0)
    approach = {"id": 3, "coords": np.column_stack([ys, np.zeros_like(ys), np.full_like(ys, 100.0)]),
                "osm_tags": {"highway": "primary"}}
    before = approach["coords"].copy()

    assert fix_underpass_elevations([_bridge(), approach], _half, **KW) == 0
    assert np.array_equal(approach["coords"], before)


def test_road_that_ends_within_the_margin_is_skipped_and_bridges_are_never_changed():
    short = _road_across(lambda y: 98.0 if abs(y) <= HALF else 90.0, y0=-6.0, y1=6.0)  # ends 1 m past the deck edge
    bridge = _bridge()
    bridge_before = bridge["coords"].copy()

    assert fix_underpass_elevations([bridge, short], _half, **KW) == 0
    assert np.array_equal(bridge["coords"], bridge_before)


def test_two_bridges_over_one_road_are_both_fixed():
    def z_of_y(y):
        return 98.0 if (abs(y) <= HALF or abs(y - 60.0) <= HALF) else 90.0

    second = _bridge()
    second["id"] = 4
    second["coords"] = second["coords"].copy()
    second["coords"][:, 1] = 60.0
    road = _road_across(z_of_y, y0=-100.0, y1=140.0)

    assert fix_underpass_elevations([_bridge(), second, road], _half, **KW) == 1  # one road, two crossings
    z = {round(y): zz for (_, y, zz) in road["coords"]}
    assert z[0] == pytest.approx(90.0) and z[60] == pytest.approx(90.0)


def _split_road(z_of_y, split_y=0.0, y0=-100.0, y1=100.0, first_id=2):
    """The road is split into two pieces exactly at the crossing (junction detection splits it under the bridge)."""
    def piece(ys, road_id):
        return {"id": road_id, "coords": np.column_stack([np.zeros_like(ys), ys, [z_of_y(y) for y in ys]]),
                "osm_tags": {"highway": "service"}}

    return (piece(np.arange(y0, split_y + 0.1, 1.0), first_id), piece(np.arange(split_y, y1 + 0.1, 1.0), first_id + 1))


def test_road_split_under_the_bridge_is_interpolated_across_both_pieces():
    # A2 Tremola ramp: the service road under the bridge is split at the crossing point
    before, after = _split_road(lambda y: 98.0 if abs(y) <= HALF else 90.0)

    fixed = fix_underpass_elevations([_bridge(), before, after], _half, **KW)

    z_before = {round(y): zz for (_, y, zz) in before["coords"]}
    z_after = {round(y): zz for (_, y, zz) in after["coords"]}
    assert fixed == 2
    assert z_before[0] == pytest.approx(90.0) and z_after[0] == pytest.approx(90.0)  # the shared node agrees
    assert z_before[-3] == pytest.approx(90.0) and z_after[3] == pytest.approx(90.0)
    assert z_before[-50] == pytest.approx(90.0) and z_after[50] == pytest.approx(90.0)


def test_pieces_are_chained_in_either_direction():
    before, after = _split_road(lambda y: 98.0 if abs(y) <= HALF else 90.0)
    after["coords"] = after["coords"][::-1].copy()  # digitized toward the crossing as well

    fix_underpass_elevations([_bridge(), after, before], _half, **KW)

    assert {round(y): zz for (_, y, zz) in after["coords"]}[3] == pytest.approx(90.0)
    assert {round(y): zz for (_, y, zz) in before["coords"]}[-3] == pytest.approx(90.0)


def test_a_side_road_that_only_ends_at_the_crossing_is_not_chained_across():
    before, after = _split_road(lambda y: 98.0 if abs(y) <= HALF else 90.0)
    sideways = {"id": 9, "coords": np.array([[x, 0.0, 98.0] for x in np.arange(0.0, 60.0, 1.0)]),
                "osm_tags": {"highway": "service"}}  # starts under the bridge, runs off at a right angle

    fix_underpass_elevations([_bridge(), before, after, sideways], _half, **KW)

    assert {round(y): zz for (_, y, zz) in before["coords"]}[0] == pytest.approx(90.0)


# --- the terrain model raises the road before the deck and lets it fall behind it: the reference is the stable height -------
def _flanked(y):
    """DGM profile of an underpass: flat 90 m, a steep flank 5 m before the deck, the deck level, a flank 8 m behind it."""
    if y < -HALF - 5.0 or y > HALF + 8.0:
        return 90.0 + 0.03 * y
    if y < -HALF:
        return 90.0 + 0.03 * y + 8.0 * (y + HALF + 5.0) / 5.0
    if y <= HALF:
        return 90.0 + 0.03 * y + 8.0
    return 90.0 + 0.03 * y + 8.0 * (HALF + 8.0 - y) / 8.0


def test_reference_heights_are_taken_where_the_terrain_model_is_stable_not_on_the_flanks():
    road = _road_across(_flanked)

    fix_underpass_elevations([_bridge(), road], _half, **KW)

    z = {round(y): zz for (_, y, zz) in road["coords"]}
    grades = np.abs(np.diff([z[y] for y in range(-30, 31)]))
    assert grades.max() < 0.06  # smooth through the whole underpass - no remaining flank
    assert z[0] == pytest.approx(90.0, abs=0.3) and z[-30] == pytest.approx(90.0 - 0.9, abs=1e-6)  # ends stay


def test_search_reaches_beyond_the_flank_even_when_it_is_long():
    def long_flank(y):
        base = 90.0 + 0.03 * y
        if abs(y) <= HALF:
            return base + 8.0
        distance = abs(y) - HALF
        return base + 8.0 * max(0.0, 1.0 - distance / 20.0)  # 20 m long flanks on both sides (0.4 m per m)

    road = _road_across(long_flank)

    fix_underpass_elevations([_bridge(), road], _half, **KW)

    z = {round(y): zz for (_, y, zz) in road["coords"]}
    assert abs(z[0] - 90.0) < 0.5 and abs(z[-15] - (90.0 - 0.45)) < 0.5


def test_unstable_terrain_within_the_search_distance_leaves_the_road_alone():
    rng = np.random.default_rng(1)
    noisy = {round(y): 90.0 + 4.0 * rng.standard_normal() for y in range(-200, 201)}  # never stable
    road = _road_across(lambda y: noisy[round(y)] + (8.0 if abs(y) <= HALF else 0.0))
    before = road["coords"].copy()

    assert fix_underpass_elevations([_bridge(), road], _half, **KW) == 0
    assert np.array_equal(road["coords"], before)


def test_corrected_roads_are_marked_for_daylight_slopes():
    road = _road_across(lambda y: 98.0 if abs(y) <= HALF else 90.0)
    untouched = _road_across(lambda y: 90.0, road_id=5)
    untouched["coords"] = untouched["coords"] + np.array([200.0, 0.0, 0.0])  # far from the bridge

    fix_underpass_elevations([_bridge(), road, untouched], _half, **KW)

    assert road["underpass"] is True and "underpass" not in untouched
