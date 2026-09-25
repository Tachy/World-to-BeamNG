"""Tests for world_to_beamng.terrain.tunnel_terrain: terrain only immediately at the tube and portal - earth 1.20 m
above the tube shell where the terrain protrudes into the tube; no embankments/fills; hole cells that hide the
collar or shell."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math

import numpy as np
import pytest
from shapely.geometry import box

from world_to_beamng.terrain.tunnel_terrain import shape_terrain_for_tunnels
from world_to_beamng.tunnels.tunnel_portal import plan_tunnels, portal_local_coords

FLOOR = 100.0
COVER = 1.2
SHELL_RATIO = 0.1  # wall thickness : diameter
COLLAR_RATIO = 0.1  # portal collar: wall thickness at the thinnest point : diameter


def _plans(coords, galleries=None):
    tunnel = {"id": 1, "coords": coords, "width": 7.0, "floor_material": "f"}
    return plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, collar_ratio=COLLAR_RATIO, flat_depth=1.5, length=3.5,
                        shell_ratio=SHELL_RATIO, galleries=galleries)


def _setup(natural=105.0, size=120):
    # Tunnel along y=60 from x=30 to x=90, floor at 100 m; in between mountain at `natural`, in front of both portals
    # road level.
    plans = _plans([(30.0, 60.0, FLOOR), (90.0, 60.0, FLOOR)])
    heights = np.full((size, size), natural)
    heights[:, :30] = FLOOR
    heights[:, 91:] = FLOOR
    return plans, heights


def _shape(plans, heights, protected=None):
    return shape_terrain_for_tunnels(heights, 0.0, 0.0, 1.0, plans, cover=COVER, protected=protected)


def _cover_height(plan, across):
    """Earth COVER above the round outer shell (radius + shell) at lateral distance `across`."""
    radius = plan["radius"]
    return FLOOR + radius / 2.0 + math.sqrt((radius + plan["shell"]) ** 2 - across**2) + COVER


def test_terrain_cutting_into_the_tube_gets_1_20_m_earth_over_the_shell():
    plans, heights = _setup(natural=105.0)

    result, _ = _shape(plans, heights)

    assert result[60, 60] == pytest.approx(_cover_height(plans[0], 0.0))  # right above the tube
    assert result[62, 60] == pytest.approx(_cover_height(plans[0], 2.0))  # follows the round cross-section


def test_terrain_already_above_the_cover_stays():
    plans, heights = _setup(natural=150.0)

    result, _ = _shape(plans, heights)

    assert result[60, 60] == pytest.approx(150.0)


def test_cover_stays_within_the_shell_footprint_no_slopes_no_dams():
    plans, heights = _setup(natural=105.0)
    heights[:60, 30:91] = 60.0  # valley side (y < 60): the terrain drops far below the carriageway
    radius = plans[0]["radius"]

    result, _ = _shape(plans, heights)

    outside = int(math.ceil(radius + plans[0]["shell"])) + 1
    assert result[60 + outside, 60] == pytest.approx(105.0)  # hill side directly next to the shell: unchanged
    assert result[60 - outside, 60] == pytest.approx(60.0)  # valley side: no fill into the valley
    assert result[60 - outside - 10, 60] == pytest.approx(60.0)


def test_tube_standing_on_or_above_the_ground_is_left_free():
    plans, heights = _setup(natural=105.0)
    heights[:, 45:55] = 80.0  # depression under the tube: it stands free (outer shell visible)
    heights[:, 65:75] = 100.2  # terrain at carriageway height: does not protrude into the tube

    result, _ = _shape(plans, heights)

    assert result[60, 50] == pytest.approx(80.0)
    assert result[60, 70] == pytest.approx(100.2)
    assert result[60, 40] == pytest.approx(_cover_height(plans[0], 0.0))  # still covered next to it


def test_where_the_tube_leaves_the_ground_the_crossing_cells_become_holes():
    # Otherwise the sloped face between the cover and the deep terrain would run as an earth wall right through the tube
    plans, heights = _setup(natural=105.0)
    heights[:, 45:55] = 80.0

    result, holes = _shape(plans, heights)

    assert holes[60, 44] and holes[60, 54]  # cells x=44..45 and x=54..55 span the transition
    assert not holes[60, 40] and not holes[60, 50]


def test_portal_zone_is_at_floor_level_then_covered_and_the_step_holes_lie_inside_the_collar():
    plans, heights = _setup(natural=105.0)
    portal = plans[0]["portals"][0]  # start at x=30, axis +x
    assert portal["axis"] == pytest.approx((1.0, 0.0))

    result, holes = _shape(plans, heights)

    assert result[60, 31] == pytest.approx(FLOOR - 0.05)  # 1 m behind the portal plane: below the tube floor
    assert result[60, 32] > FLOOR + portal["crown"]  # covered behind it
    assert holes[60, 31]  # square x=31..32 spans the step at 1.5 m
    assert not holes[60, 29]  # in front of the portal plane the approach stays closed

    # Every hole cell lies in the collar of a portal or above the tube (then it hides the shell)
    radius = portal["radius"]
    rows, cols = np.nonzero(holes)
    for r, c in zip(rows, cols):
        corners_x = np.array([c, c + 1, c, c + 1], dtype=float)
        corners_y = np.array([r, r, r + 1, r + 1], dtype=float)
        for p in plans[0]["portals"]:
            along, across = portal_local_coords(p, corners_x, corners_y)
            if np.all(along >= 0.0) and np.all(along <= p["length"]):
                assert np.all(np.abs(across) <= p["half_width"])
                break
        else:
            assert np.all(np.abs(corners_y - 60.0) <= radius + plans[0]["shell"] + 1.5)


def test_steep_hillside_behind_the_portal_is_cut_down_to_the_collar_not_the_other_way_round():
    # The portal block used to grow up to the hillside height (Banchi: 14 m above the carriageway). Now the collar stays
    # as large as planned, and the terrain within its footprint is cut down to its outer contour.
    plans, heights = _setup(natural=130.0)
    portal = plans[0]["portals"][0]
    top_before, bottom_before = portal["top_z"], portal["bottom_z"]

    result, holes = _shape(plans, heights)

    assert portal["top_z"] == top_before and portal["bottom_z"] == bottom_before
    outer = portal["half_width"]
    rows, cols = np.nonzero(holes)
    assert len(rows) > 0
    for r, c in zip(rows, cols):
        for y in (r, r + 1):
            for x in (c, c + 1):
                along, across = portal_local_coords(portal, np.array([float(x)]), np.array([float(y)]))
                if 0.0 <= along[0] <= portal["length"] and abs(across[0]) < outer:
                    assert result[y, x] <= portal["top_z"] + 1e-6  # below the top edge of the collar
    assert result[60, 40] == pytest.approx(130.0)  # further back the hillside remains


def test_surface_roads_are_protected():
    plans, heights = _setup(natural=105.0)
    road_over_tunnel = box(55.0, 0.0, 65.0, 120.0)  # path across the tube

    result, _ = _shape(plans, heights, protected=road_over_tunnel)

    assert result[60, 60] == pytest.approx(105.0)
    assert result[60, 50] > 105.0


def test_tunnel_end_inside_the_mountain_is_no_portal_and_leaves_the_terrain_alone():
    plans, heights = _setup(natural=105.0)
    heights[:, :35] = 900.0  # in front of the start end: mountain instead of open terrain
    before = heights.copy()

    result, holes = _shape(plans, heights)

    start, end = plans[0]["portals"]
    assert not start["open"] and end["open"]
    assert np.array_equal(result[:, :30], before[:, :30])  # no trench in front of/at the closed side
    assert not holes[:, :40].any()
    assert holes[:, 80:].any()  # the open end gets its portal

    from world_to_beamng.tunnels.tunnel_mesh import build_tunnels

    ids = [m["id"] for m in build_tunnels(plans, "w", "p")]
    assert ids == ["tunnel_1", "tunnel_1_portal_end"]


def test_gallery_transition_is_a_portal_even_with_mountain_in_front():
    gallery = {"id": 2, "coords": [(0.0, 60.0, FLOOR), (30.0, 60.0, FLOOR)], "width": 6.5, "floor_material": "f", "osm_tags": {}}
    plans = _plans([(30.0, 60.0, FLOOR), (90.0, 60.0, FLOOR)], galleries=[gallery])
    heights = np.full((120, 120), 150.0)  # mountain in front of the portal too - there would be no open portal here
    portal = plans[0]["portals"][0]
    top_before = portal["top_z"]

    result, holes = _shape(plans, heights)

    assert portal["open"] is True
    assert result[60, 31] == pytest.approx(FLOOR - 0.05)
    assert holes[60, 31]
    assert portal["top_z"] == top_before  # the portal does not grow with the hillside
    assert result[60, 32] <= top_before - 0.1 + 1e-6  # hillside within the portal footprint cut down below its contour


def test_without_collar_the_hillside_at_the_portal_stays_inside_the_shell_wall():
    tunnel = {"id": 1, "coords": [(30.0, 60.0, FLOOR), (90.0, 60.0, FLOOR)], "width": 7.0, "floor_material": "f"}
    plans = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, collar_ratio=0.0, flat_depth=1.5, length=3.5,
                         shell_ratio=SHELL_RATIO)
    heights = np.full((120, 120), 130.0)
    heights[:, :30] = FLOOR
    heights[:, 91:] = FLOOR
    portal = plans[0]["portals"][0]
    radius, outer = portal["radius"], portal["radius"] + portal["shell"]

    result, holes = _shape(plans, heights)

    rows, cols = np.nonzero(holes)
    assert len(rows) > 0
    for r, c in zip(rows, cols):
        for y in (r, r + 1):
            for x in (c, c + 1):
                along, across = portal_local_coords(portal, np.array([float(x)]), np.array([float(y)]))
                if 0.0 <= along[0] <= portal["length"] and abs(across[0]) < outer:
                    # below the outer surface of the shell, but above the tube interior
                    assert result[y, x] <= FLOOR + radius / 2.0 + math.sqrt(outer**2 - across[0] ** 2) + 1e-6



def test_tilted_entrance_keeps_the_whole_opening_free_and_hides_the_hole_edge_behind_the_face():
    tilt = math.tan(math.radians(20.0))
    tunnel = {"id": 1, "coords": [(30.0, 60.0, FLOOR), (90.0, 60.0, FLOOR)], "width": 7.0, "floor_material": "f"}
    plans = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, collar_ratio=0.0, flat_depth=1.5, length=3.5,
                         shell_ratio=SHELL_RATIO, tilt_deg=20.0)
    heights = np.full((120, 120), 130.0)
    heights[:, :30] = FLOOR
    heights[:, 91:] = FLOOR
    portal = plans[0]["portals"][0]
    radius, outer = portal["radius"], portal["radius"] + portal["shell"]

    result, holes = _shape(plans, heights)

    for x in range(30, 34):
        for y in range(55, 66):
            along, across = portal_local_coords(portal, np.array([float(x)]), np.array([float(y)]))
            top = (outer + radius / 2.0) - 0.0  # outer crown above the floor
            if along[0] < top * tilt and abs(across[0]) < radius:
                assert result[y, x] <= FLOOR  # in front of the tilted front face: opening free
    rows, cols = np.nonzero(holes)
    assert len(rows) > 0
    for r, c in zip(rows, cols):
        for y in (r, r + 1):
            for x in (c, c + 1):
                along, across = portal_local_coords(portal, np.array([float(x)]), np.array([float(y)]))
                height = result[y, x] - FLOOR
                if height > 0.3 and abs(across[0]) < outer and along[0] < portal["length"]:
                    assert along[0] >= height * tilt - 1e-6  # hole edge lies behind the tilted front face



@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 45.0])
def test_portal_hole_cells_stay_within_the_collar_sides_for_any_tunnel_direction(angle_deg):
    # Grid cells (1 m, diagonal 1.41 m) at the portal step reach laterally up to ~R + 1.4 m - the collar must cover them
    a = math.radians(angle_deg)
    direction = np.array([math.cos(a), math.sin(a)])
    start = np.array([40.0, 40.0])
    end = start + direction * 60.0
    tunnel = {"id": 1, "coords": [(*start, FLOOR), (*end, FLOOR)], "width": 6.5, "floor_material": "f"}
    plans = plan_tunnels([tunnel], width_margin=1.5, segment_step=10.0, flat_depth=1.5, length=3.5,
                         shell_ratio=SHELL_RATIO, tilt_deg=20.0, collar_ratio=COLLAR_RATIO, collar_min_side=1.5)
    size = 160
    gx, gy = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))
    along_total = (gx - start[0]) * direction[0] + (gy - start[1]) * direction[1]
    heights = np.where((along_total > 0.0) & (along_total < 60.0), 130.0, FLOOR)  # mountain between the portals

    result, holes = _shape(plans, heights)

    for p in [q for q in plans[0]["portals"] if q["open"]]:
        rows, cols = np.nonzero(holes)
        for r, c in zip(rows, cols):
            along, across = portal_local_coords(p, np.array([c, c + 1, c, c + 1], float), np.array([r, r, r + 1, r + 1], float))
            if along.max() >= -1.0 and along.min() <= p["length"] + 1.0:
                assert np.abs(across).max() <= p["half_width"] + 1e-6
