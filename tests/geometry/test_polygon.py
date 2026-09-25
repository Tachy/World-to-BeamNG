"""Tests for world_to_beamng.geometry.polygon.clip_road_polygons."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.geometry.polygon import clip_road_polygons


def _road(road_id, coords):
    return {"id": road_id, "coords": coords, "name": f"road_{road_id}", "osm_tags": {"highway": "track"}}


def test_clip_road_polygons_keeps_single_contiguous_road_unchanged():
    # All points lie within the clip box -> one section, ID stays the same.
    # Point spacing deliberately << config.GRID_SPACING, so that the segment subdivision
    # in clip_road_polygons() (max_seg = config.GRID_SPACING) inserts no intermediate points
    # regardless of the configured grid spacing.
    coords = [(0.0, 0.0, 100.0), (0.1, 0.0, 100.1), (0.2, 0.0, 100.2)]
    grid_bounds_local = (-10.0, 10.0, -10.0, 10.0)

    result = clip_road_polygons([_road(42, coords)], grid_bounds_local, margin=0.0)

    assert len(result) == 1
    assert result[0]["id"] == 42
    assert result[0]["coords"] == coords


def test_clip_road_polygons_splits_at_removed_gap_instead_of_bridging():
    """Regression: a road that leaves the tile and reaches back in at a completely different
    place must NOT be merged into a single, continuous centerline with an artificial "teleport"
    straight line across the gap (see root cause analysis of the hard embankment cliff at way
    77512819: two widely separated point groups, both inside the clip box, were previously
    connected directly and filled by segment subdivision with a straight, wrong Z interpolation
    over hundreds of meters).
    """
    grid_bounds_local = (-10.0, 10.0, -10.0, 10.0)

    # Point spacing within the clusters deliberately << config.GRID_SPACING (see
    # comment in test_clip_road_polygons_keeps_single_contiguous_road_unchanged).
    coords = [
        # Cluster A: inside the box
        (0.0, 0.0, 100.0),
        (0.1, 0.0, 100.1),
        # Far outside the box (gets removed)
        (500.0, 500.0, 50.0),
        (600.0, 600.0, 20.0),
        # Cluster B: inside the box again, but geometrically far from cluster A
        (-5.0, -5.0, 300.0),
        (-5.1, -5.0, 300.1),
    ]

    result = clip_road_polygons([_road(77512819, coords)], grid_bounds_local, margin=0.0)

    # Expected: TWO separate road sections, no bridge between them -
    # each section contains EXACTLY the points of its own cluster, no
    # artificially interpolated intermediate points (they would only arise if both
    # clusters were wrongly joined into one continuous centerline).
    assert len(result) == 2
    coord_lists = [road["coords"] for road in result]
    assert [(0.0, 0.0, 100.0), (0.1, 0.0, 100.1)] in coord_lists
    assert [(-5.0, -5.0, 300.0), (-5.1, -5.0, 300.1)] in coord_lists

    # IDs of the two sections must be unique
    assert result[0]["id"] != result[1]["id"]


def test_clip_road_polygons_drops_run_with_single_surviving_point():
    grid_bounds_local = (-10.0, 10.0, -10.0, 10.0)
    coords = [
        (0.0, 0.0, 100.0),
        (0.1, 0.0, 100.1),
        (500.0, 500.0, 50.0),  # removed -> ends the first section
        (600.0, 600.0, 20.0),  # removed
        (700.0, 700.0, 10.0),  # removed
        (5.0, 5.0, 200.0),  # single point -> section with only 1 point, gets discarded
        (800.0, 800.0, 5.0),  # removed
    ]

    result = clip_road_polygons([_road(1, coords)], grid_bounds_local, margin=0.0)

    assert len(result) == 1
    assert result[0]["coords"] == [(0.0, 0.0, 100.0), (0.1, 0.0, 100.1)]
