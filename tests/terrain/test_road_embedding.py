"""Tests for world_to_beamng.terrain.road_embedding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.terrain.road_embedding import (
    embed_roads_into_heightmap,
    sample_heightmap_bilinear,
    build_road_embankment_profiles,
    apply_embankment_blend,
)


def _road(polygon_xy, centerline_xyz):
    return {
        "road_polygon": np.array(polygon_xy, dtype=np.float64),
        "trimmed_centerline": np.array(centerline_xyz, dtype=np.float64),
    }


def test_embed_roads_sets_exact_road_height_only_near_road():
    # 20x20 heightmap, 1m/cell, 100m high everywhere
    size = 20
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # A flat road at Z=95, area x=[5,15], y=[5,15] (centerline at y=10)
    road = _road(
        polygon_xy=[[5, 5], [15, 5], [15, 15], [5, 15]],
        centerline_xyz=[[5, 10, 95], [15, 10, 95]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road])

    # Cells under the road must be set exactly to centerline height (95)
    # - no safety margin anymore, since the DecalRoad is projected directly onto
    # the terrain (see module docstring).
    assert np.isclose(result[10, 10], 95.0)

    # Cells far away from the road must remain unchanged at 100
    assert result[1, 1] == 100.0
    assert result[18, 18] == 100.0

    # Original array must not have been modified (function returns a copy)
    assert heights[10, 10] == 100.0


def test_embed_roads_can_raise_terrain_above_surroundings():
    # Road is HIGHER than natural terrain (e.g. fill/bridge ramp) ->
    # unlike the earlier mesh approach, the terrain MAY now be raised,
    # because it is the visible road surface itself (a DecalRoad
    # has no geometry of its own that could be poked through).
    size = 10
    heights = np.full((size, size), 50.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road = _road(
        polygon_xy=[[2, 2], [8, 2], [8, 8], [2, 8]],
        centerline_xyz=[[2, 5, 200], [8, 5, 200]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road])

    assert np.isclose(result[5, 5], 200.0)
    # Unchanged outside the road
    assert result[0, 0] == 50.0


def test_embed_roads_follows_curved_centerline_height():
    # Curved road with different height at both ends -> the
    # target height must be interpolated linearly along the centerline, not
    # constant.
    size = 20
    heights = np.zeros((size, size))
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road = _road(
        polygon_xy=[[0, 8], [20, 8], [20, 12], [0, 12]],
        centerline_xyz=[[0, 10, 0], [20, 10, 20]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road])

    assert np.isclose(result[10, 2], 2.0, atol=0.5)
    assert np.isclose(result[10, 18], 18.0, atol=0.5)
    assert result[10, 2] < result[10, 18]


# --- embed_roads_into_heightmap(clamp_to_max=True) - bridge abutments -----------------------------------


def test_clamp_to_max_lowers_terrain_above_the_deck():
    size = 20
    heights = np.full((size, size), 120.0)  # terrain protrudes completely above the deck (100)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road = _road(
        polygon_xy=[[5, 5], [15, 5], [15, 15], [5, 15]],
        centerline_xyz=[[5, 10, 100], [15, 10, 100]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road], clamp_to_max=True)

    assert np.isclose(result[10, 10], 100.0)  # clamped to deck level
    assert result[1, 1] == 120.0  # unchanged outside the bridge width


def test_clamp_to_max_leaves_terrain_below_the_deck_untouched():
    size = 20
    heights = np.full((size, size), 40.0)  # valley floor far BELOW the deck (100)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road = _road(
        polygon_xy=[[5, 5], [15, 5], [15, 15], [5, 15]],
        centerline_xyz=[[5, 10, 100], [15, 10, 100]],
    )

    result = embed_roads_into_heightmap(heights, origin_x, origin_y, square_size, [road], clamp_to_max=True)

    assert result[10, 10] == 40.0  # valley floor stays visible, not raised to deck level
    assert heights[10, 10] == 40.0  # original unchanged (function returns a copy)


def test_sample_heightmap_bilinear_matches_grid_points():
    heights = np.array([[0.0, 10.0], [20.0, 30.0]])
    # Exact grid points should return exact values
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = sample_heightmap_bilinear(heights, origin_x=0.0, origin_y=0.0, square_size=1.0, points_xy=points)
    assert np.allclose(result, [0.0, 10.0, 20.0, 30.0])

    # Midpoint should be the average of all 4 corners
    midpoint = np.array([[0.5, 0.5]])
    result_mid = sample_heightmap_bilinear(heights, 0.0, 0.0, 1.0, midpoint)
    assert np.isclose(result_mid[0], 15.0)


def test_build_road_embankment_profiles_straight_road():
    # Flat terrain at 100.0, road at 95.0 (5m cut) -> slope_width should reflect that
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # A straight road along x=20, from y=5 to y=35, road_z=95.0, width=6.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)
    poly = {"trimmed_centerline": centerline, "osm_tags": {"highway": "residential"}}

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    assert len(roads) == 1
    road = roads[0]
    # Road is at x=20, half_width=3.0 -> left edge at x=23, right edge at x=17
    assert np.allclose(road["left_edge_xyz"][:, 0], 23.0)
    assert np.allclose(road["right_edge_xyz"][:, 0], 17.0)
    # Natural height at edges should be 100.0 (flat terrain)
    assert np.allclose(road["left_natural_z"], 100.0)
    # height_diff = |100 - 95| = 5, tan(45deg) = 1 -> slope_width = 5.0
    assert np.allclose(road["left_slope_width"], 5.0)


def test_slope_width_override_replaces_the_computed_width_on_that_side_only():
    # Same setup as test_build_road_embankment_profiles_straight_road(): centerline along x=20,
    # direction +y. DEFAULT "left" (offset_points() convention: direction rotated +90 degrees) is at
    # x=17 here - that is (see docstring note) exactly "right_edge_xyz" of this function.
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}, "slope_width_override": {"left": 0.0}}
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    assert np.allclose(road["right_edge_xyz"][:, 0], 17.0)  # = DEFAULT "left", the overridden side
    assert np.allclose(road["right_slope_width"], 0.0)  # fixed width (0) instead of computed
    assert np.allclose(road["left_slope_width"], 5.0)  # other side unchanged, normal (computed)


def test_slope_width_override_can_set_both_sides_to_different_fixed_values():
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}, "slope_width_override": {"left": 0.0, "right": 5.0}}
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    assert np.allclose(road["right_slope_width"], 0.0)  # DEFAULT "left" -> this function "right"
    assert np.allclose(road["left_slope_width"], 5.0)  # DEFAULT "right" -> this function "left"


def test_slope_width_override_samples_natural_z_past_the_override_corridor_not_at_the_edge():
    """Regression: for a gallery, the DGM directly at the road edge does not show the natural
    terrain but the real valley-side structure (parapet/roof overhang) - empirically confirmed on two real
    galleries (Gotthard): height jump of 2.6-13.9m already 2m behind the edge. natural_z MUST
    therefore be sampled at the far end of the overridden corridor (edge + override width),
    otherwise the embankment "smooths" onto the raised structure height instead of descending toward the
    valley - visible as a remaining terrain spike instead of a downward slope."""
    size = 40
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    # Reconstructed DGM pattern of a real gallery: up to just behind the road edge (x=17, DEFAULT
    # "left") still the raised structure surface (95, ~road level), afterwards (x<=12, edge+5) the
    # much lower real terrain (50).
    heights = np.full((size, size), 95.0)
    heights[:, :13] = 50.0  # x < 13 -> "real terrain" beyond the corridor (edge at x=17, +5 -> x=12)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}, "slope_width_override": {"left": 5.0}}
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    # DEFAULT "left" override affects "right_*" of this function (see note at the top of the file).
    assert np.allclose(road["right_slope_width"], 5.0)
    # The edge itself (x=17) would still lie completely in the raised (95) area - without the fix,
    # right_natural_z would be sampled there and be ~95, not the real ~50 beyond the corridor.
    assert np.allclose(road["right_natural_z"], 50.0)


def test_slope_width_override_of_zero_keeps_sampling_natural_z_at_the_edge():
    """Override=0 (e.g. gallery mountain side) means "no embankment" - there is no corridor here that could
    distort natural_z, so the sampling stays at the edge (irrelevant anyway, since
    _blend_one_side touches nothing at width 0)."""
    size = 40
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)
    heights = np.full((size, size), 100.0)
    heights[:, 17] = 80.0  # value exactly at the edge (x=17)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}, "slope_width_override": {"left": 0.0}}
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    assert np.allclose(road["right_slope_width"], 0.0)
    assert np.allclose(road["right_natural_z"], 80.0)  # sampled unchanged at the edge


def test_flat_shoulder_side_stays_at_road_height_and_ignores_the_real_heightmap():
    """Gallery mountain side: 1 m flat shoulder at road height directly at the wall inner edge (no
    embankment angle, no interpolation to the terrain) - natural_z must equal the edge height itself,
    regardless of what is actually in the heightmap (deliberately chosen "unnatural" here)."""
    size = 40
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    # Deliberately NOT at 95 (road height): proves that the flat shoulder ignores the real heightmap
    # completely, instead of sampling it (as without flat_shoulder_sides) at the far end.
    heights = np.full((size, size), 40.0)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {
        "trimmed_centerline": centerline, "osm_tags": {},
        "slope_width_override": {"left": 1.0}, "flat_shoulder_sides": {"left"},
    }
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    assert np.allclose(road["right_slope_width"], 1.0)
    assert np.allclose(road["right_natural_z"], 95.0)  # = edge height (centerline Z), NOT 40 from the heightmap


def test_flat_shoulder_side_produces_a_constant_height_corridor_when_blended():
    """As above, but end-to-end via apply_embankment_blend() + embed_roads_into_heightmap() (same
    order as in terrain_workflow.py::process_tile()): the 1m corridor on the mountain side (x=16..17, beyond
    the road edge at x=17) must stay constant at road height (95), although the raw heightmap deliberately
    shows a different value (40) there - it must not show through."""
    size = 40
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    # Everywhere at road height (95) - the NOT overridden side (x=23) therefore sees diff=0 and stays
    # at min_slope_width (2m, not "running away"), only the tested flat-shoulder area (x=14..16, beyond
    # the 1m corridor at x=16..17) deliberately deviates, to prove that it is ignored.
    heights = np.full((size, size), 95.0)
    heights[:, 14:17] = 40.0

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {
        "trimmed_centerline": centerline, "osm_tags": {},
        "slope_width_override": {"left": 1.0}, "flat_shoulder_sides": {"left"},
        "road_polygon": np.array([[17.0, 5.0], [23.0, 5.0], [23.0, 35.0], [17.0, 35.0]]),
    }
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    blended = apply_embankment_blend(heights, origin_x, origin_y, square_size, roads)
    result = embed_roads_into_heightmap(blended, origin_x, origin_y, square_size, [poly])

    # DEFAULT "left" override affects the "right_xy" edge at x=17 (see docstring note) - the
    # 1m corridor extends to x=16. Every cell in it must be exactly 95 (road height), not 40.
    assert np.allclose(result[10:30, 17], 95.0)  # edge itself (set by embed)
    assert np.allclose(result[10:30, 16], 95.0)  # 1m corridor (overwritten by the flat shoulder)
    assert np.allclose(result[10:30, 14], 40.0)  # outside the corridor: unchanged raw terrain


def test_no_override_leaves_both_sides_normal():
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}}
    roads = build_road_embankment_profiles(
        [poly], heights, origin_x, origin_y, square_size, FakeMapper(),
        slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )

    road = roads[0]
    assert np.allclose(road["left_slope_width"], 5.0)
    assert np.allclose(road["right_slope_width"], 5.0)


def test_apply_embankment_blend_interpolates_correctly():
    size = 40
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # One "road" edge segment at x=20, y from 10 to 30, road_z=95, slope_width=5
    n = 21
    y_vals = np.linspace(10, 30, n)
    edge_xyz = np.column_stack([np.full(n, 20.0), y_vals, np.full(n, 95.0)])
    slope_width = np.full(n, 5.0)
    natural_z = np.full(n, 100.0)

    roads = [{
        "left_edge_xyz": edge_xyz,
        "right_edge_xyz": np.empty((0, 3)),
        "left_slope_width": slope_width,
        "right_slope_width": np.empty(0),
        "left_natural_z": natural_z,
        "right_natural_z": np.empty(0),
    }]

    result = apply_embankment_blend(heights, origin_x, origin_y, square_size, roads)

    # At the edge itself (x=20, dist~0): should be close to road height (95), not exactly
    # touched since dist>0 required, but very close cells should approach it
    near_edge_col = 21  # x=21, dist=1 from edge at x=20
    mid_row = 20
    val_near_edge = result[mid_row, near_edge_col]
    assert val_near_edge < 100.0  # pulled toward road height
    assert val_near_edge > 95.0   # not fully at road height (partway through blend)

    # Far beyond slope_width (x=30, dist=10 > slope_width=5): untouched
    far_col = 30
    assert result[mid_row, far_col] == 100.0

    # Original heights array must not be mutated
    assert heights[mid_row, near_edge_col] == 100.0


def test_apply_embankment_blend_handles_fill_and_cut():
    # Cut: road lower than terrain (tested above, terrain=100, road=95)
    # Fill: road higher than terrain
    size = 20
    heights = np.full((size, size), 50.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    n = 11
    y_vals = np.linspace(3, 13, n)
    edge_xyz = np.column_stack([np.full(n, 10.0), y_vals, np.full(n, 60.0)])  # road at 60, terrain at 50 -> fill
    slope_width = np.full(n, 3.0)
    natural_z = np.full(n, 50.0)

    roads = [{
        "left_edge_xyz": edge_xyz, "right_edge_xyz": np.empty((0, 3)),
        "left_slope_width": slope_width, "right_slope_width": np.empty(0),
        "left_natural_z": natural_z, "right_natural_z": np.empty(0),
    }]

    result = apply_embankment_blend(heights, origin_x, origin_y, square_size, roads)

    mid_row = 8
    # Near edge (x=11, dist=1): pulled UP toward road height (60), i.e. > natural 50
    near_val = result[mid_row, 11]
    assert near_val > 50.0
    assert near_val < 60.0


if __name__ == "__main__":
    test_embed_roads_sets_exact_road_height_only_near_road()
    print("[OK] test_embed_roads_sets_exact_road_height_only_near_road")
    test_embed_roads_can_raise_terrain_above_surroundings()
    print("[OK] test_embed_roads_can_raise_terrain_above_surroundings")
    test_embed_roads_follows_curved_centerline_height()
    print("[OK] test_embed_roads_follows_curved_centerline_height")
    test_sample_heightmap_bilinear_matches_grid_points()
    print("[OK] test_sample_heightmap_bilinear_matches_grid_points")
    test_build_road_embankment_profiles_straight_road()
    print("[OK] test_build_road_embankment_profiles_straight_road")
    test_apply_embankment_blend_interpolates_correctly()
    print("[OK] test_apply_embankment_blend_interpolates_correctly")
    test_apply_embankment_blend_handles_fill_and_cut()
    print("[OK] test_apply_embankment_blend_handles_fill_and_cut")
    print("Alle Tests bestanden.")


# --- Characterization: the optimization must not change the result ------------------------


def _diagonal_road(size=400, width=6.0):
    """Long, curved diagonal road: huge bounding box, but narrow strip."""
    from shapely.geometry import LineString

    t = np.linspace(20.0, size - 20.0, 120)
    xy = np.column_stack([t, 0.9 * t + 12.0 * np.sin(t / 25.0)])
    z = 100.0 + 0.05 * t + 2.0 * np.sin(t / 40.0)
    polygon = np.array(LineString(xy).buffer(width / 2.0, cap_style=2).exterior.coords[:-1])
    return _road(polygon, np.column_stack([xy, z]))


def _reference_embed(heights, origin_x, origin_y, square_size, road):
    """Brute-force reference: full bounding box, no prefiltering (old logic)."""
    from world_to_beamng.terrain.road_embedding import _points_in_polygon_2d, _project_onto_polyline

    result = heights.copy()
    polygon, centerline = road["road_polygon"], road["trimmed_centerline"]
    size_y, size_x = heights.shape
    col_start = max(0, int(np.floor((polygon[:, 0].min() - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((polygon[:, 0].max() - origin_x) / square_size)))
    row_start = max(0, int(np.floor((polygon[:, 1].min() - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((polygon[:, 1].max() - origin_y) / square_size)))
    gx, gy = np.meshgrid(
        origin_x + np.arange(col_start, col_end + 1) * square_size,
        origin_y + np.arange(row_start, row_end + 1) * square_size,
    )
    inside = _points_in_polygon_2d(gx, gy, polygon)
    target = _project_onto_polyline(gx, gy, centerline[:, 0], centerline[:, 1], centerline[:, 2])
    sub = result[row_start : row_end + 1, col_start : col_end + 1]
    sub[inside] = target[inside]
    return result


def test_embed_matches_brute_force_reference_on_long_diagonal_road():
    size = 400
    rng = np.random.RandomState(1)
    heights = 100.0 + rng.rand(size, size)
    road = _diagonal_road(size)

    result = embed_roads_into_heightmap(heights, 0.0, 0.0, 1.0, [road])
    reference = _reference_embed(heights, 0.0, 0.0, 1.0, road)

    assert np.count_nonzero(result != heights) > 500  # road was really embedded
    np.testing.assert_array_equal(result, reference)


def test_embed_with_subcell_resolution_and_offset_origin_matches_reference():
    rng = np.random.RandomState(2)
    heights = 50.0 + rng.rand(300, 300)
    road = _diagonal_road(150)

    result = embed_roads_into_heightmap(heights, 3.3, -2.7, 0.5, [road])
    reference = _reference_embed(heights, 3.3, -2.7, 0.5, road)

    np.testing.assert_array_equal(result, reference)


# --- Optimization: identical to the simple reference loop -----------------------------------


def _reference_project_onto_polyline(qx, qy, poly_x, poly_y, poly_z):
    """The original loop over all segments (reference)."""
    best_dist = np.full(qx.shape, np.inf)
    best_z = np.zeros(qx.shape)
    for i in range(len(poly_x) - 1):
        ax, ay, az = poly_x[i], poly_y[i], poly_z[i]
        bx, by, bz = poly_x[i + 1], poly_y[i + 1], poly_z[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-9:
            continue
        t = np.clip(((qx - ax) * dx + (qy - ay) * dy) / seg_len_sq, 0.0, 1.0)
        dist = np.hypot(qx - (ax + t * dx), qy - (ay + t * dy))
        z = az + t * (bz - az)
        better = dist < best_dist
        best_dist = np.where(better, dist, best_dist)
        best_z = np.where(better, z, best_z)
    return best_z


def _polylines():
    rng = np.random.RandomState(7)
    lines = {}
    t = np.linspace(0, 300, 200)
    lines["kurve"] = (t, 40 * np.sin(t / 30.0), 100 + 0.1 * t)
    # Hairpin: the line runs back and comes close to itself
    u = np.linspace(0, 1, 120)
    lines["haarnadel"] = (np.concatenate([u * 200, 200 - u * 200]), np.concatenate([np.zeros(120), np.full(120, 6.0)]), np.linspace(50, 80, 240))
    # duplicate points (zero-length segments) and few points
    lines["doppelte"] = (np.array([0, 0, 10, 10, 10, 30.0]), np.array([0, 0, 5, 5, 5, -4.0]), np.array([1, 2, 3, 4, 5, 6.0]))
    lines["zwei"] = (np.array([0.0, 50.0]), np.array([0.0, 20.0]), np.array([10.0, 30.0]))
    lines["zufall"] = (np.cumsum(rng.uniform(0.5, 4, 150)), np.cumsum(rng.uniform(-3, 3, 150)), rng.uniform(0, 100, 150))
    return lines


def test_project_onto_polyline_is_identical_to_the_reference_loop():
    from world_to_beamng.terrain.road_embedding import _project_onto_polyline

    rng = np.random.RandomState(3)
    for name, (px, py, pz) in _polylines().items():
        lo_x, hi_x, lo_y, hi_y = px.min() - 15, px.max() + 15, py.min() - 15, py.max() + 15
        qx = rng.uniform(lo_x, hi_x, 3000)
        qy = rng.uniform(lo_y, hi_y, 3000)
        # also points exactly on sample points and at equal distance to two segments
        n_on = min(5, len(px))
        qx[:n_on], qy[:n_on] = px[:n_on], py[:n_on]

        np.testing.assert_array_equal(
            _project_onto_polyline(qx, qy, px, py, pz), _reference_project_onto_polyline(qx, qy, px, py, pz), err_msg=name
        )


def test_project_onto_polyline_keeps_the_shape_and_handles_no_valid_segment():
    from world_to_beamng.terrain.road_embedding import _project_onto_polyline

    qx, qy = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0])
    same = np.array([5.0, 5.0, 5.0])
    np.testing.assert_array_equal(_project_onto_polyline(qx, qy, same, same, same), np.zeros(3))
    assert _project_onto_polyline(np.empty(0), np.empty(0), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 2.0])).shape == (0,)


def _reference_blend_one_side(heights, origin_x, origin_y, square_size, size_x, size_y, edge_xyz, slope_width, natural_z):
    """The original variant: query all cells of the bounding box (reference)."""
    from scipy.spatial import cKDTree

    if len(edge_xyz) == 0:
        return
    max_width = float(np.max(slope_width)) if len(slope_width) else 0.0
    if max_width <= 0:
        return
    min_x = float(np.min(edge_xyz[:, 0])) - max_width
    max_x = float(np.max(edge_xyz[:, 0])) + max_width
    min_y = float(np.min(edge_xyz[:, 1])) - max_width
    max_y = float(np.max(edge_xyz[:, 1])) + max_width
    col_start = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row_start = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))
    if col_start > col_end or row_start > row_end:
        return
    tree = cKDTree(edge_xyz[:, :2])
    gx, gy = np.meshgrid(origin_x + np.arange(col_start, col_end + 1) * square_size, origin_y + np.arange(row_start, row_end + 1) * square_size)
    query_points = np.column_stack([gx.ravel(), gy.ravel()])
    dist, idx = tree.query(query_points, distance_upper_bound=max_width + 1e-9)
    near = np.isfinite(dist)
    dist, idx = dist[near], idx[near]
    nearest_edge_z, nearest_slope_width, nearest_natural_z = edge_xyz[idx, 2], slope_width[idx], natural_z[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(nearest_slope_width > 0, dist / nearest_slope_width, 1.0)
    t = np.clip(t, 0.0, 1.0)
    blended = nearest_edge_z + (nearest_natural_z - nearest_edge_z) * t
    in_corridor = (dist > 0) & (dist <= nearest_slope_width)
    sub_shape = (row_end - row_start + 1, col_end - col_start + 1)
    sub = heights[row_start : row_end + 1, col_start : col_end + 1].reshape(-1)
    sub[np.flatnonzero(near)[in_corridor]] = blended[in_corridor]
    heights[row_start : row_end + 1, col_start : col_end + 1] = sub.reshape(sub_shape)


def test_blend_one_side_is_identical_to_the_full_bounding_box_reference():
    from world_to_beamng.terrain.road_embedding import _blend_one_side

    rng = np.random.RandomState(11)
    size = 500
    heights = 200.0 + rng.rand(size, size) * 5
    cases = []
    t = np.linspace(20, 470, 300)
    diagonal = np.column_stack([t, 0.9 * t + 12 * np.sin(t / 25.0), 100 + 0.05 * t])  # long diagonal
    cases.append((diagonal, rng.uniform(1.0, 9.0, len(t)), 200 + rng.rand(len(t))))
    short = np.column_stack([np.linspace(100, 130, 30), np.linspace(50, 60, 30), np.full(30, 210.0)])
    cases.append((short, np.full(30, 4.0), np.full(30, 205.0)))
    cases.append((diagonal[::-1].copy(), np.zeros(300), np.zeros(300)))  # embankment width 0
    cases.append((np.array([[-50.0, -50.0, 1.0], [-40.0, -45.0, 1.0]]), np.array([5.0, 5.0]), np.array([2.0, 2.0])))  # outside

    for origin_x, origin_y, square in ((0.0, 0.0, 1.0), (3.3, -2.7, 0.5), (-10.0, 5.0, 2.0)):
        for edge, width, natural in cases:
            expected, actual = heights.copy(), heights.copy()
            _reference_blend_one_side(expected, origin_x, origin_y, square, size, size, edge, width, natural)
            _blend_one_side(actual, origin_x, origin_y, square, size, size, edge, width, natural)
            np.testing.assert_array_equal(actual, expected)


def test_slope_width_override_may_vary_per_centerline_point():
    # Gallery valley side: the embankment width is searched per station toward the valley up to behind the roof
    size = 40
    centerline = np.array([[20.0, y, 95.0] for y in range(5, 36)], dtype=float)
    widths = np.linspace(4.0, 8.0, len(centerline))
    heights = np.tile(np.arange(size, dtype=float), (size, 1))  # height = x: natural_z reveals the sampling position

    class FakeMapper:
        def get_road_properties(self, tags):
            return {"width": 6.0}

    poly = {"trimmed_centerline": centerline, "osm_tags": {}, "slope_width_override": {"left": widths}}
    road = build_road_embankment_profiles(
        [poly], heights, 0.0, 0.0, 1.0, FakeMapper(), slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0,
    )[0]

    # DEFAULT "left" = -x (direction of travel +y) and affects "right_*" of this function
    assert np.allclose(road["right_slope_width"], widths)
    assert np.allclose(road["right_natural_z"], 20.0 - 3.0 - widths)


def _blend_straight_road(cuts=None):
    """Road along x 0..20 at y=20 (6 m wide, z=95) on flat terrain at 100, embankment 5 m on both sides."""
    heights = np.full((40, 40), 100.0)
    centerline = np.column_stack([np.linspace(0.0, 20.0, 21), np.full(21, 20.0), np.full(21, 95.0)])
    road = {"trimmed_centerline": centerline, "osm_tags": {"width": "6"}}
    if cuts is not None:
        road["embankment_cuts"] = cuts
    from world_to_beamng import config

    profiles = build_road_embankment_profiles([road], heights, 0.0, 0.0, 1.0, config.OSM_MAPPER, 45.0, 5.0, max_slope_width=5.0)
    return apply_embankment_blend(heights, 0.0, 0.0, 1.0, profiles)


def test_embankment_extends_past_an_uncut_road_end():
    result = _blend_straight_road()

    assert result[24, 22] < 100.0  # x=22 beyond the end, beside the edge: round cap of the last edge point


def test_embankment_cut_ends_flush_at_the_cut_line():
    # A cut (point, outward normal) at the road end: nothing beyond the line is touched, the rest blends as before
    result = _blend_straight_road(cuts=[((20.0, 20.0), (1.0, 0.0))])

    assert result[24, 22] == 100.0 and result[25, 21] == 100.0
    assert result[24, 19] < 100.0  # still blended in front of the line
    assert result[24, 0] < 100.0  # the uncut start is blended as before


def test_near_deck_mask_marks_only_terrain_close_below_the_bridge_deck():
    from world_to_beamng.terrain.road_embedding import near_deck_mask

    # Hillside bridge along x at y=20, deck at z=100: terrain z = 90 + 0.5 * y rises across the (wide) footprint
    heights = np.tile((90.0 + 0.5 * np.arange(41.0))[:, None], (1, 60))
    bridge = _road([(10, 10), (50, 10), (50, 30), (10, 30)], [(10, 20, 100.0), (50, 20, 100.0)])

    mask = near_deck_mask(heights, 0.0, 0.0, 1.0, [bridge], clearance=2.5)

    assert mask[20, 30] and mask[26, 30]  # at / above deck level
    assert mask[16, 30]  # 2 m below the deck: grass would still reach through
    assert not mask[14, 30] and not mask[11, 30]  # 3 m / 4.5 m below: grass stays
    assert not mask[20, 55] and not mask[35, 30]  # outside the footprint


# --- roads under a bridge: 45 degree slopes cut to the height where the terrain is met ---------------------------------------
class _Mapper6m:
    def get_road_properties(self, tags):
        return {"width": 6.0}


def _underpass_terrain():
    """Terrain at 90 except beside the road: a wall (deck level 96) starts 1 m from each edge (3 m from the centerline y=20)."""
    heights = np.full((60, 60), 90.0)
    rows = np.arange(60)
    wall = np.abs(rows - 20) >= 4  # the road polygon covers |y-20| <= 3, the wall starts at y-20 = 4
    heights[wall, :] = 96.0
    return heights


def _profiles(daylight, heights=None):
    heights = _underpass_terrain() if heights is None else heights
    centerline = np.array([[x, 20.0, 90.0] for x in range(10, 51)], dtype=float)
    poly = {"trimmed_centerline": centerline, "osm_tags": {"highway": "service"}, "daylight_slopes": daylight}
    return heights, build_road_embankment_profiles(
        [poly], heights, 0.0, 0.0, 1.0, _Mapper6m(), slope_angle_deg=45.0, min_slope_width=2.0, max_slope_width=30.0
    )


def test_default_embankment_width_follows_the_terrain_at_the_edge_only():
    _, roads = _profiles(daylight=False)

    # at the edge (y = 23 / 17) the terrain is still 90 m: the width falls to the minimum - the wall beside stays steep
    assert np.allclose(roads[0]["left_slope_width"], 2.0) and np.allclose(roads[0]["right_slope_width"], 2.0)


def test_daylight_slopes_meet_the_terrain_at_45_degrees_on_both_sides():
    heights, roads = _profiles(daylight=True)

    assert np.allclose(roads[0]["left_slope_width"], 6.0, atol=0.5) and np.allclose(roads[0]["right_slope_width"], 6.0, atol=0.5)
    blended = apply_embankment_blend(heights, 0.0, 0.0, 1.0, roads)
    for distance in range(1, 6):
        assert blended[23 + distance, 30] == pytest.approx(90.0 + distance, abs=0.75)  # 45 degrees up to the wall height
        assert blended[17 - distance, 30] == pytest.approx(90.0 + distance, abs=0.75)


def test_daylight_slope_stays_minimal_where_the_terrain_is_already_at_road_level():
    heights, roads = _profiles(daylight=True, heights=np.full((60, 60), 90.0))

    assert np.allclose(roads[0]["left_slope_width"], 2.0) and np.allclose(roads[0]["right_slope_width"], 2.0)


class _Mapper2m:
    def get_road_properties(self, tags):
        return {"width": 2.0}


def test_slope_corridors_of_an_underpass_do_not_reach_across_a_narrow_road():
    # 2 m wide road at y=20 (edges y=19 / y=21). South wall 96 m starts at y<=18 (6 m of slope), north wall 93 m at y>=22 (3 m).
    heights = np.full((60, 60), 90.0)
    heights[:19, :] = 96.0
    heights[22:, :] = 93.0
    centerline = np.array([[x, 20.0, 90.0] for x in range(10, 51)], dtype=float)
    poly = {"trimmed_centerline": centerline, "osm_tags": {"highway": "service"}, "daylight_slopes": True}

    roads = build_road_embankment_profiles([poly], heights, 0.0, 0.0, 1.0, _Mapper2m(), slope_angle_deg=45.0,
                                           min_slope_width=2.0, max_slope_width=30.0)
    blended = apply_embankment_blend(heights, 0.0, 0.0, 1.0, roads)

    for distance in (1, 2, 3):  # north side: 45 degrees up to its 3 m wall
        assert blended[21 + distance, 30] == pytest.approx(90.0 + distance, abs=0.8)
    assert blended[25, 30] == 93.0 and blended[26, 30] == 93.0  # beyond its slope: natural - not raised by the south corridor
    for distance in (1, 2, 3, 4, 5):  # south side: 45 degrees up to its 6 m wall
        assert blended[19 - distance, 30] == pytest.approx(90.0 + distance, abs=0.8)
