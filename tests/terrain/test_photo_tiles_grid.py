"""
Tests for terrain.photo_tiles.build_processing_tile_grid() - fixed photo/material tile grid
over a total area, independent of the size/number of the raw elevation data tiles.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.terrain.photo_tiles import build_processing_tile_grid, photo_tile_specs


def test_exact_multiple_of_tile_size_gives_uniform_tiles():
    # 4x4 km area, 2 km tiles -> exactly 2x2 = 4 equally sized tiles (LGL-BW case)
    bbox = (399000.0, 403000.0, 5296000.0, 5300000.0)

    tiles = build_processing_tile_grid(bbox, tile_size_m=2000.0)

    assert len(tiles) == 4
    assert {t["bbox_utm"] for t in tiles} == {
        (399000.0, 401000.0, 5296000.0, 5298000.0),
        (401000.0, 403000.0, 5296000.0, 5298000.0),
        (399000.0, 401000.0, 5298000.0, 5300000.0),
        (401000.0, 403000.0, 5298000.0, 5300000.0),
    }


def test_result_is_independent_of_the_number_of_source_tiles():
    # 16 loose 1 km source tiles (like the Swiss test data) yield the same 2 km grid as
    # 4 raw 2 km tiles, because build_processing_tile_grid() only knows the TOTAL BBox
    bbox_from_16_source_tiles = (2685000.0, 2689000.0, 1153000.0, 1157000.0)

    tiles = build_processing_tile_grid(bbox_from_16_source_tiles, tile_size_m=2000.0)

    assert len(tiles) == 4
    widths = {round(t["bbox_utm"][1] - t["bbox_utm"][0], 3) for t in tiles}
    heights = {round(t["bbox_utm"][3] - t["bbox_utm"][2], 3) for t in tiles}
    assert widths == {2000.0}
    assert heights == {2000.0}


def test_last_row_and_column_are_clamped_to_the_bbox_edge_not_oversized():
    # 3999m width, 2000m tiles -> second column is clamped to 1999m instead of 2000m
    bbox = (0.0, 3999.0, 0.0, 2000.0)

    tiles = build_processing_tile_grid(bbox, tile_size_m=2000.0)

    xs = sorted(t["bbox_utm"][:2] for t in tiles)
    assert xs == [(0.0, 2000.0), (2000.0, 3999.0)]  # last tile 1999m instead of 2000m


def test_area_smaller_than_one_tile_yields_a_single_tile():
    bbox = (0.0, 500.0, 0.0, 500.0)

    tiles = build_processing_tile_grid(bbox, tile_size_m=2000.0)

    assert len(tiles) == 1
    assert tiles[0]["bbox_utm"] == (0.0, 500.0, 0.0, 500.0)


def test_non_positive_tile_size_raises():
    with pytest.raises(ValueError):
        build_processing_tile_grid((0.0, 100.0, 0.0, 100.0), tile_size_m=0.0)


def test_output_is_compatible_with_photo_tile_specs():
    bbox = (399000.0, 403000.0, 5296000.0, 5300000.0)
    tiles = build_processing_tile_grid(bbox, tile_size_m=2000.0)

    specs = photo_tile_specs(tiles, global_offset=(401000.0, 5298000.0))

    assert len(specs) == 4
    assert all(len(s["bounds"]) == 4 for s in specs)
