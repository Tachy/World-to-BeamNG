"""
Tests: DGM30 is clipped to the horizon area (whole 1° tiles would otherwise make the horizon too large) and
multiple files are combined; missing coverage is reported.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.terrain import horizon
from world_to_beamng.terrain.horizon import clip_dgm30_to_area, load_dgm30_tiles

OFFSET = (401000.0, 5298000.0, 0.0)
HALF = 3000.0
AREA = (OFFSET[0] - HALF, OFFSET[0] + HALF, OFFSET[1] - HALF, OFFSET[1] + HALF)
CENTER_LON, CENTER_LAT = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True).transform(OFFSET[0], OFFSET[1])


@pytest.fixture(autouse=True)
def _isolate_cache_dir(tmp_path, monkeypatch):
    """load_dgm30_tiles() now writes a per-tile cache even without tile_hash (see
    horizon._cached_geotiff_as_xyz()) - without isolation, tests would pollute the real cache/ directory
    of this repo."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


def _grid(half=4000.0, step=200.0):
    xs = np.arange(-half, half + step, step)
    x, y = np.meshgrid(xs, xs)
    return np.column_stack([x.ravel(), y.ravel()]), np.full(x.size, 500.0)


def _tile(path, lon_min, lon_max):
    """DGM30 file in WGS84 (like the Copernicus DEM), elevation = 100 m + 1000 m per degree of longitude."""
    bounds = (lon_min, CENTER_LAT - 0.04, lon_max, CENTER_LAT + 0.04)
    cols, rows = 60, 80
    lon = np.linspace(lon_min, lon_max, cols)[None, :].repeat(rows, axis=0)
    with rasterio.open(
        path, "w", driver="GTiff", width=cols, height=rows, count=1, dtype="float32", crs="EPSG:4326", transform=from_bounds(*bounds, cols, rows)
    ) as dst:
        dst.write((100.0 + 1000.0 * (lon - CENTER_LON)).astype("float32"), 1)
    return path


# ---------------------------------------------------------------- Clipping


def test_points_outside_the_area_are_dropped():
    points, elevations = _grid(half=6000.0)

    clipped, heights, missing = clip_dgm30_to_area(points, elevations, AREA, local_offset=OFFSET)

    assert len(heights) == len(clipped) < len(points)
    assert np.abs(clipped).max() <= HALF
    assert missing == []


def test_utm_points_are_clipped_when_no_offset_is_given():
    points, elevations = _grid(half=6000.0)
    utm = points + np.array(OFFSET[:2])

    clipped, _, _ = clip_dgm30_to_area(utm, elevations, AREA)

    assert np.abs(clipped - np.array(OFFSET[:2])).max() <= HALF


@pytest.mark.parametrize(
    "keep, expected",
    [
        (lambda p: p[:, 0] < 0, ["east"]),
        (lambda p: p[:, 0] > 0, ["west"]),
        (lambda p: p[:, 1] < 0, ["north"]),
        (lambda p: p[:, 1] > 0, ["south"]),
        (lambda p: p[:, 0] >= -10000, []),
    ],
)
def test_missing_sides_are_reported(keep, expected):
    points, elevations = _grid()
    mask = keep(points)

    _, _, missing = clip_dgm30_to_area(points[mask], elevations[mask], AREA, local_offset=OFFSET)

    assert missing == expected


def test_a_small_gap_at_the_edge_is_not_reported():
    points, elevations = _grid(half=HALF - 400.0)  # ends 400 m before the edge: tile edge tolerance

    assert clip_dgm30_to_area(points, elevations, AREA, local_offset=OFFSET)[2] == []


def test_data_completely_outside_gives_an_empty_result():
    points, elevations = _grid(half=1000.0)
    far = points + 50000.0

    clipped, heights, missing = clip_dgm30_to_area(far, elevations, AREA, local_offset=OFFSET)

    assert len(heights) == 0 and missing == []


# ---------------------------------------------------------------- Loading from files


def test_several_files_are_combined_and_clipped(tmp_path):
    _tile(tmp_path / "west.tif", CENTER_LON - 0.06, CENTER_LON)
    _tile(tmp_path / "east.tif", CENTER_LON, CENTER_LON + 0.06)

    points, elevations = load_dgm30_tiles(tmp_path, AREA, local_offset=OFFSET)

    assert np.abs(points).max() <= HALF  # 4.5 km wide tiles, clipped to ±3 km
    assert points[:, 0].min() < -2000 and points[:, 0].max() > 2000  # both files contribute
    assert elevations.min() < 100 < elevations.max()  # elevations from both halves (linear in longitude)


def test_one_file_alone_leaves_the_other_half_uncovered(tmp_path):
    _tile(tmp_path / "west.tif", CENTER_LON - 0.06, CENTER_LON)

    points, _ = load_dgm30_tiles(tmp_path, AREA, local_offset=OFFSET)

    assert points[:, 0].max() < 500  # east is missing: the horizon ends earlier there
    assert clip_dgm30_to_area(points, np.zeros(len(points)), AREA, OFFSET)[2] == ["east"]


def test_files_far_from_the_area_give_nothing(tmp_path):
    _tile(tmp_path / "far.tif", CENTER_LON + 1.0, CENTER_LON + 1.1)

    assert load_dgm30_tiles(tmp_path, AREA, local_offset=OFFSET) == (None, None)


def test_an_empty_folder_gives_nothing(tmp_path):
    assert load_dgm30_tiles(tmp_path, AREA, local_offset=OFFSET) == (None, None)


# ---------------------------------------------------------------- Cache


def test_cache_is_not_reused_after_more_tiles_are_added(tmp_path):
    tiles = tmp_path / "dgm30"
    tiles.mkdir()
    _tile(tiles / "west.tif", CENTER_LON - 0.06, CENTER_LON)

    first, _ = load_dgm30_tiles(tiles, AREA, local_offset=OFFSET, tile_hash="abc")
    again, _ = load_dgm30_tiles(tiles, AREA, local_offset=OFFSET, tile_hash="abc")
    _tile(tiles / "east.tif", CENTER_LON, CENTER_LON + 0.06)
    combined, _ = load_dgm30_tiles(tiles, AREA, local_offset=OFFSET, tile_hash="abc")

    assert len(list((tmp_path / "cache").glob("dgm30_horizon_abc_*.npz"))) == 2  # one cache per file set
    assert np.array_equal(first, again)
    assert combined[:, 0].max() > first[:, 0].max() + 1000  # the new tile counts immediately


def test_per_tile_conversion_is_reused_across_a_core_area_switch(tmp_path):
    """Regression: changing the core area (different tile_hash, e.g. between two test regions
    like BaWue and Switzerland) must not repeat the expensive GeoTIFF conversion (reading + reprojecting +
    200m grid downsampling) of an unchanged DGM30 tile - only the subsequent combination/clipping
    for the respective area depends on tile_hash (see
    horizon._cached_geotiff_as_xyz(), independent of _dgm30_cache_file())."""
    tiles = tmp_path / "dgm30"
    tiles.mkdir()
    _tile(tiles / "west.tif", CENTER_LON - 0.06, CENTER_LON)

    with patch("world_to_beamng.terrain.horizon._load_geotiff_as_xyz", wraps=horizon._load_geotiff_as_xyz) as spy:
        load_dgm30_tiles(tiles, AREA, local_offset=OFFSET, tile_hash="region-bw")
        load_dgm30_tiles(tiles, AREA, local_offset=OFFSET, tile_hash="region-ch")

    assert spy.call_count == 1  # the tile was read only once despite the area change (different tile_hash)
    assert len(list((tmp_path / "cache").glob("dgm30_tile_*.npz"))) == 1
    assert len(list((tmp_path / "cache").glob("dgm30_horizon_*.npz"))) == 2  # one combined cache per core area


def test_per_tile_cache_survives_process_restart(tmp_path):
    """The per-tile cache is a file under cache/, not in-memory state - a second,
    completely independent call (simulates a new process/run) must hit it just the same."""
    tiles = tmp_path / "dgm30"
    tiles.mkdir()
    tif_file = _tile(tiles / "west.tif", CENTER_LON - 0.06, CENTER_LON)

    points_a, elevations_a = horizon._cached_geotiff_as_xyz(tif_file)
    with patch("world_to_beamng.terrain.horizon._load_geotiff_as_xyz") as mock_load:
        points_b, elevations_b = horizon._cached_geotiff_as_xyz(tif_file)

    mock_load.assert_not_called()  # second "process" never calls the expensive original function
    assert np.array_equal(points_a, points_b)
    assert np.array_equal(elevations_a, elevations_b)
