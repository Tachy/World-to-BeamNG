"""
Tests: Sentinel-2 auto-download (EOX WMS mosaic -> horizon texture).
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from PIL import Image
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.terrain import sentinel2_fetch
from world_to_beamng.terrain.sentinel2_fetch import (
    _mercator_bbox_for_area,
    _mosaic_pixel_size,
    ensure_horizon_texture,
    fetch_eox_mosaic,
    tile_grid,
)

AREA_UTM = (400000.0, 401000.0, 5300000.0, 5301000.0)  # 1x1 km in EPSG:25832


@pytest.fixture(autouse=True)
def _isolate_caches(tmp_path, monkeypatch):
    """ALWAYS redirect both EOX caches (raw mosaic + finished texture) to a fresh tmp_path -
    otherwise tests accidentally write/read the real cache/ directory of this repo, and
    tests with the same AREA_UTM+size_px combination would influence each other via the texture cache
    (the cache key does not depend on `dest`, see ensure_horizon_texture())."""
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", tmp_path / "cache_horizon_source")
    monkeypatch.setattr(config, "EOX_TEXTURE_CACHE_DIR", tmp_path / "cache_horizon_texture")


# --- _mercator_bbox_for_area() ------------------------------------------------------------------


def test_mercator_bbox_matches_independent_pyproj_transform():
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:3857", always_xy=True)
    x_min, x_max, y_min, y_max = AREA_UTM
    # UTM -> Web Mercator is not axis-aligned (slight shear/rotation) - the BBox is therefore
    # min/max over ALL FOUR corners, not just over the lower-left/upper-right corner.
    corners_x, corners_y = [], []
    for cx, cy in [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]:
        tx, ty = transformer.transform(cx, cy)
        corners_x.append(tx)
        corners_y.append(ty)
    left, right = min(corners_x), max(corners_x)
    bottom, top = min(corners_y), max(corners_y)

    minx, miny, maxx, maxy = _mercator_bbox_for_area(AREA_UTM)

    # The margin factor widens the result around the center - choose the tolerance accordingly large.
    margin = config.EOX_FETCH_MARGIN_FACTOR - 1.0
    width = right - left
    height = top - bottom
    assert minx == pytest.approx(left - width / 2 * margin, abs=1.0)
    assert maxx == pytest.approx(right + width / 2 * margin, abs=1.0)
    assert miny == pytest.approx(bottom - height / 2 * margin, abs=1.0)
    assert maxy == pytest.approx(top + height / 2 * margin, abs=1.0)


def test_mercator_bbox_is_wider_than_the_unmargined_transform():
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:3857", always_xy=True)
    x_min, x_max, y_min, y_max = AREA_UTM
    left, bottom = transformer.transform(x_min, y_min)
    right, top = transformer.transform(x_max, y_max)

    minx, miny, maxx, maxy = _mercator_bbox_for_area(AREA_UTM)

    assert minx < left
    assert maxx > right
    assert miny < bottom
    assert maxy > top


# --- _mosaic_pixel_size() ------------------------------------------------------------------------


def test_mosaic_pixel_size_uses_target_resolution():
    bbox = (0.0, 0.0, 1000.0, 2000.0)
    w, h = _mosaic_pixel_size(bbox)

    assert w == round(1000.0 / config.EOX_TARGET_RESOLUTION_M)
    assert h == round(2000.0 / config.EOX_TARGET_RESOLUTION_M)


def test_mosaic_pixel_size_is_capped_at_mosaic_max_px(monkeypatch):
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    bbox = (0.0, 0.0, 1_000_000.0, 1_000_000.0)

    w, h = _mosaic_pixel_size(bbox)

    assert w == 100
    assert h == 100


# --- tile_grid() -----------------------------------------------------------------------------


def test_tile_grid_covers_the_full_raster_without_gap_or_overlap(monkeypatch):
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 2000)
    w, h = 5000, 3000
    bbox = (0.0, 0.0, w * 10.0, h * 10.0)  # 10 m/px, arbitrary for this test

    tiles = tile_grid(bbox, w, h)

    # Area fully covered, no overlap.
    assert sum(t.width * t.height for t in tiles) == w * h

    # Mark the pixel grid as occupied -> every pixel hit exactly once.
    covered = np.zeros((h, w), dtype=bool)
    for t in tiles:
        region = covered[t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width]
        assert not region.any(), "tile overlaps an already occupied region"
        covered[t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width] = True
    assert covered.all()

    # Edge tiles are smaller (5000 % 2000 = 1000, 3000 % 2000 = 1000).
    col_offs = sorted({t.col_off for t in tiles})
    row_offs = sorted({t.row_off for t in tiles})
    assert col_offs == [0, 2000, 4000]
    assert row_offs == [0, 2000]
    last_col_tiles = [t for t in tiles if t.col_off == 4000]
    assert all(t.width == 1000 for t in last_col_tiles)
    last_row_tiles = [t for t in tiles if t.row_off == 2000]
    assert all(t.height == 1000 for t in last_row_tiles)


def test_tile_grid_bbox_north_is_row_zero():
    # Row 0 = top edge of the image = geographic north = maxy.
    bbox = (0.0, 0.0, 100.0, 100.0)
    tiles = tile_grid(bbox, 10, 10)

    top_left = next(t for t in tiles if t.col_off == 0 and t.row_off == 0)
    assert top_left.bbox[3] == pytest.approx(100.0)  # maxy of the topmost tile = maxy of the mosaic


# --- fetch_eox_mosaic() -----------------------------------------------------------------------


def _jpeg_response(width, height, color=(100, 150, 200)):
    image = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "image/jpeg"}
    response.content = buf.getvalue()
    return response


def _error_response(status_code=500):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"Content-Type": "text/xml"}
    response.content = b"<ServiceException>boom</ServiceException>"
    return response


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_fetch_mosaic_success_writes_correct_size_and_crs(mock_get, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # a single tile

    def fake_get(url, params=None, headers=None, timeout=None):
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    dest = tmp_path / "mosaic.tif"

    success, failed_count = fetch_eox_mosaic(AREA_UTM, dest)

    assert success is True
    assert failed_count == 0
    assert dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
    with rasterio.open(dest) as ds:
        assert ds.crs.to_string() == "EPSG:3857"
        assert ds.width <= 50
        assert ds.height <= 50
        assert ds.count == 3


@patch("world_to_beamng.terrain.sentinel2_fetch.time.sleep")  # skip the backoff in the test
@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_fetch_mosaic_partial_failure_still_returns_true_failed_tile_is_black(mock_get, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # -> 2x2 tiles
    monkeypatch.setattr(config, "EOX_FETCH_MAX_RETRIES", 2)

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        call_count["n"] += 1
        # The very first requested tile (col_off=0,row_off=0) fails on every attempt,
        # all others (and all retries) deliver an image.
        if call_count["n"] <= config.EOX_FETCH_MAX_RETRIES:
            return _error_response(500)
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    dest = tmp_path / "mosaic.tif"

    success, failed_count = fetch_eox_mosaic(AREA_UTM, dest)

    assert success is True
    assert failed_count == 1
    assert dest.exists()
    with rasterio.open(dest) as ds:
        arr = ds.read()
        # Tile (0,0) stayed black (0), other tiles have the test color (100,150,200).
        assert arr[:, 0, 0].tolist() == [0, 0, 0]
        assert arr[0, -1, -1] != 0 or arr[1, -1, -1] != 0 or arr[2, -1, -1] != 0


@patch("world_to_beamng.terrain.sentinel2_fetch.time.sleep")
@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_fetch_mosaic_total_failure_returns_false_no_file_created(mock_get, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)

    mock_get.side_effect = lambda *a, **kw: _error_response(503)
    dest = tmp_path / "mosaic.tif"

    success, failed_count = fetch_eox_mosaic(AREA_UTM, dest)

    assert success is False
    assert failed_count == 1
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_fetch_mosaic_tile_with_wrong_decoded_size_stays_black_does_not_abort_mosaic(mock_get, tmp_path, monkeypatch):
    # Server responds with status 200 + Content-Type image/* (so it passes the check in
    # _fetch_one_tile()), but the decoded image does NOT have the requested pixel size - a
    # realistic failure mode of an external WMS server. This must not let dst.write() abort with an
    # exception out of the `with rasterio.open(...)` block.
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # -> 2x2 tiles

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        # Only the very first requested tile (col_off=0,row_off=0) delivers the wrong
        # pixel size, all others (and all retries, if any took place) the requested one.
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _jpeg_response(int(params["width"]) - 1, int(params["height"]))  # wrong size
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    dest = tmp_path / "mosaic.tif"

    success, failed_count = fetch_eox_mosaic(AREA_UTM, dest)

    assert success is True  # at least one (the remaining 3) tiles were successful
    assert failed_count == 1
    assert dest.exists()
    with rasterio.open(dest) as ds:
        arr = ds.read()
        # The tile with the wrong size stayed black (0,0,0 at its origin pixel).
        assert arr[:, 0, 0].tolist() == [0, 0, 0]
        # Another (correctly answered) tile wrote the test color.
        assert arr[0, -1, -1] != 0 or arr[1, -1, -1] != 0 or arr[2, -1, -1] != 0


# --- ensure_horizon_texture() -------------------------------------------------------------------


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_end_to_end_creates_valid_geotiff(mock_get, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 60)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 60)

    def fake_get(url, params=None, headers=None, timeout=None):
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    size_px = 32

    result = ensure_horizon_texture(AREA_UTM, size_px=size_px)

    assert result.parent == config.EOX_TEXTURE_CACHE_DIR
    with rasterio.open(result) as ds:
        assert ds.width == size_px
        assert ds.height == size_px
        assert ds.count == 3


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_returns_none_when_auto_download_disabled(mock_get, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", False)

    result = ensure_horizon_texture(AREA_UTM, size_px=32)

    assert result is None
    mock_get.assert_not_called()


@patch("world_to_beamng.terrain.sentinel2_fetch.fetch_eox_mosaic")
def test_ensure_horizon_texture_never_raises_on_unexpected_error(mock_fetch, monkeypatch):
    # Analogous to dgm30_fetch.test_ensure_coverage_never_raises_on_unexpected_error(): an
    # unexpected error (here: fetch_eox_mosaic() raises instead of returning False, e.g. because a
    # file system/network error was not caught cleanly) must never propagate out of
    # ensure_horizon_texture() - horizon_workflow.py relies on being able to call this
    # entry point without its own error handling.
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    mock_fetch.side_effect = OSError("permission denied")

    result = ensure_horizon_texture(AREA_UTM, size_px=32)

    assert result is None


# --- Item 1: partial success is not cached permanently -----------------------------------------


@patch("world_to_beamng.terrain.sentinel2_fetch.time.sleep")
@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_partial_failure_still_builds_texture_this_run(mock_get, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # -> 2x2 tiles
    monkeypatch.setattr(config, "EOX_FETCH_MAX_RETRIES", 2)

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        call_count["n"] += 1
        # The very first requested tile fails on every attempt (network hiccup),
        # all others deliver an image -> partial success.
        if call_count["n"] <= config.EOX_FETCH_MAX_RETRIES:
            return _error_response(500)
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get

    result = ensure_horizon_texture(AREA_UTM, size_px=32)

    # Despite the partial success, a usable texture is still generated for THIS run - under a
    # unique "_partial" name, NOT under the canonical cache name (see next test).
    assert result is not None
    assert result.name.endswith("_partial.tif")
    with rasterio.open(result) as ds:
        assert ds.width == 32
        assert ds.height == 32


@patch("world_to_beamng.terrain.sentinel2_fetch.time.sleep")
@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_partial_failure_does_not_cache_mosaic_for_next_run(mock_get, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # -> 2x2 tiles
    monkeypatch.setattr(config, "EOX_FETCH_MAX_RETRIES", 2)

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        call_count["n"] += 1
        if call_count["n"] <= config.EOX_FETCH_MAX_RETRIES:
            return _error_response(500)
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get

    ensure_horizon_texture(AREA_UTM, size_px=32)

    # Leave neither raw mosaic nor texture under their canonical cache name (regardless of whether
    # EOX_KEEP_RAW_MOSAIC is True or False) - otherwise the next run would treat the partial success as
    # "complete" forever. Only the uniquely named "_partial.tif" (from the
    # previous test) may be there.
    assert list(config.EOX_MOSAIC_CACHE_DIR.glob("*.tif")) == []
    assert [p.name for p in config.EOX_TEXTURE_CACHE_DIR.glob("*.tif")] == [
        p.name for p in config.EOX_TEXTURE_CACHE_DIR.glob("*_partial.tif")
    ]

    # No manual deletion needed anymore to force a retry (unlike earlier, when the
    # texture lay under the fixed `dest` file): neither mosaic nor texture cache has a
    # canonical hit for this area, so the next call queries the network again
    # automatically.
    call_count["n"] = 0
    mock_get.side_effect = fake_get  # fresh, so the counter starts from the beginning again

    result = ensure_horizon_texture(AREA_UTM, size_px=32)

    assert result is not None
    assert result.exists()
    assert call_count["n"] > 0  # network was actually queried again, not skipped


# --- Area change: no mixing/reuse of the wrong texture -----------------------------------------


AREA_UTM_OTHER_REGION = (2600000.0, 2601000.0, 1200000.0, 1201000.0)  # e.g. Switzerland instead of LGL


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_switching_area_gets_its_own_texture_then_switching_back_reuses_the_original(mock_get, tmp_path, monkeypatch):
    """Regression for the bug that tied data/DOP300/horizon_temp.tif to ONE area: the
    automatically generated texture is now tied to the area, no longer to a fixed file name -
    switching the area (e.g. Switzerland instead of LGL for testing) and back must neither
    lose the old area nor mix it with the new one."""
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        call_count["n"] += 1
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get

    # 1) LGL area: first call loads fresh.
    result_lgl_1 = ensure_horizon_texture(AREA_UTM, size_px=32)
    calls_after_lgl = call_count["n"]
    assert calls_after_lgl > 0

    # 2) Switch to a different area (e.g. Switzerland): its own new cache file, no conflict
    #    with the LGL texture from above - both exist simultaneously afterwards.
    result_other = ensure_horizon_texture(AREA_UTM_OTHER_REGION, size_px=32)
    assert call_count["n"] > calls_after_lgl  # real new download for the other area
    assert result_other != result_lgl_1
    assert result_lgl_1.exists()  # LGL texture stays untouched

    # 3) Back to the LGL area: cache hit, exactly the same path as on the first call,
    #    NO new network request.
    calls_before_switch_back = call_count["n"]
    result_lgl_2 = ensure_horizon_texture(AREA_UTM, size_px=32)

    assert result_lgl_2 == result_lgl_1
    assert call_count["n"] == calls_before_switch_back


# --- Item 2: attribution is logged on every use, also on a cache hit ---------------------------


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_attribution_is_logged_on_fresh_download(mock_get, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sentinel2_fetch, "_attribution_logged", False)
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", tmp_path / "cache_horizon_source")

    mock_get.side_effect = lambda url, params=None, headers=None, timeout=None: _jpeg_response(
        int(params["width"]), int(params["height"])
    )

    with caplog.at_level("INFO", logger="world_to_beamng"):
        ensure_horizon_texture(AREA_UTM, size_px=32)

    assert config.EOX_ATTRIBUTION_NOTICE in caplog.text


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_attribution_is_logged_on_cache_hit_too_but_only_once_per_process(mock_get, tmp_path, monkeypatch, caplog):
    import hashlib

    monkeypatch.setattr(sentinel2_fetch, "_attribution_logged", False)
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)
    mosaic_cache_dir = tmp_path / "cache_horizon_source"
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", mosaic_cache_dir)

    mock_get.side_effect = lambda url, params=None, headers=None, timeout=None: _jpeg_response(
        int(params["width"]), int(params["height"])
    )

    # Create the raw mosaic IN ADVANCE (simulates an earlier process/run that already loaded it
    # successfully) - directly via fetch_eox_mosaic(), NOT via ensure_horizon_texture(), so that
    # _attribution_logged is not already set by it. Cache key computation as in
    # ensure_horizon_texture().
    sig = f"{round(AREA_UTM[0])}_{round(AREA_UTM[1])}_{round(AREA_UTM[2])}_{round(AREA_UTM[3])}_{config.EOX_WMS_LAYER}"
    cache_key = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    mosaic_path = mosaic_cache_dir / f"eox_mosaic_{cache_key}.tif"
    mosaic_cache_dir.mkdir(parents=True)
    success, failed_count = fetch_eox_mosaic(AREA_UTM, mosaic_path)
    assert success is True and failed_count == 0
    assert not sentinel2_fetch._attribution_logged
    calls_from_seeding = mock_get.call_count

    # The first call of ensure_horizon_texture() in this process is already a CACHE HIT (the
    # raw mosaic already exists) - the license attribution obligation is tied to the USE of the
    # imagery, not to the download, so it must be logged here too (previously the
    # message was only logged in the cache-miss branch and would have been omitted entirely here).
    with caplog.at_level("INFO", logger="world_to_beamng"):
        ensure_horizon_texture(AREA_UTM, size_px=32)
    assert mock_get.call_count == calls_from_seeding  # no new network request -> real cache hit
    assert caplog.text.count(config.EOX_ATTRIBUTION_NOTICE) == 1
    caplog.clear()

    # Second call (again a cache hit, now the texture itself too) in the SAME process: do not
    # log again.
    with caplog.at_level("INFO", logger="world_to_beamng"):
        ensure_horizon_texture(AREA_UTM, size_px=32)
    assert config.EOX_ATTRIBUTION_NOTICE not in caplog.text
    assert sentinel2_fetch._attribution_logged is True
