"""
Tests: DGM30-Auto-Download (Copernicus-DEM-GLO-30-Kacheln von S3).
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.terrain.dgm30_fetch import (
    copernicus_tile_id,
    download_dgm30_tiles,
    ensure_dgm30_coverage,
    missing_tile_ids,
    required_tile_ids,
)


# --- copernicus_tile_id() ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "lat_deg, lon_deg, expected",
    [
        (47, 7, "Copernicus_DSM_COG_10_N47_00_E007_00_DEM"),  # N, E
        (52, -1, "Copernicus_DSM_COG_10_N52_00_W001_00_DEM"),  # N, W
        (-1, 18, "Copernicus_DSM_COG_10_S01_00_E018_00_DEM"),  # S, E
        (-33, -70, "Copernicus_DSM_COG_10_S33_00_W070_00_DEM"),  # S, W
        (0, 0, "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"),  # Nullgrad -> N/E
    ],
)
def test_copernicus_tile_id(lat_deg, lon_deg, expected):
    assert copernicus_tile_id(lat_deg, lon_deg) == expected


# --- required_tile_ids() ----------------------------------------------------------------------


def test_bbox_exactly_on_a_degree_boundary_does_not_produce_an_extra_tile():
    # lat_max = 48.0 liegt exakt auf der Gradgrenze -> keine Kachel N48
    tile_ids = required_tile_ids((7.0, 47.0, 8.0, 48.0))

    assert tile_ids == [copernicus_tile_id(47, 7)]


def test_bbox_crossing_a_degree_boundary_produces_both_tiles():
    tile_ids = required_tile_ids((7.2, 47.5, 7.8, 48.5))

    assert set(tile_ids) == {copernicus_tile_id(47, 7), copernicus_tile_id(48, 7)}


def test_bbox_spanning_two_tiles_in_both_directions_produces_all_four():
    tile_ids = required_tile_ids((6.5, 47.5, 7.5, 48.5))

    assert set(tile_ids) == {
        copernicus_tile_id(47, 6),
        copernicus_tile_id(47, 7),
        copernicus_tile_id(48, 6),
        copernicus_tile_id(48, 7),
    }


# --- missing_tile_ids() -----------------------------------------------------------------------

BBOX = (7.0, 47.0, 8.0, 48.0)  # -> genau eine Kachel: N47_00_E007_00
TILE_ID = copernicus_tile_id(47, 7)


def test_tile_with_existing_tif_file_is_not_missing(tmp_path):
    (tmp_path / f"{TILE_ID}.tif").write_bytes(b"fake dem data")

    assert missing_tile_ids(BBOX, tmp_path, {}) == []


def test_tile_with_fresh_not_found_cache_entry_is_not_missing(tmp_path):
    not_found_cache = {TILE_ID: datetime.now(timezone.utc).isoformat()}

    assert missing_tile_ids(BBOX, tmp_path, not_found_cache) == []


def test_tile_with_expired_not_found_cache_entry_is_missing_again(tmp_path):
    expired = datetime.now(timezone.utc) - timedelta(days=config.DGM30_NOT_FOUND_CACHE_TTL_DAYS + 1)
    not_found_cache = {TILE_ID: expired.isoformat()}

    assert missing_tile_ids(BBOX, tmp_path, not_found_cache) == [TILE_ID]


def test_tile_with_no_cache_entry_and_no_file_is_missing(tmp_path):
    assert missing_tile_ids(BBOX, tmp_path, {}) == [TILE_ID]


def test_missing_tile_ids_works_with_nonexistent_dgm30_dir(tmp_path):
    nonexistent = tmp_path / "does_not_exist_yet"

    assert missing_tile_ids(BBOX, nonexistent, {}) == [TILE_ID]


# --- download_dgm30_tiles() -------------------------------------------------------------------


def _mock_response(status_code=200, chunks=(b"a" * 1024, b"b" * 1024)):
    response = MagicMock()
    response.status_code = status_code
    if status_code == 200:
        response.raise_for_status.return_value = None
        response.iter_content.return_value = iter(chunks)
    else:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code}")
    return response


@patch("world_to_beamng.terrain.dgm30_fetch.requests.get")
def test_successful_download_writes_final_tif_not_part_file(mock_get, tmp_path):
    mock_get.return_value = _mock_response(200)

    result = download_dgm30_tiles(BBOX, tmp_path)

    assert result == {"downloaded": [TILE_ID], "not_found": [], "failed": []}
    assert (tmp_path / f"{TILE_ID}.tif").exists()
    assert (tmp_path / f"{TILE_ID}.tif").read_bytes() == b"a" * 1024 + b"b" * 1024
    assert not (tmp_path / f"{TILE_ID}.tif.part").exists()


@patch("world_to_beamng.terrain.dgm30_fetch.requests.get")
def test_404_is_recorded_as_not_found_and_cached(mock_get, tmp_path):
    mock_get.return_value = _mock_response(404)

    result = download_dgm30_tiles(BBOX, tmp_path)

    assert result == {"downloaded": [], "not_found": [TILE_ID], "failed": []}
    assert not (tmp_path / f"{TILE_ID}.tif").exists()
    assert mock_get.call_count == 1  # kein Retry bei 404

    cache_content = json.loads((tmp_path / ".not_found_cache.json").read_text(encoding="utf-8"))
    assert TILE_ID in cache_content


@patch("world_to_beamng.terrain.dgm30_fetch.time.sleep")  # Backoff-Wartezeit im Test überspringen
@patch("world_to_beamng.terrain.dgm30_fetch.requests.get")
def test_timeout_then_success_still_downloads_the_tile(mock_get, mock_sleep, tmp_path):
    mock_get.side_effect = [requests.exceptions.Timeout(), _mock_response(200)]

    result = download_dgm30_tiles(BBOX, tmp_path)

    assert result == {"downloaded": [TILE_ID], "not_found": [], "failed": []}
    assert (tmp_path / f"{TILE_ID}.tif").exists()
    assert mock_get.call_count == 2


@patch("world_to_beamng.terrain.dgm30_fetch.time.sleep")
@patch("world_to_beamng.terrain.dgm30_fetch.requests.get")
def test_exhausted_retries_are_recorded_as_failed_and_not_cached(mock_get, mock_sleep, tmp_path):
    mock_get.side_effect = requests.exceptions.Timeout()

    result = download_dgm30_tiles(BBOX, tmp_path)

    assert result == {"downloaded": [], "not_found": [], "failed": [TILE_ID]}
    assert mock_get.call_count == config.DGM30_FETCH_MAX_RETRIES
    assert not (tmp_path / ".not_found_cache.json").exists()  # failed != not_found -> nicht gecacht


@patch("world_to_beamng.terrain.dgm30_fetch.requests.get")
def test_no_request_when_no_tile_is_missing(mock_get, tmp_path):
    (tmp_path / f"{TILE_ID}.tif").write_bytes(b"already there")

    result = download_dgm30_tiles(BBOX, tmp_path)

    assert result == {"downloaded": [], "not_found": [], "failed": []}
    mock_get.assert_not_called()


@patch("world_to_beamng.terrain.dgm30_fetch.requests.get")
def test_corrupt_not_found_cache_file_is_ignored_not_fatal(mock_get, tmp_path):
    (tmp_path / ".not_found_cache.json").write_text("{ this is not valid json", encoding="utf-8")
    mock_get.return_value = _mock_response(200)

    result = download_dgm30_tiles(BBOX, tmp_path)

    assert result == {"downloaded": [TILE_ID], "not_found": [], "failed": []}


# --- ensure_dgm30_coverage() ------------------------------------------------------------------


@patch("world_to_beamng.terrain.dgm30_fetch.requests.get")
def test_ensure_coverage_is_a_noop_when_auto_download_disabled(mock_get, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DGM30_AUTO_DOWNLOAD", False)

    result = ensure_dgm30_coverage(BBOX, tmp_path)

    assert result == {"downloaded": [], "not_found": [], "failed": []}
    mock_get.assert_not_called()


@patch("world_to_beamng.terrain.dgm30_fetch.requests.get")
def test_ensure_coverage_downloads_when_auto_download_enabled(mock_get, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DGM30_AUTO_DOWNLOAD", True)
    mock_get.return_value = _mock_response(200)

    result = ensure_dgm30_coverage(BBOX, tmp_path)

    assert result == {"downloaded": [TILE_ID], "not_found": [], "failed": []}


@patch("world_to_beamng.terrain.dgm30_fetch.download_dgm30_tiles")
def test_ensure_coverage_never_raises_on_unexpected_error(mock_download, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DGM30_AUTO_DOWNLOAD", True)
    mock_download.side_effect = OSError("no network")

    result = ensure_dgm30_coverage(BBOX, tmp_path)

    assert result == {"downloaded": [], "not_found": [], "failed": []}
