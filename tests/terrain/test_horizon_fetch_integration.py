"""
Real network tests for the auto-download modules (dgm30_fetch, sentinel2_fetch) - they do NOT run in
the normal suite, only on request with RUN_NETWORK_TESTS=1 (there is no pytest.ini/conftest.py with
marker infrastructure, hence a simple skipif decorator instead of a pytest marker).
"""

import os
from pathlib import Path

import pytest
import rasterio

from world_to_beamng.terrain import dgm30_fetch, sentinel2_fetch

_SKIP_REASON = "real network test, only on request (RUN_NETWORK_TESTS=1)"
_run_network_tests = pytest.mark.skipif(not os.environ.get("RUN_NETWORK_TESTS"), reason=_SKIP_REASON)

# Small area around 47.8°N/7.7°E - a single 1° tile area of the Copernicus DEM GLO-30
# tile N47/E007 (known land tile, certainly exists).
_SMALL_AREA_WGS84 = (7.65, 47.75, 7.75, 47.85)  # (lon_min, lat_min, lon_max, lat_max)

# Small UTM area (2x2 km instead of the full 100x100 km horizon area), so the test stays fast.
_SMALL_AREA_UTM = (412000.0, 414000.0, 5297000.0, 5299000.0)  # (x_min, x_max, y_min, y_max)


@_run_network_tests
def test_ensure_dgm30_coverage_downloads_a_real_tile(tmp_path):
    dgm30_fetch.ensure_dgm30_coverage(_SMALL_AREA_WGS84, tmp_path)

    tif_files = list(tmp_path.glob("*.tif")) + list(tmp_path.glob("*.tiff"))
    assert tif_files, f"no .tif file in {tmp_path} after ensure_dgm30_coverage()"


@_run_network_tests
def test_ensure_horizon_texture_downloads_a_real_geotiff(tmp_path):
    dest = tmp_path / "horizon_temp.tif"

    result = sentinel2_fetch.ensure_horizon_texture(_SMALL_AREA_UTM, dest=dest, size_px=256)

    assert result == dest
    assert dest.exists()
    with rasterio.open(dest) as src:
        assert src.count >= 3
        assert src.crs is not None
