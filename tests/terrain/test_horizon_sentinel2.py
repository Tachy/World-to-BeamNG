"""
Tests: The horizon loader reads exactly the configured Sentinel-2 file, not just any .tif in the folder.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.terrain.horizon import load_sentinel2_geotiff

BBOX = (0.0, 1.0, 0.0, 1.0)
CONFIGURED_FILENAME = "horizon_texture_deadbeef.tif"  # arbitrary name - the caller decides the path


def _write_geotiff(path: Path, bounds):
    data = np.full((3, 8, 8), 120, dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=8, height=8, count=3, dtype="uint8", crs="EPSG:25832", transform=from_bounds(*bounds, 8, 8)
    ) as dst:
        dst.write(data)


def test_configured_file_is_read_even_if_another_tif_lies_next_to_it(tmp_path):
    wanted = tmp_path / CONFIGURED_FILENAME
    _write_geotiff(wanted, (350000, 5249000, 450000, 5349000))
    _write_geotiff(tmp_path / "a_first_alphabetically.tif", (1000, 2000, 3000, 4000))
    _write_geotiff(tmp_path / "rohdaten.tif", (5000, 6000, 7000, 8000))

    image, bounds_utm, _ = load_sentinel2_geotiff(wanted, BBOX)

    assert bounds_utm == pytest.approx((350000, 5249000, 450000, 5349000))
    assert image.shape == (8, 8, 3)


def test_missing_file_returns_none_even_if_other_tifs_exist(tmp_path):
    _write_geotiff(tmp_path / "rohdaten.tif", (5000, 6000, 7000, 8000))

    assert load_sentinel2_geotiff(tmp_path / CONFIGURED_FILENAME, BBOX) is None

