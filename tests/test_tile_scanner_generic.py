"""
Tests for world_to_beamng.utils.tile_scanner.scan_elevation_tiles() - detects elevation data tiles
(ASCII XYZ in ZIP, loose GeoTIFF, GeoTIFF in ZIP) regardless of the file name, as well as
resolve_source_crs_epsg() and compute_global_bbox()/compute_global_center().
"""

import sys
import zipfile
from pathlib import Path

import pytest
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng import config
from world_to_beamng.utils.tile_scanner import (
    compute_global_bbox,
    compute_global_center,
    resolve_source_crs_epsg,
    scan_elevation_tiles,
)


def _xyz_zip(path, x0, y0):
    """ZIP with an XYZ point file, arbitrary file name (no LGL naming scheme needed)."""
    rows = [(x0, y0, 10.0), (x0 + 1.0, y0, 10.5), (x0, y0 + 1.0, 11.0)]
    text = "\n".join(f"{x} {y} {z}" for x, y, z in rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.xyz", text)
    return path


def _geotiff(path, bounds, crs="EPSG:25832", size=(4, 4)):
    width, height = size
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
        crs=crs, transform=from_bounds(*bounds, width, height),
    ) as dst:
        import numpy as np

        dst.write(np.full((height, width), 100.0, dtype="float32"), 1)
    return path


@pytest.fixture(autouse=True)
def _fixed_grid_spacing(monkeypatch):
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)


# ---------------------------------------------------------------- scan_elevation_tiles


def test_scans_xyz_zip_with_arbitrary_filename(tmp_path):
    _xyz_zip(tmp_path / "beliebiger_name_ohne_lgl_schema.zip", x0=0.0, y0=0.0)

    tiles = scan_elevation_tiles(tmp_path, cache_dir=tmp_path / "cache")

    assert len(tiles) == 1
    assert tiles[0]["crs_epsg"] is None
    # from the real data, not the file name; points at 0.0/1.0 on the 1 m grid -> coverage
    # extends 0.5 m beyond the outer points (cell centers, see elevation_io)
    assert tiles[0]["bbox_utm"] == pytest.approx((-0.5, 1.5, -0.5, 1.5))


def test_scans_loose_geotiff_with_lgl_like_filename(tmp_path):
    # Deliberately a file name in the LGL scheme although it is a GeoTIFF - the file name must not matter
    _geotiff(tmp_path / "dgm1_32_399_5296_2_bw.tif", bounds=(0.0, 0.0, 4.0, 4.0), crs="EPSG:25832")

    tiles = scan_elevation_tiles(tmp_path, cache_dir=tmp_path / "cache")

    assert len(tiles) == 1
    assert tiles[0]["crs_epsg"] == 25832
    # bbox_utm is the REAL raster coverage (0..4), not derived from pixel centers
    assert tiles[0]["bbox_utm"] == pytest.approx((0.0, 4.0, 0.0, 4.0))


def test_scans_both_formats_mixed_in_one_directory(tmp_path):
    _xyz_zip(tmp_path / "a.zip", x0=0.0, y0=0.0)
    _geotiff(tmp_path / "b.tif", bounds=(10.0, 10.0, 14.0, 14.0), crs="EPSG:25832")

    tiles = scan_elevation_tiles(tmp_path, cache_dir=tmp_path / "cache")

    assert len(tiles) == 2
    assert sorted(t["filename"] for t in tiles) == ["a.zip", "b.tif"]


def test_missing_directory_returns_empty_list(tmp_path):
    assert scan_elevation_tiles(tmp_path / "does_not_exist", cache_dir=tmp_path / "cache") == []


def test_empty_directory_returns_empty_list(tmp_path):
    assert scan_elevation_tiles(tmp_path, cache_dir=tmp_path / "cache") == []


# ---------------------------------------------------------------- resolve_source_crs_epsg


def test_resolve_falls_back_to_config_when_no_tile_has_a_crs():
    tiles = [{"crs_epsg": None}, {"crs_epsg": None}]

    assert resolve_source_crs_epsg(tiles) == config.SOURCE_CRS_EPSG


def test_resolve_uses_the_detected_crs_when_consistent():
    tiles = [{"crs_epsg": None}, {"crs_epsg": 2056}, {"crs_epsg": 2056}]

    assert resolve_source_crs_epsg(tiles) == 2056


def test_resolve_raises_on_conflicting_crs():
    tiles = [{"crs_epsg": 25832}, {"crs_epsg": 2056}]

    with pytest.raises(ValueError, match="different CRS"):
        resolve_source_crs_epsg(tiles)


# ---------------------------------------------------------------- compute_global_bbox/_center


def test_compute_global_bbox_handles_non_square_tiles():
    tiles = [
        {"bbox_utm": (0.0, 10.0, 0.0, 20.0)},   # 10x20
        {"bbox_utm": (10.0, 15.0, 0.0, 5.0)},   # 5x5, offset
    ]

    assert compute_global_bbox(tiles) == (0.0, 15.0, 0.0, 20.0)


def test_compute_global_center_is_the_bbox_midpoint():
    tiles = [{"bbox_utm": (0.0, 10.0, 0.0, 20.0)}]

    assert compute_global_center(tiles) == (5.0, 10.0)


def test_compute_global_bbox_of_empty_list_is_none():
    assert compute_global_bbox([]) is None
