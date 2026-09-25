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
    align_tiles_to_crs,
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


def test_resolve_picks_the_crs_of_most_tiles():
    tiles = [{"crs_epsg": 2056}, {"crs_epsg": 32632}, {"crs_epsg": 32632}, {"crs_epsg": None}]

    assert resolve_source_crs_epsg(tiles) == 32632


def test_resolve_breaks_a_tie_with_the_configured_crs_else_the_lowest_code(monkeypatch):
    monkeypatch.setattr(config, "SOURCE_CRS_EPSG", 25832)
    assert resolve_source_crs_epsg([{"crs_epsg": 32632}, {"crs_epsg": 25832}]) == 25832
    assert resolve_source_crs_epsg([{"crs_epsg": 32632}, {"crs_epsg": 2056}]) == 2056


def test_align_reprojects_the_bbox_of_foreign_crs_tiles_only():
    from pyproj import Transformer

    native = {"filename": "a.tif", "crs_epsg": 25832, "bbox_utm": (400000.0, 401000.0, 5300000.0, 5301000.0)}
    foreign = {"filename": "b.tif", "crs_epsg": 32632, "bbox_utm": (400000.0, 401000.0, 5300000.0, 5301000.0)}
    xyz = {"filename": "c.zip", "crs_epsg": None, "bbox_utm": (0.0, 1.0, 0.0, 1.0)}

    reprojected = align_tiles_to_crs([native, foreign, xyz], 25832)

    assert reprojected == [foreign]
    assert native["bbox_utm"] == (400000.0, 401000.0, 5300000.0, 5301000.0) and "reproject_from" not in native
    assert "reproject_from" not in xyz
    assert foreign["reproject_from"] == 32632 and foreign["target_epsg"] == 25832
    x, y = Transformer.from_crs(32632, 25832, always_xy=True).transform(400000.0, 5300000.0)
    assert foreign["bbox_utm"][0] == pytest.approx(x, abs=0.01) and foreign["bbox_utm"][2] == pytest.approx(y, abs=0.01)
    assert foreign["easting"] == foreign["bbox_utm"][0]


def test_reprojected_tiles_load_in_the_target_crs_and_change_the_cache_hash(tmp_path):
    from pyproj import Transformer

    from world_to_beamng.core.cache_manager import CacheManager
    from world_to_beamng.io.cache import calculate_global_tiles_hash
    from world_to_beamng.workflow.tile_processor import TileProcessor

    _geotiff(tmp_path / "a.tif", bounds=(400000.0, 5300000.0, 400004.0, 5300004.0), crs="EPSG:25832")
    _geotiff(tmp_path / "b.tif", bounds=(400004.0, 5300000.0, 400008.0, 5300004.0), crs="EPSG:32632")
    tiles = scan_elevation_tiles(tmp_path, cache_dir=tmp_path / "cache")
    hash_before = calculate_global_tiles_hash(tiles)

    target = resolve_source_crs_epsg(tiles)  # tie -> config.SOURCE_CRS_EPSG (25832)
    align_tiles_to_crs(tiles, target)
    points, _ = TileProcessor(CacheManager(tmp_path / "cache")).load_height_data(tiles[1])

    x, y = Transformer.from_crs(32632, target, always_xy=True).transform(400004.5, 5300003.5)
    assert points[:, 0].min() == pytest.approx(x, abs=0.01) and points[:, 1].max() == pytest.approx(y, abs=0.01)
    assert calculate_global_tiles_hash(tiles) != hash_before


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
