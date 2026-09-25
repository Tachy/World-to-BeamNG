"""
Tests: world_to_beamng.terrain.elevation_io.read_elevation_tile() - unified reader for
ASCII-XYZ point clouds (in ZIP) and GeoTIFF rasters (loose file or in ZIP), dispatching on the
actual file content.
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.core.cache_manager import CacheManager
from world_to_beamng.terrain.elevation_io import read_elevation_tile, read_elevation_tile_cached


def _write_geotiff(path, bounds, size, crs="EPSG:25832", nodata=None, fill=None):
    """Small single-band GeoTIFF; values = running index unless `fill` specifies otherwise."""
    width, height = size
    data = np.arange(width * height, dtype="float32").reshape(height, width) + 100.0
    if fill is not None:
        data = np.full((height, width), fill, dtype="float32")
    profile = dict(
        driver="GTiff", width=width, height=height, count=1, dtype="float32",
        crs=crs, transform=from_bounds(*bounds, width, height), nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return data


def _write_xyz_zip(path, rows):
    """ZIP with a single a.xyz file (LGL format: 'X Y Z' per line)."""
    text = "\n".join(f"{x} {y} {z}" for x, y, z in rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.xyz", text)
    return path


# ---------------------------------------------------------------- ASCII-XYZ (ZIP, LGL format)


def test_xyz_in_zip_is_read_and_has_no_crs():
    rows = [(0.0, 0.0, 10.0), (1.0, 0.0, 10.5), (0.0, 1.0, 11.0)]
    zip_path = "test_xyz.zip"

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / zip_path
        _write_xyz_zip(p, rows)

        points, elevations, crs_epsg, bbox_utm = read_elevation_tile(p)

    assert points.shape == (3, 2)
    assert list(elevations) == [10.0, 10.5, 11.0]
    assert crs_epsg is None
    # Points are cell centers on the 1m grid (0.0/1.0) -> coverage extends 0.5m beyond the
    # outer points, not just to their plain min/max (see module docstring)
    assert bbox_utm == pytest.approx((-0.5, 1.5, -0.5, 1.5))


def test_zip_without_xyz_or_raster_members_returns_none(tmp_path):
    p = tmp_path / "empty.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("readme.txt", "no data here")

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(p)

    assert (points, elevations, crs_epsg, bbox_utm) == (None, None, None, None)


# ---------------------------------------------------------------- GeoTIFF (loose file)


def test_loose_geotiff_pixel_centres_match_rasterio_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)
    path = tmp_path / "dem.tif"
    bounds = (0.0, 0.0, 4.0, 4.0)  # 4x4 pixels @ 1 m native = config.GRID_SPACING -> no resampling
    data = _write_geotiff(path, bounds, size=(4, 4), crs="EPSG:25832")

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(path)

    assert crs_epsg == 25832
    assert len(points) == 16  # 4x4, no NoData, no resampling
    # bbox_utm is the REAL raster coverage (0..4), not the pixel-center BBox that is smaller by
    # half a pixel (0.5..3.5)
    assert bbox_utm == pytest.approx((0.0, 4.0, 0.0, 4.0))

    with rasterio.open(path) as src:
        expected_xs, expected_ys = rasterio.transform.xy(src.transform, [0], [0])
    # upper-left pixel center (row 0, column 0) must be among the returned points
    assert any(
        pytest.approx(expected_xs[0], abs=1e-6) == x and pytest.approx(expected_ys[0], abs=1e-6) == y
        for x, y in points
    )
    assert set(elevations.tolist()) == set(data.ravel().tolist())


def test_crs_without_epsg_or_missing_crs_reports_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)
    path = tmp_path / "no_crs.tif"
    _write_geotiff(path, (0.0, 0.0, 2.0, 2.0), size=(2, 2), crs=None)

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(path)

    assert crs_epsg is None
    assert len(points) == 4


def test_nodata_pixels_are_masked_out(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)
    path = tmp_path / "with_nodata.tif"
    width, height = 3, 3
    data = np.full((height, width), 50.0, dtype="float32")
    data[1, 1] = -9999.0  # center = NoData (like swissALTI3D)
    profile = dict(
        driver="GTiff", width=width, height=height, count=1, dtype="float32",
        crs="EPSG:25832", transform=from_bounds(0.0, 0.0, 3.0, 3.0, width, height), nodata=-9999.0,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(path)

    assert len(points) == 8  # 9 pixels minus the one NoData cell
    assert -9999.0 not in elevations


def test_geotiff_finer_than_grid_spacing_is_downsampled_to_grid_spacing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)
    path = tmp_path / "fine.tif"
    # 0.25 m native over an 8x8 m area = 32x32 pixels -> should be downsampled to about 1 m (8x8 points)
    _write_geotiff(path, (0.0, 0.0, 8.0, 8.0), size=(32, 32), crs="EPSG:25832")

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(path)

    # Significantly fewer than the 1024 native pixels, close to 8x8=64 (target resolution 1 m)
    assert 50 <= len(points) <= 100
    xs = np.unique(np.round(points[:, 0], 3))
    spacing = np.diff(np.sort(xs)).mean()
    assert spacing == pytest.approx(1.0, abs=0.05)
    # bbox_utm stays the REAL area (0..8), independent of the resampling
    assert bbox_utm == pytest.approx((0.0, 8.0, 0.0, 8.0))


def test_geotiff_coarser_than_grid_spacing_stays_native(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)
    path = tmp_path / "coarse.tif"
    # 2 m native, coarser than GRID_SPACING=1 m -> stays unchanged (no upsampling here)
    _write_geotiff(path, (0.0, 0.0, 8.0, 8.0), size=(4, 4), crs="EPSG:25832")

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(path)

    assert len(points) == 16  # 4x4 native pixels, unchanged
    xs = np.unique(np.round(points[:, 0], 3))
    spacing = np.diff(np.sort(xs)).mean()
    assert spacing == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------- GeoTIFF in ZIP


def test_geotiff_embedded_in_zip_is_read_via_vsizip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)
    tif_path = tmp_path / "inner.tif"
    _write_geotiff(tif_path, (0.0, 0.0, 3.0, 3.0), size=(3, 3), crs="EPSG:2056")
    zip_path = tmp_path / "raster.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tif_path, arcname="inner.tif")

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(zip_path)

    assert crs_epsg == 2056
    assert len(points) == 9


def test_unknown_extension_is_reported_and_returns_none(tmp_path):
    p = tmp_path / "data.xyz"  # loose .xyz file (no ZIP, no raster extension) is not supported
    p.write_text("0 0 1")

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile(p)

    assert (points, elevations, crs_epsg, bbox_utm) == (None, None, None, None)


# ---------------------------------------------------------------- Cache round trip (read_elevation_tile_cached)


def test_cached_read_preserves_bbox_and_crs_across_a_cache_hit(tmp_path, monkeypatch):
    """Regression test: bbox_utm/crs_epsg must also come back correctly from the cache (not only on a
    fresh parse) - a missing bbox_utm in the cache previously shifted the aerial photo tiles by
    0.5m relative to the real terrain area (see module docstring)."""
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)
    path = tmp_path / "dem.tif"
    _write_geotiff(path, (0.0, 0.0, 4.0, 4.0), size=(4, 4), crs="EPSG:2056")
    cache = CacheManager(tmp_path / "cache")

    fresh = read_elevation_tile_cached(path, cache)
    cached = read_elevation_tile_cached(path, cache)  # second call -> cache hit

    assert fresh[2] == cached[2] == 2056  # crs_epsg
    assert fresh[3] == pytest.approx(cached[3])  # bbox_utm
    assert cached[3] == pytest.approx((0.0, 4.0, 0.0, 4.0))
    assert np.array_equal(fresh[0], cached[0])  # points


def test_a_stale_cache_entry_without_bbox_utm_is_healed_not_treated_as_valid(tmp_path, monkeypatch):
    """Cache entries from before bbox_utm (only points/elevations) must NOT count as a hit -
    otherwise bbox_utm=None and the tile is silently skipped by scan_elevation_tiles(),
    even though the file is there and has valid data."""
    monkeypatch.setattr(config, "GRID_SPACING", 1.0)
    path = tmp_path / "dem.tif"
    _write_geotiff(path, (0.0, 0.0, 4.0, 4.0), size=(4, 4), crs="EPSG:25832")
    cache = CacheManager(tmp_path / "cache")
    tile_hash = cache.hash_file(path)
    # Simulates an old cache entry from before this function (no bbox_utm key)
    cache.set_npz(f"height_raw_{tile_hash}", points=np.zeros((1, 2)), elevations=np.zeros(1))

    points, elevations, crs_epsg, bbox_utm = read_elevation_tile_cached(path, cache)

    assert bbox_utm == pytest.approx((0.0, 4.0, 0.0, 4.0))
    assert crs_epsg == 25832
    assert len(points) == 16  # freshly read from the real file, not the fake cache entry
