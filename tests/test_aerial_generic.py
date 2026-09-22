"""
Tests: generische Bilderkennung in io/aerial.py - eingebettete GeoTIFF-Georeferenz (lose Datei
und in ZIP, ohne .tfw), .tfw-Fallback, lose Rasterdateien in der Cache-Signatur, Reprojektion bei
abweichendem CRS.
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from world_to_beamng.io.aerial import (
    AERIAL_PHOTO_FILENAME,
    SINGLE_PHOTO_NAME,
    _aerial_source_files,
    _reproject_image_to_source_crs,
    ensure_aerial_photos,
    extract_georeferenced_images,
    extract_images_from_zips,
    extract_loose_images,
)


def _write_geotiff(path, bounds, crs="EPSG:25832", size=(4, 4)):
    width, height = size
    data = np.random.default_rng(0).integers(0, 255, size=(3, height, width), dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8",
        crs=crs, transform=from_bounds(*bounds, width, height),
    ) as dst:
        dst.write(data)
    return path


def _write_tfw(path, pixel_size, x_origin, y_origin):
    path.write_text(f"{pixel_size}\n0.0\n0.0\n{-pixel_size}\n{x_origin}\n{y_origin}\n")


# ---------------------------------------------------------------- lose Dateien


def test_loose_geotiff_with_embedded_crs_is_recognised_without_tfw(tmp_path):
    _write_geotiff(tmp_path / "ortho.tif", (399000.0, 5296000.0, 399004.0, 5296004.0), crs="EPSG:25832")

    images = extract_loose_images(tmp_path)

    assert len(images) == 1
    name, source, world_info = images[0]
    assert name == "ortho.tif"
    assert isinstance(source, Path)
    assert world_info["crs_epsg"] == 25832
    assert world_info["x_origin"] == pytest.approx(399000.0)


def test_loose_tiff_without_embedded_crs_falls_back_to_tfw(tmp_path):
    # Plain TIFF ohne CRS (write_bytes eines Platzhalters wäre kein gültiges TIFF - stattdessen ein
    # GeoTIFF ohne crs schreiben, das hat ebenfalls keine "echte" Geotransform -> world_info None
    # -> .tfw-Fallback greift)
    path = tmp_path / "plain.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=3, dtype="uint8") as dst:
        dst.write(np.zeros((3, 2, 2), dtype="uint8"))
    _write_tfw(tmp_path / "plain.tfw", 0.2, 399000.0, 5296004.0)

    images = extract_loose_images(tmp_path)

    assert len(images) == 1
    _, _, world_info = images[0]
    assert world_info["crs_epsg"] is None  # .tfw kennt kein CRS - Quell-CRS wird angenommen
    assert world_info["x_origin"] == pytest.approx(399000.0)


def test_empty_directory_yields_no_images(tmp_path):
    assert extract_loose_images(tmp_path) == []
    assert extract_loose_images(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------- GeoTIFF in ZIP ohne .tfw


def test_geotiff_in_zip_without_tfw_is_recognised_via_embedded_crs(tmp_path):
    tif_path = tmp_path / "inner.tif"
    _write_geotiff(tif_path, (399000.0, 5296000.0, 399004.0, 5296004.0), crs="EPSG:25832")
    zip_path = tmp_path / "dop.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tif_path, arcname="inner.tif")  # bewusst KEINE .tfw daneben

    images = extract_images_from_zips(tmp_path)

    assert len(images) == 1
    name, data, world_info = images[0]
    assert isinstance(data, bytes)
    assert world_info is not None
    assert world_info["crs_epsg"] == 25832


def test_extract_georeferenced_images_combines_zip_and_loose_sources(tmp_path):
    tif_path = tmp_path / "inner.tif"
    _write_geotiff(tif_path, (0.0, 0.0, 4.0, 4.0))
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as zf:
        zf.write(tif_path, arcname="inner.tif")
    _write_geotiff(tmp_path / "loose.tif", (10.0, 10.0, 14.0, 14.0))

    images = extract_georeferenced_images(tmp_path)

    assert {name for name, _, _ in images} == {"inner.tif", "loose.tif"}


# ---------------------------------------------------------------- Cache-Signatur (lose Rasterdateien)


def test_aerial_source_files_includes_loose_rasters_not_only_zips(tmp_path):
    _write_geotiff(tmp_path / "a.tif", (0.0, 0.0, 4.0, 4.0))
    (tmp_path / "b.zip").write_bytes(b"x")

    files = _aerial_source_files(tmp_path)

    assert {f.name for f in files} == {"a.tif", "b.zip"}


def test_ensure_aerial_photos_does_not_report_none_for_a_directory_with_only_loose_geotiffs(tmp_path):
    # Regressionstest: vorher wurde nur nach *.zip gesucht - ein Verzeichnis mit AUSSCHLIESSLICH
    # losen GeoTIFFs galt faelschlich als "keine Quellbilder" ("none").
    aerial_dir = tmp_path / "satellite"
    aerial_dir.mkdir()
    _write_geotiff(aerial_dir / "ortho.tif", (0.0, 0.0, 4.0, 4.0))
    textures = tmp_path / "textures"
    textures.mkdir()

    status = ensure_aerial_photos(
        aerial_dir, textures, [{"name": SINGLE_PHOTO_NAME, "bounds": (-2.0, 2.0, -2.0, 2.0)}], (2.0, 2.0, 0.0)
    )

    assert status != "none"


# ---------------------------------------------------------------- Reprojektion


def test_reproject_image_to_source_crs_changes_crs_and_pixel_scale(tmp_path):
    # WGS84 (Grad) -> EPSG:25832 (Meter): Pixelgroesse muss danach im Meter-Maßstab liegen, nicht
    # mehr im Grad-Maßstab (mehrere Größenordnungen Unterschied - ein grober, aber robuster Check)
    path = tmp_path / "wgs84.tif"
    _write_geotiff(path, (7.60, 47.80, 7.61, 47.81), crs="EPSG:4326", size=(20, 20))

    image, world_info = _reproject_image_to_source_crs(path, dst_epsg=25832)

    assert world_info["crs_epsg"] == 25832
    assert abs(world_info["pixel_size_x"]) > 0.01  # Meter, nicht mehr Bruchteile eines Grads
    assert image.mode == "RGB"
    assert image.width > 0 and image.height > 0


def test_reproject_image_to_source_crs_works_from_zip_bytes(tmp_path):
    tif_path = tmp_path / "inner.tif"
    _write_geotiff(tif_path, (7.60, 47.80, 7.61, 47.81), crs="EPSG:4326", size=(10, 10))
    img_bytes = tif_path.read_bytes()

    image, world_info = _reproject_image_to_source_crs(img_bytes, dst_epsg=25832)

    assert world_info["crs_epsg"] == 25832
    assert image.mode == "RGB"
