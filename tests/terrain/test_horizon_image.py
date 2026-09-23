"""
Tests: Horizont-Fläche und Horizont-Bild (Ausschneiden und Umprojizieren eines beliebigen georeferenzierten Bilds).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.terrain.horizon_image import _dst_crs, build_horizon_image, horizon_area, horizon_area_wgs84

UTM_CRS = _dst_crs()  # Default-Quell-CRS (EPSG:25832), solange kein set_source_crs() aufgerufen wurde

CENTER = (401000.0, 5298000.0)
AREA = horizon_area(CENTER)
# WGS84-Quellbild, das die Horizont-Fläche reichlich überdeckt
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = 6.8, 47.2, 8.6, 48.5


def _write(path, crs="EPSG:4326", bounds=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX), bands=3, size=(180, 130)):
    """Rot = Länge (0..255 über die Bildbreite), Grün = Breite (0..255 über die Bildhöhe), Blau konstant."""
    height, width = size
    cols = np.linspace(0, 255, width)[None, :].repeat(height, axis=0)
    rows = np.linspace(255, 0, height)[:, None].repeat(width, axis=1)  # Zeile 0 = Norden = 255
    data = np.stack([cols, rows, np.full((height, width), 50.0)][:bands]).astype("uint8")
    profile = dict(driver="GTiff", width=width, height=height, count=bands, dtype="uint8", transform=from_bounds(*bounds, width, height))
    if crs:
        profile["crs"] = crs
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_area_is_centred_on_the_area_and_has_the_configured_size():
    x_min, x_max, y_min, y_max = AREA

    assert (x_min + x_max) / 2 == CENTER[0] and (y_min + y_max) / 2 == CENTER[1]
    assert x_max - x_min == 2 * config.HORIZON_HALF_SIZE_M == y_max - y_min


def test_area_wgs84_contains_the_centre_point_and_spans_roughly_the_expected_extent():
    lon_min, lat_min, lon_max, lat_max = horizon_area_wgs84(CENTER)
    lon, lat = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True).transform(*CENTER)

    # transform_bounds() der UTM-Eckpunkte ergibt wegen Meridiankonvergenz kein exakt auf `lon`/`lat`
    # zentriertes Rechteck (die UTM-Fläche ist quadratisch, ~2*HORIZON_HALF_SIZE_M breit) - deshalb
    # hier nur eine grobe Lage-/Größenprüfung statt exakter Zentrierung.
    assert lon_min < lon < lon_max and lat_min < lat < lat_max
    approx_deg_per_m = 1 / 111_000
    expected_span = 2 * config.HORIZON_HALF_SIZE_M * approx_deg_per_m
    assert lat_max - lat_min == pytest.approx(expected_span, rel=0.05)


def test_result_is_utm_32n_with_exactly_the_requested_area_and_size(tmp_path):
    out = tmp_path / "horizon.tif"

    coverage = build_horizon_image(_write(tmp_path / "src.tif"), out, AREA, size_px=64)

    assert coverage == pytest.approx(1.0)
    with rasterio.open(out) as result:
        assert result.crs.to_string() == UTM_CRS
        assert (result.width, result.height, result.count) == (64, 64, 3)
        assert tuple(result.bounds) == pytest.approx((AREA[0], AREA[2], AREA[1], AREA[3]))


def test_pixels_end_up_at_the_right_coordinates(tmp_path):
    out = tmp_path / "horizon.tif"
    build_horizon_image(_write(tmp_path / "src.tif"), out, AREA, size_px=128)
    lon, lat = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True).transform(*CENTER)

    with rasterio.open(out) as result:
        red, green, blue = next(result.sample([CENTER]))

    assert red == pytest.approx((lon - LON_MIN) / (LON_MAX - LON_MIN) * 255, abs=3)
    assert green == pytest.approx((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * 255, abs=3)
    assert blue == 50


def test_north_is_up(tmp_path):
    out = tmp_path / "horizon.tif"
    build_horizon_image(_write(tmp_path / "src.tif"), out, AREA, size_px=64)

    with rasterio.open(out) as result:
        green = result.read(2).astype(int)

    assert green[5].mean() > green[-5].mean()  # Quelle: Norden = größerer Grünwert; obere Bildzeile = Norden


def test_a_source_that_covers_only_part_of_the_area_warns_and_leaves_black(tmp_path, caplog):
    partial = _write(tmp_path / "src.tif", bounds=(LON_MIN, LAT_MIN, 7.7, LAT_MAX))  # bis etwa zur Gebietsmitte
    out = tmp_path / "horizon.tif"

    with caplog.at_level("WARNING", logger="world_to_beamng"):
        coverage = build_horizon_image(partial, out, AREA, size_px=64)

    assert 0.4 < coverage < 0.7
    assert "deckt nur" in caplog.text
    with rasterio.open(out) as result:
        assert result.read(1)[:, -3:].max() == 0  # rechter Rand liegt außerhalb der Quelle


def test_source_without_coordinate_system_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Koordinatensystem"):
        build_horizon_image(_write(tmp_path / "src.tif", crs=None), tmp_path / "out.tif", AREA, size_px=32)


def test_source_with_fewer_than_three_bands_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="RGB"):
        build_horizon_image(_write(tmp_path / "src.tif", bands=2), tmp_path / "out.tif", AREA, size_px=32)


def test_source_far_away_is_rejected(tmp_path):
    far = _write(tmp_path / "src.tif", bounds=(100.0, 10.0, 101.0, 11.0))

    with pytest.raises(ValueError, match="überdeckt"):
        build_horizon_image(far, tmp_path / "out.tif", AREA, size_px=32)
