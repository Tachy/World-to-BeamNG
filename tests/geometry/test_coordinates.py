"""Tests für world_to_beamng.geometry.coordinates: das Lazy-CRS-Proxy."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.geometry import coordinates


@pytest.fixture(autouse=True)
def _reset_source_crs():
    """Jeder Test startet mit dem Default-Zustand (kein explizit gesetztes CRS)."""
    coordinates.set_source_crs(config.SOURCE_CRS_EPSG)
    coordinates._source_epsg = None
    yield
    coordinates._source_epsg = None


def test_default_behaviour_matches_the_old_fixed_epsg_25832():
    assert coordinates.get_source_crs_epsg() == 25832 == config.SOURCE_CRS_EPSG

    # Karlsruhe-Innenstadt in UTM32/ETRS89 -> WGS84 (bekannter Referenzwert)
    lon, lat = coordinates.transformer_to_wgs84.transform(456000.0, 5428000.0)

    assert lon == pytest.approx(8.4, abs=0.2)
    assert lat == pytest.approx(49.0, abs=0.2)


def test_set_source_crs_changes_the_result():
    lon_before, lat_before = coordinates.transformer_to_wgs84.transform(2685500.0, 1153500.0)

    coordinates.set_source_crs(2056)  # CH1903+/LV95 (Schweiz)
    lon_after, lat_after = coordinates.transformer_to_wgs84.transform(2685500.0, 1153500.0)

    # Dieselben rohen Koordinaten ergeben in unterschiedlichem CRS unterschiedliche WGS84-Punkte
    assert (lon_before, lat_before) != (lon_after, lat_after)
    # 2685500/1153500 in EPSG:2056 liegt in der Schweiz
    assert lon_after == pytest.approx(8.55, abs=0.3)
    assert lat_after == pytest.approx(46.5, abs=0.3)


def test_transformer_is_rebuilt_after_a_crs_change_not_cached_stale():
    coordinates.set_source_crs(25832)
    first = coordinates.transformer_to_utm.transform(8.4, 49.0)

    coordinates.set_source_crs(2056)
    second = coordinates.transformer_to_utm.transform(8.4, 49.0)

    assert first != second

    coordinates.set_source_crs(25832)
    third = coordinates.transformer_to_utm.transform(8.4, 49.0)

    assert third == pytest.approx(first, abs=1e-6)


def test_arbitrary_pyproj_attributes_are_proxied_through():
    # workflow/forest_workflow.py greift direkt auf .source_crs/.target_crs zu (kein .transform()-Aufruf)
    assert coordinates.transformer_to_wgs84.source_crs is not None
    assert coordinates.transformer_to_wgs84.target_crs is not None


def test_get_source_crs_epsg_falls_back_to_config_without_an_explicit_set():
    coordinates._source_epsg = None

    assert coordinates.get_source_crs_epsg() == config.SOURCE_CRS_EPSG
