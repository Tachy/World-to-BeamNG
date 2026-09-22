"""
Echte Netzwerktests für die Auto-Download-Module (dgm30_fetch, sentinel2_fetch) - laufen NICHT in
der normalen Suite, nur auf Wunsch mit RUN_NETWORK_TESTS=1 (kein pytest.ini/conftest.py mit
Marker-Infrastruktur vorhanden, daher ein einfacher skipif-Decorator statt eines pytest-Markers).
"""

import os
from pathlib import Path

import pytest
import rasterio

from world_to_beamng.terrain import dgm30_fetch, sentinel2_fetch

_SKIP_REASON = "echter Netzwerktest, nur auf Wunsch (RUN_NETWORK_TESTS=1)"
_run_network_tests = pytest.mark.skipif(not os.environ.get("RUN_NETWORK_TESTS"), reason=_SKIP_REASON)

# Kleines Gebiet um 47.8°N/7.7°E - ein einzelnes 1°-Kachel-Gebiet der Copernicus-DEM-GLO-30-
# Kachel N47/E007 (bekannte Land-Kachel, existiert sicher).
_SMALL_AREA_WGS84 = (7.65, 47.75, 7.75, 47.85)  # (lon_min, lat_min, lon_max, lat_max)

# Kleines UTM-Gebiet (2x2 km statt der vollen 100x100 km Horizont-Fläche), damit der Test schnell bleibt.
_SMALL_AREA_UTM = (412000.0, 414000.0, 5297000.0, 5299000.0)  # (x_min, x_max, y_min, y_max)


@_run_network_tests
def test_ensure_dgm30_coverage_downloads_a_real_tile(tmp_path):
    dgm30_fetch.ensure_dgm30_coverage(_SMALL_AREA_WGS84, tmp_path)

    tif_files = list(tmp_path.glob("*.tif")) + list(tmp_path.glob("*.tiff"))
    assert tif_files, f"keine .tif-Datei in {tmp_path} nach ensure_dgm30_coverage()"


@_run_network_tests
def test_ensure_horizon_texture_downloads_a_real_geotiff(tmp_path):
    dest = tmp_path / "horizon_temp.tif"

    result = sentinel2_fetch.ensure_horizon_texture(_SMALL_AREA_UTM, dest=dest, size_px=256)

    assert result == dest
    assert dest.exists()
    with rasterio.open(dest) as src:
        assert src.count >= 3
        assert src.crs is not None
