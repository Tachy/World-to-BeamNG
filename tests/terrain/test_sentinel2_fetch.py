"""
Tests: Sentinel-2-Auto-Download (EOX-WMS-Mosaik -> Horizont-Textur).
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from PIL import Image
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.terrain import sentinel2_fetch
from world_to_beamng.terrain.sentinel2_fetch import (
    _mercator_bbox_for_area,
    _mosaic_pixel_size,
    ensure_horizon_texture,
    fetch_eox_mosaic,
    tile_grid,
)

AREA_UTM = (400000.0, 401000.0, 5300000.0, 5301000.0)  # 1x1 km bei EPSG:25832


# --- _mercator_bbox_for_area() ------------------------------------------------------------------


def test_mercator_bbox_matches_independent_pyproj_transform():
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:3857", always_xy=True)
    x_min, x_max, y_min, y_max = AREA_UTM
    # UTM -> Web-Mercator ist nicht achsparallel (leichte Scherung/Rotation) - die BBox ist daher
    # min/max über ALLE VIER Ecken, nicht nur über die untere-linke/obere-rechte Ecke.
    corners_x, corners_y = [], []
    for cx, cy in [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]:
        tx, ty = transformer.transform(cx, cy)
        corners_x.append(tx)
        corners_y.append(ty)
    left, right = min(corners_x), max(corners_x)
    bottom, top = min(corners_y), max(corners_y)

    minx, miny, maxx, maxy = _mercator_bbox_for_area(AREA_UTM)

    # Margin-Faktor weitet das Ergebnis um den Mittelpunkt auf - Toleranz entsprechend groß wählen.
    margin = config.EOX_FETCH_MARGIN_FACTOR - 1.0
    width = right - left
    height = top - bottom
    assert minx == pytest.approx(left - width / 2 * margin, abs=1.0)
    assert maxx == pytest.approx(right + width / 2 * margin, abs=1.0)
    assert miny == pytest.approx(bottom - height / 2 * margin, abs=1.0)
    assert maxy == pytest.approx(top + height / 2 * margin, abs=1.0)


def test_mercator_bbox_is_wider_than_the_unmargined_transform():
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:3857", always_xy=True)
    x_min, x_max, y_min, y_max = AREA_UTM
    left, bottom = transformer.transform(x_min, y_min)
    right, top = transformer.transform(x_max, y_max)

    minx, miny, maxx, maxy = _mercator_bbox_for_area(AREA_UTM)

    assert minx < left
    assert maxx > right
    assert miny < bottom
    assert maxy > top


# --- _mosaic_pixel_size() ------------------------------------------------------------------------


def test_mosaic_pixel_size_uses_target_resolution():
    bbox = (0.0, 0.0, 1000.0, 2000.0)
    w, h = _mosaic_pixel_size(bbox)

    assert w == round(1000.0 / config.EOX_TARGET_RESOLUTION_M)
    assert h == round(2000.0 / config.EOX_TARGET_RESOLUTION_M)


def test_mosaic_pixel_size_is_capped_at_mosaic_max_px(monkeypatch):
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    bbox = (0.0, 0.0, 1_000_000.0, 1_000_000.0)

    w, h = _mosaic_pixel_size(bbox)

    assert w == 100
    assert h == 100


# --- tile_grid() -----------------------------------------------------------------------------


def test_tile_grid_covers_the_full_raster_without_gap_or_overlap(monkeypatch):
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 2000)
    w, h = 5000, 3000
    bbox = (0.0, 0.0, w * 10.0, h * 10.0)  # 10 m/px, beliebig fuer diesen Test

    tiles = tile_grid(bbox, w, h)

    # Fläche vollständig abgedeckt, keine Überlappung.
    assert sum(t.width * t.height for t in tiles) == w * h

    # Pixelraster als belegt markieren -> jedes Pixel genau einmal getroffen.
    covered = np.zeros((h, w), dtype=bool)
    for t in tiles:
        region = covered[t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width]
        assert not region.any(), "Kachel überlappt eine bereits belegte Region"
        covered[t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width] = True
    assert covered.all()

    # Randkacheln sind kleiner (5000 % 2000 = 1000, 3000 % 2000 = 1000).
    col_offs = sorted({t.col_off for t in tiles})
    row_offs = sorted({t.row_off for t in tiles})
    assert col_offs == [0, 2000, 4000]
    assert row_offs == [0, 2000]
    last_col_tiles = [t for t in tiles if t.col_off == 4000]
    assert all(t.width == 1000 for t in last_col_tiles)
    last_row_tiles = [t for t in tiles if t.row_off == 2000]
    assert all(t.height == 1000 for t in last_row_tiles)


def test_tile_grid_bbox_north_is_row_zero():
    # Zeile 0 = Bildoberkante = geografisch Norden = maxy.
    bbox = (0.0, 0.0, 100.0, 100.0)
    tiles = tile_grid(bbox, 10, 10)

    top_left = next(t for t in tiles if t.col_off == 0 and t.row_off == 0)
    assert top_left.bbox[3] == pytest.approx(100.0)  # maxy der obersten Kachel = maxy des Mosaiks


# --- fetch_eox_mosaic() -----------------------------------------------------------------------


def _jpeg_response(width, height, color=(100, 150, 200)):
    image = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Type": "image/jpeg"}
    response.content = buf.getvalue()
    return response


def _error_response(status_code=500):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"Content-Type": "text/xml"}
    response.content = b"<ServiceException>boom</ServiceException>"
    return response


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_fetch_mosaic_success_writes_correct_size_and_crs(mock_get, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # eine einzige Kachel

    def fake_get(url, params=None, headers=None, timeout=None):
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    dest = tmp_path / "mosaic.tif"

    success, failed_count = fetch_eox_mosaic(AREA_UTM, dest)

    assert success is True
    assert failed_count == 0
    assert dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
    with rasterio.open(dest) as ds:
        assert ds.crs.to_string() == "EPSG:3857"
        assert ds.width <= 50
        assert ds.height <= 50
        assert ds.count == 3


@patch("world_to_beamng.terrain.sentinel2_fetch.time.sleep")  # Backoff im Test überspringen
@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_fetch_mosaic_partial_failure_still_returns_true_failed_tile_is_black(mock_get, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # -> 2x2 Kacheln
    monkeypatch.setattr(config, "EOX_FETCH_MAX_RETRIES", 2)

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        call_count["n"] += 1
        # Die allererste angefragte Kachel (col_off=0,row_off=0) schlägt bei jedem Versuch fehl,
        # alle anderen (und alle Retries) liefern ein Bild.
        if call_count["n"] <= config.EOX_FETCH_MAX_RETRIES:
            return _error_response(500)
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    dest = tmp_path / "mosaic.tif"

    success, failed_count = fetch_eox_mosaic(AREA_UTM, dest)

    assert success is True
    assert failed_count == 1
    assert dest.exists()
    with rasterio.open(dest) as ds:
        arr = ds.read()
        # Kachel (0,0) blieb schwarz (0), andere Kacheln haben die Testfarbe (100,150,200).
        assert arr[:, 0, 0].tolist() == [0, 0, 0]
        assert arr[0, -1, -1] != 0 or arr[1, -1, -1] != 0 or arr[2, -1, -1] != 0


@patch("world_to_beamng.terrain.sentinel2_fetch.time.sleep")
@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_fetch_mosaic_total_failure_returns_false_no_file_created(mock_get, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)

    mock_get.side_effect = lambda *a, **kw: _error_response(503)
    dest = tmp_path / "mosaic.tif"

    success, failed_count = fetch_eox_mosaic(AREA_UTM, dest)

    assert success is False
    assert failed_count == 1
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_fetch_mosaic_tile_with_wrong_decoded_size_stays_black_does_not_abort_mosaic(mock_get, tmp_path, monkeypatch):
    # Server antwortet mit Status 200 + Content-Type image/* (besteht also die Prüfung in
    # _fetch_one_tile()), aber das dekodierte Bild hat NICHT die angefragte Pixelgröße - ein
    # realistischer Fehlermodus eines externen WMS-Servers. Das darf dst.write() nicht mit einer
    # Exception aus dem `with rasterio.open(...)`-Block heraus abbrechen lassen.
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # -> 2x2 Kacheln

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        # Nur die allererste angefragte Kachel (col_off=0,row_off=0) liefert die falsche
        # Pixelgröße, alle anderen (und alle Retries, falls welche stattfänden) die angeforderte.
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _jpeg_response(int(params["width"]) - 1, int(params["height"]))  # falsche Größe
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    dest = tmp_path / "mosaic.tif"

    success, failed_count = fetch_eox_mosaic(AREA_UTM, dest)

    assert success is True  # mindestens eine (die restlichen 3) Kacheln waren erfolgreich
    assert failed_count == 1
    assert dest.exists()
    with rasterio.open(dest) as ds:
        arr = ds.read()
        # Die Kachel mit falscher Größe blieb schwarz (0,0,0 an ihrem Ursprungspixel).
        assert arr[:, 0, 0].tolist() == [0, 0, 0]
        # Eine andere (korrekt beantwortete) Kachel hat die Testfarbe geschrieben.
        assert arr[0, -1, -1] != 0 or arr[1, -1, -1] != 0 or arr[2, -1, -1] != 0


# --- ensure_horizon_texture() -------------------------------------------------------------------


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_end_to_end_creates_valid_geotiff(mock_get, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 60)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 60)
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", tmp_path / "cache_horizon_source")

    def fake_get(url, params=None, headers=None, timeout=None):
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get

    dest = tmp_path / "horizon_temp.tif"
    size_px = 32

    result = ensure_horizon_texture(AREA_UTM, dest=dest, size_px=size_px)

    assert result == dest
    assert dest.exists()
    with rasterio.open(dest) as ds:
        assert ds.width == size_px
        assert ds.height == size_px
        assert ds.count == 3


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_skips_download_when_dest_already_exists(mock_get, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    dest = tmp_path / "horizon_temp.tif"
    dest.write_bytes(b"already there - not a real geotiff")

    result = ensure_horizon_texture(AREA_UTM, dest=dest, size_px=32)

    assert result == dest
    mock_get.assert_not_called()


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_returns_none_when_auto_download_disabled(mock_get, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", False)
    dest = tmp_path / "horizon_temp.tif"

    result = ensure_horizon_texture(AREA_UTM, dest=dest, size_px=32)

    assert result is None
    assert not dest.exists()
    mock_get.assert_not_called()


@patch("world_to_beamng.terrain.sentinel2_fetch.fetch_eox_mosaic")
def test_ensure_horizon_texture_never_raises_on_unexpected_error(mock_fetch, tmp_path, monkeypatch):
    # Analog zu dgm30_fetch.test_ensure_coverage_never_raises_on_unexpected_error(): ein
    # unerwarteter Fehler (hier: fetch_eox_mosaic() wirft statt False zurückzugeben, z. B. weil ein
    # Dateisystem-/Netzwerkfehler nicht sauber abgefangen wurde) darf niemals aus
    # ensure_horizon_texture() herauspropagieren - horizon_workflow.py verlässt sich darauf, diesen
    # Einstiegspunkt ohne eigene Fehlerbehandlung aufrufen zu können.
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", tmp_path / "cache_horizon_source")
    mock_fetch.side_effect = OSError("permission denied")

    dest = tmp_path / "horizon_temp.tif"

    result = ensure_horizon_texture(AREA_UTM, dest=dest, size_px=32)

    assert result is None
    assert not dest.exists()


# --- Item 1: Teilerfolg wird nicht dauerhaft gecacht -------------------------------------------


@patch("world_to_beamng.terrain.sentinel2_fetch.time.sleep")
@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_partial_failure_still_builds_texture_this_run(mock_get, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # -> 2x2 Kacheln
    monkeypatch.setattr(config, "EOX_FETCH_MAX_RETRIES", 2)
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", tmp_path / "cache_horizon_source")

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        call_count["n"] += 1
        # Die allererste angefragte Kachel schlägt bei jedem Versuch fehl (Netzwerk-Hänger),
        # alle anderen liefern ein Bild -> Teilerfolg.
        if call_count["n"] <= config.EOX_FETCH_MAX_RETRIES:
            return _error_response(500)
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    dest = tmp_path / "horizon_temp.tif"

    result = ensure_horizon_texture(AREA_UTM, dest=dest, size_px=32)

    # Trotz Teilerfolg wird für DIESEN Lauf noch eine benutzbare Textur erzeugt.
    assert result == dest
    assert dest.exists()
    with rasterio.open(dest) as ds:
        assert ds.width == 32
        assert ds.height == 32


@patch("world_to_beamng.terrain.sentinel2_fetch.time.sleep")
@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_ensure_horizon_texture_partial_failure_does_not_cache_mosaic_for_next_run(mock_get, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 100)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)  # -> 2x2 Kacheln
    monkeypatch.setattr(config, "EOX_FETCH_MAX_RETRIES", 2)
    mosaic_cache_dir = tmp_path / "cache_horizon_source"
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", mosaic_cache_dir)

    call_count = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        call_count["n"] += 1
        if call_count["n"] <= config.EOX_FETCH_MAX_RETRIES:
            return _error_response(500)
        return _jpeg_response(int(params["width"]), int(params["height"]))

    mock_get.side_effect = fake_get
    dest = tmp_path / "horizon_temp.tif"

    ensure_horizon_texture(AREA_UTM, dest=dest, size_px=32)

    # Kein Rohmosaik im permanenten Cache-Verzeichnis hinterlassen (egal ob EOX_KEEP_RAW_MOSAIC
    # True oder False ist) - sonst würde der nächste Lauf den Teilerfolg für immer als
    # "vollständig" behandeln.
    assert list(mosaic_cache_dir.glob("*.tif")) == []

    # Simuliert den vom Warnhinweis vorgeschlagenen nächsten Lauf: Nutzer löscht die Textur, um
    # einen Retry zu erzwingen. Da auch das Mosaik nicht mehr gecacht ist, muss das Netzwerk
    # tatsächlich erneut angefragt werden statt über den Cache-Hit-Pfad überzuspringen.
    dest.unlink()
    call_count["n"] = 0
    mock_get.side_effect = fake_get  # frisch, damit der Zähler wieder von vorne beginnt

    result = ensure_horizon_texture(AREA_UTM, dest=dest, size_px=32)

    assert result == dest
    assert dest.exists()
    assert call_count["n"] > 0  # Netzwerk wurde tatsächlich erneut angefragt, nicht übersprungen


# --- Item 2: Attribution wird bei jeder Nutzung geloggt, auch beim Cache-Hit -------------------


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_attribution_is_logged_on_fresh_download(mock_get, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sentinel2_fetch, "_attribution_logged", False)
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", tmp_path / "cache_horizon_source")

    mock_get.side_effect = lambda url, params=None, headers=None, timeout=None: _jpeg_response(
        int(params["width"]), int(params["height"])
    )
    dest = tmp_path / "horizon_temp.tif"

    with caplog.at_level("INFO"):
        ensure_horizon_texture(AREA_UTM, dest=dest, size_px=32)

    assert config.EOX_ATTRIBUTION_NOTICE in caplog.text


@patch("world_to_beamng.terrain.sentinel2_fetch.requests.get")
def test_attribution_is_logged_on_cache_hit_too_but_only_once_per_process(mock_get, tmp_path, monkeypatch, caplog):
    import hashlib

    monkeypatch.setattr(sentinel2_fetch, "_attribution_logged", False)
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(config, "EOX_MOSAIC_MAX_PX", 50)
    monkeypatch.setattr(config, "EOX_MAX_REQUEST_PX", 50)
    mosaic_cache_dir = tmp_path / "cache_horizon_source"
    monkeypatch.setattr(config, "EOX_MOSAIC_CACHE_DIR", mosaic_cache_dir)

    mock_get.side_effect = lambda url, params=None, headers=None, timeout=None: _jpeg_response(
        int(params["width"]), int(params["height"])
    )

    # Rohmosaik VORAB anlegen (simuliert einen früheren Prozess/Lauf, der es bereits erfolgreich
    # geladen hat) - direkt über fetch_eox_mosaic(), NICHT über ensure_horizon_texture(), damit
    # _attribution_logged dabei nicht schon gesetzt wird. Cache-Schlüssel-Berechnung wie in
    # ensure_horizon_texture().
    sig = f"{round(AREA_UTM[0])}_{round(AREA_UTM[1])}_{round(AREA_UTM[2])}_{round(AREA_UTM[3])}_{config.EOX_WMS_LAYER}"
    cache_key = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    mosaic_path = mosaic_cache_dir / f"eox_mosaic_{cache_key}.tif"
    mosaic_cache_dir.mkdir(parents=True)
    success, failed_count = fetch_eox_mosaic(AREA_UTM, mosaic_path)
    assert success is True and failed_count == 0
    assert not sentinel2_fetch._attribution_logged
    calls_from_seeding = mock_get.call_count

    # Erster Aufruf von ensure_horizon_texture() in diesem Prozess ist bereits ein CACHE-HIT (das
    # Rohmosaik existiert schon) - die Lizenz-Attributionspflicht knüpft an die NUTZUNG des
    # Bildmaterials, nicht an den Download, also muss auch hier geloggt werden (vorher wurde die
    # Meldung nur im Cache-Miss-Zweig geloggt und wäre hier komplett ausgeblieben).
    dest1 = tmp_path / "horizon_temp_1.tif"
    with caplog.at_level("INFO"):
        ensure_horizon_texture(AREA_UTM, dest=dest1, size_px=32)
    assert mock_get.call_count == calls_from_seeding  # kein neuer Netzwerk-Request -> echter Cache-Hit
    assert caplog.text.count(config.EOX_ATTRIBUTION_NOTICE) == 1
    caplog.clear()

    # Zweiter Aufruf (anderes dest, wieder ein Cache-Hit) im SELBEN Prozess: nicht nochmal loggen.
    dest2 = tmp_path / "horizon_temp_2.tif"
    with caplog.at_level("INFO"):
        ensure_horizon_texture(AREA_UTM, dest=dest2, size_px=32)
    assert config.EOX_ATTRIBUTION_NOTICE not in caplog.text
    assert sentinel2_fetch._attribution_logged is True
