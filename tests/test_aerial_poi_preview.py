"""
Tests für das POI-Spawn-Vorschaubild (io/aerial.py::build_poi_preview_image()): schneidet einen Ausschnitt
um den POI aus dem bereits gebauten Luftbild aus - baut kein Mosaik/keine Quellbilder neu (wie
test_aerial_minimap.py für die Minimap).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

import json

from world_to_beamng.io.aerial import POI_PREVIEW_SIGNATURE_FILENAME, PoiPreviewBuilder, build_poi_preview_image


def _photo(dir_, name, color, size=200):
    Image.new("RGB", (size, size), color).save(dir_ / f"{name}.png", "PNG")


def _dominant_color(image_path):
    a = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
    return a.reshape(-1, 3).mean(axis=0)


def test_crop_is_centered_on_the_poi_and_resized(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", (250, 20, 20), size=200)  # 200 px, 200 m -> 1 px/m
    out = tmp_path / "preview.jpg"

    ok = build_poi_preview_image(
        textures, out, [{"name": "aerial_photo", "bounds": (0.0, 200.0, 0.0, 200.0)}],
        position_xy=(100.0, 100.0), crop_size_m=40.0, target_pixel_size=64,
    )

    assert ok is True
    assert Image.open(out).size == (64, 64)


def test_picks_the_photo_tile_containing_the_poi(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "west", (250, 20, 20))  # x 0..200
    _photo(textures, "east", (20, 20, 250))  # x 200..400
    photos = [
        {"name": "west", "bounds": (0.0, 200.0, 0.0, 200.0)},
        {"name": "east", "bounds": (200.0, 400.0, 0.0, 200.0)},
    ]
    out = tmp_path / "preview.jpg"

    build_poi_preview_image(textures, out, photos, position_xy=(300.0, 100.0), crop_size_m=40.0, target_pixel_size=32)

    means = _dominant_color(out)
    assert means[2] > 150 and means[0] < 60  # aus der östlichen (blauen) Kachel geschnitten


def test_poi_near_the_tile_edge_is_clamped_instead_of_failing(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", (250, 20, 20), size=200)
    out = tmp_path / "preview.jpg"

    # POI direkt am Bildrand - der volle 40m-Ausschnitt würde über den Bildrand hinausragen.
    ok = build_poi_preview_image(
        textures, out, [{"name": "aerial_photo", "bounds": (0.0, 200.0, 0.0, 200.0)}],
        position_xy=(2.0, 2.0), crop_size_m=40.0, target_pixel_size=32,
    )

    assert ok is True
    assert Image.open(out).size == (32, 32)


def test_missing_source_photo_returns_false_and_writes_nothing(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    out = tmp_path / "preview.jpg"

    ok = build_poi_preview_image(
        textures, out, [{"name": "aerial_photo", "bounds": (0.0, 200.0, 0.0, 200.0)}], position_xy=(100.0, 100.0)
    )

    assert ok is False
    assert not out.exists()


def test_no_matching_photo_tile_returns_false(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    out = tmp_path / "preview.jpg"

    ok = build_poi_preview_image(textures, out, [], position_xy=(100.0, 100.0))

    assert ok is False


def test_image_cache_is_populated_and_reused_across_calls(tmp_path):
    """Mehrere POIs auf derselben Foto-Kachel: das dekodierte Bild landet im Cache und wird
    wiederverwendet statt erneut von der Platte gelesen/dekodiert zu werden (siehe
    io/aerial.py::_load_rgb_photo()-Docstring - bei ~140-MB-Luftbildern sonst der Hauptzeitfresser
    beim Export)."""
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", (250, 20, 20), size=200)
    photos = [{"name": "aerial_photo", "bounds": (0.0, 200.0, 0.0, 200.0)}]
    cache = {}

    ok1 = build_poi_preview_image(
        textures, tmp_path / "a.jpg", photos, position_xy=(50.0, 50.0),
        crop_size_m=40.0, target_pixel_size=32, image_cache=cache,
    )
    ok2 = build_poi_preview_image(
        textures, tmp_path / "b.jpg", photos, position_xy=(150.0, 150.0),
        crop_size_m=40.0, target_pixel_size=32, image_cache=cache,
    )

    assert ok1 is True and ok2 is True
    assert len(cache) == 1  # dieselbe Foto-Kachel, nur einmal im Cache
    assert Image.open(tmp_path / "a.jpg").size == (32, 32)
    assert Image.open(tmp_path / "b.jpg").size == (32, 32)


def test_image_cache_avoids_reopening_the_source_file_a_second_time(tmp_path, monkeypatch):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", (250, 20, 20), size=200)
    photos = [{"name": "aerial_photo", "bounds": (0.0, 200.0, 0.0, 200.0)}]
    cache = {}

    import world_to_beamng.io.aerial as aerial_module

    real_open = aerial_module.Image.open
    opened_paths = []

    def counting_open(path, *args, **kwargs):
        opened_paths.append(Path(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(aerial_module.Image, "open", counting_open)

    build_poi_preview_image(
        textures, tmp_path / "a.jpg", photos, position_xy=(50.0, 50.0),
        crop_size_m=40.0, target_pixel_size=32, image_cache=cache,
    )
    build_poi_preview_image(
        textures, tmp_path / "b.jpg", photos, position_xy=(150.0, 150.0),
        crop_size_m=40.0, target_pixel_size=32, image_cache=cache,
    )

    assert len(opened_paths) == 1  # zweiter Aufruf bedient sich aus dem Cache


def test_without_a_cache_behaviour_is_unchanged(tmp_path):
    """image_cache=None (Default) - Rückwärtskompatibilität, jeder Aufruf dekodiert frisch."""
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", (250, 20, 20), size=200)
    photos = [{"name": "aerial_photo", "bounds": (0.0, 200.0, 0.0, 200.0)}]

    ok = build_poi_preview_image(
        textures, tmp_path / "a.jpg", photos, position_xy=(50.0, 50.0),
        crop_size_m=40.0, target_pixel_size=32,
    )

    assert ok is True
    assert Image.open(tmp_path / "a.jpg").size == (32, 32)


def _builder_setup(tmp_path, color=(250, 20, 20)):
    textures = tmp_path / "textures"
    textures.mkdir(exist_ok=True)
    _photo(textures, "aerial_photo", color, size=200)
    level = tmp_path / "level"
    photos = [{"name": "aerial_photo", "bounds": (0.0, 200.0, 0.0, 200.0)}]
    return textures, level, photos


def test_preview_builder_writes_the_preview_and_returns_the_relative_path(tmp_path):
    textures, level, photos = _builder_setup(tmp_path)
    builder = PoiPreviewBuilder(textures, level, photos)

    path = builder("spawn_dorf", (100.0, 100.0))
    builder.close()

    assert path == "spawn_previews/spawn_dorf.jpg"
    assert (level / path).exists()
    assert builder.built == 1 and builder.reused == 0
    assert "spawn_dorf" in json.loads((textures / POI_PREVIEW_SIGNATURE_FILENAME).read_text(encoding="utf-8"))


def test_preview_builder_reuses_an_unchanged_preview(tmp_path):
    textures, level, photos = _builder_setup(tmp_path)
    first = PoiPreviewBuilder(textures, level, photos)
    first("spawn_dorf", (100.0, 100.0))
    first.close()
    stamp = (level / "spawn_previews/spawn_dorf.jpg").stat().st_mtime_ns

    second = PoiPreviewBuilder(textures, level, photos)
    path = second("spawn_dorf", (100.0, 100.0))
    second.close()

    assert path == "spawn_previews/spawn_dorf.jpg"
    assert second.reused == 1 and second.built == 0
    assert (level / path).stat().st_mtime_ns == stamp


def test_preview_builder_rebuilds_when_the_spawn_moved(tmp_path):
    textures, level, photos = _builder_setup(tmp_path)
    first = PoiPreviewBuilder(textures, level, photos)
    first("spawn_dorf", (100.0, 100.0))
    first.close()

    second = PoiPreviewBuilder(textures, level, photos)
    second("spawn_dorf", (120.0, 100.0))
    second.close()

    assert second.built == 1 and second.reused == 0


def test_preview_builder_rebuilds_after_the_aerial_photo_changed(tmp_path):
    import os

    textures, level, photos = _builder_setup(tmp_path)
    first = PoiPreviewBuilder(textures, level, photos)
    first("spawn_dorf", (100.0, 100.0))
    first.close()

    _photo(textures, "aerial_photo", (20, 20, 250), size=200)
    source = textures / "aerial_photo.png"
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 10**9))
    second = PoiPreviewBuilder(textures, level, photos)
    path = second("spawn_dorf", (100.0, 100.0))
    second.close()

    assert second.built == 1
    means = _dominant_color(level / path)
    assert means[2] > 150 and means[0] < 60  # neues (blaues) Foto


def test_preview_builder_rebuilds_a_deleted_preview(tmp_path):
    textures, level, photos = _builder_setup(tmp_path)
    first = PoiPreviewBuilder(textures, level, photos)
    first("spawn_dorf", (100.0, 100.0))
    first.close()
    (level / "spawn_previews/spawn_dorf.jpg").unlink()

    second = PoiPreviewBuilder(textures, level, photos)
    second("spawn_dorf", (100.0, 100.0))
    second.close()

    assert second.built == 1
    assert (level / "spawn_previews/spawn_dorf.jpg").exists()


def test_preview_builder_without_a_matching_photo_returns_none(tmp_path):
    textures, level, _ = _builder_setup(tmp_path)
    builder = PoiPreviewBuilder(textures, level, [{"name": "missing", "bounds": (0.0, 200.0, 0.0, 200.0)}])

    assert builder("spawn_dorf", (100.0, 100.0)) is None
    builder.close()


def test_preview_builder_uses_several_tiles(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "west", (250, 20, 20))
    _photo(textures, "east", (20, 20, 250))
    photos = [
        {"name": "west", "bounds": (0.0, 200.0, 0.0, 200.0)},
        {"name": "east", "bounds": (200.0, 400.0, 0.0, 200.0)},
    ]
    builder = PoiPreviewBuilder(textures, tmp_path / "level", photos)
    west = builder("spawn_west", (100.0, 100.0))
    east = builder("spawn_east", (300.0, 100.0))
    builder.close()

    assert _dominant_color(tmp_path / "level" / west)[0] > 150
    assert _dominant_color(tmp_path / "level" / east)[2] > 150
