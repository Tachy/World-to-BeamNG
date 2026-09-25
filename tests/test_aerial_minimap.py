"""
Tests for the BigMap preview image (io/aerial.py::build_minimap_image()/minimap_info_json_fields()): builds the
minimap from the already finished aerial_photo*.png (no source images/zips needed, unlike
tests/test_aerial_photo_tiles.py, which assembles the aerial photos itself first).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

import os

from world_to_beamng.io.aerial import build_minimap_image, ensure_minimap_image, minimap_info_json_fields

RED, BLUE, GREEN = (250, 20, 20), (20, 20, 250), (20, 250, 20)


def _photo(dir_, name, color, size=20):
    Image.new("RGB", (size, size), color).save(dir_ / f"{name}.png", "PNG")


def _dominant(image_path):
    a = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
    means = a.reshape(-1, 3).mean(axis=0)
    return ("red", "green", "blue")[int(np.argmax(means))], means


def test_single_mosaic_is_resized_onto_the_minimap_canvas(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", RED, size=100)
    out = tmp_path / "minimap.png"

    ok = build_minimap_image(textures, out, [{"name": "aerial_photo", "bounds": (0.0, 40.0, 0.0, 40.0)}], (0.0, 40.0, 0.0, 40.0), target_pixel_size=16)

    assert ok is True
    assert Image.open(out).size == (16, 16)
    assert _dominant(out)[0] == "red"


def test_tiles_are_placed_side_by_side_east_west(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo_0", RED)  # west: x 0..20
    _photo(textures, "aerial_photo_1", BLUE)  # east: x 20..40
    photos = [
        {"name": "aerial_photo_0", "bounds": (0.0, 20.0, 0.0, 20.0)},
        {"name": "aerial_photo_1", "bounds": (20.0, 40.0, 0.0, 20.0)},
    ]
    out = tmp_path / "minimap.png"

    build_minimap_image(textures, out, photos, (0.0, 40.0, 0.0, 20.0), target_pixel_size=40)

    image = np.asarray(Image.open(out).convert("RGB"), dtype=float)
    left, right = image[:, :20].reshape(-1, 3).mean(axis=0), image[:, 20:].reshape(-1, 3).mean(axis=0)
    assert left[0] > 150 and left[2] < 60  # left: red
    assert right[2] > 150 and right[0] < 60  # right: blue


def test_tiles_are_stacked_north_south_with_row_0_as_north(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo_south", RED)  # y 0..20 (south)
    _photo(textures, "aerial_photo_north", GREEN)  # y 20..40 (north)
    photos = [
        {"name": "aerial_photo_south", "bounds": (0.0, 20.0, 0.0, 20.0)},
        {"name": "aerial_photo_north", "bounds": (0.0, 20.0, 20.0, 40.0)},
    ]
    out = tmp_path / "minimap.png"

    build_minimap_image(textures, out, photos, (0.0, 20.0, 0.0, 40.0), target_pixel_size=40)

    image = np.asarray(Image.open(out).convert("RGB"), dtype=float)
    top, bottom = image[:20].reshape(-1, 3).mean(axis=0), image[20:].reshape(-1, 3).mean(axis=0)
    assert top[1] > 150 and top[0] < 60  # top (row 0) = north = green
    assert bottom[0] > 150 and bottom[1] < 60  # bottom = south = red


def test_missing_source_photo_returns_false_and_writes_nothing(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    out = tmp_path / "minimap.png"

    ok = build_minimap_image(textures, out, [{"name": "aerial_photo", "bounds": (0.0, 20.0, 0.0, 20.0)}], (0.0, 20.0, 0.0, 20.0))

    assert ok is False
    assert not out.exists()


def test_minimap_info_json_fields_uses_the_northwest_corner_as_offset():
    fields = minimap_info_json_fields(x_min=-500.0, y_max=1200.0, size_m=2048.0)

    assert fields["size"] == [2048.0, 2048.0]
    assert fields["minimap"] == [{"file": "minimap/terrain.png", "size": [2048.0, 2048.0], "offset": [-500.0, 1200.0]}]


def test_minimap_info_json_fields_accepts_a_custom_file_path():
    fields = minimap_info_json_fields(x_min=0.0, y_max=0.0, size_m=100.0, relative_file="mini/custom.png")

    assert fields["minimap"][0]["file"] == "mini/custom.png"


def _ensure(textures, out):
    photos = [{"name": "aerial_photo", "bounds": (0.0, 40.0, 0.0, 40.0)}]
    return ensure_minimap_image(textures, out, photos, (0.0, 40.0, 0.0, 40.0), target_pixel_size=16)


def test_ensure_minimap_reuses_a_current_minimap(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", RED)
    out = tmp_path / "minimap.png"

    assert _ensure(textures, out) == "built"
    stamp = out.stat().st_mtime_ns
    assert _ensure(textures, out) == "current"
    assert out.stat().st_mtime_ns == stamp


def test_ensure_minimap_rebuilds_after_the_aerial_photo_changed(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", RED)
    out = tmp_path / "minimap.png"
    assert _ensure(textures, out) == "built"

    _photo(textures, "aerial_photo", BLUE)
    source = textures / "aerial_photo.png"
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 10**9))  # guaranteed newer mtime

    assert _ensure(textures, out) == "built"
    assert _dominant(out)[0] == "blue"


def test_ensure_minimap_rebuilds_when_the_minimap_file_is_gone(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()
    _photo(textures, "aerial_photo", RED)
    out = tmp_path / "minimap.png"
    assert _ensure(textures, out) == "built"

    out.unlink()

    assert _ensure(textures, out) == "built"
    assert out.exists()


def test_ensure_minimap_reports_a_missing_source_photo(tmp_path):
    textures = tmp_path / "textures"
    textures.mkdir()

    assert _ensure(textures, tmp_path / "minimap.png") == "missing"
