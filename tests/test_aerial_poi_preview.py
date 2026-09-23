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

from world_to_beamng.io.aerial import build_poi_preview_image


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
