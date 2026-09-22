"""
Tests für den Vier-Bilder-Modus im Luftbild-Code (io/aerial.py): ein Foto je DGM1-Kachel.

Mit kleinen synthetischen Quell-Zips (1 m/px): jedes Foto muss genau den Ausschnitt seiner Kachel enthalten,
auch wenn ein Quellbild über eine Kachelgrenze reicht.
"""

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from PIL import Image

from world_to_beamng.io import aerial
from world_to_beamng.io.aerial import (
    aerial_photo_is_current,
    aerial_photos_signature,
    ensure_aerial_photos,
    process_aerial_tiles,
    write_aerial_photo_signature,
)

OFFSET = (1000.0, 2000.0, 0.0)
RED, BLUE, GREEN = (250, 20, 20), (20, 20, 250), (20, 250, 20)


def _png(width, height, left_color, right_color=None):
    """PNG, links/rechts in zwei Farben (right_color=None: einfarbig)."""
    image = Image.new("RGB", (width, height), left_color)
    if right_color:
        image.paste(right_color, (width // 2, 0, width, height))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _zip(path, name, png_bytes, x_origin, y_origin):
    tfw = f"1.0\n0.0\n0.0\n-1.0\n{x_origin}\n{y_origin}\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}.png", png_bytes)
        z.writestr(f"{name}.tfw", tfw)


def _dominant(image_path):
    a = np.asarray(Image.open(image_path).convert("RGB"), dtype=float)
    means = a.reshape(-1, 3).mean(axis=0)
    return ("red", "green", "blue")[int(np.argmax(means))], means


def _photos():
    # zwei Kacheln nebeneinander, je 20 m breit und 20 m hoch (lokale Koordinaten)
    return [
        {"name": "aerial_photo_0", "bounds": (0.0, 20.0, 0.0, 20.0)},
        {"name": "aerial_photo_1", "bounds": (20.0, 40.0, 0.0, 20.0)},
    ]


def test_each_photo_shows_only_the_area_of_its_own_tile(tmp_path):
    src = tmp_path / "satellite"
    src.mkdir()
    # ein Quellbild pro Kachel (links rot, rechts blau) - obere linke Ecke = (x_origin, y_origin)
    _zip(src / "a.zip", "a", _png(20, 20, RED), OFFSET[0] + 0.0, OFFSET[1] + 20.0)
    _zip(src / "b.zip", "b", _png(20, 20, BLUE), OFFSET[0] + 20.0, OFFSET[1] + 20.0)
    out = tmp_path / "out"

    saved = process_aerial_tiles(src, out, _photos(), OFFSET, target_pixel_size=20)

    assert saved == 2
    assert _dominant(out / "aerial_photo_0.png")[0] == "red"
    assert _dominant(out / "aerial_photo_1.png")[0] == "blue"
    assert Image.open(out / "aerial_photo_0.png").size == (20, 20)


def test_a_source_image_that_crosses_the_tile_border_is_split_correctly(tmp_path):
    src = tmp_path / "satellite"
    src.mkdir()
    # EIN Quellbild über beide Kacheln: linke Hälfte rot, rechte Hälfte blau
    _zip(src / "wide.zip", "wide", _png(40, 20, RED, BLUE), OFFSET[0] + 0.0, OFFSET[1] + 20.0)
    out = tmp_path / "out"

    process_aerial_tiles(src, out, _photos(), OFFSET, target_pixel_size=20)

    _, left = _dominant(out / "aerial_photo_0.png")
    _, right = _dominant(out / "aerial_photo_1.png")
    assert left[0] > 150 and left[2] < 60  # Kachel 0: rot
    assert right[2] > 150 and right[0] < 60  # Kachel 1: blau


def test_tiles_stacked_north_south_use_their_own_rows_of_the_source(tmp_path):
    src = tmp_path / "satellite"
    src.mkdir()
    # Quellbild 20x40 px: oben (Norden) grün, unten (Süden) rot - Kachel Nord: y 20..40, Kachel Süd: y 0..20
    image = Image.new("RGB", (20, 40), GREEN)
    image.paste(RED, (0, 20, 20, 40))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    _zip(src / "tall.zip", "tall", buffer.getvalue(), OFFSET[0], OFFSET[1] + 40.0)
    photos = [
        {"name": "aerial_photo_0", "bounds": (0.0, 20.0, 0.0, 20.0)},  # Süd
        {"name": "aerial_photo_1", "bounds": (0.0, 20.0, 20.0, 40.0)},  # Nord
    ]
    out = tmp_path / "out"

    process_aerial_tiles(src, out, photos, OFFSET, target_pixel_size=20)

    assert _dominant(out / "aerial_photo_0.png")[0] == "red"
    assert _dominant(out / "aerial_photo_1.png")[0] == "green"


def test_photo_is_downscaled_to_the_target_size(tmp_path):
    src = tmp_path / "satellite"
    src.mkdir()
    _zip(src / "a.zip", "a", _png(20, 20, RED), OFFSET[0], OFFSET[1] + 20.0)
    out = tmp_path / "out"

    process_aerial_tiles(src, out, _photos()[:1], OFFSET, target_pixel_size=8)

    assert Image.open(out / "aerial_photo_0.png").size == (8, 8)


# --- Cache/Signatur für mehrere Fotos ---------------------------------------------------------------


@pytest.fixture
def dirs(tmp_path):
    src = tmp_path / "satellite"
    src.mkdir()
    (src / "x.zip").write_bytes(b"x" * 10)
    out = tmp_path / "out"
    out.mkdir()
    return src, out


def test_signature_lists_every_photo_and_changes_with_the_tile_set(dirs):
    src, _ = dirs

    two = aerial_photos_signature(src, _photos(), OFFSET, 8192)
    one = aerial_photos_signature(src, _photos()[:1], OFFSET, 8192)

    assert [p["name"] for p in two["photos"]] == ["aerial_photo_0", "aerial_photo_1"]
    assert two != one


def test_current_requires_every_photo_file(dirs):
    src, out = dirs
    signature = aerial_photos_signature(src, _photos(), OFFSET, 8192)
    write_aerial_photo_signature(out, signature)
    (out / "aerial_photo_0.png").write_bytes(b"1")

    assert not aerial_photo_is_current(out, signature)  # Foto 1 fehlt

    (out / "aerial_photo_1.png").write_bytes(b"2")
    assert aerial_photo_is_current(out, signature)


def test_ensure_builds_all_photos_once_and_removes_stale_photos_of_the_other_mode(dirs, monkeypatch):
    src, out = dirs
    (out / "aerial_photo.png").write_bytes(b"old single photo")  # Gesamtfoto aus dem anderen Modus
    calls = []

    def fake(aerial_dir, output_dir, photos, global_offset, target_pixel_size=None):
        calls.append([p["name"] for p in photos])
        for p in photos:
            (Path(output_dir) / f"{p['name']}.png").write_bytes(b"new")
        return len(photos)

    monkeypatch.setattr(aerial, "process_aerial_tiles", fake)

    first = ensure_aerial_photos(src, out, _photos(), OFFSET, target_pixel_size=8192)
    second = ensure_aerial_photos(src, out, _photos(), OFFSET, target_pixel_size=8192)

    assert (first, second) == ("built", "current")
    assert calls == [["aerial_photo_0", "aerial_photo_1"]]
    assert not (out / "aerial_photo.png").exists()  # veraltetes Gesamtfoto entfernt
    assert (out / "aerial_photo_0.png").exists() and (out / "aerial_photo_1.png").exists()


def test_ensure_removes_tile_photos_when_switching_back_to_one_photo(dirs, monkeypatch):
    src, out = dirs
    (out / "aerial_photo_0.png").write_bytes(b"tile")
    (out / "aerial_photo_3.png").write_bytes(b"tile")
    monkeypatch.setattr(aerial, "process_aerial_images", lambda *a, **k: (out / "aerial_photo.png").write_bytes(b"single") or 1)

    status = ensure_aerial_photos(src, out, [{"name": "aerial_photo", "bounds": (-2000.0, 2096.0, -2000.0, 2096.0)}], OFFSET, 8192)

    assert status == "built"
    assert sorted(p.name for p in out.glob("aerial_photo*.png")) == ["aerial_photo.png"]


def test_ensure_does_not_touch_other_files(dirs, monkeypatch):
    src, out = dirs
    (out / "horizon_sentinel2.dds").write_bytes(b"h")
    (out / "_flat_normal_8192.png").write_bytes(b"f")
    monkeypatch.setattr(aerial, "process_aerial_tiles", lambda a, o, photos, g, target_pixel_size=None: [
        (Path(o) / f"{p['name']}.png").write_bytes(b"n") for p in photos] and len(photos))

    ensure_aerial_photos(src, out, _photos(), OFFSET, 8192)

    assert (out / "horizon_sentinel2.dds").exists() and (out / "_flat_normal_8192.png").exists()
