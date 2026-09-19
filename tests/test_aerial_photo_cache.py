"""
Tests für den Cache des Gesamt-Luftbilds (io/aerial.py).

Fehlerbild: Wurde von einer auf vier DGM1-Kacheln umgestellt, blieb das alte Ein-Kachel-Foto
(2 km) im Level liegen, weil der Exporter das Foto nur an seiner Existenz erkannte. Es wurde dann auf
die 4 km gestreckt. Das Foto muss deshalb neu gebaut werden, sobald sich Fläche, Ursprung, Größe oder
die Quellbilder ändern.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from world_to_beamng.io import aerial
from world_to_beamng.io.aerial import (
    AERIAL_PHOTO_FILENAME,
    aerial_photo_is_current,
    aerial_photo_signature,
    ensure_aerial_photo,
    write_aerial_photo_signature,
)

ONE_TILE = (-1000.0, 1048.0, -1000.0, 1048.0)
FOUR_TILES = (-2000.0, 2096.0, -2000.0, 2096.0)
OFFSET = (401000.0, 5298000.0, 0.0)


@pytest.fixture
def dirs(tmp_path):
    aerial_dir = tmp_path / "DOP20"
    aerial_dir.mkdir()
    (aerial_dir / "dop20rgb_32_399_5296_2_bw.zip").write_bytes(b"a" * 100)
    (aerial_dir / "dop20rgb_32_401_5298_2_bw.zip").write_bytes(b"b" * 200)
    textures = tmp_path / "textures"
    textures.mkdir()
    return aerial_dir, textures


def _signature(aerial_dir, bounds=FOUR_TILES, offset=OFFSET, size=8192):
    return aerial_photo_signature(aerial_dir, bounds, offset, size)


def test_signature_changes_with_area_origin_size_and_source_images(dirs):
    aerial_dir, _ = dirs
    base = _signature(aerial_dir)

    assert _signature(aerial_dir) == base  # deterministisch
    assert _signature(aerial_dir, bounds=ONE_TILE) != base  # andere Fläche
    assert _signature(aerial_dir, offset=(402000.0, 5299000.0, 0.0)) != base  # anderer Ursprung
    assert _signature(aerial_dir, size=16384) != base  # andere Auflösung
    (aerial_dir / "dop20rgb_32_399_5298_2_bw.zip").write_bytes(b"c" * 50)
    assert _signature(aerial_dir) != base  # weiteres Quellbild


def test_a_legacy_photo_without_signature_is_stale(dirs):
    # Genau der Fehlerfall: ein Foto von gestern ohne Signatur darf nicht als aktuell gelten
    aerial_dir, textures = dirs
    (textures / AERIAL_PHOTO_FILENAME).write_bytes(b"old photo")

    assert not aerial_photo_is_current(textures, _signature(aerial_dir))


def test_photo_with_matching_signature_is_current(dirs):
    aerial_dir, textures = dirs
    (textures / AERIAL_PHOTO_FILENAME).write_bytes(b"photo")
    write_aerial_photo_signature(textures, _signature(aerial_dir))

    assert aerial_photo_is_current(textures, _signature(aerial_dir))


def test_photo_is_stale_when_the_area_changed_or_the_file_is_missing(dirs):
    aerial_dir, textures = dirs
    write_aerial_photo_signature(textures, _signature(aerial_dir, bounds=ONE_TILE))
    (textures / AERIAL_PHOTO_FILENAME).write_bytes(b"photo")

    assert not aerial_photo_is_current(textures, _signature(aerial_dir, bounds=FOUR_TILES))

    write_aerial_photo_signature(textures, _signature(aerial_dir))
    (textures / AERIAL_PHOTO_FILENAME).unlink()
    assert not aerial_photo_is_current(textures, _signature(aerial_dir))


def test_corrupt_signature_file_counts_as_stale(dirs):
    aerial_dir, textures = dirs
    (textures / AERIAL_PHOTO_FILENAME).write_bytes(b"photo")
    (textures / "aerial_photo.json").write_text("{kaputt", encoding="utf-8")

    assert not aerial_photo_is_current(textures, _signature(aerial_dir))


class _Recorder:
    def __init__(self, textures):
        self.calls = []
        self.textures = textures

    def __call__(self, aerial_dir, output_dir, grid_bounds, global_offset, target_pixel_size=None):
        self.calls.append(grid_bounds)
        (Path(output_dir) / AERIAL_PHOTO_FILENAME).write_bytes(b"new photo")
        return 1


def test_ensure_rebuilds_a_legacy_photo_and_then_leaves_it_alone(dirs, monkeypatch):
    aerial_dir, textures = dirs
    (textures / AERIAL_PHOTO_FILENAME).write_bytes(b"old one-tile photo")
    recorder = _Recorder(textures)
    monkeypatch.setattr(aerial, "process_aerial_images", recorder)

    first = ensure_aerial_photo(aerial_dir, textures, FOUR_TILES, OFFSET, target_pixel_size=8192)
    second = ensure_aerial_photo(aerial_dir, textures, FOUR_TILES, OFFSET, target_pixel_size=8192)

    assert first == "built" and second == "current"
    assert len(recorder.calls) == 1
    assert (textures / AERIAL_PHOTO_FILENAME).read_bytes() == b"new photo"


def test_ensure_rebuilds_when_switching_between_one_and_four_tiles(dirs, monkeypatch):
    aerial_dir, textures = dirs
    recorder = _Recorder(textures)
    monkeypatch.setattr(aerial, "process_aerial_images", recorder)

    assert ensure_aerial_photo(aerial_dir, textures, ONE_TILE, OFFSET, target_pixel_size=8192) == "built"
    assert ensure_aerial_photo(aerial_dir, textures, FOUR_TILES, OFFSET, target_pixel_size=8192) == "built"
    assert ensure_aerial_photo(aerial_dir, textures, ONE_TILE, OFFSET, target_pixel_size=8192) == "built"
    assert recorder.calls == [ONE_TILE, FOUR_TILES, ONE_TILE]


def test_ensure_keeps_an_existing_photo_when_there_are_no_source_images(tmp_path):
    empty = tmp_path / "DOP20"
    empty.mkdir()
    textures = tmp_path / "textures"
    textures.mkdir()
    (textures / AERIAL_PHOTO_FILENAME).write_bytes(b"photo")

    assert ensure_aerial_photo(empty, textures, FOUR_TILES, OFFSET, target_pixel_size=8192) == "none"
    assert (textures / AERIAL_PHOTO_FILENAME).read_bytes() == b"photo"


def test_ensure_does_not_write_a_signature_when_building_failed(dirs, monkeypatch):
    aerial_dir, textures = dirs
    monkeypatch.setattr(aerial, "process_aerial_images", lambda *a, **k: 0)

    assert ensure_aerial_photo(aerial_dir, textures, FOUR_TILES, OFFSET, target_pixel_size=8192) == "failed"
    assert not (textures / "aerial_photo.json").exists()
