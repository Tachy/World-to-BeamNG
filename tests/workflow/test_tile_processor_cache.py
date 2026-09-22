"""
Tests für world_to_beamng.workflow.tile_processor.TileProcessor.load_height_data() als dünner
Cache-Wrapper um terrain.elevation_io.read_elevation_tile() - Cache-Key bleibt dateibasiert,
ein Cache-Hit vermeidet ein zweites Parsen der Quelldatei.
"""

import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng.core.cache_manager import CacheManager
from world_to_beamng.terrain import elevation_io
from world_to_beamng.workflow.tile_processor import TileProcessor


def _xyz_zip(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.xyz", "0 0 10.0\n1 0 10.5\n")
    return path


def test_load_height_data_reads_the_file_once_and_then_serves_from_cache(tmp_path, monkeypatch):
    zip_path = _xyz_zip(tmp_path / "tile.zip")
    processor = TileProcessor(CacheManager(tmp_path / "cache"))
    tile = {"filepath": zip_path}

    calls = []
    real_read = elevation_io.read_elevation_tile

    def spy(filepath):
        calls.append(filepath)
        return real_read(filepath)

    monkeypatch.setattr(elevation_io, "read_elevation_tile", spy)

    points1, elevations1 = processor.load_height_data(tile)
    points2, elevations2 = processor.load_height_data(tile)  # zweiter Aufruf -> Cache-Hit

    assert len(calls) == 1  # nur einmal tatsächlich geparst
    assert np.array_equal(points1, points2)
    assert np.array_equal(elevations1, elevations2)
    assert list(elevations1) == [10.0, 10.5]


def test_load_height_data_cache_key_is_file_hash_based(tmp_path):
    zip_a = _xyz_zip(tmp_path / "a.zip")
    zip_b = _xyz_zip(tmp_path / "b.zip")  # inhaltlich identisch, anderer Dateiname/-pfad
    processor = TileProcessor(CacheManager(tmp_path / "cache"))

    points_a, elevations_a = processor.load_height_data({"filepath": zip_a})
    points_b, elevations_b = processor.load_height_data({"filepath": zip_b})

    # gleicher Inhalt -> gleicher Hash -> derselbe Cache-Datei liefert dasselbe Ergebnis
    assert np.array_equal(points_a, points_b)
    assert np.array_equal(elevations_a, elevations_b)


def test_missing_file_returns_none_without_reading(tmp_path, monkeypatch):
    processor = TileProcessor(CacheManager(tmp_path / "cache"))
    calls = []
    monkeypatch.setattr(elevation_io, "read_elevation_tile", lambda fp: calls.append(fp))

    points, elevations = processor.load_height_data({"filepath": tmp_path / "does_not_exist.zip"})

    assert (points, elevations) == (None, None)
    assert calls == []
