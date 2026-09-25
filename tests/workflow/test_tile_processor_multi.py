"""Tests for world_to_beamng.workflow.tile_processor.TileProcessor.load_height_data_multi()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.core.cache_manager import CacheManager
from world_to_beamng.workflow.tile_processor import TileProcessor


def _processor(tmp_path):
    return TileProcessor(CacheManager(tmp_path))


def test_load_height_data_multi_combines_tiles(tmp_path, monkeypatch):
    processor = _processor(tmp_path)

    fake_data = {
        "tile_a": (np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([100.0, 101.0])),
        "tile_b": (np.array([[2.0, 0.0], [3.0, 0.0]]), np.array([102.0, 103.0])),
    }

    def fake_load(tile):
        return fake_data[tile["filename"]]

    monkeypatch.setattr(processor, "load_height_data", fake_load)

    points, elevations = processor.load_height_data_multi(
        [{"filename": "tile_a"}, {"filename": "tile_b"}]
    )

    assert points.shape == (4, 2)
    assert np.array_equal(points, np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert np.array_equal(elevations, np.array([100.0, 101.0, 102.0, 103.0]))


def test_load_height_data_multi_fails_if_one_tile_missing(tmp_path, monkeypatch):
    processor = _processor(tmp_path)

    def fake_load(tile):
        if tile["filename"] == "tile_bad":
            return None, None
        return np.array([[0.0, 0.0]]), np.array([100.0])

    monkeypatch.setattr(processor, "load_height_data", fake_load)

    points, elevations = processor.load_height_data_multi(
        [{"filename": "tile_ok"}, {"filename": "tile_bad"}]
    )

    assert points is None
    assert elevations is None


def test_load_height_data_multi_empty_list(tmp_path):
    processor = _processor(tmp_path)

    points, elevations = processor.load_height_data_multi([])

    assert points is None
    assert elevations is None


if __name__ == "__main__":
    import tempfile
    from unittest import mock

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        class _Patcher:
            def __init__(self):
                self._targets = []

            def setattr(self, obj, name, value):
                self._targets.append((obj, name))
                setattr(obj, name, value)

        test_load_height_data_multi_combines_tiles(tmp_path, _Patcher())
        print("[OK] test_load_height_data_multi_combines_tiles")
        test_load_height_data_multi_fails_if_one_tile_missing(tmp_path, _Patcher())
        print("[OK] test_load_height_data_multi_fails_if_one_tile_missing")
        test_load_height_data_multi_empty_list(tmp_path)
        print("[OK] test_load_height_data_multi_empty_list")
        print("Alle Tests bestanden.")
