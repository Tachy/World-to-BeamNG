"""
Tests: The finished tree instances (Poisson disk sampling + elevation interpolation +
instance generation, the most expensive part of ForestWorkflow.process_tile()) are cached per
area/elevation data version - see forest_workflow.py::_forest_cache_key() & co.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.workflow.forest_workflow import ForestWorkflow

TILE_BOUNDS = (-2000.0, -2000.0, 2000.0, 2000.0)
OFFSET = (412000.0, 5297000.0, 250.0)


def _workflow():
    return ForestWorkflow(config)


# --------------------------------------------------------------- _forest_cache_key()


def test_cache_key_is_none_without_height_hash():
    workflow = _workflow()
    assert workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash=None) is None
    assert workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="") is None


def test_cache_key_is_stable_for_identical_inputs():
    workflow = _workflow()
    key_a = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="abc123")
    key_b = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="abc123")
    assert key_a is not None
    assert key_a == key_b


def test_cache_key_keeps_height_hash_visible_as_a_prefix():
    """height_hash must appear readably in the key/file name, like everywhere else in this
    pipeline (osm_all_<height_hash>.json, grid_v3_grid_<height_hash>_..., dgm30_horizon_<tile_hash>_...)
    - it makes the related cache files of a run recognizable at a glance, instead of hiding everything in
    a single opaque hash."""
    workflow = _workflow()
    key = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="4113e78937c1")
    assert key.startswith("4113e78937c1_")


def test_cache_key_changes_when_height_hash_changes():
    workflow = _workflow()
    key_a = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="abc123")
    key_b = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="xyz789")
    assert key_a != key_b  # e.g. switching between two test regions (BaWue <-> Switzerland)


def test_cache_key_changes_when_tile_bounds_change():
    workflow = _workflow()
    key_a = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="abc123")
    other_bounds = (-1000.0, -1000.0, 1000.0, 1000.0)
    key_b = workflow._forest_cache_key(other_bounds, OFFSET, height_hash="abc123")
    assert key_a != key_b


def test_cache_key_changes_when_global_offset_changes():
    workflow = _workflow()
    key_a = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="abc123")
    other_offset = (413000.0, 5298000.0, 250.0)
    key_b = workflow._forest_cache_key(TILE_BOUNDS, other_offset, height_hash="abc123")
    assert key_a != key_b


# ------------------------------------------------- _load/_save_cached_tree_instances()


def test_save_and_load_cached_tree_instances_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    workflow = _workflow()
    cache_key = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="abc123")
    instances = [
        {"type": "oak", "pos": [1.0, 2.0, 3.0], "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "scale": 1.1}
    ]

    assert workflow._load_cached_tree_instances(cache_key) is None  # no cache yet

    workflow._save_cached_tree_instances(cache_key, instances, forests_count=3)
    loaded_instances, loaded_forests_count = workflow._load_cached_tree_instances(cache_key)

    assert loaded_instances == instances
    assert loaded_forests_count == 3


def test_load_cached_tree_instances_returns_none_without_a_cache_key():
    workflow = _workflow()
    assert workflow._load_cached_tree_instances(None) is None


def test_save_cached_tree_instances_is_a_noop_without_a_cache_key(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    workflow = _workflow()

    workflow._save_cached_tree_instances(None, [{"type": "oak"}], forests_count=1)

    assert not (tmp_path / "cache").exists()  # nothing written, no crash


def test_corrupt_cache_file_is_ignored_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    workflow = _workflow()
    cache_key = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="abc123")

    (tmp_path / "cache").mkdir(parents=True)
    workflow._forest_cache_path(cache_key).write_text("not valid json", encoding="utf-8")

    assert workflow._load_cached_tree_instances(cache_key) is None


# ------------------------------------------------------- process_tile() cache short-circuit


def test_process_tile_uses_cached_tree_instances_and_skips_osm_load(tmp_path, monkeypatch):
    """Regression: a cache hit must not even load the OSM data (which is itself already
    cached, but would still be a noticeable detour) - process_tile() must return directly with the
    cached tree instances."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    workflow = _workflow()
    # set_forest_config() is not needed for the cache short-circuit - only the two object checks
    # at the start of process_tile() must be truthy.
    workflow.normalizer = object()
    workflow.instance_generator = object()

    cached_instances = [
        {"type": "oak", "pos": [1.0, 2.0, 3.0], "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "scale": 1.1},
        {"type": "beech", "pos": [4.0, 5.0, 6.0], "rotationMatrix": [1, 0, 0, 0, 1, 0, 0, 0, 1], "scale": 0.9},
    ]
    cache_key = workflow._forest_cache_key(TILE_BOUNDS, OFFSET, height_hash="abc123")
    workflow._save_cached_tree_instances(cache_key, cached_instances, forests_count=5)

    with patch("world_to_beamng.osm.downloader.get_osm_data") as mock_get_osm:
        result = workflow.process_tile(
            tile_bounds=TILE_BOUNDS,
            tile_name="combined_area",
            height_hash="abc123",
            global_offset=OFFSET,
        )

    mock_get_osm.assert_not_called()  # cache hit -> OSM loading skipped entirely
    assert result["status"] == "success"
    assert result["tree_count"] == 2
    assert result["forests_count"] == 5
    assert result["tree_instances"] == cached_instances
    assert workflow.all_tree_instances == cached_instances  # for the later forest.forest4.json finalization
