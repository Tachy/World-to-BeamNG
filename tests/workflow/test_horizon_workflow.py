"""
Tests: HorizonWorkflow.generate_horizon() wires the auto-download modules
(dgm30_fetch.ensure_dgm30_coverage, sentinel2_fetch.ensure_horizon_texture) correctly in front of the
respective existing loaders - pure wiring, no network, everything mocked.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng import config
from world_to_beamng.workflow.horizon_workflow import HorizonWorkflow

GLOBAL_OFFSET = (412000.0, 5297000.0, 250.0)


def _stub():
    return SimpleNamespace(materials=MagicMock(), items=MagicMock())


def test_dgm30_auto_download_runs_before_load_dgm30_tiles_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "DGM30_AUTO_DOWNLOAD", True)
    calls = []
    with patch(
        "world_to_beamng.terrain.dgm30_fetch.ensure_dgm30_coverage",
        side_effect=lambda *a, **k: calls.append("ensure_dgm30_coverage"),
    ) as mock_ensure, patch(
        "world_to_beamng.terrain.horizon.load_dgm30_tiles",
        side_effect=lambda *a, **k: (calls.append("load_dgm30_tiles"), (None, None))[1],
    ) as mock_load:
        result = HorizonWorkflow.generate_horizon(_stub(), global_offset=GLOBAL_OFFSET)

    assert calls == ["ensure_dgm30_coverage", "load_dgm30_tiles"]
    assert result is None  # (None, None) -> phase 5 skipped, regardless of the auto-download

    # Arguments: horizon_area_wgs84(global_offset) and config.DGM30_CACHE_DIR
    mock_ensure.assert_called_once()
    (area_wgs84, dgm30_dir), _kwargs = mock_ensure.call_args
    assert dgm30_dir == config.DGM30_CACHE_DIR
    assert len(area_wgs84) == 4
    mock_load.assert_called_once()


def test_dgm30_auto_download_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "DGM30_AUTO_DOWNLOAD", False)
    with patch("world_to_beamng.terrain.dgm30_fetch.ensure_dgm30_coverage") as mock_ensure, patch(
        "world_to_beamng.terrain.horizon.load_dgm30_tiles", return_value=(None, None)
    ) as mock_load:
        result = HorizonWorkflow.generate_horizon(_stub(), global_offset=GLOBAL_OFFSET)

    mock_ensure.assert_not_called()
    mock_load.assert_called_once()  # the existing loader still runs completely normally
    assert result is None


def _mesh_stub():
    vertices = np.array(
        [[0.0, 0.0, 100.0], [10.0, 0.0, 100.0], [0.0, 10.0, 100.0], [10.0, 10.0, 100.0]]
    )
    mesh = SimpleNamespace(vertex_manager=SimpleNamespace(vertices=vertices), uvs=[])
    return mesh, 2, 2


def _run_past_dgm30(monkeypatch, eox_auto_download):
    """Lets DGM30 run through with mocked but 'successful' points/heights so that the code
    reaches the Sentinel-2 block - load_sentinel2_geotiff returns None so that texture_horizon_mesh
    (and thus texconv.exe) is never reached (see horizon.py:637-653: with horizon_image=None
    texture_horizon_mesh immediately returns {"texture_path": None, "uv_map": None} without being called)."""
    monkeypatch.setattr(config, "EOX_AUTO_DOWNLOAD", eox_auto_download)
    monkeypatch.setattr(config, "DGM30_AUTO_DOWNLOAD", False)  # irrelevant for this test case, no noise

    height_points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    height_elevations = np.array([100.0, 100.0, 100.0, 100.0])
    mesh, nx, ny = _mesh_stub()

    return patch(
        "world_to_beamng.terrain.horizon.load_dgm30_tiles", return_value=(height_points, height_elevations)
    ), patch(
        "world_to_beamng.terrain.horizon.generate_horizon_mesh", return_value=(mesh, nx, ny)
    ), patch(
        "world_to_beamng.terrain.horizon.load_sentinel2_geotiff", return_value=None
    ), patch(
        "world_to_beamng.terrain.horizon.export_horizon_dae", return_value="horizon.dae"
    )


def test_sentinel2_auto_download_runs_before_load_sentinel2_geotiff_when_enabled(monkeypatch, tmp_path):
    p_dgm30, p_mesh, p_sentinel_load, p_export = _run_past_dgm30(monkeypatch, eox_auto_download=True)
    fetched_path = tmp_path / "cache_horizon_texture" / "horizon_texture_deadbeef.tif"
    calls = []

    def fake_ensure(*a, **k):
        calls.append("ensure_horizon_texture")
        return fetched_path

    with p_dgm30, p_mesh, p_export, patch(
        "world_to_beamng.terrain.sentinel2_fetch.ensure_horizon_texture", side_effect=fake_ensure
    ) as mock_ensure, patch(
        "world_to_beamng.terrain.horizon.load_sentinel2_geotiff",
        side_effect=lambda *a, **k: (calls.append("load_sentinel2_geotiff"), None)[1],
    ) as mock_load:
        result = HorizonWorkflow.generate_horizon(_stub(), global_offset=GLOBAL_OFFSET)

    assert calls == ["ensure_horizon_texture", "load_sentinel2_geotiff"]
    assert result is not None  # export ran through to the end (export_horizon_dae mocked)

    # Purely automatic: only horizon_bbox, no dest parameter anymore (no manual override).
    mock_ensure.assert_called_once()
    (horizon_bbox,), kwargs = mock_ensure.call_args
    assert len(horizon_bbox) == 4
    assert kwargs == {}
    mock_load.assert_called_once()
    (loaded_path, _bbox), _load_kwargs = mock_load.call_args
    assert loaded_path == fetched_path


def test_sentinel2_auto_download_is_skipped_when_disabled(monkeypatch):
    p_dgm30, p_mesh, p_sentinel_load, p_export = _run_past_dgm30(monkeypatch, eox_auto_download=False)
    with p_dgm30, p_mesh, p_sentinel_load, p_export, patch(
        "world_to_beamng.terrain.sentinel2_fetch.ensure_horizon_texture"
    ) as mock_ensure:
        result = HorizonWorkflow.generate_horizon(_stub(), global_offset=GLOBAL_OFFSET)

    mock_ensure.assert_not_called()
    assert result is not None


def test_dgm30_missing_still_skips_phase5_regardless_of_auto_download(monkeypatch):
    """The existing skip path remains fully functional: if load_dgm30_tiles returns (None, None),
    generate_horizon() still returns None - regardless of whether ensure_dgm30_coverage ran."""
    monkeypatch.setattr(config, "DGM30_AUTO_DOWNLOAD", True)
    with patch("world_to_beamng.terrain.dgm30_fetch.ensure_dgm30_coverage"), patch(
        "world_to_beamng.terrain.horizon.load_dgm30_tiles", return_value=(None, None)
    ):
        result = HorizonWorkflow.generate_horizon(_stub(), global_offset=GLOBAL_OFFSET)

    assert result is None
