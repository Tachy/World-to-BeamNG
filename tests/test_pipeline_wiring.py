"""Signature regression test: ensures that BeamNGExporter/TerrainWorkflow still pass on
a Pipeline/PipelineTask."""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.export.beamng_exporter import BeamNGExporter
from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow


def test_beamng_exporter_requires_a_pipeline():
    params = inspect.signature(BeamNGExporter.__init__).parameters
    assert "pipeline" in params


def test_terrain_workflow_process_tile_requires_a_task():
    params = inspect.signature(TerrainWorkflow.process_tile).parameters
    assert "task" in params


def test_terrain_workflow_export_tile_requires_a_task():
    params = inspect.signature(TerrainWorkflow.export_tile).parameters
    assert "task" in params
