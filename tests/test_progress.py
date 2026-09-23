"""Tests für world_to_beamng/progress.py: Pipeline/PipelineTask/Subtask."""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rich.console import Console

from world_to_beamng import progress as progress_module
from world_to_beamng.progress import Pipeline


@pytest.fixture
def buffer(monkeypatch):
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, width=120)
    monkeypatch.setattr(progress_module, "console", test_console)
    return buf


def test_task_prints_start_and_done_summary(buffer):
    pipeline = Pipeline()
    with pipeline.task("Texturen") as task:
        task.done("5 Texturen geprüft")

    output = buffer.getvalue()
    assert "Texturen" in output
    assert "✓" in output
    assert "5 Texturen geprüft" in output


def test_task_auto_finishes_without_explicit_done(buffer):
    pipeline = Pipeline()
    with pipeline.task("Vorbereitung"):
        pass

    output = buffer.getvalue()
    assert "✓ Vorbereitung" in output


def test_task_failure_marks_red_cross_and_reraises(buffer):
    pipeline = Pipeline()
    with pytest.raises(ValueError):
        with pipeline.task("Terrain + Straßen") as task:
            raise ValueError("kaputt")

    output = buffer.getvalue()
    assert "✗" in output
    assert "kaputt" in output


def test_subtask_with_total_reaches_100_percent(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + Straßen") as task:
        with task.subtask("Bäume platzieren", total=3) as sub:
            for _ in range(3):
                sub.advance()

    output = buffer.getvalue()
    assert "✓ Bäume platzieren" in output


def test_subtask_without_total_is_indeterminate(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + Straßen") as task:
        with task.subtask("OSM-Daten laden"):
            pass

    output = buffer.getvalue()
    assert "✓ OSM-Daten laden" in output


def test_begin_subtask_flat_style_allows_custom_finish_summary(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + Straßen") as task:
        sub = task.begin_subtask("Brücken")
        sub.finish("3 Brücken")

    output = buffer.getvalue()
    assert "✓ Brücken - 3 Brücken" in output


def test_subtask_failure_marks_red_cross_and_reraises(buffer):
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        with pipeline.task("Terrain + Straßen") as task:
            with task.subtask("Wasser"):
                raise RuntimeError("boom")

    output = buffer.getvalue()
    assert "✗ Wasser" in output
    assert "boom" in output


def test_pipeline_skip_prints_marker_without_opening_a_task(buffer):
    pipeline = Pipeline()
    pipeline.skip("Horizont exportieren", "PHASE5_ENABLED=False")

    output = buffer.getvalue()
    assert "Horizont exportieren" in output
    assert "übersprungen" in output
    assert "PHASE5_ENABLED=False" in output


def test_pipeline_banner_prints_bold_text_outside_any_task(buffer):
    pipeline = Pipeline()
    pipeline.banner("BeamNG Level Export - 4 Tiles")

    output = buffer.getvalue()
    assert "BeamNG Level Export - 4 Tiles" in output
