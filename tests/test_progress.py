"""Tests for world_to_beamng/progress.py: Pipeline/PipelineTask/Subtask."""

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
    with pipeline.task("Textures") as task:
        task.done("5 textures checked")

    output = buffer.getvalue()
    assert "Textures" in output
    assert "✓" in output
    assert "5 textures checked" in output


def test_task_auto_finishes_without_explicit_done(buffer):
    pipeline = Pipeline()
    with pipeline.task("Preparation"):
        pass

    output = buffer.getvalue()
    assert "✓ Preparation" in output


def test_task_failure_marks_red_cross_and_reraises(buffer):
    pipeline = Pipeline()
    with pytest.raises(ValueError):
        with pipeline.task("Terrain + roads") as task:
            raise ValueError("broken")

    output = buffer.getvalue()
    assert "✗" in output
    assert "broken" in output


def test_subtask_with_total_reaches_100_percent(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + roads") as task:
        with task.subtask("Place trees", total=3) as sub:
            for _ in range(3):
                sub.advance()

    output = buffer.getvalue()
    assert "✓ Place trees" in output


def test_subtask_without_total_is_indeterminate(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + roads") as task:
        with task.subtask("Load OSM data"):
            pass

    output = buffer.getvalue()
    assert "✓ Load OSM data" in output


def test_begin_subtask_flat_style_allows_custom_finish_summary(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + roads") as task:
        sub = task.begin_subtask("Bridges")
        sub.finish("3 bridges")

    output = buffer.getvalue()
    assert "✓ Bridges - 3 bridges" in output


def test_subtask_failure_marks_red_cross_and_reraises(buffer):
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        with pipeline.task("Terrain + roads") as task:
            with task.subtask("Water"):
                raise RuntimeError("boom")

    output = buffer.getvalue()
    assert "✗ Water" in output
    assert "boom" in output


def test_pipeline_skip_prints_marker_without_opening_a_task(buffer):
    pipeline = Pipeline()
    pipeline.skip("Export horizon", "PHASE5_ENABLED=False")

    output = buffer.getvalue()
    assert "Export horizon" in output
    assert "skipped" in output
    assert "PHASE5_ENABLED=False" in output


def test_pipeline_banner_prints_bold_text_outside_any_task(buffer):
    pipeline = Pipeline()
    pipeline.banner("BeamNG Level Export - 4 Tiles")

    output = buffer.getvalue()
    assert "BeamNG Level Export - 4 Tiles" in output


def test_task_warn_prints_yellow_marker(buffer):
    pipeline = Pipeline()
    with pipeline.task("Water") as task:
        task.warn("no sources found")

    output = buffer.getvalue()
    assert "⚠" in output
    assert "Water" in output
    assert "no sources found" in output


def test_subtask_warn_prints_yellow_marker(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + roads") as task:
        sub = task.begin_subtask("Bridges")
        sub.warn("geometry implausible")

    output = buffer.getvalue()
    assert "⚠" in output
    assert "Bridges" in output
    assert "geometry implausible" in output


def test_task_fail_with_brackets_in_summary_prints_literally_and_does_not_raise(buffer):
    """Regression: summary strings (e.g. str(exc)) can contain literal square brackets
    (file paths, error messages) - they must not be parsed as rich markup
    and must neither be swallowed nor raise a MarkupError."""
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        with pipeline.task("Terrain + roads") as task:
            raise RuntimeError("Path not found: [/tmp/missing]")

    output = buffer.getvalue()
    assert "[/tmp/missing]" in output


def test_task_done_with_brackets_in_summary_prints_literally(buffer):
    pipeline = Pipeline()
    with pipeline.task("Textures") as task:
        task.done("loaded from [cache]")

    output = buffer.getvalue()
    assert "loaded from [cache]" in output


def test_task_warn_with_brackets_in_summary_prints_literally(buffer):
    pipeline = Pipeline()
    with pipeline.task("Water") as task:
        task.warn("unknown tag [waterway=weird]")

    output = buffer.getvalue()
    assert "unknown tag [waterway=weird]" in output


def test_subtask_finish_with_brackets_in_summary_prints_literally(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + roads") as task:
        sub = task.begin_subtask("Bridges")
        sub.finish("Texture [/data/textures/x.png] loaded")

    output = buffer.getvalue()
    assert "Texture [/data/textures/x.png] loaded" in output


def test_subtask_fail_with_brackets_in_summary_prints_literally_and_does_not_raise(buffer):
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        with pipeline.task("Terrain + roads") as task:
            with task.subtask("Water"):
                raise RuntimeError("boom [x]")

    output = buffer.getvalue()
    assert "boom [x]" in output


def test_task_name_with_brackets_is_escaped_and_does_not_raise(buffer):
    pipeline = Pipeline()
    with pipeline.task("Import [test]") as task:
        task.done()

    output = buffer.getvalue()
    assert "Import [test]" in output


def test_task_exit_without_message_uses_exception_type_name(buffer):
    """An error without a message (e.g. a bare assert - str(exc) == "") must not
    lead to a line that simply shows nothing after the separator dash.
    (AssertionError() constructed directly instead of `assert False`, so that pytest's
    assertion rewriting does not insert a message into the exception here.)"""
    pipeline = Pipeline()
    with pytest.raises(AssertionError):
        with pipeline.task("Terrain + roads") as task:
            raise AssertionError()

    output = buffer.getvalue()
    assert "AssertionError" in output
    assert " - \n" not in output
    assert not output.rstrip().endswith(" -")


def test_task_exit_does_not_double_report_after_explicit_done(buffer):
    """If done()/warn() was already called inside the with block, an exception
    propagated afterwards must not print the line again (as fail)."""
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        with pipeline.task("Terrain + roads") as task:
            task.warn("partially done")
            raise RuntimeError("boom afterwards")

    output = buffer.getvalue()
    assert output.count("⚠") == 1
    assert "✗" not in output


def test_task_and_subtask_report_elapsed_seconds(buffer):
    pipeline = Pipeline()
    with pipeline.task("Textures") as task:
        with task.subtask("Loading") as sub:
            sub.finish("ok")
        task.done("done")

    output = buffer.getvalue()
    import re

    assert re.search(r"Loading.*\(\d+\.\d+s\)", output)
    assert re.search(r"Textures.*\(\d+\.\d+s\)", output)


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(progress_module.time, "perf_counter", fake)
    return fake


def test_time_outside_subtasks_is_reported_as_unassigned(buffer, clock):
    # The individual times must add up to the total time - unmeasured work is made visible
    pipeline = Pipeline()
    with pipeline.task("Terrain + roads") as task:
        with task.subtask("DecalRoads"):
            clock.now += 2.0
        clock.now += 7.0  # work without a subtask (e.g. formerly the minimap image)

    output = buffer.getvalue()
    assert "unassigned (7.0s)" in output
    assert "✓ Terrain + roads (9.0s)" in output


def test_seamless_subtasks_report_no_unassigned_time(buffer, clock):
    pipeline = Pipeline()
    with pipeline.task("Finalization") as task:
        with task.subtask("Materials"):
            clock.now += 1.0
        with task.subtask("Items"):
            clock.now += 2.0
        clock.now += 0.05  # small stuff below the reporting threshold

    assert "unassigned" not in buffer.getvalue()


def test_task_without_subtasks_reports_no_unassigned_time(buffer, clock):
    pipeline = Pipeline()
    with pipeline.task("Textures"):
        clock.now += 3.0

    assert "unassigned" not in buffer.getvalue()


def test_subtasks_are_contiguous_so_preparation_time_counts_to_the_next_subtask(buffer, clock):
    # Seamless: the time between two subtasks (imports, hand-offs) belongs to the following subtask
    pipeline = Pipeline()
    with pipeline.task("Terrain + roads") as task:
        clock.now += 0.5  # imports before the first subtask
        with task.subtask("Load OSM data"):
            clock.now += 2.0
        clock.now += 0.5  # hand-off to the exporter
        with task.subtask("DecalRoads"):
            clock.now += 1.0

    output = buffer.getvalue()
    assert "Load OSM data (2.5s)" in output
    assert "DecalRoads (1.5s)" in output
    assert "✓ Terrain + roads (4.0s)" in output
    assert "unassigned" not in output
