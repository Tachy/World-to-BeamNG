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
    with pipeline.task("Wasser") as task:
        task.warn("keine Quellen gefunden")

    output = buffer.getvalue()
    assert "⚠" in output
    assert "Wasser" in output
    assert "keine Quellen gefunden" in output


def test_subtask_warn_prints_yellow_marker(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + Straßen") as task:
        sub = task.begin_subtask("Brücken")
        sub.warn("Geometrie unplausibel")

    output = buffer.getvalue()
    assert "⚠" in output
    assert "Brücken" in output
    assert "Geometrie unplausibel" in output


def test_task_fail_with_brackets_in_summary_prints_literally_and_does_not_raise(buffer):
    """Regression: summary strings (z.B. str(exc)) können literale eckige Klammern
    enthalten (Dateipfade, Fehlermeldungen) - die dürfen nicht als rich-Markup geparst
    werden und weder verschluckt werden noch eine MarkupError auslösen."""
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        with pipeline.task("Terrain + Straßen") as task:
            raise RuntimeError("Pfad nicht gefunden: [/tmp/missing]")

    output = buffer.getvalue()
    assert "[/tmp/missing]" in output


def test_task_done_with_brackets_in_summary_prints_literally(buffer):
    pipeline = Pipeline()
    with pipeline.task("Texturen") as task:
        task.done("geladen aus [cache]")

    output = buffer.getvalue()
    assert "geladen aus [cache]" in output


def test_task_warn_with_brackets_in_summary_prints_literally(buffer):
    pipeline = Pipeline()
    with pipeline.task("Wasser") as task:
        task.warn("unbekanntes Tag [waterway=weird]")

    output = buffer.getvalue()
    assert "unbekanntes Tag [waterway=weird]" in output


def test_subtask_finish_with_brackets_in_summary_prints_literally(buffer):
    pipeline = Pipeline()
    with pipeline.task("Terrain + Straßen") as task:
        sub = task.begin_subtask("Brücken")
        sub.finish("Textur [/data/textures/x.png] geladen")

    output = buffer.getvalue()
    assert "Textur [/data/textures/x.png] geladen" in output


def test_subtask_fail_with_brackets_in_summary_prints_literally_and_does_not_raise(buffer):
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        with pipeline.task("Terrain + Straßen") as task:
            with task.subtask("Wasser"):
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
    """Ein Fehler ohne Nachricht (z.B. ein bloßes assert - str(exc) == "") darf nicht
    zu einer Zeile führen, die nach dem Trennstrich einfach nichts mehr zeigt.
    (AssertionError() direkt konstruiert statt `assert False`, damit pytests
    Assertion-Rewriting hier keine Nachricht in die Exception einfügt.)"""
    pipeline = Pipeline()
    with pytest.raises(AssertionError):
        with pipeline.task("Terrain + Straßen") as task:
            raise AssertionError()

    output = buffer.getvalue()
    assert "AssertionError" in output
    assert " - \n" not in output
    assert not output.rstrip().endswith(" -")


def test_task_exit_does_not_double_report_after_explicit_done(buffer):
    """Wenn done()/warn() bereits im with-Block aufgerufen wurde, darf eine
    anschließend durchgereichte Exception die Zeile nicht nochmal (als fail) drucken."""
    pipeline = Pipeline()
    with pytest.raises(RuntimeError):
        with pipeline.task("Terrain + Straßen") as task:
            task.warn("teilweise fertig")
            raise RuntimeError("boom danach")

    output = buffer.getvalue()
    assert output.count("⚠") == 1
    assert "✗" not in output


def test_task_and_subtask_report_elapsed_seconds(buffer):
    pipeline = Pipeline()
    with pipeline.task("Texturen") as task:
        with task.subtask("Laden") as sub:
            sub.finish("ok")
        task.done("fertig")

    output = buffer.getvalue()
    import re

    assert re.search(r"Laden.*\(\d+\.\d+s\)", output)
    assert re.search(r"Texturen.*\(\d+\.\d+s\)", output)


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
    # Die Einzelzeiten müssen sich zur Gesamtzeit summieren - ungemessene Arbeit wird sichtbar gemacht
    pipeline = Pipeline()
    with pipeline.task("Terrain + Straßen") as task:
        with task.subtask("DecalRoads"):
            clock.now += 2.0
        clock.now += 7.0  # Arbeit ohne Teilaufgabe (z.B. früher das Minimap-Bild)

    output = buffer.getvalue()
    assert "unassigned (7.0s)" in output
    assert "✓ Terrain + Straßen (9.0s)" in output


def test_seamless_subtasks_report_no_unassigned_time(buffer, clock):
    pipeline = Pipeline()
    with pipeline.task("Finalisierung") as task:
        with task.subtask("Materials"):
            clock.now += 1.0
        with task.subtask("Items"):
            clock.now += 2.0
        clock.now += 0.05  # Kleinkram unter der Meldeschwelle

    assert "unassigned" not in buffer.getvalue()


def test_task_without_subtasks_reports_no_unassigned_time(buffer, clock):
    pipeline = Pipeline()
    with pipeline.task("Texturen"):
        clock.now += 3.0

    assert "unassigned" not in buffer.getvalue()


def test_subtasks_are_contiguous_so_preparation_time_counts_to_the_next_subtask(buffer, clock):
    # Nahtlos: die Zeit zwischen zwei Teilaufgaben (Imports, Übergaben) gehört zur folgenden Teilaufgabe
    pipeline = Pipeline()
    with pipeline.task("Terrain + Straßen") as task:
        clock.now += 0.5  # Imports vor der ersten Teilaufgabe
        with task.subtask("OSM-Daten laden"):
            clock.now += 2.0
        clock.now += 0.5  # Übergabe an den Exporter
        with task.subtask("DecalRoads"):
            clock.now += 1.0

    output = buffer.getvalue()
    assert "OSM-Daten laden (2.5s)" in output
    assert "DecalRoads (1.5s)" in output
    assert "✓ Terrain + Straßen (4.0s)" in output
    assert "unassigned" not in output
