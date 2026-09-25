"""
Progress/structure display for the export pipeline.

Groups the pipeline into named main tasks; each main task can open
subtasks with progress bars (known item count) or spinners
(unknown item count). Results are marked with colored UTF-8
status symbols (✓ green / ⚠ yellow / ✗ red).

`console` is the only shared rich console of the process - logging_config.py
also attaches its handler to it, so that normal
logger.info()/logger.debug() output appears cleanly above the active
bars/spinners instead of tearing them apart.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from typing import Iterator, Optional

from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()


def _status_line(style: str, symbol: str, name: str, summary: str, elapsed: float, indent: str = "") -> str:
    """Build an escaped, colored status line (name/summary may contain arbitrary text)."""
    text = f"[{style}]{indent}{symbol} {escape(name)}[/{style}]"
    if summary:
        text += f" - {escape(summary)}"
    text += f" ({elapsed:.1f}s)"
    return text


class Subtask:
    """Handle for a single subtask (bar or spinner)."""

    def __init__(self, progress: Progress, task_id, name: str, on_close=None, start: Optional[float] = None):
        self._progress = progress
        self._task_id = task_id
        self._name = name
        self._finalized = False
        self._on_close = on_close  # reports (duration, end time) to the main task
        # Seamless: starts where the previous subtask ended (preparation counts toward the following subtask)
        self._start = time.perf_counter() if start is None else start

    def _close(self, style: str, symbol: str, summary: str) -> None:
        self._finalized = True
        self._progress.remove_task(self._task_id)
        end = time.perf_counter()
        elapsed = end - self._start
        if self._on_close:
            self._on_close(elapsed, end)
        console.print(_status_line(style, symbol, self._name, summary, elapsed, indent="  "))

    def advance(self, n: int = 1) -> None:
        self._progress.advance(self._task_id, n)

    def finish(self, summary: str = "") -> None:
        if self._finalized:
            return
        self._close("green", "✓", summary)

    def warn(self, summary: str) -> None:
        if self._finalized:
            return
        self._close("yellow", "⚠", summary)

    def fail(self, summary: str) -> None:
        if self._finalized:
            return
        self._close("red", "✗", summary)


class PipelineTask:
    """Handle for a main task; holds the shared rich Progress instance for its subtasks."""

    # From this gap between total time and the sum of the subtasks on, "unassigned" is reported, in seconds
    UNASSIGNED_REPORT_MIN = 0.1

    def __init__(self, name: str):
        self.name = name
        self._finalized = False
        self._subtask_count = 0
        self._subtask_time = 0.0
        self._progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

    def __enter__(self) -> "PipelineTask":
        console.print(f"[bold cyan]▶ {escape(self.name)}[/bold cyan]")
        self._start = time.perf_counter()
        self._mark = self._start  # end of the most recently completed subtask
        self._progress.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._progress.stop()
        if exc_type is not None:
            if not self._finalized:
                self.fail(str(exc) or type(exc).__name__)
            return False
        if not self._finalized:
            self.done()
        return False

    def begin_subtask(self, name: str, total: Optional[int] = None) -> Subtask:
        task_id = self._progress.add_task(name, total=total)
        self._subtask_count += 1
        return Subtask(self._progress, task_id, name, on_close=self._add_subtask_time, start=self._mark)

    def _add_subtask_time(self, elapsed: float, end: float) -> None:
        self._subtask_time += elapsed
        self._mark = end

    def _report_unassigned(self, elapsed: float) -> None:
        """If the task has subtasks, their times must add up to the total time - report the remainder visibly."""
        gap = elapsed - self._subtask_time
        if self._subtask_count and gap >= self.UNASSIGNED_REPORT_MIN:
            console.print(_status_line("yellow", "⚠", "unassigned", "", gap, indent="  "))

    @contextmanager
    def subtask(self, name: str, total: Optional[int] = None) -> Iterator[Subtask]:
        sub = self.begin_subtask(name, total)
        try:
            yield sub
        except Exception as exc:
            sub.fail(str(exc))
            raise
        else:
            sub.finish()

    def done(self, summary: str = "") -> None:
        self._finalized = True
        elapsed = time.perf_counter() - self._start
        self._report_unassigned(elapsed)
        console.print(_status_line("bold green", "✓", self.name, summary, elapsed))

    def warn(self, summary: str) -> None:
        self._finalized = True
        elapsed = time.perf_counter() - self._start
        self._report_unassigned(elapsed)
        console.print(_status_line("bold yellow", "⚠", self.name, summary, elapsed))

    def fail(self, summary: str) -> None:
        self._finalized = True
        elapsed = time.perf_counter() - self._start
        self._report_unassigned(elapsed)
        console.print(_status_line("bold red", "✗", self.name, summary, elapsed))


def optional_subtask(task: Optional["PipelineTask"], name: str):
    """task.subtask(name) - or an empty context if the caller passes no main task (e.g. tests)."""
    if task is None:
        return nullcontext(_NullSubtask())
    return task.subtask(name)


class _NullSubtask:
    """Stand-in without display for optional_subtask() without a main task."""

    def advance(self, n: int = 1) -> None:
        pass

    def finish(self, summary: str = "") -> None:
        pass

    def warn(self, summary: str) -> None:
        pass

    def fail(self, summary: str) -> None:
        pass


class Pipeline:
    """A pipeline run; the only entry point for main tasks."""

    @contextmanager
    def task(self, name: str) -> Iterator[PipelineTask]:
        t = PipelineTask(name)
        with t:
            yield t

    def banner(self, text: str) -> None:
        """One-off, bold header line outside any main task (e.g. run summary)."""
        console.print(f"[bold]{text}[/bold]")

    def skip(self, name: str, reason: str) -> None:
        console.print(f"[dim]⏭ {name} - skipped ({reason})[/dim]")
