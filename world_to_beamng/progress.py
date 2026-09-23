"""
Fortschritts-/Struktur-Anzeige für die Export-Pipeline.

Bündelt die Pipeline in benannte Hauptaufgaben; jede Hauptaufgabe kann
Unteraufgaben mit Fortschrittsbalken (bekannte Stückzahl) oder Spinnern
(unbekannte Stückzahl) öffnen. Ergebnisse werden mit farbigen UTF-8-
Status-Symbolen markiert (✓ grün / ⚠ gelb / ✗ rot).

`console` ist die einzige geteilte rich-Konsole des Prozesses - auch
logging_config.py hängt seinen Handler daran, damit normale
logger.info()/logger.debug()-Ausgaben sauber oberhalb der aktiven
Balken/Spinner erscheinen statt sie zu zerreißen.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional

from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()


def _status_line(style: str, symbol: str, name: str, summary: str, elapsed: float, indent: str = "") -> str:
    """Baue eine escapte, farbige Status-Zeile (Name/Summary können beliebigen Text enthalten)."""
    text = f"[{style}]{indent}{symbol} {escape(name)}[/{style}]"
    if summary:
        text += f" - {escape(summary)}"
    text += f" ({elapsed:.1f}s)"
    return text


class Subtask:
    """Handle für eine einzelne Unteraufgabe (Balken oder Spinner)."""

    def __init__(self, progress: Progress, task_id, name: str):
        self._progress = progress
        self._task_id = task_id
        self._name = name
        self._finalized = False
        self._start = time.perf_counter()

    def advance(self, n: int = 1) -> None:
        self._progress.advance(self._task_id, n)

    def finish(self, summary: str = "") -> None:
        if self._finalized:
            return
        self._finalized = True
        self._progress.remove_task(self._task_id)
        elapsed = time.perf_counter() - self._start
        console.print(_status_line("green", "✓", self._name, summary, elapsed, indent="  "))

    def warn(self, summary: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._progress.remove_task(self._task_id)
        elapsed = time.perf_counter() - self._start
        console.print(_status_line("yellow", "⚠", self._name, summary, elapsed, indent="  "))

    def fail(self, summary: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._progress.remove_task(self._task_id)
        elapsed = time.perf_counter() - self._start
        console.print(_status_line("red", "✗", self._name, summary, elapsed, indent="  "))


class PipelineTask:
    """Handle für eine Hauptaufgabe; hält die geteilte rich-Progress-Instanz für ihre Unteraufgaben."""

    def __init__(self, name: str):
        self.name = name
        self._finalized = False
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
        return Subtask(self._progress, task_id, name)

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
        console.print(_status_line("bold green", "✓", self.name, summary, elapsed))

    def warn(self, summary: str) -> None:
        self._finalized = True
        elapsed = time.perf_counter() - self._start
        console.print(_status_line("bold yellow", "⚠", self.name, summary, elapsed))

    def fail(self, summary: str) -> None:
        self._finalized = True
        elapsed = time.perf_counter() - self._start
        console.print(_status_line("bold red", "✗", self.name, summary, elapsed))


class Pipeline:
    """Ein Pipeline-Lauf; einziger Einstiegspunkt für Hauptaufgaben."""

    @contextmanager
    def task(self, name: str) -> Iterator[PipelineTask]:
        t = PipelineTask(name)
        with t:
            yield t

    def banner(self, text: str) -> None:
        """Einmalige, fett gedruckte Kopfzeile außerhalb jeder Hauptaufgabe (z.B. Lauf-Zusammenfassung)."""
        console.print(f"[bold]{text}[/bold]")

    def skip(self, name: str, reason: str) -> None:
        console.print(f"[dim]⏭ {name} - übersprungen ({reason})[/dim]")
