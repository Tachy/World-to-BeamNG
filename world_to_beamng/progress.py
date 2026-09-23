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

from contextlib import contextmanager
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()


class Subtask:
    """Handle für eine einzelne Unteraufgabe (Balken oder Spinner)."""

    def __init__(self, progress: Progress, task_id, name: str):
        self._progress = progress
        self._task_id = task_id
        self._name = name
        self._finalized = False

    def advance(self, n: int = 1) -> None:
        self._progress.advance(self._task_id, n)

    def finish(self, summary: str = "") -> None:
        if self._finalized:
            return
        self._finalized = True
        self._progress.remove_task(self._task_id)
        text = f"[green]  ✓ {self._name}[/green]" + (f" - {summary}" if summary else "")
        console.print(text)

    def warn(self, summary: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._progress.remove_task(self._task_id)
        console.print(f"[yellow]  ⚠ {self._name} - {summary}[/yellow]")

    def fail(self, summary: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._progress.remove_task(self._task_id)
        console.print(f"[red]  ✗ {self._name} - {summary}[/red]")


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
        console.print(f"[bold cyan]▶ {self.name}[/bold cyan]")
        self._progress.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._progress.stop()
        if exc_type is not None:
            self.fail(str(exc))
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
        text = f"[bold green]✓ {self.name}[/bold green]" + (f" - {summary}" if summary else "")
        console.print(text)

    def warn(self, summary: str) -> None:
        self._finalized = True
        console.print(f"[bold yellow]⚠ {self.name} - {summary}[/bold yellow]")

    def fail(self, summary: str) -> None:
        self._finalized = True
        console.print(f"[bold red]✗ {self.name} - {summary}[/bold red]")


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
