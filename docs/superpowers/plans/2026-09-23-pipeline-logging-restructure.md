# Pipeline-Logging-Restrukturierung Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Die Export-Pipeline zeigt ihren Ablauf als klar gegliederte Hauptaufgaben mit Fortschrittsbalken/Spinnern und farbigen ✓/⚠/✗-Status auf stdout, und der Bug, durch den die meisten Detail-Logs bisher lautlos verworfen wurden, ist behoben.

**Architecture:** Ein neues, schlankes `world_to_beamng/progress.py`-Modul (Pipeline/PipelineTask/Subtask, auf `rich` aufgebaut) kapselt die komplette Anzeige-Logik und ist der einzige Ort, der `rich` importiert. `logging_config.py` bekommt einen Root-Cause-Fix (Paket-Logger statt Logger-Name `"w2b"`) und hängt seinen Console-Handler an dieselbe `rich`-Konsole wie `progress.py`, damit normale `logger.*`-Aufrufe sauber oberhalb der aktiven Balken erscheinen. `beamng_exporter.py` und `terrain_workflow.py` werden an den bestehenden Phasengrenzen mit `pipeline.task()`/`task.subtask()`-Aufrufen instrumentiert - die Geschäftslogik selbst ändert sich nicht.

**Tech Stack:** Python 3, `rich` (neu), `logging` (Standardbibliothek), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-23-pipeline-logging-design.md`

## Global Constraints

- `rich` ist eine neue Abhängigkeit; die exakte Version wird durch `pip install` im Projekt-`.venv` ermittelt und 1:1 in `requirements.txt` übernommen (nicht raten).
- `LoggerConfig` konfiguriert künftig den Logger mit Namen `"world_to_beamng"` (Paket-Logger), nicht mehr `"w2b"`.
- Der Console-`RichHandler` läuft mit `markup=False` - Log-Nachrichten enthalten literale eckige Klammern (`[OK]`, `[i]`, `[!]`, `[✓]`, ...) und dürfen NIEMALS als rich-Markup geparst werden.
- `rich`-Markup (`[bold green]...[/bold green]` etc.) wird ausschließlich in `world_to_beamng/progress.py` für selbst verfasste Status-/Task-Zeilen verwendet, nie für durchgereichten Log-Nachrichtentext.
- Diese Restrukturierung ändert keine Geschäftslogik, keine Rückgabewerte und keine Exportergebnisse - nur Konsolenausgabe/Logging. Der bestehende Test-Suite-Umfang muss unverändert grün bleiben.
- Die Pipeline bleibt strikt sequenziell (keine Parallelisierung von Tasks/Subtasks).
- Reihenfolge der 10 Unteraufgaben unter Hauptaufgabe 4 folgt der TATSÄCHLICHEN Code-Ausführungsreihenfolge (ermittelt in der Analyse, s.u.), nicht der in der Spec nur illustrativ angegebenen Nummerierung.

---

## Task 1: `world_to_beamng/progress.py` - Pipeline/PipelineTask/Subtask-API

**Files:**
- Create: `world_to_beamng/progress.py`
- Create: `tests/test_progress.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `world_to_beamng.progress.console` (geteilte `rich.console.Console`-Instanz), `world_to_beamng.progress.Pipeline` (Klasse, kein `__init__`-Parameter), `Pipeline.task(name: str) -> ContextManager[PipelineTask]`, `Pipeline.skip(name: str, reason: str) -> None`, `PipelineTask.subtask(name: str, total: Optional[int] = None) -> ContextManager[Subtask]`, `PipelineTask.begin_subtask(name: str, total: Optional[int] = None) -> Subtask`, `PipelineTask.done(summary: str = "") -> None`, `PipelineTask.warn(summary: str) -> None`, `PipelineTask.fail(summary: str) -> None`, `Subtask.advance(n: int = 1) -> None`, `Subtask.finish(summary: str = "") -> None`, `Subtask.warn(summary: str) -> None`, `Subtask.fail(summary: str) -> None`.

- [ ] **Step 1: `rich` installieren und exakte Version ermitteln**

Run:
```
D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\pip.exe install rich
D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\pip.exe freeze --local > requirements.txt
```

`pip freeze` regeneriert die komplette Datei (inkl. bestehender Pakete) in derselben UTF-16LE-Kodierung, in der sie bereits vorliegt (PowerShell-Redirect) - `requirements.txt` NICHT manuell mit einem Text-Editor-Tool anfassen, sonst droht Encoding-Korruption. Danach kurz prüfen, dass `rich==<version>` in der Datei steht:

```
Select-String -Path "D:\Eigene_Programme\World-to-BeamNG\requirements.txt" -Pattern "^rich=="
```

- [ ] **Step 2: Fehlschlagenden Test für `PipelineTask.done()` schreiben**

Create `tests/test_progress.py`:

```python
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
```

- [ ] **Step 3: Test ausführen, Fehlschlag bestätigen**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_progress.py -v`
Expected: FAIL mit `ModuleNotFoundError: No module named 'world_to_beamng.progress'`

- [ ] **Step 4: `world_to_beamng/progress.py` implementieren**

Create `world_to_beamng/progress.py`:

```python
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
        text = f"[green]  \u2713 {self._name}[/green]" + (f" - {summary}" if summary else "")
        console.print(text)

    def warn(self, summary: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._progress.remove_task(self._task_id)
        console.print(f"[yellow]  \u26a0 {self._name} - {summary}[/yellow]")

    def fail(self, summary: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._progress.remove_task(self._task_id)
        console.print(f"[red]  \u2717 {self._name} - {summary}[/red]")


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
        console.print(f"[bold cyan]\u25b6 {self.name}[/bold cyan]")
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
        text = f"[bold green]\u2713 {self.name}[/bold green]" + (f" - {summary}" if summary else "")
        console.print(text)

    def warn(self, summary: str) -> None:
        self._finalized = True
        console.print(f"[bold yellow]\u26a0 {self.name} - {summary}[/bold yellow]")

    def fail(self, summary: str) -> None:
        self._finalized = True
        console.print(f"[bold red]\u2717 {self.name} - {summary}[/bold red]")


class Pipeline:
    """Ein Pipeline-Lauf; einziger Einstiegspunkt für Hauptaufgaben."""

    @contextmanager
    def task(self, name: str) -> Iterator[PipelineTask]:
        t = PipelineTask(name)
        with t:
            yield t

    def skip(self, name: str, reason: str) -> None:
        console.print(f"[dim]\u23ed {name} - übersprungen ({reason})[/dim]")
```

Hinweis: `\u2713`/`\u26a0`/`\u2717`/`\u25b6`/`\u23ed` sind die Unicode-Escapes für ✓/⚠/✗/▶/⏭ - im echten Dateiinhalt die literalen UTF-8-Zeichen schreiben, nicht die Escape-Sequenzen (hier nur zur eindeutigen Übertragung in diesem Plandokument verwendet).

- [ ] **Step 5: Test ausführen, Erfolg bestätigen**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_progress.py -v`
Expected: PASS

- [ ] **Step 6: Restliche Tests ergänzen (auto-finish, Fehlschlag, Balken, Spinner, flache API, skip)**

An `tests/test_progress.py` anhängen:

```python
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
```

- [ ] **Step 7: Alle Tests ausführen**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_progress.py -v`
Expected: PASS (8 Tests)

- [ ] **Step 8: Commit**

```bash
git add world_to_beamng/progress.py tests/test_progress.py requirements.txt
git commit -m "feat: Add Pipeline/PipelineTask/Subtask progress API on top of rich"
```

---

## Task 2: Root-Cause-Fix in `logging_config.py`

**Files:**
- Modify: `world_to_beamng/logging_config.py`
- Create: `tests/test_logging_config.py`

**Interfaces:**
- Consumes: `world_to_beamng.progress.console` (aus Task 1)
- Produces: `LoggerConfig.get_logger()` liefert weiterhin ein `logging.Logger`-Objekt, jetzt mit Namen `"world_to_beamng"` statt `"w2b"`. Jedes Modul, das `logging.getLogger(__name__)` benutzt, propagiert ab jetzt korrekt zu den konfigurierten Handlern.

- [ ] **Step 1: Fehlschlagenden Test für die Root-Cause schreiben**

Create `tests/test_logging_config.py`:

```python
"""Tests für world_to_beamng/logging_config.py: Root-Cause-Fix für stumme Modul-Logger.

Vorher konfigurierte LoggerConfig nur den Logger "w2b" - Module, die das
Standard-Idiom logging.getLogger(__name__) nutzen (z.B. world_to_beamng.
workflow.terrain_workflow), propagierten NICHT zu "w2b" und verwarfen
INFO/DEBUG-Meldungen lautlos (Root-Logger-Default-Level WARNING).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from world_to_beamng.logging_config import LoggerConfig


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    LoggerConfig._instance = None
    LoggerConfig._logger = None
    yield
    LoggerConfig._instance = None
    LoggerConfig._logger = None


def test_module_style_logger_inherits_configured_level():
    LoggerConfig.get_instance(log_file=None, level=logging.INFO)

    module_logger = logging.getLogger("world_to_beamng.workflow.terrain_workflow")

    assert module_logger.getEffectiveLevel() == logging.INFO
    assert module_logger.isEnabledFor(logging.INFO)
```

- [ ] **Step 2: Test ausführen, Fehlschlag bestätigen**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_logging_config.py -v`
Expected: FAIL - `module_logger.getEffectiveLevel()` ist `30` (WARNING), nicht `20` (INFO), weil nur `"w2b"` konfiguriert ist.

- [ ] **Step 3: `logging_config.py` lesen und Root-Cause-Fix einbauen**

In `world_to_beamng/logging_config.py`, Import-Block ergänzen:

```python
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

from .progress import console
```

`_setup_logger()` ersetzen durch:

```python
    def _setup_logger(self) -> None:
        """Konfiguriere Logger mit Console- und optional File-Handler."""
        logger_instance = logging.getLogger("world_to_beamng")
        logger_instance.setLevel(self.level)
        logger_instance.handlers.clear()  # Verhindere Duplikate bei mehrfachen Calls

        # File-Formatter mit Zusatzinfos; die Konsole übernimmt Level-Farbe/-Badge über RichHandler
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s() | %(message)s",
            datefmt="%H:%M:%S",
        )

        # Console-Handler (immer aktiv, stdout) - teilt sich die Konsole mit progress.py, damit
        # Log-Zeilen sauber oberhalb aktiver Balken/Spinner erscheinen. markup=False ist Pflicht:
        # bestehende Log-Nachrichten enthalten literale eckige Klammern ("[OK]", "[i]", "[!]", ...),
        # die NICHT als rich-Markup geparst werden dürfen.
        console_handler = RichHandler(
            console=console,
            markup=False,
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(self.level)
        logger_instance.addHandler(console_handler)

        # File-Handler (optional)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(file_formatter)
            logger_instance.addHandler(file_handler)
            logger_instance.info(
                f"Logger initialisiert: Datei={self.log_file}, Level={logging.getLevelName(self.level)}"
            )

        # Geschwätzige Third-Party-Logger dämpfen: sie nutzen ebenfalls logging.getLogger(__name__)
        # und würden sonst durch den jetzt korrekt propagierenden Root-Cause-Fix mitgeloggt.
        for noisy in ("urllib3", "PIL", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Setze Klassen-Variable damit get_logger() es findet
        LoggerConfig._logger = logger_instance
```

Docstring der Klasse (Zeilen 17-26) und von `get_logger()` von "w2b" auf "world_to_beamng" (Paket-Logger) aktualisieren - reine Kommentarpflege, keine Codeänderung nötig darüber hinaus.

- [ ] **Step 4: Test ausführen, Erfolg bestätigen**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_logging_config.py -v`
Expected: PASS

- [ ] **Step 5: Restliche Tests ergänzen (Verschachtelung, get_logger-Name, Third-Party-Dämpfung, Markup-Schutz)**

An `tests/test_logging_config.py` anhängen:

```python
def test_deeply_nested_module_logger_also_inherits_the_level():
    LoggerConfig.get_instance(log_file=None, level=logging.DEBUG, verbose=True)

    module_logger = logging.getLogger("world_to_beamng.forest.forest_instance_generator")

    assert module_logger.getEffectiveLevel() == logging.DEBUG


def test_get_logger_returns_the_package_logger():
    logger = LoggerConfig.get_logger()

    assert logger.name == "world_to_beamng"


def test_noisy_third_party_loggers_are_dampened_to_warning():
    LoggerConfig.get_instance(log_file=None, level=logging.DEBUG, verbose=True)

    for name in ("urllib3", "PIL", "matplotlib"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING


def test_console_handler_does_not_choke_on_literal_square_brackets(capsys):
    LoggerConfig.get_instance(log_file=None, level=logging.INFO)
    logger = logging.getLogger("world_to_beamng.textures.registry")

    logger.info("  [OK] Textur foo (prozedural)")  # darf NICHT als rich-Markup interpretiert werden

    captured = capsys.readouterr()
    assert "[OK] Textur foo" in captured.out
```

- [ ] **Step 6: Alle Tests ausführen**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_logging_config.py -v`
Expected: PASS (5 Tests)

- [ ] **Step 7: Vollen Testlauf zur Regressionskontrolle**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest -x -q`
Expected: PASS, keine Regressionen durch den Logger-Namenswechsel (kein Test greift bisher auf den Logger-Namen `"w2b"` zu - falls doch, an den neuen Namen `"world_to_beamng"` anpassen).

- [ ] **Step 8: Commit**

```bash
git add world_to_beamng/logging_config.py tests/test_logging_config.py
git commit -m "fix: Configure the world_to_beamng package logger instead of the unreachable w2b logger

Root cause of most pipeline logs never appearing on stdout: LoggerConfig
only attached handlers to a logger literally named \"w2b\". Modules using
the standard logging.getLogger(__name__) idiom (terrain_workflow.py,
utils/timing.py's StepTimer, forest/*, textures/*, ...) got a different
logger name, never propagated to \"w2b\", and were silently dropped by
the root logger's default WARNING level."
```

---

## Task 3: Hauptaufgaben-Orchestrierung in `world_to_beamng.py` + `beamng_exporter.py`

**Files:**
- Modify: `world_to_beamng.py`
- Modify: `world_to_beamng/export/beamng_exporter.py`
- Modify: `world_to_beamng/utils/timing.py` (Datei wird geleert / `StepTimer` entfernt)
- Test: bestehende Test-Suite (siehe Step 6) - kein neuer Testort nötig, da `process_tile()`/`export_complete_level()` schon vor diesem Plan keine direkten Unit-Test-Aufrufer hatten (zu IO-lastig; verifiziert per Analyse: einziger Aufrufer beider Methoden ist `beamng_exporter.py`).

**Interfaces:**
- Consumes: `world_to_beamng.progress.Pipeline`, `.task()`, `PipelineTask.subtask()`/`.done()`/`.warn()`/`.fail()` (aus Task 1)
- Produces: `BeamNGExporter.__init__(self, pipeline: Pipeline)` - `pipeline` wird als `self.pipeline` gespeichert und in `export_complete_level()` verwendet. `TerrainWorkflow.process_tile(..., task: PipelineTask)` und `TerrainWorkflow.export_tile(tile_x, tile_y, mesh_data, task: PipelineTask)` erwarten künftig zwingend ein `task`-Argument (für Task 4).

- [ ] **Step 1: `utils/timing.py` leeren (StepTimer entfernen)**

`world_to_beamng/utils/timing.py` komplett durch einen leeren Docstring ersetzen (Datei bleibt als Platzhalter bestehen, falls andere Importe noch darauf verweisen - Step 5 prüft das):

```python
"""Ehemals StepTimer - ersetzt durch world_to_beamng.progress.Pipeline (siehe docs/superpowers/specs/2026-09-23-pipeline-logging-design.md)."""
```

- [ ] **Step 2: `beamng_exporter.py` - Konstruktor um `pipeline` erweitern**

In `world_to_beamng/export/beamng_exporter.py`, Import-Block (nach `from world_to_beamng.logging_config import LoggerConfig`) ergänzen:

```python
from ..progress import Pipeline, PipelineTask
```

`__init__`-Signatur ändern von:

```python
    def __init__(self):
        """
        Initialisiere BeamNGExporter.
        """
```

zu:

```python
    def __init__(self, pipeline: Pipeline):
        """
        Initialisiere BeamNGExporter.

        Args:
            pipeline: Pipeline-Instanz für die Hauptaufgaben-Anzeige (siehe progress.py)
        """
        self.pipeline = pipeline
```

- [ ] **Step 3: `export_complete_level()` - Kopfbereich + Hauptaufgaben 1-3 umstellen**

Den Block ab `from ..utils.timing import StepTimer` bis zum Ende des Aerial-Photo-Try/Except (vor `timer.begin("Terrain + Straßen (Gesamtfläche)")`) ersetzen.

Alt (Auszug, Kopf):
```python
        from ..utils.timing import StepTimer

        timer = StepTimer()
        stats = {
```

Neu:
```python
        stats = {
```

Alt (Banner-Block):
```python
        logger.info(f"\n{'='*60}")
        logger.info(f"BEAMNG LEVEL EXPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Tiles: {len(tiles)}")
        logger.info(f"Global Offset: {global_offset}")
        logger.info(
            f"Forests: {'Yes' if forests_enabled else 'No'} | Config: {'on' if config.FORESTS_ENABLED else 'off'}"
        )  # NEU
        logger.info(f"{'='*60}\n")
```

Neu:
```python
        from .. import progress as progress_module

        progress_module.console.print(
            f"[bold]BeamNG Level Export[/bold] - {len(tiles)} Tiles, Offset {global_offset}, "
            f"Forests: {'ein' if forests_enabled else 'aus'}"
        )
```

Direkt danach, um `registry.prepare_textures()` (Hauptaufgabe 1: Texturen):

Alt:
```python
        registry.prepare_textures()
```

Neu:
```python
        with self.pipeline.task("Texturen") as task:
            registry.prepare_textures()
            task.done()
```

Um die Forest-Asset-Initialisierung (Hauptaufgabe 2), alt:
```python
        registered_trees = {}
        vineyard_assets_ready = False
        if forests_enabled:
            timer.begin("Forest Asset Initialization")

            # Reben-Assets für Weinberge sicherstellen (idempotent) - VOR dem Laden von
            # managedItemData.json, damit die Reben als Forest-Items registriert sind.
            if config.VINEYARDS_ENABLED:
```

Neu (der Block bleibt inhaltlich INLINE - keine Hilfsmethode, da `registered_trees`/`vineyard_assets_ready` von umgebendem Code weiterverwendet werden und eine Auslagerung nur unnötige Rückgabewert-Logik hinzufügen würde):

```python
        registered_trees = {}
        vineyard_assets_ready = False
        if forests_enabled:
            with self.pipeline.task("Forest-Assets") as task:
                # Reben-Assets für Weinberge sicherstellen (idempotent) - VOR dem Laden von
                # managedItemData.json, damit die Reben als Forest-Items registriert sind.
                if config.VINEYARDS_ENABLED:
                    try:
                        ensure_vineyard_assets(config.BEAMNG_DIR, get_beamng_install_dir(), config.LEVEL_NAME)
                        vineyard_assets_ready = True
                    except Exception as e:
                        logger.warning(f"Reben-Assets nicht verfügbar - Weinberge bleiben ohne Reben: {e}")

                # Lade managedItemData.json (wird von generate_forest_assets.py erzeugt)
                forest_item_data_path = config.BEAMNG_DIR / "art" / "forest" / "managedItemData.json"

                if forest_item_data_path.exists():
                    try:
                        with open(forest_item_data_path, "r", encoding="utf-8") as f:
                            forest_item_data = json.load(f)

                        for item_key, item_info in forest_item_data.items():
                            internal_name = item_info.get("internalName", item_key)
                            if internal_name in VINEYARD_ITEM_NAMES:
                                continue
                            registered_trees[internal_name] = {
                                "name": internal_name,
                                "dae_path": item_info.get("shapeFile", ""),
                                "radius": item_info.get("radius", 1.5),
                            }

                    except Exception as e:
                        logger.error(f"Fehler beim Laden von managedItemData.json: {e}")
                        registered_trees = {}
                else:
                    logger.warning(f"managedItemData.json nicht gefunden: {forest_item_data_path}")
                    logger.warning("  Bitte führen Sie zuerst aus: python tools/generate_forest_assets.py")

                stats["forests_registered"] = len(registered_trees)

                if registered_trees:
                    self.forests.set_forest_config(
                        self.forest_config,
                        osm_mapper=config.OSM_MAPPER,
                        registered_trees=registered_trees,
                    )
                task.done(f"{len(registered_trees)} Tree-Items")
        else:
            self.pipeline.skip("Forest-Assets", "FORESTS_ENABLED=False")
```

(Die bisherige `logger.info(f"✓ {len(registered_trees)} Tree-Items...")`-Zeile entfällt - identischer Inhalt steht jetzt in `task.done(...)`.)

Für Hauptaufgabe 3 (Luftbild), alt:
```python
        status = "none"  # Fallback, falls ensure_aerial_photos() unten eine Ausnahme wirft (siehe Minimap-Schritt weiter unten)
        try:
            status = ensure_aerial_photos(
                aerial_dir=aerial_dir, output_dir=textures_dir, photos=photos, global_offset=global_offset
            )
            if status == "current":
                logger.info(f"[i] Luftbild(er) passen zur Fläche ({len(photos)}) - werden übernommen")
            elif status == "built":
                logger.info(f"[OK] {len(photos)} Luftbild(er) neu gebaut und exportiert")
            elif status == "failed":
                logger.error("[!] Luftbild konnte nicht gebaut werden")
        except Exception as e:
            logger.error(f"[!] Fehler bei Luftbild-Verarbeitung: {e}")
```

Neu:
```python
        status = "none"  # Fallback, falls ensure_aerial_photos() unten eine Ausnahme wirft (siehe Minimap-Schritt weiter unten)
        with self.pipeline.task("Luftbild") as task:
            try:
                status = ensure_aerial_photos(
                    aerial_dir=aerial_dir, output_dir=textures_dir, photos=photos, global_offset=global_offset
                )
                if status == "current":
                    task.done(f"{len(photos)} Luftbild(er) passen zur Fläche - übernommen")
                elif status == "built":
                    task.done(f"{len(photos)} Luftbild(er) neu gebaut")
                elif status == "failed":
                    task.fail("Luftbild konnte nicht gebaut werden")
            except Exception as e:
                task.fail(str(e))
```

- [ ] **Step 4: Hauptaufgabe 4 (Terrain + Straßen) inkl. Forest-Platzierung (4.9)**

Alt:
```python
        timer.begin("Terrain + Straßen (Gesamtfläche)")
        result = self.terrain.process_tile(tiles=tiles, global_offset=global_offset[:2], bbox_margin=50.0)

        if result["status"] != "success":
            stats["tiles_failed"] = len(tiles)
            logger.error(f"[!] Terrain-Verarbeitung fehlgeschlagen: {result.get('reason')}")
        else:
            stats["tiles_processed"] = len(tiles)

            self.road_polygons = result.get("road_slope_polygons_2d")
            self.poi_points = result.get("poi_points")

            self.terrain.export_tile(0, 0, result)
```

Neu:
```python
        with self.pipeline.task("Terrain + Straßen") as task:
            result = self.terrain.process_tile(tiles=tiles, global_offset=global_offset[:2], bbox_margin=50.0, task=task)

            if result["status"] != "success":
                stats["tiles_failed"] = len(tiles)
                task.fail(f"Terrain-Verarbeitung fehlgeschlagen: {result.get('reason')}")
            else:
                stats["tiles_processed"] = len(tiles)

                self.road_polygons = result.get("road_slope_polygons_2d")
                self.poi_points = result.get("poi_points")

                self.terrain.export_tile(0, 0, result, task=task)
```

Der restliche Inhalt des `else`-Zweigs (Minimap, `terrain_height_at`, `tile_bounds_local`, Forest-Platzierung, Gebäude-Sammlung) bleibt inhaltlich unverändert und rutscht eine Ebene tiefer in den `with`-Block (Einrückung um 4 Leerzeichen erhöhen, da `if result["status"] != "success": ... else: ...` jetzt innerhalb von `with self.pipeline.task(...) as task:` liegt statt auf Methodenebene). Innerhalb dieses `else`-Zweigs den Forest-Platzierungs-Block (4.9) ersetzen:

Alt:
```python
            # Phase 1b: Forest Processing (für die Gesamtfläche, nicht mehr pro Kachel)
            if forests_enabled:
                forest_result = self.forests.process_tile(
                    tile_bounds=(x_min, y_min, x_max, y_max),
                    tile_name="combined_area",
                    elevation_data=result.get("height_points"),
                    height_grid_info={
                        "origin": (x_min, y_min),
                        "spacing": 1.0,
                        "elevations": result.get("height_elevations"),
                    },
                    height_hash=result.get("height_hash"),  # Für Cache-Konsistenz
                    global_offset=global_offset,  # NEU: Für WGS84-Transformation
                    height_at=terrain_height_at_1d,
                    road_surfaces=result.get("road_surface_union"),
                )
                if forest_result["status"] == "success":
                    stats["trees_generated"] += forest_result.get("tree_count", 0)

                # Weinberg-Reben (Forest-Items) zusammen mit den Bäumen in forest.forest4.json
                if vineyard_assets_ready and result.get("vineyard_instances"):
                    stats["vine_segments"] += self.forests.add_instances(result["vineyard_instances"])
```

Neu:
```python
            # Phase 1b: Forest Processing (für die Gesamtfläche, nicht mehr pro Kachel)
            if forests_enabled:
                with task.subtask("Forest-Platzierung") as sub:
                    forest_result = self.forests.process_tile(
                        tile_bounds=(x_min, y_min, x_max, y_max),
                        tile_name="combined_area",
                        elevation_data=result.get("height_points"),
                        height_grid_info={
                            "origin": (x_min, y_min),
                            "spacing": 1.0,
                            "elevations": result.get("height_elevations"),
                        },
                        height_hash=result.get("height_hash"),  # Für Cache-Konsistenz
                        global_offset=global_offset,  # NEU: Für WGS84-Transformation
                        height_at=terrain_height_at_1d,
                        road_surfaces=result.get("road_surface_union"),
                    )
                    if forest_result["status"] == "success":
                        stats["trees_generated"] += forest_result.get("tree_count", 0)

                    # Weinberg-Reben (Forest-Items) zusammen mit den Bäumen in forest.forest4.json
                    vine_segments = 0
                    if vineyard_assets_ready and result.get("vineyard_instances"):
                        vine_segments = self.forests.add_instances(result["vineyard_instances"])
                        stats["vine_segments"] += vine_segments

                    sub.finish(f"{forest_result.get('tree_count', 0)} Bäume, {vine_segments} Rebzeilen-Segmente")
```

Am Ende des `with self.pipeline.task("Terrain + Straßen") as task:`-Blocks (nach der Gebäude-Sammlung `all_buildings.extend(...)`, noch innerhalb von `else:`) steht kein explizites `task.done()` nötig - der automatische Abschluss beim Verlassen des `with`-Blocks (`PipelineTask.__exit__`) übernimmt das mit einer leeren Zusammenfassung, was hier passend ist (die interessanten Zahlen stehen bereits in den Unteraufgaben-Zeilen).

- [ ] **Step 5: Hauptaufgaben 5-7 (Buildings, Horizon, Finalisierung)**

Alt:
```python
        if include_buildings and all_buildings:

            timer.begin("Buildings Export")

            from ..workflow.building_workflow import plan_building_shapes, remove_stale_building_daes

            shapes = plan_building_shapes(
                all_buildings,
                None if config.BUILDINGS_AS_ONE_OBJECT else config.TILE_SIZE,
                config.MAX_BUILDINGS_PER_SHAPE,
            )

            written = set()
            for tile_x, tile_y, name, tile_buildings in shapes:
                dae_path = self.buildings.export_buildings(tile_buildings, tile_x, tile_y, grid_bounds=None, name=name)
                if dae_path:
                    written.add(Path(dae_path).stem)
                    self.buildings.add_items(tile_buildings, tile_x, tile_y, name=name)
                    stats["buildings_exported"] += len(tile_buildings)

            remove_stale_building_daes(config.BEAMNG_DIR_BUILDINGS, keep=written)

            self._add_lod2_materials()

        timer.begin("Horizon Export")

        # Phase 3: Horizon-Layer (optional)
        if include_horizon:
            horizon_dae = self.horizon.generate_horizon(
                global_offset=global_offset,
                tile_hash=tile_hash,
                tile_bounds=tile_bounds_local,
                terrain_height_at=terrain_height_at,
            )
            stats["horizon_exported"] = horizon_dae is not None

        timer.begin("Finalisierung")

        # Phase 4: Finalisierung
        self._finalize_export(forests_enabled)

        timer.report()

        return stats
```

Neu:
```python
        if include_buildings and all_buildings:
            with self.pipeline.task("Gebäude exportieren") as task:
                from ..workflow.building_workflow import plan_building_shapes, remove_stale_building_daes

                shapes = plan_building_shapes(
                    all_buildings,
                    None if config.BUILDINGS_AS_ONE_OBJECT else config.TILE_SIZE,
                    config.MAX_BUILDINGS_PER_SHAPE,
                )

                written = set()
                for tile_x, tile_y, name, tile_buildings in shapes:
                    dae_path = self.buildings.export_buildings(tile_buildings, tile_x, tile_y, grid_bounds=None, name=name)
                    if dae_path:
                        written.add(Path(dae_path).stem)
                        self.buildings.add_items(tile_buildings, tile_x, tile_y, name=name)
                        stats["buildings_exported"] += len(tile_buildings)

                remove_stale_building_daes(config.BEAMNG_DIR_BUILDINGS, keep=written)

                self._add_lod2_materials()
                task.done(f"{stats['buildings_exported']} Gebäude")
        elif not include_buildings:
            self.pipeline.skip("Gebäude exportieren", "LOD2_ENABLED=False")
        else:
            self.pipeline.skip("Gebäude exportieren", "keine Gebäudedaten gefunden")

        # Phase 3: Horizon-Layer (optional)
        if include_horizon:
            with self.pipeline.task("Horizont exportieren") as task:
                horizon_dae = self.horizon.generate_horizon(
                    global_offset=global_offset,
                    tile_hash=tile_hash,
                    tile_bounds=tile_bounds_local,
                    terrain_height_at=terrain_height_at,
                )
                stats["horizon_exported"] = horizon_dae is not None
                if horizon_dae:
                    task.done(Path(horizon_dae).name)
                else:
                    # deckungsgleich mit der Warnung in horizon_workflow.py::generate_horizon()
                    task.warn("DGM30-Daten nicht gefunden - kein Horizont erzeugt")
        else:
            self.pipeline.skip("Horizont exportieren", "PHASE5_ENABLED=False")

        # Phase 4: Finalisierung
        with self.pipeline.task("Finalisierung") as task:
            self._finalize_export(forests_enabled)
            task.done()

        return stats
```

- [ ] **Step 6: `world_to_beamng.py` - Hauptaufgabe 0 + Pipeline-Erzeugung**

`world_to_beamng.py` komplett lesen (91 Zeilen) und wie folgt ersetzen:

```python
"""
WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator

Refactored Version mit modularer Architektur.
Main Entry Point für die Anwendung.

Benötigte Pakete:
  pip install requests numpy scipy pyproj pyvista shapely rtree rich
"""

import sys
import time

# UTF-8 Encoding für Windows Console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from world_to_beamng import config
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()
from world_to_beamng.export import BeamNGExporter
from world_to_beamng.textures.registry import MissingTexturesError
from world_to_beamng.geometry import coordinates
from world_to_beamng.progress import Pipeline
from world_to_beamng.utils.tile_scanner import scan_elevation_tiles, compute_global_center, resolve_source_crs_epsg


def main():
    """Hauptfunktion - verwendet neue BeamNGExporter API."""

    start_time = time.time()

    pipeline = Pipeline()
    exporter = BeamNGExporter(pipeline)

    with pipeline.task("Vorbereitung") as task:
        tiles = scan_elevation_tiles(dgm_dir=config.HEIGHT_DATA_DIR)

        if not tiles:
            task.fail("keine DGM1-Kacheln gefunden")
            return

        # Quell-CRS auflösen (aus GeoTIFF-Kacheln automatisch erkannt, sonst config.SOURCE_CRS_EPSG) -
        # MUSS vor jeder weiteren Koordinatentransformation gesetzt werden (OSM-BBox, LoD2, Horizont, ...)
        source_epsg = resolve_source_crs_epsg(tiles)
        coordinates.set_source_crs(source_epsg)

        global_center = compute_global_center(tiles)
        # 3-Tupel: (x, y, z) - z ist der Mittelwert der Höhen oder 0
        global_offset = (global_center[0], global_center[1], global_center[2] if len(global_center) > 2 else 0.0)

        task.done(f"{len(tiles)} Tiles, EPSG:{source_epsg}, Offset {global_offset}")

    # Export durchführen
    try:
        stats = exporter.export_complete_level(
            tiles=tiles,
            global_offset=global_offset,
            include_buildings=config.LOD2_ENABLED,
            include_horizon=config.PHASE5_ENABLED,
        )
    except MissingTexturesError as error:  # Foto-Textur fehlt: klare Meldung statt Traceback, Exit-Code 1
        logger.error(f"\n[!] Export abgebrochen:\n{error}")
        sys.exit(1)

    # Statistiken
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("EXPORT ABGESCHLOSSEN")
    logger.info(f"{'='*60}")
    logger.info(f"Tiles verarbeitet: {stats['tiles_processed']}")
    logger.info(f"Tiles fehlgeschlagen: {stats['tiles_failed']}")
    logger.info(f"Gebäude exportiert: {stats['buildings_exported']}")
    logger.info(f"Horizon exportiert: {'Ja' if stats['horizon_exported'] else 'Nein'}")
    logger.info(f"Gesamtzeit: {elapsed:.1f}s")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
```

(Der Abschluss-Block bleibt bewusst `logger.info`-basiert - er ist eine einmalige Gesamtzusammenfassung außerhalb jeder Hauptaufgabe, kein Kandidat für die Pipeline-API.)

- [ ] **Step 7: Signaturprüfung per Test absichern**

An `tests/test_progress.py` (oder eine neue, kleine `tests/test_pipeline_wiring.py`) folgenden Test anhängen - er prüft NICHT das Laufzeitverhalten (dafür fehlen reale DGM/OSM-Daten in einer schnellen Unit-Test-Umgebung), sondern dass die Verdrahtung nicht versehentlich zurückgedreht wird:

Create `tests/test_pipeline_wiring.py`:

```python
"""Signatur-Regressionstest: stellt sicher, dass BeamNGExporter/TerrainWorkflow weiterhin
eine Pipeline/PipelineTask durchreichen (siehe docs/superpowers/plans/2026-09-23-pipeline-logging-restructure.md)."""

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
```

Dieser Test schlägt nach Step 6 dieses Tasks für `BeamNGExporter` bereits fehl-frei durch (Konstruktor ist fertig), für `TerrainWorkflow` erst nach Task 4 - das ist beabsichtigt und wird in Task 4 Step 1 verifiziert (RED), dann grün gemacht.

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_pipeline_wiring.py::test_beamng_exporter_requires_a_pipeline -v`
Expected: PASS

- [ ] **Step 8: Vollen Testlauf zur Regressionskontrolle**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest -x -q`
Expected: Alle bisher grünen Tests bleiben grün; `test_terrain_workflow_process_tile_requires_a_task` und `test_terrain_workflow_export_tile_requires_a_task` schlagen erwartungsgemäß fehl (werden in Task 4 behoben) - falls der Testrunner das als Gesamtfehlschlag zählt, das für diesen Zwischenstand als bekannt/erwartet protokollieren und mit Task 4 fortfahren.

- [ ] **Step 9: Commit**

```bash
git add world_to_beamng.py world_to_beamng/export/beamng_exporter.py world_to_beamng/utils/timing.py tests/test_pipeline_wiring.py
git commit -m "feat: Wire the 8 pipeline main tasks (0-7) through Pipeline/PipelineTask

Retires StepTimer's ad-hoc '='*60 banners in favor of world_to_beamng.
progress.Pipeline. Terrain+Straßen's internal 10 subtasks land in the
next commit (terrain_workflow.py needs a task parameter first)."
```

---

## Task 4: Unteraufgaben 4.1-4.8+4.10 in `terrain_workflow.py`

**Files:**
- Modify: `world_to_beamng/workflow/terrain_workflow.py`

**Interfaces:**
- Consumes: `PipelineTask` (aus Task 1/3) - wird jetzt zwingend an `process_tile()` und `export_tile()` durchgereicht.
- Produces: `process_tile(..., task: PipelineTask) -> Dict` (unveränderter Rückgabewert), `export_tile(tile_x, tile_y, mesh_data, task: PipelineTask) -> int` (unveränderter Rückgabewert).

Reihenfolge der Unteraufgaben folgt der tatsächlichen Code-Ausführung: 4.1 OSM-Daten laden, 4.2 Gebäude normalisieren, 4.3 Straßennetz + Infrastruktur-Geometrie (alle in `process_tile()`), dann 4.4 DecalRoads, 4.5 Wasser, 4.6 Mauern, 4.7 Brücken, 4.8 Tunnel/Galerien, 4.9 Terrain-Export inkl. GroundCover (alle in `export_tile()`). Die im Design-Spec genannte Forest-Platzierung (4.9 dort) ist bereits in Task 3 als eigene Unteraufgabe in `beamng_exporter.py` verdrahtet - hier nicht nochmal anfassen.

- [ ] **Step 1: Wiring-Test aus Task 3 als RED bestätigen**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_pipeline_wiring.py -v`
Expected: `test_terrain_workflow_process_tile_requires_a_task` und `test_terrain_workflow_export_tile_requires_a_task` schlagen fehl (`task` noch nicht in der Signatur).

- [ ] **Step 2: Import + Signaturen ergänzen**

In `world_to_beamng/workflow/terrain_workflow.py`, nach `from .tile_processor import TileProcessor`:

```python
from ..progress import PipelineTask
```

`process_tile`-Signatur:

Alt:
```python
    def process_tile(
        self,
        tiles: List[Dict],
        global_offset: Tuple[float, float],
        bbox_margin: float = 50.0,
        buildings_data: Optional[Dict] = None,
    ) -> Dict:
```

Neu:
```python
    def process_tile(
        self,
        tiles: List[Dict],
        global_offset: Tuple[float, float],
        task: PipelineTask,
        bbox_margin: float = 50.0,
        buildings_data: Optional[Dict] = None,
    ) -> Dict:
```

`export_tile`-Signatur (später in der Datei):

Alt:
```python
    def export_tile(self, tile_x: int, tile_y: int, mesh_data: Dict) -> int:
```

Neu:
```python
    def export_tile(self, tile_x: int, tile_y: int, mesh_data: Dict, task: PipelineTask) -> int:
```

- [ ] **Step 3: 4.1 OSM-Daten laden**

In `process_tile()`, unmittelbar nach dem lokalen Import-Block (nach `from ..io.cache import calculate_global_tiles_hash`) und vor `# 1. Höhendaten aller Kacheln zu einer Punktwolke kombinieren`:

```python
        sub = task.begin_subtask("OSM-Daten laden")
```

Die bestehende Fehlerbehandlung erweitern:

Alt:
```python
        if not osm_data:
            logger.warning("  [!] Keine OSM-Daten")
            return {"status": "failed", "reason": "no_osm_data"}
```

Neu:
```python
        if not osm_data:
            logger.warning("  [!] Keine OSM-Daten")
            sub.fail("keine OSM-Daten")
            return {"status": "failed", "reason": "no_osm_data"}
        sub.finish()
```

- [ ] **Step 4: 4.2 Gebäude normalisieren**

Vor `if buildings_data is None and config.LOD2_ENABLED:` einfügen:

```python
        sub = task.begin_subtask("Gebäude normalisieren")
```

Nach dem Kirchturm-Block (nach `logger.info(f"  [OK] {towers} Kirchen mit Turm erkannt (Turmuhr statt Fenster)")`), vor `# Berechne Grid-Bounds aus lokalen Punkten für Clipping` einfügen:

```python
        sub.finish(f"{len(buildings_data)} Gebäude" if buildings_data else "keine LoD2-Gebäude")
```

- [ ] **Step 5: 4.3 Straßennetz + Infrastruktur-Geometrie**

Vor `# Berechne Grid-Bounds aus lokalen Punkten für Clipping` (direkt nach dem in Step 4 eingefügten `sub.finish(...)`):

```python
        sub = task.begin_subtask("Straßennetz + Infrastruktur")
```

Direkt vor `z_min = float(heights.min())` (letzte Zeilen von `process_tile()` vor `return {...}`):

```python
        sub.finish(f"{len(road_slope_polygons_2d)} Straßensegmente")
```

- [ ] **Step 6: 4.4-4.9 in `export_tile()`**

Alt:
```python
        road_count = self.export_decal_roads(mesh_data)
        self.export_water(mesh_data)
        self.export_walls(mesh_data)
        self.export_bridges(mesh_data)
        self.export_tunnels(mesh_data)
        self.export_merged_terrain(
            heights=mesh_data["heightmap"],
            layer_map=mesh_data["layer_map"],
            terrain_material_names=list(mesh_data["terrain_material_names"]),
            terrain_origin_x=mesh_data["terrain_origin_x"],
            terrain_origin_y=mesh_data["terrain_origin_y"],
            terrain_size=mesh_data["terrain_size"],
            z_min=mesh_data["z_min"],
            max_height=mesh_data["max_height"],
            photo_tile_names=mesh_data["photo_tile_names"],
            layer_variants=mesh_data.get("layer_variants"),
            variant_parents=mesh_data.get("variant_parents"),
            photo_extents=mesh_data.get("photo_extents"),
        )
        return road_count
```

Neu:
```python
        with task.subtask("DecalRoads") as sub:
            road_count = self.export_decal_roads(mesh_data)
            sub.finish(f"{road_count} Straßen" if road_count else "keine Straßen")

        with task.subtask("Wasser") as sub:
            count = self.export_water(mesh_data)
            sub.finish(f"{count} Objekte" if count else "keine Wasserflächen")

        with task.subtask("Mauern") as sub:
            count = self.export_walls(mesh_data)
            sub.finish(f"{count} Mauern" if count else "keine Mauern")

        with task.subtask("Brücken") as sub:
            count = self.export_bridges(mesh_data)
            sub.finish(f"{count} Brücken" if count else "keine Brücken")

        with task.subtask("Tunnel/Galerien") as sub:
            count = self.export_tunnels(mesh_data)
            sub.finish(f"{count} Mesh(e)" if count else "keine Tunnel/Galerien")

        with task.subtask("Terrain-Export") as sub:
            self.export_merged_terrain(
                heights=mesh_data["heightmap"],
                layer_map=mesh_data["layer_map"],
                terrain_material_names=list(mesh_data["terrain_material_names"]),
                terrain_origin_x=mesh_data["terrain_origin_x"],
                terrain_origin_y=mesh_data["terrain_origin_y"],
                terrain_size=mesh_data["terrain_size"],
                z_min=mesh_data["z_min"],
                max_height=mesh_data["max_height"],
                photo_tile_names=mesh_data["photo_tile_names"],
                layer_variants=mesh_data.get("layer_variants"),
                variant_parents=mesh_data.get("variant_parents"),
                photo_extents=mesh_data.get("photo_extents"),
            )
            sub.finish()

        return road_count
```

- [ ] **Step 7: Wiring-Test als GREEN bestätigen**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/test_pipeline_wiring.py -v`
Expected: PASS (alle 3 Tests)

- [ ] **Step 8: Bestehende Terrain-Workflow-Tests prüfen**

Die Tests in `tests/workflow/test_terrain_workflow_*.py` rufen `_build_bridges()`/`export_bridges()`/`_build_wall_meshes()`/etc. DIREKT auf (nicht über `process_tile()`/`export_tile()`) - sie sollten von diesem Task unberührt bleiben. Zur Kontrolle:

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest tests/workflow/ -v`
Expected: PASS, unverändert gegenüber dem Stand vor diesem Task.

- [ ] **Step 9: Vollen Testlauf**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest -x -q`
Expected: PASS, alle Tests grün.

- [ ] **Step 10: Commit**

```bash
git add world_to_beamng/workflow/terrain_workflow.py
git commit -m "feat: Split Terrain+Straßen into 10 visible subtasks (OSM, buildings, roads, DecalRoads, water, walls, bridges, tunnels, terrain export)"
```

---

## Task 5: Log-Level-Politik - geschwätzige Zwischenschritte auf DEBUG

**Files:**
- Modify: `world_to_beamng/workflow/terrain_workflow.py`
- Modify: `world_to_beamng/workflow/horizon_workflow.py`
- Modify: `world_to_beamng/workflow/forest_workflow.py`
- Modify: `world_to_beamng/textures/registry.py`
- Modify: `world_to_beamng/utils/tile_scanner.py`
- Modify: `world_to_beamng/facade/building_textures.py`

Diese Änderungen sind rein mechanisch (`logger.info(` → `logger.debug(` an genau benannten Stellen, plus eine Duplikat-Bereinigung) - kein neuer Test nötig, die Wirkung wird in Task 6 per manuellem Lauf sichtbar geprüft. Jeder Schritt: Zeile(n) suchen, ersetzen, Datei kompiliert weiterhin (`python -c "import world_to_beamng.workflow.terrain_workflow"` als schneller Syntax-Check).

- [ ] **Step 1: `terrain_workflow.py` - zwei Build-Phase-Ergebniszeilen demoten**

Diese beiden Zeilen berichten die Geometrie-BERECHNUNG (innerhalb der neuen Unteraufgabe 4.3 "Straßennetz + Infrastruktur"), nicht den tatsächlichen Export - das echte, für den Nutzer relevante Ergebnis steht bereits in den Unteraufgaben 4.5 (Wasser, Zeile ~984) und 4.6 (Mauern, Zeile ~1022). Doppelmeldung vermeiden.

Alt:
```python
        logger.info(
            f"  [OK] Wasser: {len(rivers)} River-Objekt(e) ({length:.0f} m Bachlauf), "
            f"{len(ponds)} Wasserfläche(n) mit {sum(len(p['blocks']) for p in ponds)} WaterBlocks"
        )
```
Neu: `logger.info(` → `logger.debug(` (restlicher Inhalt unverändert).

Alt:
```python
        logger.info(
            f"  [OK] Mauern: {stats['built']} Bruchsteinmauer(n) mit Höhenangabe ({stats['length']:.0f} m), "
            f"{stats['without_height']} ohne Höhenangabe übersprungen"
        )
```
Neu: `logger.info(` → `logger.debug(` (restlicher Inhalt unverändert).

- [ ] **Step 2: `horizon_workflow.py` - Zwischenschritt-Narration demoten + Duplikat entfernen**

Hauptaufgabe 6 (Horizont exportieren) hat KEINE Unteraufgaben (siehe Spec) - INFO bleibt daher nur für Start/Ende + das eine Endergebnis.

Zeilen 79-81, alt:
```python
        logger.info(f"  [i] Horizont-BBOX: ±{config.HORIZON_HALF_SIZE_M / 1000:.0f}km um ({ox:.0f}, {oy:.0f})")
        logger.info(f"      UTM (EPSG:25832): X=[{x_min:.0f}..{x_max:.0f}], Y=[{y_min:.0f}..{y_max:.0f}]")
        logger.info(f"      Breite: {x_max - x_min:.0f}m, Höhe: {y_max - y_min:.0f}m")
```
Neu: alle drei `logger.info(` → `logger.debug(`.

Zeile 86, alt: `logger.info("  [i] Prüfe DGM30-Abdeckung (lädt fehlende Kacheln bei Bedarf)...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 89, alt: `logger.info("  [i] Lade DGM30-Daten (30m)...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeilen 98-102, alt:
```python
        # === Mesh generieren ===
        logger.info("  [i] Generiere Horizont-Mesh...")

        # === STEP 1: Generiere Horizont-Mesh (separater VM, OHNE UVs noch) ===
        logger.info("  [i] Generiere Horizont-Mesh...")
```
Neu (Duplikat entfernt, eine Zeile auf DEBUG):
```python
        # === STEP 1: Generiere Horizont-Mesh (separater VM, OHNE UVs noch) ===
        logger.debug("  [i] Generiere Horizont-Mesh...")
```

Zeile 118, alt: `logger.info("  [i] Prüfe Sentinel-2-Textur (lädt bei Bedarf automatisch)...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 122, alt: `logger.info("  [i] Lade Sentinel-2 Satellitenbilder...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeilen 138-143, alt:
```python
            logger.info(
                f"      Mesh Bounds (lokal): X=[{mesh_x_min:.0f}..{mesh_x_max:.0f}], Y=[{mesh_y_min:.0f}..{mesh_y_max:.0f}]"
            )
            logger.info(
                f"      Texture Bounds (UTM): X=[{bounds_utm[0]:.0f}..{bounds_utm[2]:.0f}], Y=[{bounds_utm[1]:.0f}..{bounds_utm[3]:.0f}]"
            )
```
Neu: beide `logger.info(` → `logger.debug(`.

Zeile 146, alt: `logger.info("  [i] Texturiere Horizont-Mesh...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 150, alt: `logger.info("  [i] Generiere UVs für Horizont-Mesh...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 167, alt: `logger.info(f"  [✓] {len(horizon_mesh.uvs)} UVs generiert für {len(horizon_vertices)} Vertices")`
Neu: `logger.debug(...)` (Text unverändert) - Zwischenergebnis ohne eigenes sichtbares Artefakt.

Zeile 170, alt: `logger.info("  [i] Exportiere Horizont DAE...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 183, alt: `logger.info("  [i] Registriere Materials & Items...")`
Neu: `logger.debug(...)` (Text unverändert).

Unverändert bleiben (echte Meilensteine/Warnung): Zeile 71 (`Phase 5 ist deaktiviert`), Zeile 95 (`logger.warning`, nicht betroffen), Zeile 129 (`Sentinel-2 nicht vorhanden...`), Zeile 180 (`[✓] Horizon DAE: ...` - das Endergebnis der Hauptaufgabe).

- [ ] **Step 3: `textures/registry.py` - Pro-Textur-Zeile demoten**

Zeile 116, alt:
```python
        logger.info(f"  [OK] Textur {spec.name:<22} {library.texture_tile_m(spec.name, 0.0, library_dir):5.2f} m  ({kind})  -> {spec.used_by}")
```
Neu: `logger.info(` → `logger.debug(` (feuert pro Textur, Dutzende Male - Hauptaufgabe 1 "Texturen" bekommt stattdessen über `task.done()` in Task 3 bereits eine aggregierte Abschlusszeile). Zeile 107 (`Textur '...' fehlt, erzeuge sie...`) bleibt INFO (einmaliges, wichtiges Ereignis).

- [ ] **Step 4: `utils/tile_scanner.py` - Pro-Kachel-Zeile demoten**

Zeile 82, alt:
```python
            logger.info(f"  - {tile['filename']} → X={x0:.0f}..{x1:.0f}, Y={y0:.0f}..{y1:.0f}")
```
Neu: `logger.info(` → `logger.debug(` (Zeile 79, `[INFO] N Höhendaten-Kacheln gefunden`, bleibt INFO als Ergebnis von Hauptaufgabe 0).

- [ ] **Step 5: `facade/building_textures.py` - Narration demoten**

Zeile 87, alt: `logger.info("  [i] Erzeuge Putz- und Fenstertexturen ...")`
Neu: `logger.debug(...)` (Text unverändert). Zeile 98 (`[✓] Gebäude-Texturen in {output_dir}`) bleibt INFO (Ergebnis).

- [ ] **Step 6: `workflow/forest_workflow.py` - Zwischenschritt-Narration demoten**

Diese Datei nutzte bereits vorher `LoggerConfig.get_logger()` (kein "stummes" Modul aus dem Root-Cause-Audit), ist aber laut Spec explizit als geschwätzig genannt. Die Unteraufgabe "Forest-Platzierung" (Task 3, innerhalb Hauptaufgabe 4) hat jetzt eine eigene `task.subtask(...)`-Kopf-/Ergebniszeile - die folgenden Narrations-Zeilen sind dadurch redundant geworden und werden auf DEBUG demotet.

Zeile 474, alt:
```python
            logger.info(f"\n[Forest Phase 1b] Starte für {tile_name} (bounds: {tile_bounds})")
```
Neu: `logger.debug(...)` (Text unverändert) - der `▶`-Kopf der Unteraufgabe "Forest-Platzierung" übernimmt das jetzt.

Zeile 513, alt: `logger.info(f"  [→] Lade OSM-Daten aus Cache...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 537, alt: `logger.info(f"  [→] {len(osm_data) if osm_data else 0} OSM-Elemente geladen")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 551, alt: `logger.info(f"  [→] Normalisiere OSM-Waldpolygone...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 561, alt: `logger.info(f"  [→] Transformiere OSM-Daten zu lokalen Koordinaten...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeilen 573-575, alt:
```python
            logger.info(
                f"  [Forest] Normalisierung: {normalized.get('status')} - {normalized.get('forest_count')} Wälder"
            )
```
Neu: `logger.info(` → `logger.debug(` (restlicher Inhalt unverändert).

Zeile 610, alt: `logger.info(f"  [→] {len(forests)} Waldpolygone zu bearbeiten")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 613, alt: `logger.info(f"  [→] Generiere Tree-Positionen (Poisson-Disk)...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeilen 618-622, alt:
```python
                logger.info(
                    f"  [Forest] Road Buffer erstellt - Bounds: {road_buffer.bounds}, Area: {road_buffer.area:.0f}m²"
                )
            else:
                logger.info(f"  [Forest] Road Buffer ist None!")
```
Neu: beide `logger.info(` → `logger.debug(` (restlicher Inhalt unverändert).

Zeile 626, alt: `logger.info(f"  [Forest] Gebäude-Puffer erstellt - Fläche: {building_buffer.area:.0f}m²")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 659, alt: `logger.info(f"  [Forest] {len(singles)} Einzelbäume (natural=tree)")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 662, alt: `logger.info(f"  [→] {total_points} Baumpositionen generiert")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 665, alt: `logger.info(f"  [→] Interpoliere Höhen...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 674, alt: `logger.info(f"  [→] Höhen für {total_points} Punkte interpoliert")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 677, alt: `logger.info(f"  [→] Generiere Baum-Instanzen...")`
Neu: `logger.debug(...)` (Text unverändert).

Zeile 702, alt: `logger.info(f"  [✓] {len(tree_instances)} Baum-Instanzen generiert für {tile_name}")`
Neu: `logger.debug(...)` (Text unverändert) - identischer Inhalt steht jetzt in der `task.subtask("Forest-Platzierung")`-Abschlusszeile aus Task 3.

Zeile 765, alt: `logger.info(f"[Forest] Finalisiere Export ({len(self.all_tree_instances)} Instanzen)...")`
Neu: `logger.debug(...)` (Text unverändert).

Unverändert bleiben: Zeilen 481/598/721 (`logger.error`, echte Fehler), Zeile 500 (`[OK] Forest-Cache gefunden...` - erklärt einen Cache-Kurzschluss, bleibt INFO), Zeilen 540/769 (`logger.warning`, unverändert), Zeilen 717/780/823 (bereits als `[Forest ERROR]`-Text formulierte `logger.info`-Aufrufe in except-Blöcken - de facto Fehlermeldungen, außerhalb des Umfangs dieser reinen Level-Demotion), Zeilen 806-812 (das eigentliche Forest-Export-Ergebnis der Hauptaufgabe 7 "Finalisierung" - bleibt INFO).

- [ ] **Step 7: Syntax-Check aller sechs Dateien**

Run:
```
D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -c "import world_to_beamng.workflow.terrain_workflow, world_to_beamng.workflow.horizon_workflow, world_to_beamng.workflow.forest_workflow, world_to_beamng.textures.registry, world_to_beamng.utils.tile_scanner, world_to_beamng.facade.building_textures"
```
Expected: kein Fehler.

- [ ] **Step 8: Vollen Testlauf**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest -x -q`
Expected: PASS (Log-Level-Änderungen betreffen kein von Tests geprüftes Verhalten).

- [ ] **Step 9: Commit**

```bash
git add world_to_beamng/workflow/terrain_workflow.py world_to_beamng/workflow/horizon_workflow.py world_to_beamng/workflow/forest_workflow.py world_to_beamng/textures/registry.py world_to_beamng/utils/tile_scanner.py world_to_beamng/facade/building_textures.py
git commit -m "refactor: Demote narration log lines to DEBUG, keep only milestones at INFO

Also removes an exact duplicate 'Generiere Horizont-Mesh...' log call in
horizon_workflow.py discovered during the audit."
```

---

## Task 6: Regressionslauf, manueller Realdaten-Lauf, Abschluss

**Files:** keine Code-Änderungen - reine Verifikation.

- [ ] **Step 1: Vollständige Test-Suite**

Run: `D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe -m pytest -q`
Expected: Alle Tests PASS, keine Regressionen über alle 5 Tasks hinweg.

- [ ] **Step 2: Realdaten-Export als manueller Smoke-Test**

Echte Höhendaten liegen unter `data/height/` (Swiss ALTI3D-Kacheln) vor, ein realer Lauf ist also möglich und ist der eigentliche Abnahmetest für dieses Vorhaben (sichtbare Hauptaufgaben, Balken/Spinner, farbige ✓/⚠/✗-Zeilen, vorher stumme Detail-Logs jetzt sichtbar). Run (Performance-Notiz aus dem Projektgedächtnis: ein 4x4-km-Export dauert optimiert ca. 50s):

```
D:\Eigene_Programme\World-to-BeamNG\.venv\Scripts\python.exe world_to_beamng.py
```

Von Hand gegen die Spec prüfen (`docs/superpowers/specs/2026-09-23-pipeline-logging-design.md`):
- Erscheinen alle aktiven Hauptaufgaben (0, 1, 2 oder Skip-Zeile, 3, 4, 5 oder Skip-Zeile, 6 oder Skip-Zeile, 7) sichtbar mit `▶`-Start und `✓`/`✗`-Abschluss?
- Zeigt Hauptaufgabe 4 ("Terrain + Straßen") die 9 erwarteten Unteraufgaben-Zeilen (OSM, Gebäude, Straßennetz+Infrastruktur, Forest-Platzierung, DecalRoads, Wasser, Mauern, Brücken, Tunnel/Galerien, Terrain-Export - 10 insgesamt)?
- Ist die Ausgabe insgesamt spürbar kürzer/übersichtlicher als vorher (keine geschwätzigen `→ ...`-Zeilen mehr im Standardlauf ohne `DEBUG_VERBOSE`)?
- Läuft `python world_to_beamng.py` mit `config.DEBUG_VERBOSE = True` (kurz manuell in `config.py` umschalten, nach dem Test zurücksetzen - NICHT committen) durch, ohne dass die vorher stummen Module (terrain_workflow.py-Details, forest_workflow.py `→`-Zeilen) das Bild sprengen?
- Falls BeamNG.drive verfügbar ist: exportiertes Level laden und `C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\beamng.log` auf neue `|E|`/`Fatal-ISV`-Zeilen prüfen (Regressionscheck laut CLAUDE.md) - dieser Task ändert keine Export-Artefakte, ein Fehlschlag hier wäre ein Alarmsignal für einen versehentlichen Verhaltensfehler.

- [ ] **Step 3: `config.py` auf unveränderten Zustand prüfen**

Falls in Step 2 `DEBUG_VERBOSE` testweise umgeschaltet wurde:

Run: `git diff world_to_beamng/config.py`
Expected: keine Änderung (oder zurücksetzen, falls doch).

- [ ] **Step 4: Spec-Abdeckung gegenprüfen**

Gegen `docs/superpowers/specs/2026-09-23-pipeline-logging-design.md` Punkt für Punkt abhaken: Root-Cause-Fix (Task 2) ✓, 8 Hauptaufgaben (Task 3) ✓, 10 Unteraufgaben unter Hauptaufgabe 4 (Task 4) ✓, Progress-API mit Balken/Spinner/✓⚠✗ (Task 1) ✓, Log-Level-Politik (Task 5) ✓, `rich`-Abhängigkeit (Task 1) ✓. "Out of Scope"-Punkte der Spec (Log-Datei-Umbau, `tools/*.py`, Parallelisierung) - bestätigen, dass keiner davon versehentlich angefasst wurde:

Run: `git diff main --stat` (oder `git log --oneline` seit Task 1, je nachdem wie das Team commitet)

- [ ] **Step 5: Finaler Commit (falls noch offene Änderungen, z.B. aus Step 2/3)**

Nur falls `git status` nach den vorherigen Schritten noch etwas zeigt - normalerweise ist Task 6 reine Verifikation ohne eigenen Commit.
