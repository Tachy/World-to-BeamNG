# Pipeline-Logging: Struktur + Fortschrittsanzeige

## Ziel

Das Logging der Export-Pipeline (`world_to_beamng.py` → `BeamNGExporter.export_complete_level()`)
soll nach Hauptaufgaben gegliedert werden, jede Hauptaufgabe zeigt ihren Fortschritt sichtbar an
(Balken, wo eine Stückzahl bekannt ist; Spinner sonst), und Ergebnisse werden mit farbigen
UTF-8-Status-Symbolen markiert (✓ grün / ⚠ gelb / ✗ rot). Alles landet auf stdout. Zukünftige
Pipeline-Schritte müssen sich in dieselbe Struktur einreihen können, ohne dass man sich eine
Sonderregel merken muss.

## Root Cause: Logs verschwinden lautlos

`LoggerConfig` (`world_to_beamng/logging_config.py`) hängt Console-/File-Handler ausschließlich an
den Logger mit dem exakten Namen `"w2b"`. Ein großer Teil der Module ruft aber stattdessen das
Standard-Python-Idiom `logging.getLogger(__name__)` auf, was einen Logger mit dem Modulpfad
erzeugt (z.B. `world_to_beamng.workflow.terrain_workflow`). Dieser Logger ist **kein Kind** von
`"w2b"` (andere Namenshierarchie) und propagiert stattdessen zum Root-Logger, der nie konfiguriert
wird. Ohne Handler und mit dem Default-Level `WARNING` werden alle `logger.info(...)`- und
`logger.debug(...)`-Aufrufe dieser Module lautlos verworfen - weder auf stdout noch (falls aktiv)
in die Logdatei.

Betroffen sind u.a.:
- `utils/timing.py` - **der `StepTimer` selbst**, dessen Phasen-Banner deshalb nie erscheinen
- `workflow/terrain_workflow.py` (1305 Zeilen, die meisten `logger.*`-Aufrufe der Pipeline)
- `workflow/horizon_workflow.py`, `workflow/building_workflow.py`
- `textures/registry.py`, `textures/library.py`
- `utils/tile_scanner.py`, `utils/debug_exporter.py`
- `forest/forest_height_calculator.py`, `forest/forest_instance_generator.py`,
  `forest/forest_json_writer.py`, `forest/tree_footprints.py`
- `facade/church_towers.py`, `facade/building_textures.py`, `terrain/horizon_image.py`

Nur Module, die explizit `LoggerConfig.get_logger()` aufrufen (z.B. `beamng_exporter.py`,
`forest_workflow.py`), kommen bisher überhaupt durch.

### Fix

`LoggerConfig` konfiguriert künftig den **Paket-Logger `"world_to_beamng"`** statt `"w2b"`. Jedes
Modul in `world_to_beamng/` erzeugt über `logging.getLogger(__name__)` automatisch einen Logger
unterhalb dieses Namens (`world_to_beamng.<submodule>`) und propagiert damit automatisch zu den
konfigurierten Handlern - ganz ohne Sonderaufruf. Das ist der Fix UND das Schema zugleich: neuer
Code schreibt einfach das Standard-Idiom, keine Ausnahme zu merken.

- `LoggerConfig.get_logger()` bleibt als Alias erhalten (liefert jetzt den `"world_to_beamng"`-
  Logger direkt) für Callsites, die keinen eigenen Modulnamen brauchen (z.B. `world_to_beamng.py`
  selbst, das kein Package-Submodul ist).
- Alle bestehenden `logger = logging.getLogger(__name__)`-Zeilen bleiben unverändert - sie sind ab
  jetzt korrekt.
- Root-Logger selbst bleibt unangetastet (kein `logging.basicConfig()`), damit Third-Party-Libs
  (rasterio, shapely, urllib3, PIL, vtk/pyvista) nicht plötzlich mitloggen. Bekannte geschwätzige
  Third-Party-Logger (`urllib3`, `PIL`, `matplotlib`) werden vorsorglich explizit auf `WARNING`
  gesetzt.

## Gliederung in Hauptaufgaben

Ersetzt die verstreuten `logger.info('='*60)`-Banner in `beamng_exporter.py` und den alten
`StepTimer` (`utils/timing.py`) durch eine einheitliche Liste. Deaktivierte Schritte (z.B.
`PHASE5_ENABLED=False`) werden als "übersprungen" markiert statt einfach zu fehlen.

```
0. Vorbereitung        (Tiles scannen, CRS auflösen, Global Offset)
1. Texturen             (Registry prüfen/generieren)
2. Forest-Assets        (managedItemData + Weinreben laden, nur wenn FORESTS_ENABLED)
3. Luftbild              (Aerial-Fotos sicherstellen/bauen)
4. Terrain + Straßen     (Gesamtfläche) - mit Unteraufgaben:
   4.1 OSM-Daten laden
   4.2 Gebäude normalisieren (LoD2 + Kirchtürme)
   4.3 Straßennetz (Grid/Junctions/Road-Surfaces)
   4.4 Brücken
   4.5 Tunnel/Galerien
   4.6 Mauern
   4.7 Wasser
   4.8 DecalRoads + GroundCover
   4.9 Forest-Platzierung (Baumpositionen)
   4.10 Terrain-Export (.ter)
5. Gebäude exportieren   (DAE + Items)
6. Horizont exportieren
7. Finalisierung         (Materials/Items/Forest-JSON speichern)
```

Jede Nummer entspricht einer sichtbaren Zeile/Balken auf Top-Level; 4.1-4.10 sind Unteraufgaben
innerhalb von Hauptaufgabe 4, da `terrain_workflow.process_tile()` diese Schritte intern
nacheinander ausführt.

## Progress-API (`world_to_beamng/progress.py`, neu)

Dünner Wrapper um `rich.progress.Progress` + `rich.console.Console`, ersetzt `StepTimer`
vollständig (inkl. dessen Abschluss-Zeitübersicht, die künftig eine `rich.table.Table` wird).

```python
pipeline = Pipeline()  # ein Console/Live-Kontext für den ganzen Lauf

with pipeline.task("Terrain + Straßen") as task:
    with task.subtask("OSM-Daten laden"):                    # Spinner: keine Stückzahl bekannt
        ...
    with task.subtask("Bäume platzieren", total=n) as sub:    # Balken: Stückzahl bekannt
        for tree in trees:
            ...
            sub.advance()
    task.done(f"{n} Bäume platziert")                         # ✓ grüne Zeile mit Kurzresultat
```

Regeln:
- `subtask(name)` ohne `total` → Spinner mit verstrichener Zeit (für Aufrufe ohne bekannte
  Schrittzahl: OSM laden, Straßennetz bauen, Brücken/Tunnel/Mauern/Wasser generieren).
- `subtask(name, total=n)` → echter Prozent-Balken (Bäume, Gebäude, Downloads, Chunks - überall
  dort, wo vorab eine Stückzahl feststeht).
- `task.done(summary)` / `task.warn(summary)` / `task.fail(summary)` schreiben die
  Abschlusszeile mit ✓ (grün) / ⚠ (gelb) / ✗ (rot) und beenden die Live-Anzeige der Aufgabe.
- Normale `logger.info/debug`-Aufrufe *innerhalb* eines `task`/`subtask`-Blocks bleiben
  unverändert bestehen (kein Zwang, sie in die neue API zu pressen) und erscheinen dank
  `rich`s `Live`+gemeinsamer `Console` sauber oberhalb der aktiven Balken/Spinner statt sie zu
  zerreißen.
- `Pipeline` ist der einzige Ort, der `rich` importiert und die Console hält; alle anderen Module
  bleiben bei reinem `logging`.

Das ist das Schema für künftige Erweiterungen: neue Hauptaufgabe → `pipeline.task(...)` an der
Aufrufstelle in `beamng_exporter.py`; neue Unteraufgabe mit bekannter Stückzahl →
`task.subtask(name, total=n)` + `.advance()`; alles andere bleibt spinner-basiert oder normales
Logging.

## Log-Level-Politik (Entrümpelung)

- **INFO** (Standardausgabe): Start/Ende jeder Haupt-/Unteraufgabe inkl. Ein-Zeilen-
  Zusammenfassung (Counts, ✓/⚠/✗) sowie alle `warning`/`error`-Aufrufe.
- **DEBUG** (nur mit `config.DEBUG_VERBOSE=True`): die aktuell sehr geschwätzigen
  Zwischenschritt-Zeilen (z.B. die `→ ...`-Zeilen in `forest_workflow.py`, die Detail-Logs in
  `terrain_workflow.py`) wandern von INFO auf DEBUG. Bestehender `config.DEBUG_VERBOSE`-Schalter
  bleibt unverändert die Steuerung dafür.
- Farbschema folgt Log-Level (über `rich`s `RichHandler`, gemeinsame Console mit der
  Progress-Anzeige): INFO neutral, WARNING gelb, ERROR rot - konsistent mit den ✓/⚠/✗-Symbolen
  der Task-Zusammenfassungen.

## Betroffene Dateien (Umsetzungsplan folgt separat)

- `world_to_beamng/logging_config.py` - Root-Cause-Fix (Paket-Logger statt `"w2b"`),
  Third-Party-Logger dämpfen, `RichHandler` einhängen
- `world_to_beamng/progress.py` - neu, `Pipeline`/`PipelineTask`-API
- `world_to_beamng/utils/timing.py` - `StepTimer` entfernen (durch `Pipeline` ersetzt)
- `world_to_beamng/export/beamng_exporter.py` - `export_complete_level()` auf
  `pipeline.task(...)`-Aufrufe für die 8 Hauptaufgaben umstellen
- `world_to_beamng/workflow/terrain_workflow.py` - `process_tile()` intern auf die 10
  Unteraufgaben (4.1-4.10) umstellen
- Diverse Module mit sehr geschwätzigen INFO-Logs (`forest_workflow.py`,
  `terrain_workflow.py`, u.a.) - Log-Level einzelner Zeilen auf DEBUG absenken
- `requirements.txt` - `rich` als neue Abhängigkeit ergänzen

## Out of Scope

- Kein Umbau der Log-Datei-Funktion selbst (`config.LOGGING_FILE` bleibt optional/`None`
  per Default) - nur der Root-Cause-Fix, der sie (falls aktiviert) ebenfalls korrekt befüllt.
- Keine Änderung an `tools/*.py`-Skripten (eigene Einstiegspunkte, nutzen die Pipeline-API nicht).
- Keine Parallelisierung der Pipeline-Schritte - die Reihenfolge bleibt sequenziell, nur die
  Anzeige wird gegliedert.
