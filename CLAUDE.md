# CLAUDE.md

## Sprache

- Antworten im Chat: auf Deutsch.
- Code-Kommentare und Docstrings: auf Englisch (das Projekt ist öffentlich).
- Code selbst (Bezeichner, Funktionsnamen, Variablen, Strings im Code, Commit-Titel-Präfixe wie `fix:`/`feat:` etc.): auf Englisch.
- Log-Meldungen und alle anderen Laufzeit-Ausgaben: auf Englisch - `logger.*`-Aufrufe, Namen und Statustexte der
  Fortschrittsanzeige (`pipeline.task()`, `subtask()`, `finish()`/`done()`/`warn()`/`fail()`) und Texte von
  Exceptions. Neue Meldungen nie auf Deutsch schreiben.

## BeamNG-Logs zum Auswerten von Crashes/Ladefehlern

Das aktuell laufende BeamNG.drive (aktive Version, nicht die versionierten
`0.35`/`0.36`-Ordner) schreibt sein Log hierhin:

```
C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\beamng.log
```

- Ältere Sessions liegen daneben als `beamng.1.log`, `beamng.2.log`, ... (neueste zuerst).
- Crash-Reports (inkl. älterer Logs zum Zeitpunkt des Crashes) liegen unter
  `C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\temp\crashReports\`.
- Der exportierte Level liegt unter
  `C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\levels\world_to_beamng\`
  (u.a. `main\materials.json`, `main\items.level.json`, `*.ter`, `forest\forest.forest4.json`).

Beim Start einer Map immer zuerst `beamng.log` (nicht die `0.35`/`0.36`-Logs)
auf `|E|`-Zeilen und `Fatal-ISV`/`assert`-Meldungen gegen Ende der Datei prüfen.
