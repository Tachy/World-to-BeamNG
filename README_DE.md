[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

# 🗺️ World-to-BeamNG

🇬🇧 [English version](README.md) · 🇩🇪 Deutsche Version (diese Datei)

Erzeugt aus offenen Geodaten ein **spielbares BeamNG.drive-Level**: Gelände, Luftbild, Straßen, Wald, Weinberge,
Bäche und Teiche, Gebäude und einen Horizont. Alpha-Status, in Entwicklung.

## 🎯 Was wird erzeugt

| Bestandteil | Quelle |
|---|---|
| **Terrain** (natives `.terrain`, 1 m Raster) mit Luftbild-Textur | DGM1 + DOP20 |
| **Straßen** als BeamNG-`DecalRoad`, eingebettet ins Gelände | OpenStreetMap + DGM1 |
| **Wald, Einzelbäume, Weinberge, Bodenbewuchs** | OpenStreetMap |
| **Bäche und Teiche** (`River`, `WaterBlock`) | OpenStreetMap + DGM1 |
| **Bruchsteinmauern** (50 cm dick, Steinplatten oben drauf, folgen dem Gelände; neben einer Straße stehen sie auf deren Centerline-Höhe) für `barrier=wall`/`retaining_wall` mit `height`-Tag | OpenStreetMap + DGM1 |
| **Gebäude** aus LOD2: verputzte Wände in Weiß/Beige/vereinzelt Rot, Fenster, Türen, Kellerfenster, Dächer mit Überstand, Kies auf Flachdächern, Kirchtürme mit Turmuhr | LOD2 (+ OpenStreetMap für Kirchen) |
| **Horizont** bis 50 km (optional) | DGM30 + Satellitenbild |

## 📋 Voraussetzungen

| | |
|---|---|
| Betriebssystem | **Windows 10/11** (BeamNG-Pfade unter `%LOCALAPPDATA%`, `texconv.exe`) |
| BeamNG.drive | installiert und **mindestens einmal gestartet** (legt den Benutzerordner an) |
| Python | **3.11 oder neuer**, getestet mit 3.13 |
| Internet | für OpenStreetMap (Overpass API) und den einmaligen Download von `texconv.exe` |
| Gebiet | Jede Region mit georeferenziertem GeoTIFF-Höhenmodell und -Orthophoto: CRS und Ausdehnung werden automatisch aus dem Dateiinhalt erkannt, kein festes Namensschema oder feste Kachelgröße nötig. Reine ASCII-XYZ-Punktwolken in einem ZIP (das LGL-Baden-Württemberg-Format) werden genauso am Inhalt statt am Namen erkannt (siehe „Daten aus anderen Regionen verwenden" unten). Gebäude (LoD2/CityGML) bleiben Baden-Württemberg-spezifisch und müssen anderswo deaktiviert werden (`LOD2_ENABLED = False`). |
| Speicher | pro 2×2-km-Kachel etwa 250 MB Rohdaten (siehe unten) plus Cache und Ergebnis |

## 🚀 Schnellstart

```powershell
# 1. Repository holen
git clone https://github.com/Tachy/World-to-BeamNG.git
cd World-to-BeamNG

# 2. Virtuelle Umgebung, Pakete und texconv.exe
python -m venv .venv
.\.venv\Scripts\python.exe setup_project.py

# 3. Basisdaten nach data/ legen (siehe "Basisdaten")

# 4. Level erzeugen
.\.venv\Scripts\python.exe world_to_beamng.py
```

Danach BeamNG.drive starten und das Level **„World to BeamNG"** wählen. Der Level-Ordner liegt automatisch im
BeamNG-Benutzerordner (`%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\levels\world_to_beamng`), es muss kein Pfad
eingestellt werden.

`setup_project.py` installiert `requirements.txt` in den Python, mit dem es gestartet wird (hier die `.venv`) und
lädt `texconv.exe` (Microsoft DirectXTex) nach `bin/`. Das Programm braucht `texconv.exe` für alle DDS-Texturen;
fehlt es, lädt der Export es beim ersten Bedarf selbst herunter.
Die Tests brauchen zusätzlich `pytest` (`.\.venv\Scripts\pip install pytest`).

## 📦 Basisdaten

Die Daten sind **nicht im Repository** (rund 1 GB je 4×4 km). Sie kommen vom
**Landesamt für Geoinformation und Landentwicklung Baden-Württemberg (LGL)** aus dem Open-GeoData-Portal
(<https://opengeodata.lgl-bw.de>) und werden als ZIP-Dateien **unverändert** in die Ordner unter `data/` gelegt. Die
drei Ordner (`data/height/`, `data/satellite/`, `data/buildings/`) existieren bereits im Repository, jeder mit einer
kurzen `README.md`, die beschreibt, was dort hingehört; nur ihr Inhalt ist von Git ignoriert.

### Das Gebiet ergibt sich aus den DGM1-Kacheln

Alle Produkte kommen in **2×2-km-Kacheln**. Der Dateiname enthält die Koordinate der Südwest-Ecke in Kilometern
(UTM 32, ETRS89): `…_32_399_5296_…` ist Rechtswert 399 000 m, Hochwert 5 296 000 m. Das Programm liest **alle**
vorhandenen DGM1-ZIPs und bearbeitet genau diese Fläche. Für dieselben Kacheln müssen auch Luftbild und
Gebäude vorliegen. Beispiel für ein 4×4-km-Gebiet: `399`/`401` × `5296`/`5298`.

### Pflicht

| Ordner | Inhalt | Dateiname | Größe je Kachel |
|---|---|---|---|
| `data/height/` | Digitales Geländemodell 1 m (ZIP mit XYZ-Punkten) | `dgm1_32_<x>_<y>_2_bw.zip` | ca. 14 MB |
| `data/satellite/` | Digitale Orthophotos 20 cm, RGB (ZIP mit TIF + TFW) | `dop20rgb_32_<x>_<y>_2_bw.zip` | ca. 230 MB |

Der Dateiname spielt nur für das LGL-BW-Format oben eine Rolle. Jedes andere georeferenzierte GeoTIFF-Höhenmodell
(lose Datei oder in einem ZIP) funktioniert ebenfalls, unter beliebigem Dateinamen, und wird immer auf
`GRID_SPACING` umgetastet, unabhängig von seiner nativen Auflösung; entsprechend für jedes georeferenzierte
Orthophoto (eingebettete GeoTIFF-Tags oder eine `.tfw`-Weltdatei) unter `data/satellite/`. Siehe „Daten aus anderen
Regionen verwenden" unten.

Ohne DGM1 bricht der Export ab („no DGM1 tiles found" - keine DGM1-/GeoTIFF-Kacheln gefunden). Fehlt das
Luftbild, meldet der Export einen Fehler im Log.

### Optional

| Ordner | Inhalt | Dateiname | Wenn es fehlt |
|---|---|---|---|
| `data/buildings/` | 3D-Gebäudemodelle LoD2 (ZIP mit CityGML) | `LoD2_32_<x>_<y>_2_bw.zip` | keine Gebäude (`LOD2_ENABLED`) |

Fertiges Beispiel-Layout:

```
World-to-BeamNG/
└── data/
    ├── height/     dgm1_32_399_5296_2_bw.zip   dgm1_32_399_5298_2_bw.zip   …
    ├── satellite/  dop20rgb_32_399_5296_2_bw.zip   …
    └── buildings/  LoD2_32_399_5296_2_bw.zip   …
```

### Horizont (optional, `PHASE5_ENABLED`)

Der Horizont bis 50 km braucht zwei Dinge - 30-m-Höhendaten (**Copernicus DEM GLO-30**) und ein Satellitenbild
(**Sentinel-2 cloudless** über den EOX-WMS) - und beide werden beim ersten Lauf **vollständig automatisch** geladen
(`config.DGM30_AUTO_DOWNLOAD` / `config.EOX_AUTO_DOWNLOAD`, beide standardmäßig `True`). Nichts zum Herunterladen
oder Ablegen von Hand; beide liegen unter `cache/` (nicht `data/`), weil das Programm sie vollständig selbst
verwaltet:

- `cache/dgm30/` - die rohen Copernicus-DEM-GLO-30-Kacheln (`Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif` usw., eine
  je 1°×1°-Kachel, je etwa 43 MB). Fehlt eine Kachel (z. B. kein Internet), meldet der Export, in welcher
  Himmelsrichtung die Daten enden, und der Horizont ist dort kürzer.
- `cache/horizon_source/` - das rohe Sentinel-2-Mosaik vor dem Zuschnitt, und `cache/horizon_texture/` - die
  fertige, zugeschnittene Textur, die tatsächlich verwendet wird. Beide sind gebietsabhängig benannt (bei der Textur
  zusätzlich nach Zielgröße/Resampling), sodass ein Wechsel des Quellgebiets (`data/height/` usw.) und zurück die
  Bilder der beiden Gebiete nie vermischt.

Alle drei sind gewöhnliche Caches: jederzeit sicher löschbar, werden beim nächsten Lauf automatisch neu aufgebaut.
Falls der Auto-Download eine Kachel nicht bekommen konnte (z. B. Offline-Betrieb), kann man weiterhin von Hand ein
Copernicus-DEM-GLO-30-GeoTIFF in `cache/dgm30/` ablegen - der öffentliche AWS-Bucket `copernicus-dem-30m`
(Region eu-central-1) hat eins, benannt nach seiner Südwest-Ecke:

```
https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/Copernicus_DSM_COG_10_N47_00_E007_00_DEM/Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif
                                                                                  └ 47° N ┘ └ 7° O ┘
```
(`aws s3 cp --no-sign-request s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N47_00_E007_00_DEM/Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif cache/dgm30/`
funktioniert auch ohne Konto.) Für das Horizont-Satellitenbild gibt es keinen solchen manuellen Fallback mehr - es
ist ausschließlich Sentinel-2 cloudless.

Die Sentinel-2-cloudless-Bilder stammen von © **EOX IT Services GmbH** (<https://cloudless.eox.at>), lizenziert unter
**CC BY-NC-SA 4.0** (nicht-kommerzielle Nutzung, Namensnennung + Weitergabe unter gleichen Bedingungen erforderlich) -
den genauen Pflicht-Attributionstext siehe „Lizenz und Quellenangaben" unten.

### Daten aus anderen Regionen verwenden

`data/height/` und `data/satellite/` akzeptieren jedes georeferenzierte GeoTIFF (Einzelband-Höhenmodell, RGB-Orthophoto),
lose oder in einem ZIP, unter beliebigem Dateinamen - Koordinatensystem und abgedeckte Fläche werden aus der Datei
selbst gelesen, nicht aus einem Namensschema erraten. So werden Daten einer anderen Region als Baden-Württemberg
genutzt:

- Die DGM-GeoTIFFs nach `data/height/` legen, die Orthophoto-GeoTIFFs nach `data/satellite/`. Das Höhenmodell wird immer
  auf `config.GRID_SPACING` (Standard 1 m) umgetastet, unabhängig von seiner nativen Auflösung; die Orthophotos
  werden zu `config.PHOTO_TILE_SIZE_M`-großen Kacheln (Standard 2000 m) mit `TERRAIN_BASE_TEX_PIXEL_SIZE`
  zusammengesetzt, unabhängig davon, wie viele Quelldateien es gibt oder wie das Quellportal selbst kachelt.
- Die Quell-CRS wird automatisch aus den GeoTIFFs erkannt; `config.SOURCE_CRS_EPSG` ist nur der Fallback für
  Höhendaten ohne eingebettetes CRS (reine ASCII-XYZ-Punktwolken, z. B. das LGL-BW-Format) und wird sonst ignoriert.
  Weicht das CRS des Orthophotos vom CRS der Höhendaten ab, wird es automatisch umprojiziert.
- `LOD2_ENABLED = False` in `config.py` setzen - Gebäude (LoD2/CityGML) bleiben spezifisch für das
  Baden-Württemberg-CityGML-1.0-Schema.
- Höhendaten-GeoTIFFs in unterschiedlichen CRS lassen sich mischen: Die Fläche wird im CRS der meisten Kacheln
  verarbeitet, die übrigen werden automatisch umprojiziert. Der Horizont-Auto-Download (DGM30/Satellitenbild) bleibt
  weltweit nutzbar (siehe „Horizont" oben).

Durchgehend getestet mit echten Daten außerhalb Baden-Württembergs: swissALTI3D (0,5 m Höhenmodell) und
SwissImage DOP10 (0,1 m Orthophoto), beide EPSG:2056 (CH1903+/LV95), lose GeoTIFFs ohne feste Kachelgröße.

### Was das Programm selbst besorgt

- **OpenStreetMap** (Straßen, Wald, Wasser, Landnutzung, Kirchen) über die Overpass API mit Ersatz-Servern. Die
  Antworten liegen danach in `cache/`.
- **Höhen für Gelände, Straßen und Wasser** kommen aus DGM1, die der Gebäude aus LOD2; dafür gibt es keinen
  Download. Nur der Horizont lädt DGM30 + Satellitenbild automatisch (siehe „Horizont" oben).
- **Putz- und Fenstertexturen** der Gebäude erzeugt das Programm selbst. Einmalig erzeugte Texturen (Dachkies und die
  Bruchsteinmauer aus einem Foto) liegen in `data/textures/` im Repository, je Textur ein Ordner plus `manifest.json`;
  der Export wandelt sie nur noch in DDS um. `textures/registry.py` listet, welche Texturen der Export braucht, und
  prüft sie vorab: fehlende prozedurale (Kies) werden einmalig erzeugt, eine fehlende Foto-Textur (Bruchsteinmauer)
  **bricht den Export ab**, samt Befehl zum Erzeugen. Neue Foto-Texturen: `tools\make_seamless_texture.py`.

### Aus der BeamNG-Installation

Der Installationspfad wird aus `%LOCALAPPDATA%\BeamNG\BeamNG.drive.ini` gelesen. Daraus werden BeamNG-eigene Inhalte
übernommen, die nicht ins Repository dürfen:

| Was | Wie | Wenn es fehlt |
|---|---|---|
| Baum-Modelle (`east_coast_usa`) und `managedItemData.json` | automatisch beim Export | kein Wald (Warnung im Log) |
| Standard-Texturen (Straßen, Dachziegel, Terrain-Details) | automatisch beim Export | BeamNG zeigt „no Texture" |
| Weinreben aus dem `italy`-Level | automatisch beim Export | Weinberge ohne Reben |

Kopiert wird nur, was fehlt oder sich in BeamNG geändert hat (z. B. nach einem Update); spätere Exporte überspringen
den Schritt. Die Waldtyp-Vorlagen in `data/osm_to_beamng.json` lassen sich mit `tools\generate_forest_types.py` aus den
Baum-Modellen neu erzeugen (nur nötig, wenn sich die Baumauswahl ändern soll).

## ⚙️ Konfiguration

Alle Einstellungen stehen in `world_to_beamng/config.py`.

| Einstellung | Bedeutung |
|---|---|
| `LOD2_ENABLED`, `FORESTS_ENABLED`, `VINEYARDS_ENABLED`, `WATER_ENABLED`, `GROUND_COVER_ENABLED`, `PHASE5_ENABLED` | einzelne Bestandteile ein- und ausschalten (`PHASE5_ENABLED` ist der Horizont) |
| `BEAMNG_DIR` | Zielordner des Levels; wird aus `%LOCALAPPDATA%` abgeleitet, nur bei Sonderfällen ändern |
| `GRID_SPACING` | Terrain-Auflösung in Metern (Standard 1,0); GeoTIFF-Höhendaten werden immer darauf umgetastet, unabhängig von ihrer nativen Auflösung |
| `TERRAIN_BASE_TEX_PIXEL_SIZE` | Größe des Luftbilds je Kachel |
| `PHOTO_TILE_SIZE_M` | Kachelgröße (Meter) des Luftbild-/Material-Rasters (Standard 2000), unabhängig von der Kachelung der Rohdaten |
| `SOURCE_CRS_EPSG` | Fallback-Quell-CRS (Standard 25832) für Höhendaten ohne eingebettetes CRS (reine ASCII-XYZ); wird für GeoTIFF-Quellen ignoriert, deren CRS automatisch erkannt wird |
| `SUN_REFERENCE_LATLON` | Ungefähre `(Breite, Länge)` nur für den Sonnenstand (Datum/Uhrzeit) - **nicht** der Fahrzeug-Spawn-Punkt, der automatisch platziert wird (siehe unten) |
| `ENV_DATE`, `ENV_CLOCK_TIME` | Datum und Uhrzeit für den Sonnenstand |
| `DGM30_AUTO_DOWNLOAD`, `EOX_AUTO_DOWNLOAD` | die DGM30-Kacheln bzw. das Sentinel-2-Horizontbild beim ersten Lauf automatisch laden (Standard jeweils `True`); `False` deaktiviert diese Quelle und degradiert auf das bestehende Überspringen-Verhalten (Horizont übersprungen / Horizont ohne Textur) |
| `DGM30_S3_BUCKET`, `DGM30_S3_REGION`, `DGM30_FETCH_MAX_RETRIES`, `DGM30_FETCH_TIMEOUT_S`, `DGM30_NOT_FOUND_CACHE_TTL_DAYS` | Feinabstimmung für den Copernicus-DEM-GLO-30-Auto-Download (Bucket/Region, Retry/Timeout, wie lange eine bestätigt fehlende Kachel - z. B. offenes Meer - vor einem erneuten Versuch als „fehlt" gemerkt wird) |
| `EOX_WMS_URL`, `EOX_WMS_LAYER`, `EOX_WMS_VERSION`, `EOX_WMS_FORMAT`, `EOX_MAX_REQUEST_PX`, `EOX_TARGET_RESOLUTION_M`, `EOX_MOSAIC_MAX_PX`, `EOX_FETCH_MARGIN_FACTOR`, `EOX_FETCH_MAX_RETRIES`, `EOX_FETCH_TIMEOUT_S`, `EOX_KEEP_RAW_MOSAIC`, `EOX_MOSAIC_CACHE_DIR`, `EOX_TEXTURE_CACHE_DIR` | Feinabstimmung für den EOX-Sentinel-2-cloudless-WMS-Auto-Download (Endpunkt/Layer/Version, Größenlimit je Anfrage, Zielauflösung, Mosaik-Größenobergrenze, Rohmosaik- und Fertig-Textur-Cache) |

Siehe „Horizont" oben für die Caches `cache/dgm30/`, `cache/horizon_source/` und `cache/horizon_texture/`, die diese
Einstellungen feinabstimmen - alle drei sind jederzeit sicher löschbar und werden automatisch neu aufgebaut.

**Fahrzeug-Spawn:** das Auto spawnt automatisch auf der Straße, die der Gebietsmitte am nächsten liegt, in eine der
beiden Richtungen entlang dieser Straße (beliebig) - nichts zu konfigurieren.

## ⏱️ Ablauf und Dauer

Ein Lauf liest die DGM1-Kacheln, lädt (beim ersten Mal) die OSM-Daten, baut das Luftbild und das Terrain und
schreibt Straßen, Wald, Wasser, Gebäude und Horizont in den Level-Ordner. Bei 4×4 km dauert ein Lauf mit gefüllten
Caches etwa eine Minute. Der erste Lauf ist länger, weil OSM geladen und die Caches aufgebaut werden. Die Caches
(`cache/`) werden bei geänderten Daten automatisch ungültig. Bei seltsamen Ergebnissen hilft es, `cache/` zu löschen.

## 🐛 Fehlersuche

Die Meldungen des Programms sind englisch; sie werden hier im Wortlaut zitiert.

| Meldung / Symptom | Ursache und Lösung |
|---|---|
| `no DGM1 tiles found` | `data/height/` fehlt, ist leer, oder enthält keine lesbaren ZIPs/GeoTIFFs (ein GeoTIFF ohne Koordinatensystem wird mit Warnung übersprungen) |
| `texconv.exe not found … download failed` | kein Internet bei der ersten DDS-Umwandlung; `texconv.exe` manuell nach `bin\` laden |
| `BeamNG.drive.ini not found` | BeamNG.drive wurde noch nie gestartet |
| `Tree assets not available …` / kein Wald | BeamNG-Installation nicht gefunden (`BeamNG.drive.ini`) oder `east_coast_usa.zip` fehlt in `content/levels` |
| Straßen oder Dächer mit „no Texture" | Warnung `… stock texture(s) not found in the BeamNG content zips` im Log prüfen |
| Level erscheint nicht in BeamNG | prüfen, ob `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\levels\world_to_beamng` entstanden ist; sonst `BEAMNG_DIR` in `config.py` anpassen |
| `The DGM30 files do not cover the horizon area …` | eine Kachel, die der Auto-Download nicht bekommen konnte (z. B. kein Internet für diese Anfrage) - wird beim nächsten Lauf automatisch erneut versucht, oder die Kachel für die genannte Himmelsrichtung von Hand nach `cache/dgm30/` legen (siehe „Horizont" oben) |
| `No DGM30 files (*.tif) in …` | `cache/dgm30/` ist leer und der Auto-Download lief noch nicht oder ist komplett fehlgeschlagen; der Horizont wird sonst übersprungen |
| `DGM30 auto-download failed` / Sentinel-2-Auto-Download funktioniert nicht, kein Internet | der Auto-Download protokolliert eine Warnung und fällt auf das bisherige Überspringen-Verhalten zurück (Horizont übersprungen / Horizont ohne Textur) - er bricht den Export nie ganz ab; er versucht es beim nächsten Lauf erneut, sobald das Netzwerk wieder da ist |
| Horizont ohne Textur oder verschoben | `cache/horizon_texture/` löschen (und `cache/horizon_source/`, falls schon das Rohmosaik falsch aussieht), um beim nächsten Lauf einen Neuaufbau zu erzwingen |
| Absturz oder Fehler beim Laden des Levels | `C:\Users\<NAME>\AppData\Local\BeamNG\BeamNG.drive\current\beamng.log` auf `\|E\|`-Zeilen prüfen |
| OSM-Zeitüberschreitung | das Programm probiert Ersatz-Server; erneut starten, erfolgreiche Antworten sind gecacht |

## 🧪 Tests

```powershell
.\.venv\Scripts\pip install pytest
.\.venv\Scripts\python.exe -m pytest tests
```

## 🔍 Level-Viewer

Ein Entwickler-Viewer zeigt den exportierten Level außerhalb von BeamNG (Terrain mit Luftbild, Straßen und
Markierungen, Brücken, Tunnel, Wasser, Wald, Tunnel-Zonen, Spawn-Punkte, Horizont, Debug-Netzwerk). Standardmäßig
zeigt er das Terrain in voller Auflösung mit den vollen Luftbild-Kacheln (lädt in wenigen Sekunden, braucht etwa
4 GB Arbeitsspeicher):

```powershell
.\.venv\Scripts\python.exe -m tools.level_viewer                # Level aus config.BEAMNG_DIR
.\.venv\Scripts\python.exe -m tools.level_viewer --terrain-step 4 --minimap-photo   # schneller, weniger Speicher
.\.venv\Scripts\python.exe -m tools.level_viewer --help         # alle Optionen (Ebenen, Terrain-Schritt, Screenshot)
```

Maus: Linke Taste ziehen dreht um die senkrechte Achse und neigt den Blick (das Gelände kippt nie), Shift + links
ziehen oder die mittlere Taste verschiebt, Mausrad oder rechte Taste ziehen zoomt. **Doppelklick** auf ein Objekt zeigt
Name, Klasse, Material, Position und die Terrainhöhe darunter; die Kamera fliegt dorthin (40 m Abstand). `Leertaste`
zeigt das ganze ausgewählte Objekt, `Esc` hebt die Auswahl auf, `f` fliegt zum Punkt unter dem Mauszeiger,
`Pfeil hoch/runter` zoomt, `r` setzt die Kamera zurück, `v` zeigt eine Schrägansicht, `+`/`-` machen Linien und
Punkte dicker/dünner. `i` blendet diese Übersicht im Fenster ein und aus.

Tasten: `g` Terrain, `x` Luftbild/Höhenfarben, `a` Straßen, `m` Markierungen, `o` Wasser, `b` Bauwerke, `h` Horizont,
`c` Wald, `z` Zonen + Spawns, `n` Beschriftungen, `d` Debug-Netzwerk, `l` neu laden, `i` Hilfe, `q` beenden. Fenster,
Kamera und Ebenen werden in `tools/level_viewer.cfg` gespeichert.

## 🏗️ Projektstruktur

```
world_to_beamng.py        Einstiegspunkt
setup_project.py          Pakete und texconv.exe installieren
world_to_beamng/
├── config.py             alle Einstellungen
├── export/               Level-Export (BeamNGExporter)
├── workflow/             Terrain-, Gebäude-, Wald- und Horizont-Ablauf
├── terrain/              Höhenmodell, Straßeneinbettung, Luftbild, Horizont
├── osm/                  OpenStreetMap-Download und -Auswertung
├── io/                   LOD2-Einlesen, Luftbild, Caches
├── facade/               Gebäude: Putzwände, Fenster, Dächer, Kirchtürme
├── forest/               Wald, Weinberge, Bodenbewuchs
├── walls/                Bruchsteinmauern (OSM barrier=wall mit height), Steinplatten oben drauf
├── textures/             Textur-Bibliothek (data/textures) und Werkzeuge für kachelnde Texturen
├── managers/             Materialien, Level-Objekte, DAE-Export
├── builders/, core/      Mesh-Builder, Cache-Verwaltung
└── geometry/, mesh/, analysis/, utils/
data/                     Konfigurations-JSONs (im Repository) und Basisdaten (nicht im Repository)
tools/                    Hilfsskripte (Assets übernehmen, Horizont-Bild erzeugen, Prüfungen, level_viewer/)
tests/                    pytest
docs/                     technische Dokumentation (z. B. MATERIAL_TEMPLATES.md)
```

## 📄 Lizenz und Quellenangaben

Der Code steht unter der **MIT License** (siehe [LICENSE](LICENSE)). Für die Daten gilt:

- **LGL Baden-Württemberg** (DGM1, DOP20, LoD2): **Datenlizenz Deutschland – Namensnennung – Version 2.0**
  (dl-de/by-2.0, <https://www.govdata.de/dl-de/by-2-0>). Der Quellenvermerk lautet:
  *„Datenquelle: LGL, www.lgl-bw.de, dl-de/by-2-0"*, dazu der Hinweis, dass die Daten verändert wurden. Wer
  erzeugte Level weitergibt, muss das angeben. Die Lizenztexte liegen auch in den ZIP-Dateien.
- **OpenStreetMap**: © OpenStreetMap-Mitwirkende, [ODbL](https://www.openstreetmap.org/copyright).
- **Copernicus DEM GLO-30** (Horizont): frei nutzbar unter den Bedingungen der Copernicus-Lizenz, siehe
  <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM>.
- **Sentinel-2 cloudless (s2cloudless)** (Horizont-Hintergrundbild): © EOX IT Services GmbH,
  <https://cloudless.eox.at>, lizenziert unter CC BY-NC-SA 4.0 (nicht-kommerzielle Nutzung, Namensnennung +
  Weitergabe unter gleichen Bedingungen erforderlich) - siehe <https://cloudless.eox.at/documentation/license>.
  Pflicht-Attribution: "EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH (Contains modified
  Copernicus Sentinel data 2025)". Kommerzielle Nutzung erfordert eine separate EOX Commercial
  Attribution-RestrictedUse-Lizenz.
- **BeamNG-Inhalte** (Texturen, Bäume) bleiben Eigentum von BeamNG und werden nur aus der eigenen Installation in das
  eigene Level kopiert. Sie gehören nicht ins Repository.

## 🤝 Beiträge

Beiträge sind willkommen: Fork, Branch anlegen, Änderungen committen (Präfixe `feat:`, `fix:`, `perf:` …), Pull
Request öffnen. Fehler und Wünsche bitte als [Issue](https://github.com/Tachy/World-to-BeamNG/issues) melden.

### Versionen und Releases

Die Commit-Präfixe bestimmen die Versionsnummer ([Conventional Commits](https://www.conventionalcommits.org/)): `feat:`
erhöht die Minor-Nummer, `fix:` und `perf:` die Patch-Nummer, `feat!:` / `BREAKING CHANGE:` die Major-Nummer (unter 1.0
die Minor-Nummer). Eine GitHub Action ([release-please](https://github.com/googleapis/release-please)) hält auf
`main` einen Release-Pull-Request aktuell; sein Merge erzeugt den Tag (`vX.Y.Z`), das GitHub-Release mit Änderungsliste
([CHANGELOG.md](CHANGELOG.md)) und aktualisiert `__version__`. Die Version steht am Ende eines Exports und im
Level-Viewer.

## 🙏 Danksagungen

**OpenStreetMap** und die **Overpass API**, das **LGL Baden-Württemberg** für die offenen Geodaten, **BeamNG** und die
Communities von **Shapely**, **NumPy**, **SciPy** und **Rasterio**.
