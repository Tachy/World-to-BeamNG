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
| Gebiet | **Baden-Württemberg**: Die Dateinamen und Formate sind die des LGL BW (UTM Zone 32, ETRS89). Andere Bundesländer oder Länder gehen nicht ohne Anpassung. |
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
# 4. SPAWN_POINT in world_to_beamng/config.py auf das eigene Gebiet setzen (siehe "Konfiguration")

# 5. Einmalig: Assets aus der BeamNG-Installation übernehmen
.\.venv\Scripts\python.exe tools\generate_forest_assets.py
.\.venv\Scripts\python.exe tools\vendor_shared_textures.py

# 6. Level erzeugen
.\.venv\Scripts\python.exe world_to_beamng.py
```

Danach BeamNG.drive starten und das Level **„World to BeamNG"** wählen. Der Level-Ordner liegt automatisch im
BeamNG-Benutzerordner (`%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\levels\world_to_beamng`), es muss kein Pfad
eingestellt werden.

`setup_project.py` installiert `requirements.txt` in den Python, mit dem es gestartet wird (hier die `.venv`) und
lädt `texconv.exe` (Microsoft DirectXTex) nach `bin/`. Das Programm braucht `texconv.exe` für alle DDS-Texturen.
Die Tests brauchen zusätzlich `pytest` (`.\.venv\Scripts\pip install pytest`).

## 📦 Basisdaten

Die Daten sind **nicht im Repository** (rund 1 GB je 4×4 km). Sie kommen vom
**Landesamt für Geoinformation und Landentwicklung Baden-Württemberg (LGL)** aus dem Open-GeoData-Portal
(<https://opengeodata.lgl-bw.de>) und werden als ZIP-Dateien **unverändert** in die Ordner unter `data/` gelegt. Die
Ordner müssen selbst angelegt werden, weil sie nicht in Git stehen.

### Das Gebiet ergibt sich aus den DGM1-Kacheln

Alle Produkte kommen in **2×2-km-Kacheln**. Der Dateiname enthält die Koordinate der Südwest-Ecke in Kilometern
(UTM 32, ETRS89): `…_32_399_5296_…` ist Rechtswert 399 000 m, Hochwert 5 296 000 m. Das Programm liest **alle**
vorhandenen DGM1-ZIPs und bearbeitet genau diese Fläche. Für dieselben Kacheln müssen auch Luftbild und
Gebäude vorliegen. Beispiel für ein 4×4-km-Gebiet: `399`/`401` × `5296`/`5298`.

### Pflicht

| Ordner | Inhalt | Dateiname | Größe je Kachel |
|---|---|---|---|
| `data/DGM1/` | Digitales Geländemodell 1 m (ZIP mit XYZ-Punkten) | `dgm1_32_<x>_<y>_2_bw.zip` | ca. 14 MB |
| `data/DOP20/` | Digitale Orthophotos 20 cm, RGB (ZIP mit TIF + TFW) | `dop20rgb_32_<x>_<y>_2_bw.zip` | ca. 230 MB |

Ohne DGM1 bricht der Export ab („Keine DGM1-Kacheln gefunden"). Fehlt das Luftbild, meldet der Export einen Fehler im Log.

### Optional

| Ordner | Inhalt | Dateiname | Wenn es fehlt |
|---|---|---|---|
| `data/LOD2/` | 3D-Gebäudemodelle LoD2 (ZIP mit CityGML) | `LoD2_32_<x>_<y>_2_bw.zip` | keine Gebäude (`LOD2_ENABLED`) |
| `data/DGM30/` | Höhenmodell 30 m als GeoTIFF: **Copernicus DEM GLO-30**, selbst herunterladen (siehe unten). Es dürfen mehrere `*.tif` im Ordner liegen. | beliebig, z. B. `Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif` | Horizont wird übersprungen |
| `data/DOP300/` | Satellitenbild für die Horizont-Textur: **ein** georeferenziertes RGB-GeoTIFF in **UTM 32N (EPSG:25832)**, das die ±50 km um die Gebietsmitte abdeckt. Beliebige Auflösung, es wird auf 8192×8192 skaliert. | `horizon_temp.tif` (Name in `config.SENTINEL2_FILE`) | Horizont ohne Textur |

**DGM30 herunterladen:** Das Programm lädt es nicht selbst. Verwendet wird das **Copernicus DEM GLO-30** (30 m,
weltweit, kostenlos). Am einfachsten ohne Konto aus dem öffentlichen AWS-Bucket `copernicus-dem-30m`
(Region eu-central-1, Cloud-Optimized GeoTIFFs, je Kachel etwa 43 MB). Eine Kachel deckt 1° × 1° ab, der Name
enthält ihre Südwest-Ecke:

```
https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/Copernicus_DSM_COG_10_N47_00_E007_00_DEM/Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif
                                                                                  └ 47° N ┘ └ 7° O ┘
```

Benötigt werden **alle Kacheln, die die Horizont-Fläche berühren** (±50 km um die Gebietsmitte, etwa ±0,7° in der
Länge und ±0,45° in der Breite). Für ein Gebiet bei 47,8° N / 7,7° O sind das `N47` und `N48` jeweils mit `E007` und
`E008` (vier Kacheln, zusammen etwa 170 MB). Die Dateien einfach in `data/DGM30/` legen; das Programm kombiniert sie und
schneidet sie auf die Horizont-Fläche zu. Fehlt eine Kachel, meldet der Export, in welcher Himmelsrichtung die Daten
enden, und der Horizont ist dort kürzer. Mit dem AWS-Kommandozeilenwerkzeug geht es auch ohne Konto:
`aws s3 cp --no-sign-request s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N47_00_E007_00_DEM/Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif data/DGM30/`.
Alternativen sind das Copernicus Data Space Ecosystem (<https://dataspace.copernicus.eu>) und OpenTopography
(<https://opentopography.org>).

**Horizont-Bild:** Es gibt keinen automatischen Download für das Satellitenbild, aber ein Werkzeug, das es aus
jedem georeferenzierten RGB-Bild erzeugt (z. B. einem Sentinel-2-Export in Web-Mercator oder WGS84). Es schneidet
genau die Horizont-Fläche (±50 km um die Gebietsmitte, sie ergibt sich aus den DGM1-Kacheln) aus, projiziert nach
EPSG:25832 um und schreibt `data/DOP300/horizon_temp.tif`:

```powershell
.\.venv\Scripts\python.exe tools\make_horizon_image.py C:\pfad\zum\satellitenbild.tif
```

Deckt das Quellbild nur einen Teil der Fläche ab, warnt das Werkzeug; der Rest bleibt schwarz. Ein Bild in einem
anderen Koordinatensystem im selben Ordner wird beim Export nicht verwendet, nur die Datei `horizon_temp.tif`.

Fertiges Beispiel-Layout:

```
World-to-BeamNG/
└── data/
    ├── DGM1/    dgm1_32_399_5296_2_bw.zip   dgm1_32_399_5298_2_bw.zip   …
    ├── DOP20/   dop20rgb_32_399_5296_2_bw.zip   …
    ├── LOD2/    LoD2_32_399_5296_2_bw.zip   …
    ├── DGM30/   dgm30_copernicus.tif
    └── DOP300/  horizon_temp.tif
```

### Was das Programm selbst besorgt

- **OpenStreetMap** (Straßen, Wald, Wasser, Landnutzung, Kirchen) über die Overpass API mit Ersatz-Servern. Die
  Antworten liegen danach in `cache/`.
- **Höhen für Gelände, Straßen und Wasser** kommen aus DGM1, die der Gebäude aus LOD2; dafür gibt es keinen
  Download. Nur der Horizont braucht DGM30 (siehe oben).
- **Putz- und Fenstertexturen** der Gebäude erzeugt das Programm selbst. Einmalig erzeugte Texturen (Dachkies und die
  Bruchsteinmauer aus einem Foto) liegen in `data/textures/` im Repository, je Textur ein Ordner plus `manifest.json`;
  der Export wandelt sie nur noch in DDS um. `textures/registry.py` listet, welche Texturen der Export braucht, und
  prüft sie vorab: fehlende prozedurale (Kies) werden einmalig erzeugt, eine fehlende Foto-Textur (Bruchsteinmauer)
  **bricht den Export ab**, samt Befehl zum Erzeugen. Neue Texturen: `tools\make_seamless_texture.py` (Foto) oder
  `tools\generate_gravel_texture.py`.

### Aus der BeamNG-Installation

Der Installationspfad wird aus `%LOCALAPPDATA%\BeamNG\BeamNG.drive.ini` gelesen. Daraus werden BeamNG-eigene Inhalte
übernommen, die nicht ins Repository dürfen:

| Was | Wie | Wenn es fehlt |
|---|---|---|
| Baum-Modelle und `managedItemData.json` | einmalig `tools\generate_forest_assets.py` | kein Wald (Warnung im Log) |
| Standard-Texturen (Straßen, Dachziegel) | einmalig `tools\vendor_shared_textures.py` | BeamNG zeigt „no Texture" |
| Weinreben aus dem `italy`-Level | automatisch beim Export | Weinberge ohne Reben |

Die Skripte sind wiederholbar. Neu ausführen nach einem BeamNG-Update oder wenn der Level-Ordner gelöscht wurde.

## ⚙️ Konfiguration

Alle Einstellungen stehen in `world_to_beamng/config.py`.

| Einstellung | Bedeutung |
|---|---|
| **`SPAWN_POINT`** | Startposition als `(Breite, Länge)` in Grad. Muss im eigenen Gebiet liegen. |
| `LOD2_ENABLED`, `FORESTS_ENABLED`, `VINEYARDS_ENABLED`, `WATER_ENABLED`, `GROUND_COVER_ENABLED`, `PHASE5_ENABLED` | einzelne Bestandteile ein- und ausschalten (`PHASE5_ENABLED` ist der Horizont) |
| `BEAMNG_DIR` | Zielordner des Levels; wird aus `%LOCALAPPDATA%` abgeleitet, nur bei Sonderfällen ändern |
| `GRID_SPACING` | Terrain-Auflösung in Metern (Standard 1,0 = native DGM1-Auflösung) |
| `TERRAIN_BASE_TEX_PIXEL_SIZE` | Größe des Luftbilds je Kachel |
| `ENV_DATE`, `ENV_CLOCK_TIME` | Datum und Uhrzeit für den Sonnenstand |

## ⏱️ Ablauf und Dauer

Ein Lauf liest die DGM1-Kacheln, lädt (beim ersten Mal) die OSM-Daten, baut das Luftbild und das Terrain und
schreibt Straßen, Wald, Wasser, Gebäude und Horizont in den Level-Ordner. Bei 4×4 km dauert ein Lauf mit gefüllten
Caches etwa eine Minute. Der erste Lauf ist länger, weil OSM geladen und die Caches aufgebaut werden. Die Caches
(`cache/`) werden bei geänderten Daten automatisch ungültig. Bei seltsamen Ergebnissen hilft es, `cache/` zu löschen.

## 🐛 Fehlersuche

| Meldung / Symptom | Ursache und Lösung |
|---|---|
| `Keine DGM1-Kacheln gefunden` | `data/DGM1/` fehlt oder enthält keine ZIPs im Schema `dgm1_32_<x>_<y>_2_bw.zip` |
| `texconv.exe nicht gefunden` | `setup_project.py` nicht gelaufen; oder Datei manuell nach `bin\texconv.exe` legen |
| `BeamNG.drive.ini nicht gefunden` | BeamNG.drive wurde noch nie gestartet |
| `managedItemData.json nicht gefunden` | einmalig `tools\generate_forest_assets.py` ausführen |
| Straßen oder Dächer mit „no Texture" | einmalig `tools\vendor_shared_textures.py` ausführen |
| Level erscheint nicht in BeamNG | prüfen, ob `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\levels\world_to_beamng` entstanden ist; sonst `BEAMNG_DIR` in `config.py` anpassen |
| `DGM30-Dateien decken die Horizont-Fläche … nicht ab` | Kacheln für die genannte Himmelsrichtung nach `data/DGM30/` legen (siehe oben) |
| `Keine DGM30-Dateien` | `data/DGM30/` ist leer; der Horizont wird sonst übersprungen |
| Horizont ohne Textur oder verschoben | `tools\make_horizon_image.py` verwenden; die Datei muss genau die Horizont-Fläche in EPSG:25832 zeigen |
| Absturz oder Fehler beim Laden des Levels | `C:\Users\<NAME>\AppData\Local\BeamNG\BeamNG.drive\current\beamng.log` auf `\|E\|`-Zeilen prüfen |
| OSM-Zeitüberschreitung | das Programm probiert Ersatz-Server; erneut starten, erfolgreiche Antworten sind gecacht |

## 🧪 Tests

```powershell
.\.venv\Scripts\pip install pytest
.\.venv\Scripts\python.exe -m pytest tests
```

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
tools/                    Hilfsskripte (Assets übernehmen, Horizont-Bild erzeugen, Prüfungen, Viewer)
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
- **BeamNG-Inhalte** (Texturen, Bäume) bleiben Eigentum von BeamNG und werden nur aus der eigenen Installation in das
  eigene Level kopiert. Sie gehören nicht ins Repository.

## 🤝 Beiträge

Beiträge sind willkommen: Fork, Branch anlegen, Änderungen committen (Präfixe `feat:`, `fix:`, `perf:` …), Pull
Request öffnen. Fehler und Wünsche bitte als [Issue](https://github.com/Tachy/World-to-BeamNG/issues) melden.

## 🙏 Danksagungen

**OpenStreetMap** und die **Overpass API**, das **LGL Baden-Württemberg** für die offenen Geodaten, **BeamNG** und die
Communities von **Shapely**, **NumPy**, **SciPy** und **Rasterio**.
