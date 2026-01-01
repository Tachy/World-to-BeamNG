[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

# 🗺️ World-to-BeamNG

Konvertiere OpenStreetMap-Straßen und Höhenmodelle zu hochdetaillierten **BeamNG.drive**-kompatiblen Gelände-Maps!

## 🎯 Beschreibung

**World-to-BeamNG** ist ein leistungsstarker Pipeline-Generator, der:

- ✅ **OpenStreetMap-Daten** abruft und Straßen extrahiert
- ✅ **Digitale Höhenmodelle** (XYZ-Dateien) als Terrain-Basis nutzt
- ✅ **Realistische Straßen-Geometrie** mit Böschungen generiert
- ✅ **Zentral dedupliziertes Vertex-Management** für optimierte Meshes nutzt
- ✅ **BeamNG.drive-kompatible OBJ-Dateien** mit Debug-Ebenen exportiert

### 🌍 Anwendungsbeispiele

- Erstellung von **Custom-Maps** aus echten GPS-Koordinaten
- Automatische **Straßen-Mesh-Generierung** mit realistischen Höhenprofilen
- Integration von **realen Kartendaten** in Simulationen
- **Batch-Verarbeitung** mehrerer Regions mit konsistenter Zeitmessung

## 🚀 Features

### Pipeline & Optimierung
- **Modular aufgebaut** mit klaren, importierbaren Modulen
- **Zentrale Zeitmessung** per `StepTimer` mit automatischer Schritt-Verwaltung
- **Vertex-Deduplication** für speicher- und speicherplatz-optimierte Meshes
- **Smart Caching** für Höhendaten und OSM-Abfragen (Cache-Verzeichnis: `cache/`)

### Robustheit
- **Fehlertoleranz** bei Overpass API mit Fallback-Endpoints
- **Automatische Cache-Invalidierung** bei Datenänderungen
- **CCW-Normalisierung** für konsistente Face-Orientierung
- **Optionales Stitching** und Boundary-Checks für geschlossene Meshes

### Flexibilität
- **Konfigurierbare Parameter** (Straßenbreite, Böschungswinkel, Grid-Spacing, etc.)
- **Junction-Remesh** mit lokaler Delaunay-Triangulation
- **Layer-basierter Output** (Road Surface, Slopes, Terrain separat klassifiziert)
- **Debug-Optionen** für einzelne Junctions oder Boundary-Analyse

## 📋 Anforderungen

### Pakete
```bash
pip install requests numpy scipy pyproj pyvista shapely rtree
```

### Daten
- **Höhendaten**: XYZ-Dateien im `height-data/` Verzeichnis
  - Format: X Y Z pro Zeile (Leerzeichen/Tab-getrennt)
  - Koordinaten in UTM Zone 32N (anpassbar via `config.py`)
  - Z-Werte in Metern über NN

## 🛠️ Installation

### 1. Repository clonen/öffnen
```bash
cd World-to-BeamNG
```

### 2. Virtual Environment (Windows PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install requests numpy scipy pyproj pyvista shapely rtree
```

### 3. Höhendaten vorbereiten
```
World-to-BeamNG/
├── height-data/
│   ├── dgm1_32_506_5644_1_de.xyz     ← Deine XYZ-Daten
│   ├── dgm1_32_507_5644_1_de.xyz
│   └── ...
```

## 📖 Verwendung

### Basis-Ausführung (Windows PowerShell)
```bash
.\.venv\Scripts\python.exe world_to_beamng.py
```

Das Skript wird:
1. Höhendaten aus `height-data/` laden und transformieren
2. BBOX automatisch berechnen
3. OSM-Straßen, Polygone und Junctions abrufen
4. Zentral dedupliziertes Vertex-Management aufbauen
5. Straßen- und Terrain-Mesh generieren
6. Junction-Remesh mit lokaler Delaunay durchführen
7. CCW-Normalisierung und optionales Stitching anwenden
8. Faces deduplizieren und `beamng.obj` exportieren
9. Automatische Zeitübersicht per `timer.report()` anzeigen

### Optionen
```bash
.\.venv\Scripts\python.exe world_to_beamng.py --junction-id 123
```
- `--junction-id <id>`: Nur diese Junction remeshen (Debug/Profiling).

### Konfiguration

Bearbeite `world_to_beamng/config.py`:

```python
# Mesh-Parameter
ROAD_WIDTH = 7.0              # Straßenbreite in Metern
SLOPE_ANGLE = 45.0            # Böschungswinkel in Grad
GRID_SPACING = 10.0           # Terrain-Auflösung in Metern

# Terrain-Optimierung
TERRAIN_REDUCTION = 0.0       # PyVista Decimation (0.0-1.0), derzeit deaktiviert

# Stitching & Checks
HOLE_CHECK_ENABLED = True     # aktiviert Boundary-Checks und Stitching-Versuche

# Debug/Export
BOUNDARY_EDGES_EXPORT = False # exportiert Boundary-Edges zu separater Datei (falls aktiviert)
```

## ⏱️ Zeitmessung

Die Zeitmessung ist **vollständig integriert** und benötigt keine externen Variablen:

```python
timer.begin("Mein Schritt")    # Beendet vorherigen Schritt automatisch
# ... Arbeit ...
timer.report()                 # Schließt offene Schritte, zeigt formatierte Übersicht
```

**Beispiel-Output:**
```
ZEITMESSUNG (Gesamtzeit: 190.55s / 3.2 min)
  1 Lade Hoehendaten....................     18.2% [████░░░░░░░░░░░░░]
  2 Berechne BBOX........................      2.1% [█░░░░░░░░░░░░░░░░░]
  3 Lade OSM-Daten.......................      5.3% [██░░░░░░░░░░░░░░░░]
  ...
 16 Exportiere OBJ.......................      3.2% [█░░░░░░░░░░░░░░░░░]
```

## 🏗️ Architektur

Die Anwendung ist modular aufgebaut für einfache Wartung und Erweiterung:

```
world_to_beamng/
├── terrain/          ← Höhendaten-Laden, Interpolation, Grid-Generierung
├── osm/              ← OpenStreetMap Downloader & Parser
├── geometry/         ← Koordinaten-Transformationen, Junctions, Polygone
├── mesh/             ← Straßen-Mesh, Terrain-Mesh, Junction-Remesh, Cleanup
├── io/               ← Cache-Management, OBJ-Export
├── analysis/         ← Overlap-Detection, Validierung
└── utils/            ← StepTimer, Hilfsfunktionen
```

## 🐛 Debugging

### Problem: Höhendaten nicht gefunden
```bash
# Prüfe height-data Verzeichnis
Get-ChildItem height-data/
# XYZ-Dateien sollten vorhanden sein
```

### Problem: OSM-API Timeout
Das Skript versucht automatisch Fallback-Server:
```
Server 1: overpass-api.de
Server 2: overpass.kumi.systems  
Server 3: overpass.openstreetmap.ru
```
Cache speichert erfolgreiche Abfragen in `cache/`.

### Problem: Nur eine Junction debuggen
```bash
.\.venv\Scripts\python.exe world_to_beamng.py --junction-id 12345
```

### Problem: Boundary-Kanten prüfen
Setze in `config.py`:
```python
HOLE_CHECK_ENABLED = True
BOUNDARY_EDGES_EXPORT = True
```
Exportierte Kanten liegen dann in separater Datei vor.

## 📝 Pipeline-Ablauf

Vereinfachter Überblick über die Verarbeitungsschritte:

1. **Höhendaten laden** → lokalen Offset setzen, BBOX berechnen
2. **OSM-Daten abrufen** → Straßen, Polygone, Junctions extrahieren
3. **VertexManager initialisieren** → zentrale Deduplizierung
4. **Straßen-Mesh generieren** → mit Böschungen (Slopes)
5. **Grid klassifizieren** → Terrain vs. Slopes
6. **Terrain-Mesh erzeugen** → aus klassifizierten Grid-Vertices
7. **Junction-Remesh** → lokale Delaunay pro Junction
8. **CCW-Normalisierung** → konsistente Face-Orientierung
9. **Stitching (optional)** → geschlossene Meshes erzwingen
10. **Face-Deduplication** → Duplikate entfernen
11. **OBJ-Export** → mit Materials und Debug-Layers
12. **Zeitmessung-Report** → automatisch per `timer.report()`

## 📄 Lizenz

Dieses Projekt ist unter der **MIT License** lizenziert – siehe [LICENSE](LICENSE) Datei für Details.

## 🤝 Beiträge

Beiträge sind willkommen! Bitte beachte:

1. **Fork** das Repository
2. **Erstelle einen Branch** (`git checkout -b feature/AmazingFeature`)
3. **Committe deine Änderungen** (`git commit -m 'Add AmazingFeature'`)
4. **Push** zum Branch (`git push origin feature/AmazingFeature`)
5. **Öffne einen Pull Request**

### Ideen für Beiträge
- [ ] Unit-Tests für einzelne Module
- [ ] Support für weitere UTM-Zonen (z.B. 31N, 33N)
- [ ] Erweiterte Visualisierung (Mesh-Viewer-Integration)
- [ ] Performance-Optimierungen (Parallelisierung)
- [ ] Support für weitere Datenquellen (SRTM, GEBCO)

## 📞 Support & Kontakt

- 📧 **Issues**: [GitHub Issues](https://github.com/yourusername/World-to-BeamNG/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/World-to-BeamNG/discussions)

## 🙏 Danksagungen

- **OpenStreetMap** und **Overpass API** für die Geodaten
- **BeamNG** für das fantastische Simulations-Game
- **Shapely**, **PyVista**, **NumPy** Community für großartige Geometrie-Tools

---

**Made with ❤️ for the BeamNG Community**

*Weitere technische Details: siehe Quellcode-Kommentare in `world_to_beamng/`*
