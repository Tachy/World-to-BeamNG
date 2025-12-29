[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

# 🗺️ World-to-BeamNG

Konvertiere OpenStreetMap-Straßen und digitale Höhenmodelle zu hochdetaillierten BeamNG.drive Gelände-Maps!

## 🎯 Beschreibung

**World-to-BeamNG** ist ein leistungsstarker Generator, der:

- ✅ **OpenStreetMap-Daten** abruft und Straßen extrahiert
- ✅ **Digitale Höhenmodelle** (DGM1) als Terrain-Basis nutzt
- ✅ **Realistische Straßen-Geometrie** mit Böschungen generiert
- ✅ **Hochoptimierte Meshes** mit STRtree-Spatial-Indexing erstellt
- ✅ **BeamNG.drive-kompatible OBJ-Dateien** exportiert

### 🌍 Anwendungsbeispiele

- Erstellung von **Custom-Maps** aus echten GPS-Koordinaten
- Umwandlung von **Gelände-Scans** zu spielbaren Strecken
- Automatische **Straßen-Mesh-Generierung** mit realistischen Höhenprofilen
- Integration von **realen Kartendaten** in Simulationen

## 🚀 Features

### Performance & Optimierung
- **3.5x schneller** durch R-Tree Spatial Indexing (STRtree)
- **121 Straßen/Sekunde** Verarbeitungsrate
- **Smart Caching** für Höhendaten und OSM-Abfragen
- **Speicher-optimiert** mit Chunking und Garbage Collection

### Robustheit
- **Fehlertoleranz** bei Overpass API mit Retry-Logik
- **Fallback-Endpoints** für OSM-Abfragen
- **Automatische Cache-Invalidierung** bei Datenänderungen
- **Validierung** aller Geometrien

### Flexibilität
- **Konfigurierbare Parameter** (Straßenbreite, Böschungswinkel, etc.)
- **Optional Multiprocessing** für noch schnellere Verarbeitung
- **Layer-basierter Export** (Terrain, Roads, Slopes separat)
- **Verschiedene Komprimierungsgrade** für Terrain

## 📋 Anforderungen

### Pakete
```bash
pip install requests numpy scipy pyproj pyvista shapely rtree
```

### Daten
- **Höhendaten**: XYZ oder ZIP-Dateien im `height-data/` Verzeichnis
  - Format: X Y Z pro Zeile
  - Koordinaten in UTM Zone 32N
  - Z-Werte in Metern über NN

## 🛠️ Installation

### 1. Repository clonen
```bash
git clone https://github.com/yourusername/World-to-BeamNG.git
cd World-to-BeamNG
```

### 2. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 3. Höhendaten vorbereiten
```
World-to-BeamNG/
├── height-data/
│   ├── dgm1_32_506_5644_1_de.xyz     ← Deine Höhendaten
│   ├── dgm1_32_507_5644_1_de.xyz
│   └── dgm1_32_508_5644_1_de.xyz
```

## 📖 Verwendung

### Basis-Verwendung
```bash
python world_to_beamng.py
```

Das Skript wird:
1. Höhendaten aus `height-data/` laden
2. BBOX automatisch aus den Daten berechnen
3. OSM-Straßen für diesen Bereich abrufen
4. Straßen-Mesh generieren
5. Mit Terrain kombinieren
6. `beamng.obj` + `beamng.mtl` exportieren

### Konfiguration

Bearbeite `world_to_beamng/config.py`:

```python
# Mesh-Parameter
ROAD_WIDTH = 7.0              # Straßenbreite in Metern
SLOPE_ANGLE = 45.0            # Böschungswinkel in Grad
GRID_SPACING = 10.0           # Terrain-Auflösung in Metern

# Terrain-Optimierung
TERRAIN_REDUCTION = 0.0       # PyVista Decimation (0.0-1.0)

# Multiprocessing (optional)
USE_MULTIPROCESSING = False   # Für mehr Geschwindigkeit
NUM_WORKERS = 8               # Anzahl der Worker-Prozesse
```

### Output

Das Skript generiert:
- **beamng.obj** - Unified Mesh mit allen Layers
- **beamng.mtl** - Material-Definitionen
  - `road_surface` (Asphalt)
  - `road_slope` (Böschung)
  - `terrain` (Gras/Natur)

## 📊 Performance

Typische Laufzeiten für ein 10×10 km Gebiet:

| Schritt | Zeit | Anteil |
|---------|------|--------|
| Höhendaten laden | 0.6s | 2% |
| OSM-Daten abrufen | 0.1s | 0.5% |
| Mesh-Generierung | 4.5s | 15% |
| Straßen-Geometrie | 5.4s | 18% |
| **Face-Overlap-Prüfung** | **16.2s** | **53%** |
| Terrain-Vereinfachung | 0.5s | 2% |
| OBJ-Export | 0.7s | 2% |
| **Gesamt** | **~30 Sekunden** | **100%** |

*Mit STRtree Optimization (3.5x schneller als Brute-Force)*

## 🏗️ Architektur

Die Anwendung ist modular aufgebaut für einfache Wartung und Erweiterung:

```
world_to_beamng/
├── terrain/          ← Höhendaten & Grid-Generierung
├── osm/              ← OpenStreetMap Downloader & Parser
├── geometry/         ← Koordinaten-Transformationen & Polygone
├── mesh/             ← Mesh-Generierung & Overlap-Detection
└── io/               ← Cache-Management & OBJ-Export
```

Siehe [REFACTORING.md](REFACTORING.md) für technische Details.

## 🐛 Debugging

### Problem: Höhendaten nicht gefunden
```bash
# Prüfe height-data Verzeichnis
ls -la height-data/
# XYZ-Dateien sollten vorhanden sein
```

### Problem: OSM-API Timeout
Das Skript versucht automatisch Fallback-Server:
```
Server 1: overpass-api.de
Server 2: overpass.kumi.systems  
Server 3: overpass.openstreetmap.ru
```

### Problem: Zu viel Speicher
Reduziere `GRID_SPACING` oder nutze lokale Cache-Dateien in `cache/`.

## 📝 Changelog

### v1.0.0 (Dezember 2025)
- ✨ Komplette Refactorierung zu modularer Architektur
- 🚀 STRtree Spatial-Index für 3.5x Speedup
- 🎨 Saubere API mit importierbaren Modulen
- 📦 Bessere Fehlerbehandlung und Logging
- 🧪 Grundlage für Unit-Tests

### v0.9.0 (Frühere Version)
- Monolithisches Script (funktionsfähig, aber schwer zu warten)

## 🤝 Beiträge

Beiträge sind willkommen! Bitte beachte:

1. **Fork** das Repository
2. **Erstelle einen Branch** (`git checkout -b feature/AmazingFeature`)
3. **Committe deine Änderungen** (`git commit -m 'Add AmazingFeature'`)
4. **Push** zum Branch (`git push origin feature/AmazingFeature`)
5. **Öffne einen Pull Request**

### Ideen für Beiträge
- [ ] Unit-Tests für einzelne Module
- [ ] Support für weitere Koordinaten-Systeme (z.B. UTM Zone 31N, 33N)
- [ ] Web-UI für Map-Auswahl
- [ ] Automatische Textur-Generierung basierend auf OSM-Tags
- [ ] Support für weitere Datenquellen (z.B. GEBCO, SRTM)

## 📄 Lizenz

Dieses Projekt ist unter der MIT License lizenziert - siehe [LICENSE](LICENSE) Datei für Details.

## 🙏 Danksagungen

- **OpenStreetMap** und **Overpass API** für die Geodaten
- **BeamNG.drive** für das fantastische Simulations-Spiel
- **GeoPy** Community für Koordinaten-Transformationen
- **Shapely** und **PyVista** für Geometrie-Processing

## 📞 Support & Kontakt

- 📧 Issues: [GitHub Issues](https://github.com/yourusername/World-to-BeamNG/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/World-to-BeamNG/discussions)
- 🐦 Twitter: [@yourusername](https://twitter.com/yourusername)

## 🗺️ Roadmap

### Phase 2 (Q1 2026)
- [ ] Web-UI für interaktive Map-Auswahl
- [ ] Batch-Processing für mehrere Regions
- [ ] Integration von Satellite-Imagery für Texturierung
- [ ] Performance-Monitoring Dashboard

### Phase 3 (Q2 2026)
- [ ] Real-time OSM-Updates
- [ ] Custom Material-Assignment basierend auf Road-Tags
- [ ] Automatische Traffic-Light & Sign-Platzierung
- [ ] VR-Preview Integration

## ⚙️ Technische Details

### Verwendete Technologien
- **NumPy** - Array-Operationen & Numerik
- **SciPy** - Interpolation & räumliche Algorithmen
- **Shapely** - Geometrie-Processing
- **RTREE/STRtree** - Spatial Indexing
- **PyProj** - Koordinaten-Transformationen
- **PyVista** - Mesh-Verarbeitung
- **Requests** - HTTP/API-Kommunikation

### Mathematische Grundlagen
- **PCA (Principal Component Analysis)** für Centerline-Berechnung
- **KDTree Queries** für Punkt-zu-Polygon Tests
- **R-Tree Spatial Indexing** für Geometry-Intersection
- **Linear Interpolation** für Höhen-Estimation

---

**Made with ❤️ for the BeamNG Community**

*Weitere Dokumentation: Siehe [REFACTORING.md](REFACTORING.md) für technische Details*
