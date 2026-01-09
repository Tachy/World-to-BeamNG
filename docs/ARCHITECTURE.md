# Neue Architektur - Übersicht

## 🏗️ Struktur

Die Architektur wurde grundlegend überarbeitet für bessere Wartbarkeit und Übersichtlichkeit:

```
world_to_beamng/
├── core/                      # Kern-Komponenten
│   ├── config.py             # Config-Klasse (ersetzt altes config.py)
│   └── cache_manager.py      # Zentraler Cache-Manager
│
├── export/                    # Export-API
│   └── beamng_exporter.py    # Zentrale Fassade
│
├── workflow/                  # Workflow-Orchestrierung
│   ├── tile_processor.py     # Tile-Loading
│   ├── terrain_workflow.py   # Terrain-Export
│   ├── building_workflow.py  # LoD2-Export
│   └── horizon_workflow.py   # Horizon-Export
│
├── managers/                  # Manager (bereits vorhanden)
│   ├── material_manager.py
│   ├── item_manager.py
│   └── dae_exporter.py
│
├── io/                        # I/O-Operationen
├── mesh/                      # Mesh-Generierung
├── geometry/                  # Geometrie-Utilities
├── osm/                       # OSM-Verarbeitung
└── terrain/                   # Terrain-Utilities
```

## 🎯 Vorteile

### 1. **Klare Separation of Concerns**
- **core**: Zentrale Konfiguration & Cache
- **export**: High-Level API
- **workflow**: Orchestrierung der Verarbeitungsschritte
- **io**: Nur I/O-Operationen

### 2. **Einfachere API**

**Alt** (multitile.py):
```python
from world_to_beamng.utils.multitile import (
    phase1_multitile_init,
    phase2_process_tile,
    phase3_multitile_finalize,
    phase5_generate_horizon_layer
)

# Komplexer Workflow mit vielen Funktionen
tiles = phase1_multitile_init()
for tile in tiles:
    phase2_process_tile(tile, ...)
phase3_multitile_finalize(...)
phase5_generate_horizon_layer(...)
```

**Neu** (BeamNGExporter):
```python
from world_to_beamng.core import Config
from world_to_beamng.export import BeamNGExporter

config = Config(beamng_dir="...")
exporter = BeamNGExporter(config)

# Ein Aufruf für alles
stats = exporter.export_complete_level(tiles, global_offset)
```

### 3. **Bessere Testbarkeit**
- Alle Komponenten sind Klassen mit Dependency Injection
- Einfaches Mocking für Unit-Tests
- Klare Schnittstellen

### 4. **Typsicherheit**
- Config als Klasse mit typsicheren Attributen
- Keine Module-Level Globals mehr
- IDE-Autocomplete funktioniert besser

## 📖 Verwendung

### Basic Usage

```python
from world_to_beamng.core import Config
from world_to_beamng.export import BeamNGExporter
from world_to_beamng.utils.tile_scanner import scan_lgl_tiles, compute_global_center

# 1. Config erstellen
config = Config(
    beamng_dir="C:/BeamNG/levels/MyLevel",
    level_name="MyLevel"
)

# 2. Exporter initialisieren
exporter = BeamNGExporter(config)

# 3. Tiles scannen
tiles = scan_lgl_tiles("data/DGM1")
global_offset = compute_global_center(tiles)

# 4. Export
stats = exporter.export_complete_level(
    tiles=tiles,
    global_offset=(global_offset[0], global_offset[1]),
    include_buildings=True,
    include_horizon=True
)

print(f"Exportiert: {stats['tiles_processed']} Tiles")
```

### Einzelne Workflows nutzen

```python
# Nur Terrain
exporter.export_terrain_only(tiles, global_offset)

# Nur ein Tile
exporter.export_single_tile(tile, global_offset, tile_x=0, tile_y=0)

# Nur Horizon
exporter.horizon.generate_horizon(global_offset)

# Nur Buildings
buildings_data = exporter.buildings.cache_buildings(bbox, global_offset)
exporter.buildings.export_buildings(buildings_data["buildings"], 0, 0)
```

### Config anpassen

```python
config = Config(beamng_dir="...")

# Mesh-Parameter
config.mesh.grid_spacing = 1.0  # Feineres Grid
config.mesh.road_width = 10.0   # Breitere Straßen

# Workflow
config.workflow.debug_exports = True
config.workflow.debug_verbose = True

# API
config.api.opentopography_enabled = True
```

## 🔧 Migration

### Für bestehenden Code

Die neue Architektur ist **kompatibel** mit bestehendem Code:

```python
# Alt (funktioniert weiter)
from world_to_beamng import config
print(config.BEAMNG_DIR)

# Neu (empfohlen)
from world_to_beamng.core import Config
my_config = Config(beamng_dir="...")
print(my_config.paths.beamng_dir)
```

### Schrittweise Migration

1. **Phase 1**: Neue API parallel nutzen
2. **Phase 2**: Legacy-Code schrittweise ersetzen
3. **Phase 3**: Legacy-Attribute aus Config entfernen

## 🧪 Testing

```python
# Unit-Test-Beispiel
from world_to_beamng.core import Config, CacheManager
from world_to_beamng.workflow import TileProcessor

def test_tile_processor():
    config = Config(beamng_dir="/tmp/test")
    cache = CacheManager(config.paths.cache_dir)
    processor = TileProcessor(config, cache)
    
    # Test mit Mock-Daten
    result = processor.load_height_data(mock_tile)
    assert result is not None
```

## 📊 Vorher/Nachher

| Aspekt | Alt (multitile.py) | Neu (BeamNGExporter) |
|--------|-------------------|---------------------|
| **Zeilen Code** | ~1018 | ~300 (verteilt) |
| **Funktionen** | 9 globale | 0 (nur Klassen) |
| **Testbarkeit** | Schwer | Einfach |
| **API-Calls** | 4+ | 1 |
| **Dependency Injection** | Nein | Ja |
| **Typsicherheit** | Teilweise | Vollständig |

## 🚀 Performance

Keine Performance-Einbußen:
- Gleicher Code, nur besser organisiert
- Cache-Manager optimiert Cache-Zugriffe
- Lazy Loading wo möglich

## 📝 Nächste Schritte

1. ✅ Config-Klasse
2. ✅ CacheManager
3. ✅ Workflow-Module
4. ✅ BeamNGExporter-Fassade
5. ⏳ Legacy multitile.py als deprecated markieren
6. ⏳ Tests schreiben
7. ⏳ Dokumentation erweitern
