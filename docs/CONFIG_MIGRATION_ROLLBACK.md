# Config-Migration Rollback: Zurück zur flachen Struktur

**Datum:** 9. Januar 2026  
**Status:** ✅ Abgeschlossen

## Übersicht

Alle neuen Workflows wurden von der OOP-Config-Struktur zurück zur alten, userfreundlichen flachen Config-Syntax migriert.

## Änderungen

### **1. Workflow-Klassen**

Alle Workflow-Klassen nutzen jetzt wieder **`config.XXXX`** statt `self.config.mesh.xxx`:

#### **terrain_workflow.py**
- ✅ `from .. import config` (statt `from ..core.config import Config`)
- ✅ `__init__` ohne `config`-Parameter
- ✅ `config.GRID_SPACING` (statt `self.config.mesh.grid_spacing`)
- ✅ `config.ROAD_CLIP_MARGIN` (statt `self.config.mesh.road_clip_margin`)
- ✅ `config.BEAMNG_DIR_TEXTURES` (statt `str(self.config.paths.textures_dir)`)
- ✅ `config.BEAMNG_DIR_SHAPES` (statt `str(self.config.paths.shapes_dir)`)
- ✅ `config.TILE_SIZE` (statt `legacy_config.TILE_SIZE`)
- ✅ `config.LEVEL_NAME` (statt `self.config.level_name`)

#### **building_workflow.py**
- ✅ `from .. import config`
- ✅ `__init__` ohne `config`-Parameter
- ✅ `config.BEAMNG_DIR_BUILDINGS` (statt `self.config.paths.buildings_dir`)
- ✅ `config.BEAMNG_DIR` (statt `str(self.config.paths.beamng_dir)`)

#### **horizon_workflow.py**
- ✅ `from .. import config`
- ✅ `__init__` ohne `config`-Parameter
- ✅ `config.PHASE5_ENABLED` (statt `legacy_config.PHASE5_ENABLED`)
- ✅ `config.DGM30_DATA_DIR` (statt `getattr(legacy_config, "DGM30_DATA_DIR", ...)`)
- ✅ `config.DOP300_DATA_DIR` (statt `getattr(legacy_config, "DOP300_DATA_DIR", ...)`)
- ✅ `config.BEAMNG_DIR` (statt `str(self.config.paths.beamng_dir)`)
- ✅ `config.LEVEL_NAME` (statt `level_name = getattr(legacy_config, ...)`)
- ✅ `config.BEAMNG_DIR_SHAPES` (statt `str(self.config.paths.shapes_dir / ...)`)

#### **tile_processor.py**
- ✅ `from .. import config`
- ✅ `__init__` ohne `config`-Parameter

#### **beamng_exporter.py**
- ✅ `from .. import config`
- ✅ `__init__()` ohne Parameter (kein `Config`-Objekt mehr nötig)
- ✅ Manuelle Verzeichniserstellung mit `os.makedirs(config.XXX, exist_ok=True)`
- ✅ `config.CACHE_DIR` (statt `config.paths.cache_dir`)
- ✅ `config.BEAMNG_DIR` (statt `str(config.paths.beamng_dir)`)
- ✅ `config.BEAMNG_DIR_SHAPES`, `config.BEAMNG_DIR_TEXTURES`, `config.BEAMNG_DIR_BUILDINGS`

### **2. Builder-Anpassungen**

- ✅ `RoadMeshBuilder()` ohne `config`-Parameter (nutzt intern `from .. import config`)

### **3. Syntax-Validierung**

Alle geänderten Dateien wurden erfolgreich auf Syntax-Fehler geprüft:
```powershell
python -m py_compile <alle_dateien>
✓ Alle Dateien sind syntaktisch korrekt!
```

## Verwendung

### **Alt (OOP-Config - nicht mehr benötigt):**
```python
from world_to_beamng.core.config import Config
from world_to_beamng.export import BeamNGExporter

config = Config(beamng_dir="C:/BeamNG/levels/MyLevel")
exporter = BeamNGExporter(config)
```

### **Neu (Flache Config - userfreundlich):**
```python
from world_to_beamng import config
from world_to_beamng.export import BeamNGExporter

# Config direkt anpassen
config.GRID_SPACING = 1.0
config.ROAD_CLIP_MARGIN = -30.0

# Exporter ohne Parameter
exporter = BeamNGExporter()
exporter.export_complete_level(tiles, global_offset)
```

## Vorteile

✅ **Userfreundlich:** `config.GRID_SPACING` statt `config.mesh.grid_spacing`  
✅ **Weniger Code:** Keine Config-Instanzen nötig  
✅ **Konsistent:** Gleiche Syntax wie in 95% der Codebase  
✅ **Einfacher:** Direkter Zugriff ohne Objekthierarchie  

## Hinweise

- Die OOP-Config (`world_to_beamng/core/config.py`) existiert weiterhin für zukünftige Erweiterungen
- Sie wird aktuell **nicht mehr verwendet**
- Alle Workflows nutzen die flache globale Config aus `world_to_beamng/config.py`

## Betroffene Dateien

```
world_to_beamng/
├── workflow/
│   ├── terrain_workflow.py     ✅ Migriert
│   ├── building_workflow.py    ✅ Migriert
│   ├── horizon_workflow.py     ✅ Migriert
│   └── tile_processor.py       ✅ Migriert
└── export/
    └── beamng_exporter.py      ✅ Migriert
```

## Status

**✅ Migration abgeschlossen**  
Alle neuen Workflows verwenden jetzt die alte, bewährte flache Config-Struktur.
