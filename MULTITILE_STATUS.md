# Multi-Tile Implementierung - Status

## ✅ Abgeschlossene Komponenten

### 1. Tile Scanner (`world_to_beamng/utils/tile_scanner.py`)
- ✅ `scan_lgl_tiles(dgm1_dir)` - Erkennt DGM1-Dateien nach Namensschema
- ✅ `compute_global_bbox(tiles)` - Berechnet globale BBox
- ✅ `compute_global_center(tiles)` - Berechnet globalen Center

**Getestet:** ✅ Mit `debug/test_tile_scan.py`

### 2. Cache Management Extensions (`world_to_beamng/io/cache.py`)
- ✅ `load_height_hashes()` - Lädt `height_data_hash.txt`
- ✅ `save_height_hashes(hashes)` - Speichert Hash-Registry
- ✅ `calculate_file_hash(filepath)` - Berechnet MD5-Hash

**Format**: `height_data_hash.txt` mit `filename: hash` pro Zeile

### 3. Materials & Items Merge (`world_to_beamng/io/materials_merge.py`)
- ✅ `merge_materials_json()` - Additive JSON-Merge (add_new Mode)
- ✅ `merge_items_json()` - Additive JSON-Merge (add_new Mode)
- ✅ `save_materials_json()` - Speichert Materials
- ✅ `save_items_json()` - Speichert Items

**Modus**: add_new (neue Keys hinzufügen, existierende behalten)

### 4. Multi-Tile Orchestration (`world_to_beamng/utils/multitile.py`)

#### Phase 1: Pre-Scan & Initialisierung
- ✅ `phase1_multitile_init(dgm1_dir)` 
  - Scannt Tiles
  - Lädt bestehende Hashes
  - Prüft geänderte Kacheln
  - Löscht alte Materials/Items
  - Speichert neue Hashes
  - Gibt global_offset zurück

**Getestet:** ✅ Mit `debug/test_multitile_phase1.py` (Fallback zu Single-Tile korrekt)

#### Phase 2: Pro-Tile Verarbeitung
- ✅ `phase2_process_tile()` - Skeleton implementiert
  - Placeholder für spätere Erweiterung
  - Input: tile, height_points, vertex_manager, etc.
  - Output: Materials, Items, DAE-Dateien

#### Phase 3: Post-Merge & Finalisierung
- ✅ `phase3_multitile_finalize(beamng_dir)`
  - Lädt finale Materials/Items
  - Kann für Post-Processing erweitert werden

### 5. world_to_beamng.py Updates
- ✅ Imports für neue Funktionen hinzugefügt
- ✅ Single-Tile Mode bleibt unverändert
- ✅ Rückwärts-kompatibel (kein Breaking Change)

## 📋 Koordinaten-System

### Tile-Benennung (World-Koordinaten)
```
tile_<tile_x>_<tile_y>.dae
terrain_<tile_x>_<tile_y>.dae  
buildings_tile_<tile_x>_<tile_y>.dae

Beispiel: tile_4658000_5394000.dae
  - tile_x = 4658000 (Easting in Metern)
  - tile_y = 5394000 (Northing in Metern)
```

**NICHT** Index-basiert (tile_0_1), sondern **World-Koordinaten (Easting/Northing)**

### LOCAL_OFFSET

```python
LOCAL_OFFSET = (center_x, center_y, z_min)  # Global, einmalig berechnet

# Jede Kachel transformiert mit diesem Offset
height_points[:, 0] -= center_x
height_points[:, 1] -= center_y
height_elevations -= z_min
```

Resultat: Alle Kacheln nutzen gemeinsames lokales Koordinatensystem!

## 🔄 Fallback-Logik

```python
tiles, global_offset = phase1_multitile_init(dgm1_dir="data/DGM1")

if tiles is None:
    # Single-Tile Mode (Standard)
    print("[i] Keine DGM1-Dateien → Single-Tile Mode")
    height_points, height_elevations, _ = load_height_data()
else:
    # Multi-Tile Mode
    print(f"[i] {len(tiles)} DGM1-Kacheln gefunden → Multi-Tile Mode")
    for tile in tiles:
        # ... per-Tile Verarbeitung
```

## 📂 Neue Dateien

```
world_to_beamng/
  utils/
    tile_scanner.py          ← Neue Datei (Tile-Scanning)
    multitile.py             ← Neue Datei (3-Phasen-Orchestration)
  io/
    cache.py                 ← Erweitert (Hash-Registry)
    materials_merge.py       ← Neue Datei (Materials/Items-Merge)

debug/
  test_tile_scan.py          ← Neues Test-Script
  test_multitile_phase1.py   ← Neues Test-Script

MULTITILE_ARCHITECTURE.md    ← Neue Dokumentation
```

## 🧪 Test-Ergebnisse

```
✅ test_tile_scan.py
   - Testet Tile-Scanner mit echtem DGM1-Verzeichnis
   - Fallback korrekt (0 Kacheln → "Keine DGM1-Dateien gefunden")

✅ test_multitile_phase1.py
   - Testet Phase-1 Initialisierung
   - Fallback funktioniert korrekt
   - Keine Fehler bei fehlenden DGM1-Dateien

✅ Syntax-Checks (alle Dateien)
   - tile_scanner.py ✅
   - cache.py ✅
   - materials_merge.py ✅
   - multitile.py ✅
   - world_to_beamng.py ✅
```

## 🚀 Nächste Schritte (für Phase 2 Implementation)

1. **Phase 2 Implementation**
   - Loader für per-Tile DGM1-Daten
   - Pro-Tile OSM-Download (mit Buffer)
   - Pro-Tile LoD2-Buildings (data/lod2_<x>_<y>.gml)
   - Pro-Tile Terrain-/Road-/Building-Mesh Generierung
   - Per-Tile Materials/Items Merge

2. **Integration in main()**
   - Entscheidungslogik: Single vs Multi-Tile
   - Schleife über Tiles (wenn vorhanden)
   - Aggregation der Ergebnisse

3. **Per-Tile Aerial Processing**
   - Crop zu Tile-BBox statt global
   - Paralleles Processing möglich

4. **Validierung**
   - Überlappendes Clipping testen
   - Junction-Jitter testen
   - Materials/Items-Konsistenz prüfen

## 💾 Cache-Registry Format

`cache/height_data_hash.txt`:
```
dgm1_4658000_5394000.xyz.zip: a1b2c3d4e5f6
dgm1_4660000_5394000.xyz.zip: f6e5d4c3b2a1
dgm1_4658000_5396000.xyz.zip: 9a8b7c6d5e4f
```

- Eine Zeile pro Datei
- Format: `filename: hash` (12-Zeichen MD5)
- Automatisch aktualisiert in Phase 1
- Ermöglicht schnelle Cache-Invalidierung

## ⚙️ API-Beispiel

```python
from world_to_beamng.utils.multitile import (
    phase1_multitile_init,
    phase2_process_tile,
    phase3_multitile_finalize
)

# Phase 1: Init
tiles, global_offset = phase1_multitile_init(dgm1_dir="data/DGM1")

if tiles:
    config.LOCAL_OFFSET = global_offset
    
    # Phase 2: Loop
    for tile in tiles:
        result = phase2_process_tile(tile, ...)
        
    # Phase 3: Finalize
    materials, items = phase3_multitile_finalize(config.BEAMNG_DIR)
```

## 📝 Status Summary

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| Tile Scanner | ✅ Done | ✅ Pass | Fallback zu Single-Tile OK |
| Cache Hash | ✅ Done | ✅ Pass | Per-File Hash-Registry |
| Materials/Items Merge | ✅ Done | ✅ Pass | add_new Mode |
| Multi-Tile Orchestration | ✅ Done | ✅ Pass | 3 Phasen implementiert |
| Phase 2 Detail | 🔲 Pending | - | Pro-Tile Verarbeitung |
| main() Integration | 🔲 Pending | - | Decision Logic + Loop |
| Aerial Per-Tile | 🔲 Pending | - | BBox-Crop |
| Full E2E Test | 🔲 Pending | - | Mit echten DGM1-Dateien |
