# Multi-Tile Verarbeitungs-Architektur

## Übersicht

Die Implementierung unterstützt Multi-Tile-Verarbeitung für große OSM-Regionen mit mehreren DGM1-Kacheln (je 2000m × 2000m).

## Komponenten

### 1. **Tile Scanner** (`world_to_beamng/utils/tile_scanner.py`)

Funktionen:
- `scan_lgl_tiles(dgm1_dir)` - Scannt DGM1-Verzeichnis für Kacheldateien
  - Erkennt Dateinamen: `dgm1_<easting>_<northing>.xyz.zip`
  - Gibt sortierte Kachelmetadaten zurück
  
- `compute_global_bbox(tiles)` - Berechnet globale Bounding Box
  - Gibt (min_x, max_x, min_y, max_y) in UTM-Koordinaten zurück
  
- `compute_global_center(tiles)` - Berechnet globalen Mittelpunkt
  - Wird für LOCAL_OFFSET verwendet

### 2. **Cache Management** (`world_to_beamng/io/cache.py`)

Neue Funktionen für Multi-Tile Hash-Registry:

- `load_height_hashes()` - Lädt `height_data_hash.txt` 
  - Format: `filename: hash_value` (eine pro Zeile)
  - Ermöglicht Cache-Invalidierung pro Datei
  
- `save_height_hashes(hashes)` - Speichert aktualisierte Hashes
  - Wird nach Verarbeitung aufgerufen
  
- `calculate_file_hash(filepath)` - Berechnet MD5-Hash einer Datei
  - 12-stelliger Hex-String

### 3. **Materials & Items Merge** (`world_to_beamng/io/materials_merge.py`)

Funktionen für Merge-Operationen im `add_new` Mode:

- `merge_materials_json(output_path, new_materials, mode='add_new')`
  - Addiert neue Material-Definitionen
  - Überschreibt NICHT bestehende Keys
  
- `merge_items_json(output_path, new_items, mode='add_new')`
  - Addiert neue Item-Definitionen
  - Überschreibt NICHT bestehende Keys
  
- `save_materials_json(output_path, materials)` - Speichert JSON
- `save_items_json(output_path, items)` - Speichert JSON

### 4. **Multi-Tile Orchestration** (`world_to_beamng/utils/multitile.py`)

Implementiert 3-Phasen-Architektur:

#### Phase 1: Pre-Scan & Initialisierung
```python
tiles, global_offset = phase1_multitile_init(dgm1_dir="data/DGM1")
```

Was diese Phase macht:
1. Scannt DGM1-Verzeichnis
2. Lädt bestehende Hashes aus `height_data_hash.txt`
3. Prüft welche Kacheln geändert haben
4. Löscht alte `main.materials.json` und `main.items.json`
5. Speichert aktualisierte Hashes
6. Gibt globalen Offset zurück

**Fallback**: Wenn keine DGM1-Dateien gefunden, wird `None` zurückgegeben → Single-Tile Mode

#### Phase 2: Pro-Tile Verarbeitung (Schleife)
```python
for tile in tiles:
    result = phase2_process_tile(tile, height_points, ...)
```

Was diese Phase macht (pro Kachel):
1. Lädt DGM1-Daten für die Kachel
2. Lädt OSM-Daten (mit Puffer)
3. Lädt LoD2-Gebäude (parallel `data/lod2_<x>_<y>.gml`)
4. Generiert Road/Terrain/Building-Meshes
5. Erstellt Materials & Items für die Kachel
6. Merged Materials/Items additiv (add_new Mode!)

#### Phase 3: Post-Merge & Finalisierung
```python
final_materials, final_items = phase3_multitile_finalize(beamng_dir)
```

Was diese Phase macht:
1. Lädt finale Materials & Items
2. Validiert Daten
3. Kann zusätzliche Post-Processing durchführen

## Integrations-Punkte

### In `world_to_beamng.py`

Die Phase-1-Funktion kann am Anfang der main() aufgerufen werden:

```python
from world_to_beamng.utils.multitile import phase1_multitile_init

# Nach dem Parsen von Command-Line-Args
tiles, global_offset = phase1_multitile_init(dgm1_dir="data/DGM1")

if tiles is None:
    # Single-Tile Mode - nutze existiende load_height_data()
    print("[i] Single-Tile Mode")
else:
    # Multi-Tile Mode - verarbeite pro Kachel
    print(f"[i] Multi-Tile Mode: {len(tiles)} Kacheln")
```

## Cache-Strategie

### `height_data_hash.txt` Format

```
dgm1_4658000_5394000.xyz.zip: abc123def456
dgm1_4660000_5394000.xyz.zip: xyz789abc123
```

- Eine Zeile pro DGM1-Datei
- Format: `filename: hash`
- Ermöglicht schnelle Invalidierung per Datei
- Wird bei Phase 1 aktualisiert

## Koordinaten-System

### Multi-Tile Benennung

Alle Tile-Namen verwenden **World-Koordinaten (UTM)**:

```
tile_4658000_5394000.dae          ← tile_x_tile_y = easting_northing
tile_4658000_5394000.mtl
terrain_4658000_5394000.dae
buildings_tile_4658000_5394000.dae
```

**NICHT** index-basiert wie früher (tile_0_1).

### LOCAL_OFFSET Handling

```python
# Globaler Offset (einmalig berechnet aus allen Tiles)
LOCAL_OFFSET = (center_x, center_y, z_min)

# Jede Kachel transformiert ihre Höhendaten mit diesem Offset
height_points[:, 0] -= center_x
height_points[:, 1] -= center_y
height_elevations -= z_min
```

So haben alle Kacheln ein gemeinsames lokales Koordinatensystem!

## Implementierungs-Schritte (für zukünftige Erweiterung)

1. ✅ **Tile Scanner** - Erkennt DGM1-Dateien
2. ✅ **Hash Registry** - Cache-Invalidierung pro Datei
3. ✅ **Materials/Items Merge** - Additive Merged
4. ✅ **Multi-Tile Orchestration** - 3-Phasen-Framework
5. ⧬ **Phase 2 Implementation** - Pro-Tile Verarbeitungslogik
   - Loader für per-Tile DGM1-Daten
   - Per-Tile OSM-Download (mit Buffer)
   - Per-Tile LoD2-Buildings
   - Per-Tile Aerial-Processing
6. ⧬ **Integration in main()** - Entscheidungslogik Single vs Multi-Tile

## Rückwärts-Kompatibilität

Die Implementierung ist vollständig rückwärts-kompatibel:

- Single-Tile Mode ist Standard (wenn keine DGM1-Dateien vorhanden)
- Alle neuen Funktionen sind optional
- Bestehender Code ändert sich nicht
- Bei `scan_lgl_tiles()` → `None` wird Single-Tile weitergemacht

## Test-Scripts

- `debug/test_tile_scan.py` - Testet Tile-Scanner
- `debug/test_multitile_phase1.py` - Testet Phase-1 Initialisierung
