# Multi-Tile System - Visuelle Übersicht

## Architektur-Diagramm

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORLD-TO-BEAMNG                              │
│                 Multi-Tile Verarbeitung                         │
└─────────────────────────────────────────────────────────────────┘

                          ┌──────────────────┐
                          │  Eingang:        │
                          │  world_to_beamng │
                          │  .py main()      │
                          └────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   PHASE 0: Entscheidung     │
                    │  (Single vs Multi-Tile)    │
                    │                            │
                    │ tiles = scan_lgl_tiles()   │
                    │ if tiles: Multi-Tile       │
                    │ else:     Single-Tile      │
                    └────┬──────────────┬────────┘
                         │              │
              ┌──────────▼──┐    ┌──────▼──────────┐
              │   Multi-    │    │  Single-Tile    │
              │   Tile      │    │  Mode (Today)   │
              │   Mode      │    │                 │
              └──────┬──────┘    └────────┬────────┘
                     │                    │
           ┌─────────▼──────┐   ┌─────────▼─────────┐
           │    PHASE 1     │   │  load_height_data │
           │ Pre-Scan Init  │   │  get_osm_data     │
           │                │   │  create_mesh      │
           │ • scan_tiles   │   │  export_dae       │
           │ • global_bbox  │   │  ...              │
           │ • load_hashes  │   └─────────┬─────────┘
           │ • calc_offset  │             │
           │ • save_hashes  │             │
           └────────┬───────┘             │
                    │                     │
        ┌──────────▼────────┐             │
        │   PHASE 2 Loop    │             │
        │                   │             │
        │ for tile in tiles:│             │
        │   ├─ load_dgm1    │             │
        │   ├─ get_osm      │             │
        │   ├─ load_lod2    │             │
        │   ├─ gen_mesh     │             │
        │   ├─ exp_dae      │             │
        │   └─ merge_json   │             │
        └────────┬──────────┘             │
                 │                        │
          ┌──────▼───────┐                │
          │   PHASE 3    │                │
          │ Finalisieren │                │
          │              │                │
          │ • save data  │                │
          │ • validate   │                │
          │ • cleanup    │                │
          └──────┬───────┘                │
                 │                        │
                 └────────────┬───────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Ausgang:         │
                    │  BeamNG-Dateien    │
                    │  (DAE/MTL/JSON)    │
                    └────────────────────┘
```

## Module & Komponenten

```
world_to_beamng/
│
├── utils/
│   ├── tile_scanner.py (NEU)
│   │   ├─ scan_lgl_tiles(dgm1_dir)
│   │   ├─ compute_global_bbox(tiles)
│   │   └─ compute_global_center(tiles)
│   │
│   └── multitile.py (NEU)
│       ├─ phase1_multitile_init(dgm1_dir)
│       ├─ phase2_process_tile(tile, ...)
│       └─ phase3_multitile_finalize(beamng_dir)
│
├── io/
│   ├── cache.py (ERWEITERT)
│   │   ├─ load_height_hashes()      (NEU)
│   │   ├─ save_height_hashes()      (NEU)
│   │   └─ calculate_file_hash()     (NEU)
│   │
│   └── materials_merge.py (NEU)
│       ├─ merge_materials_json()
│       ├─ merge_items_json()
│       ├─ save_materials_json()
│       └─ save_items_json()
│
└── world_to_beamng.py (UPDATED IMPORTS)
    ├─ Import tile_scanner
    ├─ Import cache extensions
    ├─ Import materials_merge
    └─ main()
```

## Datenfluss: Single-Tile Verarbeitung (heute)

```
data/DEM1/ (einzelne Datei)
    │
    └──> load_height_data()
        │
        ├──> get_osm_data(bbox)
        │
        ├──> create_mesh()
        │
        └──> export_dae/materials/items
             │
             └──> cache/main.dae
                  cache/main.materials.json
                  cache/main.items.json
```

## Datenfluss: Multi-Tile Verarbeitung (zukünftig)

```
data/DGM1/
├── dgm1_4658000_5394000.xyz.zip  ┐
├── dgm1_4660000_5394000.xyz.zip  │
├── dgm1_4658000_5396000.xyz.zip  │ (2000m × 2000m Kacheln)
└── dgm1_4660000_5396000.xyz.zip  ┘
    │
    └──> scan_lgl_tiles()
        │
        ├──> compute_global_bbox()   → BBox über alle Tiles
        ├──> compute_global_center() → LOCAL_OFFSET
        └──> load_height_hashes()    → Cache-Status
             │
             ├─┬─────────────────────────────┐
             │ │ TILE 1: 4658000_5394000      │
             │ │                              │
             │ ├─> load_dgm1(tile)            │
             │ ├─> get_osm_data(bbox_buffer)  │
             │ ├─> load_lod2_buildings()      │
             │ ├─> create_mesh()              │
             │ ├─> export_dae/mtl             │
             │ └─> merge_materials/items()    │
             │
             ├─┬─────────────────────────────┐
             │ │ TILE 2: 4660000_5394000      │
             │ │                              │
             │ ├─> load_dgm1(tile)            │
             │ ├─> get_osm_data(bbox_buffer)  │
             │ ├─> load_lod2_buildings()      │
             │ ├─> create_mesh()              │
             │ ├─> export_dae/mtl             │
             │ └─> merge_materials/items()    │
             │
             └─┬─────────────────────────────┐
               │ TILE 3, 4, ...               │
               │ (same pattern)               │
               │                              │
               └─> merge_materials/items()    │
                   │
                   └──> cache/main.materials.json (merged)
                        cache/main.items.json (merged)
                        terrain_4658000_5394000.dae
                        terrain_4660000_5394000.dae
                        buildings_tile_4658000_5394000.dae
                        buildings_tile_4660000_5394000.dae
                        ... (alle Tiles)
```

## Cache-Strategie

```
SINGLE-TILE (heute):
┌─────────────────────────────────┐
│ cache/                          │
├── height_data_hash.txt          │ Global Hash
│   "no_files"                    │
│                                 │
├── osm_all_abc123.json           │ Global OSM
├── elevations_abc123.json        │ Global Elevations
└── grid_v3_abc123_spacing.npz    │ Global Grid
```

```
MULTI-TILE (zukünftig):
┌─────────────────────────────────────┐
│ cache/                              │
├── height_data_hash.txt              │ Pro-File Hashes
│   "dgm1_4658000_5394000: a1b2c3" │
│   "dgm1_4660000_5394000: f6e5d4" │
│   "dgm1_4658000_5396000: 9a8b7c" │
│                                    │
├── osm_all_abc123.json              │ TILE 1
├── elevations_abc123.json           │ (bei Änderung neu)
├── grid_v3_abc123_spacing.npz       │
│                                    │
├── osm_all_def456.json              │ TILE 2
├── elevations_def456.json           │ (separate Caches)
├── grid_v3_def456_spacing.npz       │
│                                    │
└── (weitere Tiles...)
```

## Koordinaten-System

```
UTM World-Koordinaten:
┌────────────────────────────────────┐
│ tile_4658000_5394000        (2000m)│
│ ├─ tile_x = 4658000 (Easting)      │
│ └─ tile_y = 5394000 (Northing)     │
└────────────────────────────────────┘

        ↓ Globaler LOCAL_OFFSET
        (center_x, center_y, z_min)

Lokales Koordinaten-System:
┌────────────────────────────────────┐
│ Alle Tiles im gemeinsamen           │
│ lokalen XY-System                   │
│                                    │
│ • Zentrale Vertex-Verwaltung       │
│ • Unified Mesh                     │
│ • Shared Materials/Items           │
└────────────────────────────────────┘
```

## Fehlerbehandlung

```
┌─────────────────────────────────┐
│  phase1_multitile_init()        │
│                                 │
│  dgm1_dir nicht vorhanden?      │
│  └─> Warning + return None      │
│                                 │
│  Keine DGM1-Dateien?            │
│  └─> Warning + return None      │
│                                 │
│  Hash-Berechnung fehlgeschlagen?│
│  └─> Tile als "geändert"        │
│                                 │
│  ─────────────────────────────  │
│  Fallback zu Single-Tile Mode   │
│  ✅ (transparent & sicher)      │
└─────────────────────────────────┘
```

## Test-Struktur

```
debug/
├── test_tile_scan.py
│   └─ Testet: scan_lgl_tiles()
│   └─ Status: ✅ PASS
│
├── test_multitile_phase1.py
│   └─ Testet: phase1_multitile_init()
│   └─ Status: ✅ PASS (Fallback OK)
│
└── test_multitile_validation.py
    ├─ Testet: Alle Module + Funktionen
    ├─ Testet: Imports
    ├─ Testet: Funktionalität
    ├─ Testet: add_new Merge-Mode
    └─ Status: ✅ ALL PASS (10/10 Tests)
```

## Status-Zusammenfassung

```
✅ PHASE 1: Pre-Scan & Init      - IMPLEMENTIERT
   ├─ Tile-Scanner             - ✅ DONE
   ├─ Hash-Registry             - ✅ DONE
   ├─ Materials/Items Merge     - ✅ DONE
   └─ Fallback-Logik            - ✅ DONE

🔲 PHASE 2: Pro-Tile Loop        - GEPLANT
   ├─ Per-Tile DGM1-Loader      - TODO
   ├─ Per-Tile OSM-Download     - TODO
   ├─ Per-Tile LoD2-Processing  - TODO
   ├─ Per-Tile Mesh-Generation  - TODO
   └─ Per-Tile Material/Item    - TODO

🔲 PHASE 3: Post-Merge           - BASIC DONE
   ├─ Finalisierung             - ✅ SKELETON
   ├─ Validierung               - TODO
   └─ Cleanup                   - TODO

📄 Dokumentation                - ✅ DONE
   ├─ MULTITILE_ARCHITECTURE.md - ✅ DONE
   └─ MULTITILE_STATUS.md       - ✅ DONE
```

## Nächste Schritte

1. **Phase 2 Implementation**
   - Refaktoriere `load_height_data()` → `load_height_data_for_tile(tile)`
   - Wrap `get_osm_data()` für Per-Tile BBox mit Buffer
   - Refaktoriere LoD2-Loading für parallele Files
   - Implementiere Per-Tile Aerial-Processing

2. **Integration in main()**
   - Entscheidungslogik vor Schritt 1
   - Schleife nach Phase 1 Ergebnis
   - Materials/Items Merge in der Schleife

3. **Testing**
   - E2E Test mit echten DGM1-Dateien
   - Overlapping-Tests (OSM Buffer, Junctions)
   - Materials/Items-Konsistenz-Checks

4. **Optimierung**
   - Paralleles Tile-Processing (optional)
   - Memory-Optimierung für große Regionen
   - Performance-Profiling
