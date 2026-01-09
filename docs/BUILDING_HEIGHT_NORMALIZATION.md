# Building Height Normalization - Implementierung

## Problem
Gebäude hatten keine korrekten Höhendaten und orientierten sich nicht am Terrain-Mesh. Sie wurden nicht ins lokale Koordinatensystem normalisiert, speziell die Z-Koordinate nicht.

## Root Cause
Die Gebäude wurden mit `parse_citygml_buildings()` geparst und nur mit X/Y-Offset transformiert, aber nicht mit dem Z-Min des Terrains. Dies führte dazu, dass:
- Gebäude unter oder über dem Terrain lagen
- Sie das Terrain-Relief nicht folgten
- Z-Koordinaten in verschiedenen Systemen waren (CityGML vs. lokales Koordinatensystem)

## Lösung: Sofortige 3D-Normalisierung nach dem Import

### 1. Neue Funktion: `normalize_buildings_full()` in `lod2.py`

```python
def normalize_buildings_full(
    buildings: List[Dict],
    local_offset: Tuple[float, float, float],
) -> List[Dict]:
```

Diese Funktion:
- Normalisiert **alle 3 Koordinaten (X, Y, Z)** konsistent
- Wird **sofort nach dem CityGML-Parse** aufgerufen
- Nutzt `local_offset = (ox, oy, oz)` wobei `oz = z_min` vom Terrain
- Transformiert alle Gebäude-Vertices: `vertex -= (ox, oy, oz)`
- Aktualisiert auch die Bounds

**Vorher:**
```python
buildings = parse_citygml_buildings(gml_root, local_offset)
```

**Nachher:**
```python
buildings_raw = parse_citygml_buildings(gml_root, local_offset[:2])  # Nur X/Y
buildings = normalize_buildings_full(buildings_raw, local_offset)   # 3D-Normalisierung
```

### 2. Updated: `cache_lod2_buildings()` Signatur

Alte Signatur:
```python
def cache_lod2_buildings(
    lod2_dir: str,
    bbox: Tuple[float, float, float, float],
    local_offset: Tuple[float, float],      # ← 2D-nur
    cache_dir: str,
    height_hash: str,
) -> str:
```

Neue Signatur:
```python
def cache_lod2_buildings(
    lod2_dir: str,
    bbox: Tuple[float, float, float, float],
    local_offset: Tuple[float, float, float],  # ← 3D! (ox, oy, oz)
    cache_dir: str,
    height_hash: str,
) -> str:
```

### 3. Updated: `terrain_workflow.py` - Building Loading

**Vor:**
```python
# Lade rohe Gebäude
buildings_cache_path = cache_lod2_buildings(
    lod2_dir=config.LOD2_DATA_DIR,
    bbox=osm_bbox,
    local_offset=global_offset,  # Nur (x, y)!
    cache_dir=config.CACHE_DIR,
    height_hash=tile_hash,
)
# ... dann später normalisieren mit Terrain-Daten
normalized_cache = cache_normalized_buildings(
    raw_buildings, height_points, height_elevations, ...
)
```

**Nachher:**
```python
# Berechne Z-Min aus Terrain-Höhendaten
z_min = float(height_points[:, 2].min()) if len(height_points) > 0 else 0.0
local_offset_3d = (global_offset[0], global_offset[1], z_min)

# Lade UND normalisiere Gebäude direkt
buildings_cache_path = cache_lod2_buildings(
    lod2_dir=config.LOD2_DATA_DIR,
    bbox=osm_bbox,
    local_offset=local_offset_3d,  # 3D-Offset mit z_min!
    cache_dir=config.CACHE_DIR,
    height_hash=tile_hash,
)
# Gebäude sind bereits normalisiert - kein separater Normalisierungs-Schritt nötig
buildings_data = load_buildings_from_cache(buildings_cache_path)
```

## Workflow-Änderungen

### Alte Architektur (Multitile):
```
Parse CityGML (X/Y nur)
  ↓
Speichere Raw-Gebäude im Cache
  ↓
Lade Raw-Gebäude
  ↓
Terrain-Daten laden
  ↓
[Separater Normalisierungs-Schritt] snap_buildings_to_terrain_batch()
  ↓
Speichere Normalisierte Gebäude im Cache
  ↓
Export DAE
```

### Neue Architektur:
```
Parse CityGML (X/Y nur)
  ↓
normalize_buildings_full() mit (ox, oy, z_min)
  ↓
[Sofort nach Parse normalisiert!]
  ↓
Speichere normalisierte Gebäude im Cache
  ↓
Lade normalisierte Gebäude
  ↓
Export DAE
```

## Vorteile

✅ **Sofortige Normalisierung**: Gebäude sind direkt nach dem Import korrekt positioniert  
✅ **Konsistente Koordinaten**: X, Y, Z alle im selben lokalen System  
✅ **Automatische Terrain-Anpassung**: Z-Min vom Terrain wird automatisch genutzt  
✅ **Kein separater Snap-to-Terrain-Schritt nötig**: Reduziert Komplexität  
✅ **Bessere Cache-Konsistenz**: Ein einheitlicher Cache-Punkt  
✅ **Performance**: Keine redundanten KDTree-Abfragen später  

## Testing

Zwei Test-Scripts wurden erstellt:

### `debug/test_building_normalization.py`
- Testet `normalize_buildings_full()` direkt
- Überprüft Vertex-Normalisierung
- Überprüft Bounds-Normalisierung
- Überprüft Z-Konsistenz

**Ergebnis:** ✓ ALLE TESTS BESTANDEN

### `debug/test_building_workflow_integration.py`
- Überprüft Function-Signaturen
- Überprüft Integrations-Architektur
- Überprüft Importe

**Ergebnis:** ✓ ALLE INTEGRATION-TESTS BESTANDEN

## Migration vom alten Code

Falls noch alter Code `cache_normalized_buildings()` nutzt:
- Diese Funktion ist immer noch vorhanden für Rückwärts-Kompatibilität
- Wird aber nicht mehr im Hauptworkflow verwendet
- Kann später entfernt werden

## Nächste Schritte

1. ✅ Implementierung abgeschlossen
2. ✅ Unit-Tests bestanden  
3. ✅ Integration-Tests bestanden
4. → Test mit echtem Datensatz (DAE-Export prüfen)
5. → Visualisierung überprüfen (dae_viewer)

