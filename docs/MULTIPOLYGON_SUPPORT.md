# 🌲 Multipolygon-Support für Wälder - IMPLEMENTIERT

## Problem
Das OSM-Waldabdeckungs-Problem war teilweise durch **fehlende Multipolygon-Geometrien** verursacht:
- 237 einfache Forest-Ways (Polygone)
- **7 Forest-Multipolygons** (Relationen) - die bisher IGNORIERT wurden!
- Das entspricht **3% zusätzlicher Waldgeometrien**

Beispiel aus dem Cache: `type=multipolygon` Relation ID 19885340 mit `landuse=forest`

## Ursachen
1. **Overpass-Query**: Relations mit `out geom` wurden nicht rekursiv geladen
   - Multipolygon-Relations haben mehrere Member-Ways
   - Ohne `(>;)` fehlen die Member-Ways im Ergebnis
   
2. **Code**: `_extract_forests_from_osm()` verarbeitete nur einfache Ways
   - Multipolygon-Relations werden jetzt erkannt
   - Relations mit `type=multipolygon` werden verarbeitet

3. **Geometrie-Transformation**: Bei MultiPolygons wurde nur das größte Polygon genommen
   - Jetzt werden ALLE Polygone eines Multipolygons verarbeitet

## Implementierte Fixes

### 1. Overpass-Query verbessert (`osm/downloader.py`)
```python
# ALT: Relations ohne ihre Member-Ways
query = """[out:json];...; out geom;"""

# NEU: Rekursive Laden aller Member-Ways/Nodes
query = """[out:json];...; (>;); out geom;"""
```

**Effekt**: Multipolygon-Relations werden mit vollständiger Geometrie geladen

### 2. Forest-Extraktion erweitert (`forest/forest_normalizer.py`)
`_extract_forests_from_osm()` verarbeitet jetzt:
- ✓ Einfache Ways (wie vorher)
- ✓ Multipolygon-Relations mit `type=multipolygon`
- ✓ Detektiert Element-Typ (way vs. relation)

### 3. Multipolygon-Geometrie-Transformation (`forest/forest_normalizer.py`)
`_transform_forests_wgs84_to_local()` verarbeitet jetzt:
- ✓ Einfache Polygone (Polygon)
- ✓ Multipolygone (MultiPolygon) - **ALLE Komponenten werden verarbeitet**
- Nicht nur das größte, sondern JEDES Polygon des Multipolygons

## Technische Details

### Multipolygon-Struktur in OSM
```json
{
  "type": "relation",
  "id": 19885340,
  "tags": {
    "landuse": "forest",
    "leaf_type": "mixed",
    "type": "multipolygon"
  },
  "members": [
    {"type": "way", "ref": 1214295831, "role": "outer", "geometry": [...]},
    {"type": "way", "ref": 657572815, "role": "inner", "geometry": [...]}
  ]
}
```

### Verarbeitung
1. Überpass API: Relations + ihre Members recursive laden (`(>;)`)
2. Parser: type=multipolygon Relationen erkennen
3. Shapely: Multipolygon mit allen Komponenten erstellen
4. Normalisierung: Jede Komponente transformieren und clippen

## Test-Ergebnisse

Vorher:
- 237 Forest-Ways erkannt
- 7 Forest-Multipolygons **ignoriert**
- **Total: 237 Waldgeometrien**

Nachher (sobald neue Daten):
- 237 Forest-Ways
- **+ 7 Forest-Multipolygons** (mit allen ihren Komponenten)
- **Total: 244+ Waldgeometrien** (+3%)

## Auswirkungen auf Dense Forests

Das erklärt möglicherweise einige der "grünen Bereiche ohne Bäume":
- Große Waldgebiete sind oft als Multipolygons definiert
- Sie werden jetzt korrekt geometrisiert
- Mit neuen OSM-Cache-Daten sollten mehr Waldflächen erfasst werden

## Nächste Schritte

1. **Neuer OSM-Cache nötig**: Die alte Cache-Datei (`osm_all_56bf311a5f35.json`) wurde ohne `(>;)` geladen
   - Beim nächsten Export wird die neue Query verwendet
   - Neue Daten mit vollständiger Multipolygon-Geometrie

2. **Testen**: Prüfen ob dense forest coverage sich verbessert hat

3. **Raster-Daten**: Falls OSM-Vektoren weiterhin unzureichend sind
   - DEM/DOP20-basierte Walddetection
   - Separate Pipeline für rasterbasierte Waldextraktion

## Files Modified

- `world_to_beamng/osm/downloader.py`: Overpass-Query mit `(>;)`
- `world_to_beamng/forest/forest_normalizer.py`:
  - `_extract_forests_from_osm()`: Multipolygon-Support
  - `_transform_forests_wgs84_to_local()`: Alle Polygon-Komponenten verarbeiten
