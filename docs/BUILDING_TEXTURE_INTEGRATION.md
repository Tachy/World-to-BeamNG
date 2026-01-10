# Building Texture Integration - Implementierungsbericht

## Zusammenfassung

Die Gebäude-Texturen (Wände und Dächer) werden nun aus der `osm_to_beamng.json` Konfiguration ausgelesen, anstatt hardcodierte Werte zu verwenden. Dies ermöglicht eine flexible, datengetriebene Verwaltung der Texturen mit korrekten UV-Tiling-Faktoren.

## Änderungen

### 1. **OSMMapper** - Neue Methode `get_building_properties()`
**Datei:** `world_to_beamng/osm/osm_mapper.py`

```python
def get_building_properties(self, building_type="wall"):
    """
    Gibt ein Dictionary mit allen Gebäude-Material-Parametern zurück.
    
    Args:
        building_type: "wall" oder "roof"
    
    Returns:
        Dict mit:
        - internal_name: Material-Name (z.B. "lod2_wall_white")
        - groundModelName: Boden-Modell (z.B. "stone", "roof_tiles")
        - textures: Dict mit Textur-Pfaden (baseColorMap, normalMap, roughnessMap)
        - tiling_scale: Wiederholung in Metern:
            - Wände: 4.0m
            - Dächer: 2.0m
    """
```

**Funktionsweise:**
- Liest Konfiguration aus `osm_to_beamng.json` unter `buildings > {wall|roof}`
- Setzt automatisch die Tiling-Skala:
  - **Wände:** 4.0 Meter (alle 4m wiederholen)
  - **Dächer:** 2.0 Meter (alle 2m wiederholen)

### 2. **MaterialManager** - Erweiterte `add_building_material()`
**Datei:** `world_to_beamng/managers/material_manager.py`

**Neue Parameter:**
```python
def add_building_material(
    self,
    material_name: str,
    color: List[float] = None,           # Optional - Fallback für feste Farben
    textures: Dict[str, str] = None,     # NEU: Textur-Pfade
    tiling_scale: float = 1.0,           # NEU: UV-Wiederholung
    overwrite: bool = False,
    **kwargs
) -> str:
```

**Neue Funktionalität:**
- Akzeptiert Textur-Pfade statt nur Farben
- Setzt `materialFactors` für UV-Tiling (z.B. `"1 1 4.0 1"` für 4m Wiederholung)
- Fallback auf Farben, wenn keine Texturen vorhanden sind

### 3. **BeamNG Exporter** - Refaktoriert `_add_lod2_materials()`
**Datei:** `world_to_beamng/export/beamng_exporter.py`

**Vorher:** Hardcodierte Farben mit `config.LOD2_WALL_COLOR` und `config.LOD2_ROOF_COLOR`

**Nachher:** Liest Texturen und Tiling aus `OSM_MAPPER.get_building_properties()`
```python
wall_props = OSM_MAPPER.get_building_properties("wall")
self.materials.add_building_material(
    wall_props.get("internal_name", "lod2_wall_white"),
    textures=wall_props.get("textures"),
    tiling_scale=wall_props.get("tiling_scale", 4.0),
    groundType="STONE",
    materialTag0="beamng",
    materialTag1="Building",
)
```

### 4. **LoD2 Module** - Refaktoriert `create_materials_json()`
**Datei:** `world_to_beamng/io/lod2.py`

**Vorher:** Hardcodierte Farben ohne Texturen

**Nachher:** Liest Texturen aus OSM-Konfiguration mit derselben Logik wie BeamNG Exporter

## Konfigurationsdatei

**Datei:** `data/osm_to_beamng.json`

```json
{
    "buildings": {
        "wall": {
            "internal_name": "lod2_wall_white",
            "groundModelName": "stone",
            "textures": {
                "baseColorMap": "/assets/materials/building/generic/wall_plaster_white_01_d.dds",
                "normalMap": "/assets/materials/building/generic/wall_plaster_white_01_n.dds",
                "roughnessMap": "/assets/materials/building/generic/wall_plaster_white_01_r.dds"
            }
        },
        "roof": {
            "internal_name": "lod2_roof_red",
            "groundModelName": "roof_tiles",
            "textures": {
                "baseColorMap": "/assets/materials/building/generic/roof_clay_tiles_01_d.dds",
                "normalMap": "/assets/materials/building/generic/roof_clay_tiles_01_n.dds",
                "roughnessMap": "/assets/materials/building/generic/roof_clay_tiles_01_r.dds"
            }
        }
    }
}
```

## Tiling-Faktoren

Die Tiling-Skala wird als `materialFactors` in das Material geschrieben:

```
materialFactors: "1 1 <tiling_scale> 1"
```

**Beispiele:**
- **Wände (4m):** `materialFactors: "1 1 4.0 1"` → Textur wiederholt sich alle 4 Meter
- **Dächer (2m):** `materialFactors: "1 1 2.0 1"` → Textur wiederholt sich alle 2 Meter

## Test-Ergebnisse

Alle Tests erfolgreich (siehe `debug/test_building_texture_integration.py`):

✅ **TEST 1:** `OSMMapper.get_building_properties()`
- Wall: internal_name="lod2_wall_white", tiling_scale=4.0m
- Roof: internal_name="lod2_roof_red", tiling_scale=2.0m
- Alle Textur-Pfade korrekt

✅ **TEST 2:** `MaterialManager.add_building_material()` mit Texturen
- Wall-Material mit 3 Texturen + materialFactors (4.0)
- Roof-Material mit 3 Texturen + materialFactors (2.0)
- groundType korrekt gesetzt

✅ **TEST 3:** `osm_to_beamng.json` Struktur
- buildings Sektion vorhanden
- Wall und Roof Konfiguration vollständig
- Alle Textur-Pfade korrekt

## Vorteile dieser Implementierung

1. **Datengetrieben:** Texturen werden aus Config gelesen, nicht hardcodiert
2. **Erweiterbar:** Neue Gebäudetypen können leicht hinzugefügt werden
3. **Realistic Texturing:** Korrekte UV-Tiling-Faktoren für physikalisch korrekte Materialeigenschaften
4. **Fallback-Sicherheit:** Wenn keine Texturen, wird auf feste Farben zurückgegriffen
5. **Konsistent:** Gleiches Pattern wie bei Road-Materialien (bereits implementiert)

## Backwards Compatibility

- Die alten Konstanten `config.LOD2_WALL_COLOR` und `config.LOD2_ROOF_COLOR` werden als Fallback noch verwendet
- Bestehendes Code funktioniert weiterhin
- Neue Texturen überschreiben die Fallback-Farben

## Zukünftige Erweiterungen

Mögliche Verbesserungen:
1. Zusätzliche Gebäudetypen in `osm_to_beamng.json` (z.B. "damaged_wall", "metal_roof")
2. Dynamische Tiling-Faktoren pro Gebäudetyp (aktuell: fest auf 4m/2m)
3. Textur-Varianten basierend auf Gebäudealter oder Material-Tags aus OSM
