# Material Templates Konfiguration

## Übersicht

Die Material-Templates werden in `data/material_templates.json` definiert und steuern die **Struktur und Eigenschaften** aller Materialien im BeamNG-Level.

Der `MaterialManager` lädt diese Templates beim Initialisieren automatisch und nutzt sie als **Basisschablone** für neue Materialien.

---

## Struktur

### JSON-Format

```json
{
  "version": "1.0",
  "description": "...",
  "templates": {
    "template_name": {
      "class": "Material",
      "version": 2.0,
      "Stages": [{ ... }],
      "fieldName": "fieldValue",
      ...
    }
  }
}
```

### Standard-Felder

| Feld | Typ | Beschreibung |
|------|-----|---|
| `class` | string | BeamNG Material-Klasse (immer `"Material"`) |
| `version` | float | Collada/Material-Format-Version (1.5 oder 2) |
| `Stages` | array | Liste von Rendering-Stages mit Texturen/Farben |
| `groundModelName` | string | Optional: Terrain-Physik (grass, water, concrete, etc.) |
| `groundType` | string | Optional: Oberflächentyp für Fahrzeugphysik |
| `materialTag0` | string | Optional: Kategorie-Tag |
| `materialTag1` | string | Optional: Unter-Kategorie-Tag |
| `friction` | float | Optional: Reibungskoeffizient (0.1 = rutschig, 1.0+ = normal) |
| `alpha` | float | Optional: Transparenz (0 = unsichtbar, 1 = opak) |
| `materialFactors` | string | Optional: UV-Tiling-Faktor (z.B. "1 1 4.0 1" für 4m Wiederholung) |

---

## Eingebaute Templates

Terrain-Materialien entstehen in `terrain/terrain_materials.py`, Straßen-Materialien in
`OSMMapper.generate_materials_json_entry()`; beide brauchen kein Template.

### 1. **building_wall**
Für Gebäude-Wände (LoD2).

```json
{
  "class": "Material",
  "version": 1.5,
  "Stages": [{"specularPower": 1, "pixelSpecular": true}],
  "groundType": "concrete",
  "materialTag0": "beamng",
  "materialTag1": "Building"
}
```

**Nutzung:**
```python
materials.add_building_material(
    "lod2_wall_plaster_white",
    textures={"baseColorMap": "...", "normalMap": "...", "roughnessMap": "...", "useAnisotropic": True},
)
```

Die UVs der Gebäude sind metrisch: Wände tragen eine fugenlos gekachelte Putztextur (`facade/facade_mapper.py`), Dächer
wiederholen alle `config.ROOF_REPEAT_M` Meter in der Dachebene (`facade/roof_uv.py`). Es gibt deshalb keine
`tiling_scale` mehr; `tiling_scale != 1.0` würde nur ein `materialFactors` schreiben und wird für Gebäude nicht genutzt.
Zusätzliche Stage-Eigenschaften (z. B. `roughnessFactor`, `metallicFactor`) gehen über `stage_properties`.

---

### 2. **building_roof**
Für Gebäude-Dächer (LoD2).

Identisch mit `building_wall`, aber typischerweise mit:
- Andere Textur
- Andere Farbe

Genutzt für `lod2_roof_red` (Biberschwanz), `lod2_roof_flat` (Kies auf Flachdächern) und `lod2_roof_edge`
(untexturierter Blechrand um Flachdächer). Texturen der prozeduralen Materialien (Putz, Fenster-Atlas, Kies)
entstehen in `facade/building_textures.py` (`ensure_building_textures()`), nicht in `osm_to_beamng.json`.

---

### 3. **horizon**
Für Horizont-Layer (distant terrain).

```json
{
  "class": "Material",
  "version": 1.5,
  "Stages": [{"specularPower": 16, "pixelSpecular": true}]
}
```

Höhere Specularity (glänzender) weil aus der Ferne.

**Nutzung:**
```python
materials.add_horizon_material(texture_path)
```

---

## Benutzerdefinierte Templates

### Template hinzufügen

Bearbeite `data/material_templates.json` und füge ein neues Template hinzu:

```json
{
  "templates": {
    "existing_templates": {...},
    "water": {
      "description": "Wasser-Material für Seen und Flüsse",
      "class": "Material",
      "version": 2,
      "Stages": [{"specularPower": 4, "pixelSpecular": true}],
      "groundModelName": "water",
      "alpha": 0.3,
      "friction": 0.05
    }
  }
}
```

Dann nutzen:

```python
materials.add_material(
    "lake_001",
    template="water",
    Stages={"baseColorMap": "textures/water.dds"}
)
```

---

## Stage-Felder (Rendering)

Das `Stages` Array steuert das Rendering:

```json
{
  "Stages": [
    {
      "specularPower": 1,        // Wie "glänzend" - höher = glänzender
      "pixelSpecular": true,     // Pixel-basiertes Specular Mapping
      "baseColorMap": "...",     // Haupttextur (Farbe)
      "normalMap": "...",        // Normal Map (Höhen-Details)
      "roughnessMap": "...",     // Rauheits-Map
      "ambientOcclusionMap": "...", // Schatten in Falten
      "diffuseColor": [r, g, b, a]  // Fallback-Farbe wenn keine Textur
    }
  ]
}
```

---

## Beispiele

### Neue Vegetations-Materialien

```json
{
  "vegetation": {
    "description": "Gras und Sträucher",
    "class": "Material",
    "version": 1.5,
    "Stages": [{"specularPower": 0.1, "pixelSpecular": true}],
    "groundModelName": "grass",
    "groundType": "dirt"
  },
  "vegetation_dense": {
    "description": "Dichte Vegetation (Wald)",
    "class": "Material",
    "version": 1.5,
    "Stages": [{"specularPower": 0.05, "pixelSpecular": true}],
    "groundModelName": "grass",
    "groundType": "mud",
    "friction": 0.3
  }
}
```

### Benutzerdefinierte Straßen-Typen

```json
{
  "unpaved_road": {
    "description": "Feldweg, unbefestigt",
    "class": "Material",
    "version": 2,
    "Stages": [{"specularPower": 0.5, "pixelSpecular": true}],
    "groundType": "dirt",
    "friction": 0.4,
    "note": "Nutze mit Benutzerdefinierte OSM-Tags oder manuell"
  }
}
```

---

## Fehlerbehandlung

Falls `data/material_templates.json` nicht existiert oder fehlerhaft ist:

1. **MaterialManager** lädt die eingebauten Defaults
2. Keine Material-Funkion wird unterbrochen
3. Ein Warnung wird in die Konsole gedruckt:
   ```
   [i] data/material_templates.json nicht gefunden. Nutze eingebaute Template-Defaults
   ```

---

## Best Practices

1. **Template-Namen** sollten prägnant sein: `water`, `vegetation_dense`, `road_unpaved`
2. **Beschreibungen** sind wichtig für Dokumentation
3. **Spekularity-Werte**:
   - `0-0.5`: Stumpf (Erde, Gras, Beton)
   - `1-2`: Normal (Asphalt, Wände)
   - `4-8`: Glänzend (Wasser, Eis)
   - `16+`: Sehr glänzend (Horizont, Glas)
4. **Frictions** für Physics:
   - `0.05-0.1`: Rutschig (Eis, Wasser)
   - `0.3-0.5`: Unbefestigt (Erde, Gras)
   - `0.7-1.0`: Normal (Asphalt)
   - `1.0+`: Griffig (Beton, Bergung)

---

## Buildings Sektion

Die `buildings` Sektion (Top-Level in JSON) enthält Konfigurationen für LoD2-Gebäude:

```json
{
  "buildings": {
    "wall": {
      "description": "Gebäude-Wand Konfiguration",
      "template": "building_wall",
      "material_hints": {
        "groundType": "concrete",
        "materialTag0": "beamng",
        "materialTag1": "Building"
      }
    },
    "roof": {
      "description": "Gebäude-Dach Konfiguration",
      "template": "building_roof",
      "material_hints": {
        "groundType": "concrete",
        "materialTag0": "beamng",
        "materialTag1": "Building"
      }
    }
  }
}
```

### Buildings-Felder

| Feld | Beschreibung |
|------|---|
| `template` | Verweis auf Basis-Template (`building_wall`, `building_roof`) |
| `material_hints.groundType` | Physik-Oberflächentyp (concrete, brick, etc.) |
| `material_hints.materialTag0/1` | Kategorie-Tags für Fahrzeugverhalten |

### Integration im Exporter

`BeamNGExporter._add_lod2_materials()` (`export/beamng_exporter.py`) legt die Gebäude-Materialien an
(Namen in `facade/material_names.py`):

| Material | Inhalt |
|---|---|
| `lod2_wall_plaster_<Farbe>` (6x) | fugenloser Putz, je Farbe eine Albedo-Textur; Normal- und Roughness-Textur gemeinsam |
| `lod2_windows` | Sprite-Atlas: Fenster (mit/ohne Kämpfer, mit Fensterläden), Türen, Kellerfenster |
| `lod2_roof_red` | Biberschwanz-Stocktextur aus `osm_to_beamng.json` (`buildings.roof`) |
| `lod2_roof_flat` | prozedural erzeugte, kachelbare Kiestextur (Flachdächer) |
| `lod2_roof_edge` | Blechrand um Flachdächer, untexturiert (`buildings.roof_edge`) |
| `lod2_roof_trim` | Stirnbrett und Untersicht der Dachüberstände, untexturiert (`buildings.roof_trim`) |

Putzfarben und ihre Häufigkeit stehen in `facade/facade_styles.py` (`PLASTER_COLORS`, Promille): vorwiegend weiß,
vereinzelt Beigetöne, ganz vereinzelt Rottöne. Die Farbe je Gebäude ist fest (crc32 der gml:id).

Wände sind EIN Polygon je Wand (keine Zellen im Putz); Fenster sind eigene Flächen 3 cm vor der Wand
(`facade/facade_mapper.py`). Die Geschosse werden von der Traufe nach unten gezählt; ein Rest darunter ist ein
erhöhter Keller mit Kellerfenstern. Schrägdächer bekommen 60 cm Traufüberstand (waagerecht) und 30 cm am Giebel,
10 cm dick (`facade/roof_overhang.py`, Werte in `config.ROOF_*`).

Kirchtürme (`facade/church_towers.py`): Die Kirche kommt aus OSM (`building=church/cathedral/chapel`,
`amenity=place_of_worship`, sonst kennen die LOD2-Daten keine Funktion), die Turmwände aus der Geometrie (Wände, die weit
über dem Median der Wandoberkanten enden, dazu Wände in OSM-Glockenturm-Polygonen). Turmwände bekommen keine Fenster; die
Turmwand, die vom Kirchenschiff wegzeigt, trägt eine Turmuhr (Sprite `tower_clock` im Fenster-Atlas). Schwellen in
`config.CHURCH_*`.

Die Texturen liegen als DDS in `art/shapes/textures/` und werden nur neu geschrieben, wenn sich Farben, Layout oder
Generator ändern (Hash in `building_textures.hash`).

```python
generated = ensure_building_textures()
materials.add_building_material(WALL_MATERIALS[0], textures={"baseColorMap": generated["plaster_color_white"], ...})
```

---

## Integration

Der `MaterialManager` wird als Singleton initialisiert:

```python
from world_to_beamng.managers.material_manager import MaterialManager

# Erste Instanz lädt Templates + buildings Config
materials = MaterialManager.get_instance(beamng_dir=config.BEAMNG_DIR)

# Hole alle Konfigurationen
config = materials.get_templates()
buildings_config = config["buildings"]  # wall, roof Konfigurationen
material_templates = config["templates"]  # building_wall, building_roof, etc.

# Nachfolgende Aufrufe geben die gleiche Instanz
materials = MaterialManager.get_instance()  # Kein beamng_dir nötig!

# Für neuen Export: Reset
MaterialManager.reset_instance()
materials = MaterialManager.get_instance(beamng_dir=new_dir)
```

---

## Siehe auch

- [MATERIAL_MANAGER.md](MATERIAL_MANAGER.md) - MaterialManager API
- [data/osm_to_beamng.json](../data/osm_to_beamng.json) - OSM Road Properties
- [OSMMapper-Dokumentation](OSM_MAPPER.md)
