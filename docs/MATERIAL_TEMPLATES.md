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

### 1. **terrain**
Für Terrain-Tiles mit Texturen.

```json
{
  "class": "Material",
  "version": 2,
  "Stages": [{"specularPower": 1, "pixelSpecular": true}],
  "groundModelName": "grass"
}
```

**Nutzung:**
```python
materials.add_terrain_material(tile_x, tile_y, texture_path)
```

---

### 2. **road**
Für OSM-Straßen (residential, primary, motorway, etc.).

```json
{
  "class": "Material",
  "version": 2,
  "Stages": [{"specularPower": 1, "pixelSpecular": true}]
}
```

**Nutzung:**
```python
road_props = config.OSM_MAPPER.get_road_properties(osm_tags)
materials.add_road_material(road_name, road_props)
```

**OSM-Mapper fügt hinzu:**
- `friction` (aus data/osm_to_beamng.json)
- `groundType` (ASPHALT, DIRT, etc.)
- `color` oder `baseColorMap` (falls Texturen vorhanden)

---

### 3. **building_wall**
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
    "lod2_wall_white",
    color=[0.9, 0.9, 0.9, 1.0],
    textures={"baseColorMap": "..."},
    tiling_scale=4.0  # 4m Wiederholung
)
```

---

### 4. **building_roof**
Für Gebäude-Dächer (LoD2).

Identisch mit `building_wall`, aber typischerweise mit:
- Andere Textur
- Andere Farbe
- `tiling_scale=2.0` statt 4.0 (feiner Detail)

---

### 5. **horizon**
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
      "tiling_scale": 4.0,
      "uv_mode": "wall",
      "color_extraction": {
        "method": "citygml_appearance",
        "fallback_color": [0.9, 0.9, 0.9, 1.0]
      },
      "material_hints": {
        "groundType": "concrete",
        "materialTag0": "beamng",
        "materialTag1": "Building"
      }
    },
    "roof": {
      "description": "Gebäude-Dach Konfiguration",
      "template": "building_roof",
      "tiling_scale": 2.0,
      "uv_mode": "roof",
      "color_extraction": {
        "method": "citygml_appearance",
        "fallback_color": [0.6, 0.2, 0.1, 1.0]
      },
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
| `tiling_scale` | UV-Tiling-Faktor (4.0 = alle 4m Wiederholung) |
| `uv_mode` | UV-Mapping-Modus (`wall` oder `roof`) |
| `color_extraction.method` | Farb-Extraktions-Methode (`citygml_appearance` für CityGML) |
| `color_extraction.fallback_color` | Fallback RGBA falls CityGML keine Farbe hat |
| `material_hints.groundType` | Physik-Oberflächentyp (concrete, brick, etc.) |
| `material_hints.materialTag0/1` | Kategorie-Tags für Fahrzeugverhalten |

### Integration in lod2.py

Die Konfigurationen werden in `lod2.py` in `create_materials_json()` geladen und mit OSM-Properties gemergt:

```python
def create_materials_json(material_manager):
    # Hole alle Konfigurationen (templates + buildings section)
    config = material_manager.get_templates()
    buildings_config = config.get("buildings", {})
    
    # Wall-Material: Template + OSM-Properties
    wall_template = buildings_config.get("wall", {})
    wall_props = OSM_MAPPER.get_building_properties("wall")
    
    material_manager.add_building_material(
        wall_name,
        color=wall_props.get("diffuseColor"),  # Von OSM
        tiling_scale=wall_template.get("tiling_scale", 4.0),  # Von Template JSON
        groundType=wall_template.get("material_hints", {}).get("groundType"),
        # ...
    )
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
