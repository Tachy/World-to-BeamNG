# Material Templates Configuration

## Overview

The material templates are defined in `data/material_templates.json` and control the **structure and properties** of all materials in the BeamNG level.

The `MaterialManager` loads these templates automatically on initialization and uses them as a **base template** for new materials.

---

## Structure

### JSON Format

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

### Standard Fields

| Field | Type | Description |
|------|-----|---|
| `class` | string | BeamNG material class (always `"Material"`) |
| `version` | float | Collada/material format version (1.5 or 2) |
| `Stages` | array | List of rendering stages with textures/colors |
| `groundModelName` | string | Optional: terrain physics (grass, water, concrete, etc.) |
| `groundType` | string | Optional: surface type for vehicle physics |
| `materialTag0` | string | Optional: category tag |
| `materialTag1` | string | Optional: subcategory tag |
| `friction` | float | Optional: friction coefficient (0.1 = slippery, 1.0+ = normal) |
| `alpha` | float | Optional: transparency (0 = invisible, 1 = opaque) |
| `materialFactors` | string | Optional: UV tiling factor (e.g. "1 1 4.0 1" for a 4 m repeat) |

---

## Built-in Templates

Terrain materials are created in `terrain/terrain_materials.py` and road materials in
`OSMMapper.generate_materials_json_entry()`; neither needs a template.

### 1. **building_wall**
For building walls (LoD2).

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

**Usage:**
```python
materials.add_building_material(
    "lod2_wall_plaster_white",
    textures={"baseColorMap": "...", "normalMap": "...", "roughnessMap": "...", "useAnisotropic": True},
)
```

Building UVs are metric: walls carry a seamlessly tiled plaster texture (`facade/facade_mapper.py`), roofs
repeat every `config.ROOF_REPEAT_M` meters in the roof plane (`facade/roof_uv.py`). There is therefore no
`tiling_scale` anymore; `tiling_scale != 1.0` would only write a `materialFactors` and is not used for buildings.
Additional stage properties (e.g. `roughnessFactor`, `metallicFactor`) go through `stage_properties`.

---

### 2. **building_roof**
For building roofs (LoD2).

Identical to `building_wall`, but typically with:
- A different texture
- A different color

Used for `lod2_roof_red` (beaver-tail tiles), `lod2_roof_flat` (gravel on flat roofs) and `lod2_roof_edge`
(untextured sheet-metal edge around flat roofs). Textures of the procedural materials (plaster, window atlas, gravel)
are created in `facade/building_textures.py` (`ensure_building_textures()`), not in `osm_to_beamng.json`.

---

### 3. **horizon**
For the horizon layer (distant terrain).

```json
{
  "class": "Material",
  "version": 1.5,
  "Stages": [{"specularPower": 16, "pixelSpecular": true}]
}
```

Higher specularity (glossier) because it is seen from far away.

**Usage:**
```python
materials.add_horizon_material(texture_path)
```

---

## Custom Templates

### Adding a Template

Edit `data/material_templates.json` and add a new template:

```json
{
  "templates": {
    "existing_templates": {...},
    "water": {
      "description": "Water material for lakes and rivers",
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

Then use it:

```python
materials.add_material(
    "lake_001",
    template="water",
    Stages={"baseColorMap": "textures/water.dds"}
)
```

---

## Stage Fields (Rendering)

The `Stages` array controls rendering:

```json
{
  "Stages": [
    {
      "specularPower": 1,        // How "glossy" - higher = glossier
      "pixelSpecular": true,     // Pixel-based specular mapping
      "baseColorMap": "...",     // Main texture (color)
      "normalMap": "...",        // Normal map (height details)
      "roughnessMap": "...",     // Roughness map
      "ambientOcclusionMap": "...", // Shadows in creases
      "diffuseColor": [r, g, b, a]  // Fallback color if there is no texture
    }
  ]
}
```

---

## Examples

### New Vegetation Materials

```json
{
  "vegetation": {
    "description": "Grass and shrubs",
    "class": "Material",
    "version": 1.5,
    "Stages": [{"specularPower": 0.1, "pixelSpecular": true}],
    "groundModelName": "grass",
    "groundType": "dirt"
  },
  "vegetation_dense": {
    "description": "Dense vegetation (forest)",
    "class": "Material",
    "version": 1.5,
    "Stages": [{"specularPower": 0.05, "pixelSpecular": true}],
    "groundModelName": "grass",
    "groundType": "mud",
    "friction": 0.3
  }
}
```

### Custom Road Types

```json
{
  "unpaved_road": {
    "description": "Field track, unpaved",
    "class": "Material",
    "version": 2,
    "Stages": [{"specularPower": 0.5, "pixelSpecular": true}],
    "groundType": "dirt",
    "friction": 0.4,
    "note": "Use with custom OSM tags or manually"
  }
}
```

---

## Error Handling

`data/material_templates.json` is required. If it does not exist or is malformed, **MaterialManager** aborts the
export with an error instead of falling back to defaults:

- missing file: `FileNotFoundError` ("Material templates not found: <path> ...")
- invalid JSON: `ValueError` ("Error parsing <path>: ...")
- any other read error: `RuntimeError` ("Error loading <path>: ...")

Make sure the file is part of your checkout.

---

## Best Practices

1. **Template names** should be concise: `water`, `vegetation_dense`, `road_unpaved`
2. **Descriptions** are important for documentation
3. **Specularity values**:
   - `0-0.5`: Matte (soil, grass, concrete)
   - `1-2`: Normal (asphalt, walls)
   - `4-8`: Glossy (water, ice)
   - `16+`: Very glossy (horizon, glass)
4. **Friction values** for physics:
   - `0.05-0.1`: Slippery (ice, water)
   - `0.3-0.5`: Unpaved (soil, grass)
   - `0.7-1.0`: Normal (asphalt)
   - `1.0+`: Grippy (concrete, salvage)

---

## Buildings Section

The `buildings` section (top level in the JSON) contains configurations for LoD2 buildings:

```json
{
  "buildings": {
    "wall": {
      "description": "Building wall configuration",
      "template": "building_wall",
      "material_hints": {
        "groundType": "concrete",
        "materialTag0": "beamng",
        "materialTag1": "Building"
      }
    },
    "roof": {
      "description": "Building roof configuration",
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

### Buildings Fields

| Field | Description |
|------|---|
| `template` | Reference to the base template (`building_wall`, `building_roof`) |
| `material_hints.groundType` | Physics surface type (concrete, brick, etc.) |
| `material_hints.materialTag0/1` | Category tags for vehicle behavior |

### Integration in the Exporter

`BeamNGExporter._add_lod2_materials()` (`export/beamng_exporter.py`) creates the building materials
(names in `facade/material_names.py`):

| Material | Content |
|---|---|
| `lod2_wall_plaster_<color>` (6x) | seamless plaster, one albedo texture per color; normal and roughness textures shared |
| `lod2_windows` | Sprite atlas: windows (with/without transom, with shutters), doors, basement windows |
| `lod2_roof_red` | Beaver-tail tile texture from `osm_to_beamng.json` (`buildings.roof`) |
| `lod2_roof_flat` | Procedurally generated, tileable gravel texture (flat roofs) |
| `lod2_roof_edge` | Sheet-metal edge around flat roofs, untextured (`buildings.roof_edge`) |
| `lod2_roof_trim` | Fascia board and underside of the roof overhangs, untextured (`buildings.roof_trim`) |

Plaster colors and their frequency are defined in `facade/facade_styles.py` (`PLASTER_COLORS`, per mille): mostly white,
occasional beige tones, very rare red tones. The color per building is fixed (crc32 of the gml:id).

Walls are ONE polygon per wall (no cells in the plaster); windows are separate areas 3 cm in front of the wall
(`facade/facade_mapper.py`). Storeys are counted downward from the eave; any remainder below is a
raised basement with basement windows. Pitched roofs get a 60 cm eave overhang (horizontal) and 30 cm at the gable,
10 cm thick (`facade/roof_overhang.py`, values in `config.ROOF_*`).

Church towers (`facade/church_towers.py`): The church comes from OSM (`building=church/cathedral/chapel`,
`amenity=place_of_worship`; otherwise the LOD2 data has no function information), the tower walls from the geometry (walls that end far
above the median of the wall top edges, plus walls in OSM bell-tower polygons). Tower walls get no windows; the
tower wall facing away from the nave carries a tower clock (sprite `tower_clock` in the window atlas). Thresholds in
`config.CHURCH_*`.

The textures are stored as DDS in `art/shapes/textures/` and are only rewritten when colors, layout, or
generator change (hash in `building_textures.hash`).

```python
generated = ensure_building_textures()
materials.add_building_material(WALL_MATERIALS[0], textures={"baseColorMap": generated["plaster_color_white"], ...})
```

---

## Integration

The `MaterialManager` is initialized as a singleton:

```python
from world_to_beamng.managers.material_manager import MaterialManager

# First instance loads templates + buildings config
materials = MaterialManager.get_instance(beamng_dir=config.BEAMNG_DIR)

# Get all configurations
config = materials.get_templates()
buildings_config = config["buildings"]  # wall, roof configurations
material_templates = config["templates"]  # building_wall, building_roof, etc.

# Subsequent calls return the same instance
materials = MaterialManager.get_instance()  # No beamng_dir needed!

# For a new export: reset
MaterialManager.reset_instance()
materials = MaterialManager.get_instance(beamng_dir=new_dir)
```

---

## See Also

- [MATERIAL_MANAGER.md](MATERIAL_MANAGER.md) - MaterialManager API
- [data/osm_to_beamng.json](../data/osm_to_beamng.json) - OSM road properties
- [OSMMapper documentation](OSM_MAPPER.md)
