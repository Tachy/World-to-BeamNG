# Terrain-Heightmap-Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Terrain-Export von custom-generiertem, mit Straßen verschweißtem DAE-Mesh auf natives BeamNG `.terrain`-Heightmap umstellen — Straßen bleiben unverändertes Custom-Mesh, werden aber nicht mehr topologisch ins Terrain integriert, sondern liegen darauf, mit dem Terrain-Raster entlang der Straßenkorridore knapp abgesenkt ("eingebettet").

**Architecture:** Neue, kleine Terrain-Module (`terrain/heightmap.py`, `terrain/road_embedding.py`, `terrain/ter_writer.py`, `terrain/terrain_materials.py`) ersetzen den `TerrainMeshBuilder`+Stitching-Teil von `TerrainWorkflow.process_tile()`/`export_tile()`. Straßen-Mesh-Generierung (`RoadMeshBuilder`, `mesh/road_mesh.py`) bleibt unverändert; `GENERATE_SLOPES` wird reaktiviert. OSM-Landnutzung wird über den bereits vorhandenen, bisher ungenutzten `landuse_mappings`-Block in `data/osm_to_beamng.json` in die Terrain-Layer-Map gerastert.

**Tech Stack:** Python, numpy, `rasterio.features.rasterize` (bereits Dependency), `affine` (Dependency von rasterio, bereits transitiv installiert), shapely.

**Spec:** `docs/superpowers/specs/2026-09-17-terrain-heightmap-migration-design.md`

## Global Constraints

- Workflow-Aufruf bleibt `python world_to_beamng.py`, keine neuen CLI-Argumente, keine neuen Eingabedaten-Anforderungen.
- `TERRAIN_SQUARE_SIZE` (neuer Config-Wert) = `GRID_SPACING` (aktuell 2.0m) — Auflösungs-Parität, keine automatische Verfeinerung.
- Straßen-Querschnitt/Böschungs-Geometrie in `mesh/road_mesh.py` wird NICHT verändert — nur `config.GENERATE_SLOPES` wird von `False` auf `True` gesetzt.
- Rechteckige Level-Flächen sind ausreichend (keine Hole-Map-Unterstützung für nicht-rechteckige Level-Umrisse in diesem Plan).
- `.terrain`-Binärformat (Version 9) ist gegen eine echte BeamNG-`.ter`-Datei (`GridMap.ter` aus der Installation) byte-genau verifiziert — siehe Task 3.
- `TerrainBlock`- und `TerrainMaterial`-JSON-Schema sind gegen BeamNGs eigenes `template`-Level verifiziert — siehe Task 3 und Task 7.
- Aktive Landnutzungs-Kategorien für diesen Plan: **forest, meadow, farmland** (aus `landuse_mappings`) — deren `baseColorMap`/`normalMap`-Pfade in `data/osm_to_beamng.json` sind aktuell erfunden/nicht verifiziert und werden in Task 1 auf echte, geprüfte BeamNG-Content-Pfade korrigiert. `water`, `industrial`, `commercial`, `residential`, `vineyard`, `greenhouse_horticulture`, `orchard` bleiben im Code unberücksichtigt (Foto-Fallback), ihre Einträge in `landuse_mappings` bleiben unangetastet für spätere Erweiterung.

---

## Task 1: `landuse_mappings`-Texturpfade korrigieren

Die vorhandenen `baseColorMap`/`normalMap`-Pfade für `forest`, `meadow`, `farmland` in `data/osm_to_beamng.json` zeigen auf nicht-existente Dateien (z.B. `/assets/materials/terrains/forest_floor_01_d.dds` — Ordner "terrains" existiert nicht, echter Ordner heißt "terrain"). Verifiziert wurden folgende echten Pfade in `content/assets/materials/terrain.zip` der BeamNG-Installation:

- Wald: `assets/materials/terrain/forest/t_forest_ground/t_forest_ground_b.png` (+ `_nm.png`) — nur PNG-Quelle vorhanden, kein vorkompiliertes DDS.
- Wiese: `assets/materials/terrain/grass/t_grass_01/t_grass_01_b.png` (+ `_nm.png`) — nur PNG-Quelle.
- Acker: `assets/materials/terrain/soil/t_dirt_dry_grassy/t_dirt_dry_grassy_b.png` (+ `_nm.png`) — nur PNG-Quelle.

**Files:**
- Modify: `data/osm_to_beamng.json`

**Interfaces:**
- Produces: korrekte `landuse_mappings["forest"|"meadow"|"farmland"]["baseColorMap"|"normalMap"]`-Werte, die Task 8 (`terrain_materials.py`) direkt als Datei-Referenzen verwendet — Pfad-Konvention: level-lokales Format ohne führenden Slash, `levels/world_to_beamng/art/shapes/assets/materials/terrain/<category>/<subfolder>/<file>.png` (folgt der Konvention von `tools/vendor_shared_textures.py`, das dieselbe Ordnerstruktur unter `art/shapes/assets/materials/...` erwartet).

- [ ] **Step 1: `landuse_mappings`-Einträge korrigieren**

Öffne `data/osm_to_beamng.json` und ersetze in den Objekten `landuse_mappings.forest`, `landuse_mappings.meadow`, `landuse_mappings.farmland` die Felder `baseColorMap` und `normalMap`:

```json
"forest": {
  "splat_map": "A",
  "channel": 0,
  "priority": 10,
  "internal_name": "mat_forest",
  "groundModelName": "forest",
  "baseColorMap": "levels/world_to_beamng/art/shapes/assets/materials/terrain/forest/t_forest_ground/t_forest_ground_b.png",
  "normalMap": "levels/world_to_beamng/art/shapes/assets/materials/terrain/forest/t_forest_ground/t_forest_ground_nm.png",
  "groundCover": [
    {"type": "forest_fern", "probability": 0.4},
    {"type": "forest_dry_leaves", "probability": 0.8}
  ]
},
```

```json
"meadow": {
  "splat_map": "A",
  "channel": 1,
  "priority": 4,
  "internal_name": "mat_grass",
  "groundModelName": "grass",
  "baseColorMap": "levels/world_to_beamng/art/shapes/assets/materials/terrain/grass/t_grass_01/t_grass_01_b.png",
  "normalMap": "levels/world_to_beamng/art/shapes/assets/materials/terrain/grass/t_grass_01/t_grass_01_nm.png",
  "groundCover": [
    {"type": "grass_flowers", "probability": 0.5},
    {"type": "grass_long", "probability": 0.7}
  ]
},
```

```json
"farmland": {
  "splat_map": "A",
  "channel": 2,
  "priority": 5,
  "internal_name": "mat_dirt",
  "groundModelName": "dirt",
  "baseColorMap": "levels/world_to_beamng/art/shapes/assets/materials/terrain/soil/t_dirt_dry_grassy/t_dirt_dry_grassy_b.png",
  "normalMap": "levels/world_to_beamng/art/shapes/assets/materials/terrain/soil/t_dirt_dry_grassy/t_dirt_dry_grassy_nm.png",
  "groundCover": [
    {"type": "crop_stubble", "probability": 0.6}
  ]
},
```

- [ ] **Step 2: Prüfen, dass das JSON valide bleibt**

Run: `python -c "import json; json.load(open('data/osm_to_beamng.json', encoding='utf-8'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add data/osm_to_beamng.json
git commit -m "fix: Correct landuse_mappings terrain texture paths to verified BeamNG content"
```

---

## Task 2: Config-Anpassungen

**Files:**
- Modify: `world_to_beamng/config.py`

**Interfaces:**
- Produces: `config.ROAD_EMBED_MARGIN` (float), `config.TERRAIN_SQUARE_SIZE` (float), `config.TERRAIN_MAX_HEIGHT_BUFFER` (float) — von Task 5/6/9 konsumiert. `config.GENERATE_SLOPES` bleibt der existierende Name, nur der Wert ändert sich.

- [ ] **Step 1: `GENERATE_SLOPES` aktivieren und neue Terrain-Parameter ergänzen**

In `world_to_beamng/config.py`, ersetze:

```python
# Böschungs-Generierung (vorübergehend deaktiviert bis Remeshing stabil)
GENERATE_SLOPES = False
```

durch:

```python
# Böschungs-Generierung: war wegen Terrain-Stitching-Instabilität deaktiviert;
# seit der Umstellung auf natives .terrain-Heightmap (siehe
# docs/superpowers/specs/2026-09-17-terrain-heightmap-migration-design.md)
# entfällt das Stitching komplett, Böschungen sind jetzt immer aktiv.
GENERATE_SLOPES = True
```

Direkt danach (nach `TERRAIN_REDUCTION = 0`), ergänze:

```python
# === NATIVES TERRAIN (.terrain-Heightmap) ===
# Meter pro Heightmap-Rasterzelle. = GRID_SPACING für Auflösungs-Parität zum
# bisherigen Mesh-Ansatz (siehe Spec Abschnitt 2, Anforderung 2).
TERRAIN_SQUARE_SIZE = GRID_SPACING
# Sicherheitsabstand (Meter), den das Terrain unter der Straßen-/Böschungs-
# Mesh-Oberfläche bleiben muss, damit nichts durchsticht oder Z-Fighting
# entsteht (siehe Spec Abschnitt 4).
ROAD_EMBED_MARGIN = 0.1
# Puffer (Meter) über dem tatsächlichen Höhen-Max/-Min beim Berechnen von
# maxHeight für die .terrain-Datei (siehe Spec Abschnitt 8).
TERRAIN_MAX_HEIGHT_BUFFER = 50.0
```

- [ ] **Step 2: `FILL_ALL_MESH_HOLES`/`FILL_HOLES_MAX_EDGE_LENGTH` entfernen**

Entferne aus `world_to_beamng/config.py`:

```python
# === MESH-HOLE-FILLING ===
FILL_ALL_MESH_HOLES = False  # Schließe ALLE Boundary-Holes (äußer + Inseln)
FILL_HOLES_MAX_EDGE_LENGTH = 100.0  # Warnung bei Edge-Länge > X Metern
```

(Diese Werte werden nur in `mesh/stitch_gaps.py` verwendet, das in Task 10 komplett gelöscht wird.)

- [ ] **Step 3: Import-Fehler prüfen**

Run: `python -c "from world_to_beamng import config; print(config.TERRAIN_SQUARE_SIZE, config.ROAD_EMBED_MARGIN, config.GENERATE_SLOPES)"`
Expected: `2.0 0.1 True` (kein Fehler)

- [ ] **Step 4: Commit**

```bash
git add world_to_beamng/config.py
git commit -m "feat: Re-enable GENERATE_SLOPES, add native terrain config params"
```

---

## Task 3: `.ter`-Binärformat-Writer/Reader

Binärformat empirisch verifiziert gegen `GridMap.ter` aus `content/levels/GridMap.zip` der BeamNG-Installation (Byte-für-Byte nachgerechnet: `1(version) + 4(size) + 1024*1024*2(heightmap) + 1024*1024(layermap) + 4(matcount) + 63(materialnamen) = 3.145.800 Bytes`, exakte Übereinstimmung mit der realen Dateigröße).

**Files:**
- Create: `world_to_beamng/terrain/ter_writer.py`
- Test: `tests/terrain/test_ter_writer.py`

**Interfaces:**
- Produces:
  - `write_ter(path: Path, heightmap: np.ndarray[uint16, (size,size)], layer_map: np.ndarray[uint8, (size,size)], material_names: List[str]) -> None`
  - `read_ter(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]`
  - `encode_heights_to_u16(heights_m: np.ndarray[float], z_min: float, max_height: float) -> np.ndarray[uint16]`
  - `decode_heights_from_u16(encoded: np.ndarray[uint16], z_min: float, max_height: float) -> np.ndarray[float]`
  - `VALID_SIZES: Set[int]` = `{128, 256, 512, 1024, 2048, 4096, 8192}`

- [ ] **Step 1: Testverzeichnis anlegen und fehlschlagenden Test schreiben**

Erstelle `tests/terrain/__init__.py` (leer) und `tests/terrain/test_ter_writer.py`:

```python
"""Tests für world_to_beamng.terrain.ter_writer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.terrain.ter_writer import (
    write_ter,
    read_ter,
    encode_heights_to_u16,
    decode_heights_from_u16,
)


def test_round_trip_small_terrain(tmp_path):
    size = 128
    heightmap = np.random.randint(0, 65536, size=(size, size), dtype=np.uint16)
    layer_map = np.random.randint(0, 3, size=(size, size), dtype=np.uint8)
    material_names = ["tile_0_0", "mat_forest", "mat_grass"]

    ter_path = tmp_path / "test.ter"
    write_ter(ter_path, heightmap, layer_map, material_names)

    read_heightmap, read_layer_map, read_names = read_ter(ter_path)

    assert np.array_equal(read_heightmap, heightmap)
    assert np.array_equal(read_layer_map, layer_map)
    assert read_names == material_names


def test_invalid_size_rejected(tmp_path):
    heightmap = np.zeros((100, 100), dtype=np.uint16)
    layer_map = np.zeros((100, 100), dtype=np.uint8)

    try:
        write_ter(tmp_path / "bad.ter", heightmap, layer_map, [])
        assert False, "sollte ValueError werfen (100 ist keine Zweierpotenz)"
    except ValueError as e:
        assert "Zweierpotenz" in str(e)


def test_height_encode_decode_round_trip():
    heights_m = np.array([0.0, 100.0, 256.0, 1023.5], dtype=np.float64)
    z_min = 0.0
    max_height = 1024.0

    encoded = encode_heights_to_u16(heights_m, z_min, max_height)
    decoded = decode_heights_from_u16(encoded, z_min, max_height)

    # Präzision: max_height / 65536 = 1024/65536 = 0.015625m pro Schritt
    assert np.allclose(decoded, heights_m, atol=0.02)


def test_material_name_length_limit(tmp_path):
    heightmap = np.zeros((128, 128), dtype=np.uint16)
    layer_map = np.zeros((128, 128), dtype=np.uint8)
    too_many = [f"mat_{i}" for i in range(255)]

    try:
        write_ter(tmp_path / "bad.ter", heightmap, layer_map, too_many)
        assert False, "sollte ValueError werfen (>254 Materialien)"
    except ValueError as e:
        assert "254" in str(e)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_round_trip_small_terrain(tmp_path)
        print("[OK] test_round_trip_small_terrain")
        test_invalid_size_rejected(tmp_path)
        print("[OK] test_invalid_size_rejected")
        test_height_encode_decode_round_trip()
        print("[OK] test_height_encode_decode_round_trip")
        test_material_name_length_limit(tmp_path)
        print("[OK] test_material_name_length_limit")
        print("Alle Tests bestanden.")
```

- [ ] **Step 2: Test ausführen, Fehlschlag bestätigen**

Run: `python tests/terrain/test_ter_writer.py`
Expected: `ModuleNotFoundError: No module named 'world_to_beamng.terrain.ter_writer'`

- [ ] **Step 3: `ter_writer.py` implementieren**

Erstelle `world_to_beamng/terrain/ter_writer.py`:

```python
"""
Schreibt/liest BeamNG .ter Terrain-Dateien (Binärformat Version 9).

Format (empirisch verifiziert gegen eine echte .ter-Datei aus der BeamNG-
Installation, content/levels/GridMap.zip -> GridMap.ter):

    u8       version              (= 9)
    u32 LE   size                 (Kantenlänge, Zweierpotenz, 128-8192)
    u16[] LE heightmap            (size*size Werte, row-major)
    u8[]     layer_map            (size*size Werte, 255 = leer/Hole)
    u32 LE   material_count
    für jedes Material:
        u8   name_length
        ...  name (ASCII, name_length Bytes, kein Terminator)
"""

import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np

TER_VERSION = 9
VALID_SIZES = {128, 256, 512, 1024, 2048, 4096, 8192}
EMPTY_LAYER_VALUE = 255


def write_ter(
    path: Path,
    heightmap: np.ndarray,
    layer_map: np.ndarray,
    material_names: List[str],
) -> None:
    """
    Schreibt eine .ter-Datei.

    Args:
        path: Zielpfad der .ter-Datei
        heightmap: 2D uint16-Array, shape (size, size), row-major
        layer_map: 2D uint8-Array, gleiche Shape wie heightmap
        material_names: Materialnamen, Index entspricht layer_map-Werten
                        (max. 254 Einträge, Index 255 ist für "leer" reserviert)

    Raises:
        ValueError: bei ungültiger Größe, Shape-Mismatch oder zu vielen Materialien
    """
    if heightmap.shape != layer_map.shape:
        raise ValueError(f"heightmap shape {heightmap.shape} != layer_map shape {layer_map.shape}")

    if heightmap.ndim != 2 or heightmap.shape[0] != heightmap.shape[1]:
        raise ValueError(f"heightmap muss quadratisch sein, ist aber {heightmap.shape}")

    size = heightmap.shape[0]
    if size not in VALID_SIZES:
        raise ValueError(f"size muss eine Zweierpotenz zwischen 128 und 8192 sein (Spec: .ter-Format), ist {size}")
    if len(material_names) > 254:
        raise ValueError(f"maximal 254 Materialien erlaubt (255 ist für Holes reserviert), {len(material_names)} übergeben")

    heightmap_u16 = heightmap.astype("<u2", copy=False)
    layer_map_u8 = layer_map.astype("u1", copy=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<B", TER_VERSION))
        f.write(struct.pack("<I", size))
        f.write(heightmap_u16.tobytes(order="C"))
        f.write(layer_map_u8.tobytes(order="C"))
        f.write(struct.pack("<I", len(material_names)))
        for name in material_names:
            name_bytes = name.encode("ascii")
            if len(name_bytes) > 255:
                raise ValueError(f"Materialname zu lang (>255 Bytes): {name}")
            f.write(struct.pack("<B", len(name_bytes)))
            f.write(name_bytes)


def read_ter(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Liest eine .ter-Datei zurück (für Tests/Validierung).

    Returns:
        (heightmap, layer_map, material_names) - gleiche Typen wie write_ter's Input
    """
    with open(path, "rb") as f:
        data = f.read()

    version = data[0]
    if version != TER_VERSION:
        raise ValueError(f"Unerwartete .ter-Version: {version} (erwartet {TER_VERSION})")

    size = struct.unpack_from("<I", data, 1)[0]
    offset = 5

    heightmap = np.frombuffer(data, dtype="<u2", count=size * size, offset=offset).reshape(size, size).copy()
    offset += size * size * 2

    layer_map = np.frombuffer(data, dtype="u1", count=size * size, offset=offset).reshape(size, size).copy()
    offset += size * size

    material_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    material_names: List[str] = []
    for _ in range(material_count):
        name_length = data[offset]
        offset += 1
        name = data[offset : offset + name_length].decode("ascii")
        offset += name_length
        material_names.append(name)

    return heightmap, layer_map, material_names


def encode_heights_to_u16(heights_m: np.ndarray, z_min: float, max_height: float) -> np.ndarray:
    """
    Wandelt absolute Höhenwerte (Meter) in das u16-Format der .ter-Heightmap um.

    Formel (siehe Spec Abschnitt 8): heightMeters = storedHeight * (maxHeight / 65536)
    -> storedHeight = (heightMeters - z_min) / maxHeight * 65536

    Args:
        heights_m: beliebige Shape, absolute Höhenwerte in Metern
        z_min: Höhe (Meter), die u16-Wert 0 entspricht
        max_height: Höhenbereich (Meter), den u16-Wert 65535 entspricht

    Returns:
        Gleiche Shape wie heights_m, dtype uint16, auf [0, 65535] geclampt
    """
    relative = (heights_m - z_min) / max_height * 65536.0
    clamped = np.clip(relative, 0, 65535)
    return clamped.astype(np.uint16)


def decode_heights_from_u16(encoded: np.ndarray, z_min: float, max_height: float) -> np.ndarray:
    """Kehrt encode_heights_to_u16() um."""
    return z_min + encoded.astype(np.float64) * (max_height / 65536.0)
```

- [ ] **Step 4: Test erneut ausführen**

Run: `python tests/terrain/test_ter_writer.py`
Expected: `Alle Tests bestanden.`

- [ ] **Step 5: Reales `.ter` aus der BeamNG-Installation einlesen (Verifikationstest)**

Führe interaktiv aus (nicht Teil der automatisierten Suite, nur zur Bestätigung, dass `read_ter` mit einer echten Datei kompatibel ist):

```bash
python -c "
import zipfile, io
from pathlib import Path
import sys
sys.path.insert(0, '.')
from world_to_beamng.terrain.ter_writer import read_ter

z = zipfile.ZipFile(r'C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive\content\levels\GridMap.zip')
data = z.read('levels/GridMap/GridMap.ter')
tmp = Path('cache/_verify_gridmap.ter')
tmp.write_bytes(data)
heightmap, layer_map, names = read_ter(tmp)
print('size:', heightmap.shape, 'materials:', names)
tmp.unlink()
"
```

Expected: `size: (1024, 1024) materials: ['Asphalt', 'RockyDirt', 'Grass', 'Rock', 'Mud', 'BeachSand', 'Ice', 'asphalt_prepped']`

- [ ] **Step 6: Commit**

```bash
git add world_to_beamng/terrain/ter_writer.py tests/terrain/
git commit -m "feat: Add .ter binary format writer/reader (verified against real BeamNG terrain)"
```

---

## Task 4: `terrain/heightmap.py` — Heightmap aus Elevation-Grid bauen

**Files:**
- Create: `world_to_beamng/terrain/heightmap.py`
- Test: `tests/terrain/test_heightmap.py`

**Interfaces:**
- Consumes: nichts Neues — nutzt dieselbe `(grid_points, grid_elevations, nx, ny)`-Struktur, die `terrain/grid.py:create_terrain_grid()` bereits zurückgibt (siehe `world_to_beamng/workflow/terrain_workflow.py:264`: `grid_points, grid_elevations, nx, ny = grid`).
- Produces:
  - `next_power_of_two_size(min_size: int) -> int`
  - `build_heightmap(grid_points: np.ndarray, grid_elevations: np.ndarray, nx: int, ny: int, square_size: float) -> dict` mit Keys `"heights"` (float64 (size,size)), `"size"` (int), `"origin_x"` (float), `"origin_y"` (float)

- [ ] **Step 1: Fehlschlagenden Test schreiben**

Erstelle `tests/terrain/test_heightmap.py`:

```python
"""Tests für world_to_beamng.terrain.heightmap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.terrain.heightmap import build_heightmap, next_power_of_two_size


def test_next_power_of_two_size():
    assert next_power_of_two_size(100) == 128
    assert next_power_of_two_size(128) == 128
    assert next_power_of_two_size(129) == 256
    assert next_power_of_two_size(2049) == 4096


def test_next_power_of_two_size_exceeds_max():
    try:
        next_power_of_two_size(9000)
        assert False, "sollte ValueError werfen"
    except ValueError as e:
        assert "8192" in str(e)


def _make_regular_grid(nx, ny, spacing, base_height=100.0):
    """Baut ein synthetisches Grid wie terrain.grid.create_terrain_grid() es liefert."""
    x_coords = np.arange(nx) * spacing
    y_coords = np.arange(ny) * spacing
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)  # shape (ny, nx)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    # Höhe = base_height + x*0.1 (linearer Gradient, um Reshape-Reihenfolge zu prüfen)
    grid_elevations = base_height + grid_x.ravel() * 0.1
    return grid_points, grid_elevations, nx, ny


def test_build_heightmap_preserves_real_data():
    nx, ny, spacing = 50, 40, 2.0
    grid_points, grid_elevations, nx, ny = _make_regular_grid(nx, ny, spacing)

    result = build_heightmap(grid_points, grid_elevations, nx, ny, square_size=spacing)

    assert result["size"] == 128  # next_power_of_two_size(max(50, 40)) == 128
    assert result["heights"].shape == (128, 128)
    assert result["origin_x"] == 0.0
    assert result["origin_y"] == 0.0

    # Echte Daten im Bereich [0:ny, 0:nx] müssen exakt erhalten bleiben
    expected = grid_elevations.reshape(ny, nx)
    assert np.allclose(result["heights"][:ny, :nx], expected)


def test_build_heightmap_pads_edges_without_cliff():
    nx, ny, spacing = 10, 10, 2.0
    grid_points, grid_elevations, nx, ny = _make_regular_grid(nx, ny, spacing)

    result = build_heightmap(grid_points, grid_elevations, nx, ny, square_size=spacing)
    heights = result["heights"]

    # Padding-Bereich (rechts von Spalte nx-1) muss die letzte echte Spalte fortsetzen,
    # nicht auf 0 springen (das wäre eine sichtbare Kante, siehe Spec Abschnitt 8)
    last_real_col = heights[:ny, nx - 1]
    first_padded_col = heights[:ny, nx]
    assert np.allclose(last_real_col, first_padded_col)


if __name__ == "__main__":
    test_next_power_of_two_size()
    print("[OK] test_next_power_of_two_size")
    test_next_power_of_two_size_exceeds_max()
    print("[OK] test_next_power_of_two_size_exceeds_max")
    test_build_heightmap_preserves_real_data()
    print("[OK] test_build_heightmap_preserves_real_data")
    test_build_heightmap_pads_edges_without_cliff()
    print("[OK] test_build_heightmap_pads_edges_without_cliff")
    print("Alle Tests bestanden.")
```

- [ ] **Step 2: Test ausführen, Fehlschlag bestätigen**

Run: `python tests/terrain/test_heightmap.py`
Expected: `ModuleNotFoundError: No module named 'world_to_beamng.terrain.heightmap'`

- [ ] **Step 3: `heightmap.py` implementieren**

Erstelle `world_to_beamng/terrain/heightmap.py`:

```python
"""
Baut das quadratische .ter-Heightmap-Array direkt aus dem bestehenden
Elevation-Grid (terrain.grid.create_terrain_grid) - ersetzt die bisherige
Triangulierung in TerrainMeshBuilder.
"""

import numpy as np

VALID_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)


def next_power_of_two_size(min_size: int) -> int:
    """Kleinste gültige .ter-Größe >= min_size (Zweierpotenz, 128-8192)."""
    for size in VALID_SIZES:
        if size >= min_size:
            return size
    raise ValueError(
        f"Benötigte Terrain-Größe ({min_size}px) übersteigt das .ter-Maximum von 8192px. "
        f"TERRAIN_SQUARE_SIZE erhöhen oder Gebiet verkleinern."
    )


def build_heightmap(
    grid_points: np.ndarray,
    grid_elevations: np.ndarray,
    nx: int,
    ny: int,
    square_size: float,
) -> dict:
    """
    Baut ein quadratisches, auf Zweierpotenz aufgefülltes Heightmap-Array.

    Args:
        grid_points: (N, 2) lokale XY-Koordinaten, row-major (y-major) wie von
            terrain.grid.create_terrain_grid zurückgegeben (np.meshgrid mit
            default indexing="xy", dann .ravel() -> Reihenfolge ist
            [y0x0, y0x1, ..., y0x(nx-1), y1x0, ...])
        grid_elevations: (N,) Höhenwerte in Metern, gleiche Reihenfolge wie grid_points
        nx, ny: Grid-Dimensionen (Breite, Höhe) wie von create_terrain_grid zurückgegeben
        square_size: Meter pro Rasterzelle (config.TERRAIN_SQUARE_SIZE)

    Returns:
        {
            "heights": (size, size) float64 Array, absolute Weltkoordinaten-Höhen.
                       heights[row, col] entspricht Weltposition
                       (origin_x + col*square_size, origin_y + row*square_size).
            "size": int (Zweierpotenz),
            "origin_x": float (Welt-X der Zelle [*, 0]),
            "origin_y": float (Welt-Y der Zelle [0, *]),
        }
    """
    if len(grid_elevations) != nx * ny:
        raise ValueError(f"grid_elevations hat {len(grid_elevations)} Werte, erwartet nx*ny={nx * ny}")

    source_heights = grid_elevations.reshape(ny, nx)

    size = next_power_of_two_size(max(nx, ny))

    origin_x = float(grid_points[:, 0].min())
    origin_y = float(grid_points[:, 1].min())

    heights = np.empty((size, size), dtype=np.float64)
    heights[:ny, :nx] = source_heights

    # Rand-Padding: letzte echte Spalte/Zeile fortsetzen statt auf 0 zu springen
    # (verhindert eine sichtbare Kante am Datenrand, siehe Spec Abschnitt 8)
    if size > nx:
        heights[:ny, nx:] = source_heights[:, -1:]
    if size > ny:
        heights[ny:, :] = heights[ny - 1 : ny, :]

    return {
        "heights": heights,
        "size": size,
        "origin_x": origin_x,
        "origin_y": origin_y,
    }
```

- [ ] **Step 4: Test erneut ausführen**

Run: `python tests/terrain/test_heightmap.py`
Expected: `Alle Tests bestanden.`

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/terrain/heightmap.py tests/terrain/test_heightmap.py
git commit -m "feat: Add heightmap builder replacing terrain mesh triangulation"
```

---

## Task 5: `terrain/road_embedding.py` — Straßen-Einbettung

Implementiert den Kern-Mechanismus aus Spec Abschnitt 4: Terrain-Raster nahe jeder Straße wird gegen die (unveränderte) Straßen-/Böschungs-Mesh-Unterseite abgesenkt.

**Files:**
- Create: `world_to_beamng/terrain/road_embedding.py`
- Test: `tests/terrain/test_road_embedding.py`

**Interfaces:**
- Consumes: `road_mesh_data` (Liste von `{"vertices": [i0, i1, i2], ...}`, wie `TerrainWorkflow.process_tile()` es aus `RoadMeshBuilder.build()` erhält — siehe `world_to_beamng/workflow/terrain_workflow.py:284`) und `vertex_manager.get_array()` (siehe `world_to_beamng/mesh/vertex_manager.py`).
- Produces:
  - `road_mesh_to_arrays(road_mesh_data: List[Dict], all_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
  - `embed_roads_into_heightmap(heights: np.ndarray, origin_x: float, origin_y: float, square_size: float, road_vertices: np.ndarray, road_triangles: np.ndarray, margin: float) -> np.ndarray`

- [ ] **Step 1: Fehlschlagenden Test schreiben**

Erstelle `tests/terrain/test_road_embedding.py`:

```python
"""Tests für world_to_beamng.terrain.road_embedding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng.terrain.road_embedding import (
    embed_roads_into_heightmap,
    road_mesh_to_arrays,
)


def test_road_mesh_to_arrays():
    road_mesh_data = [
        {"vertices": [0, 1, 2], "road_id": 1, "uvs": {}},
        {"vertices": [1, 2, 3], "road_id": 1, "uvs": {}},
    ]
    all_vertices = np.array(
        [[0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10]], dtype=np.float64
    )

    vertices, triangles = road_mesh_to_arrays(road_mesh_data, all_vertices)

    assert np.array_equal(vertices, all_vertices)
    assert triangles.shape == (2, 3)
    assert list(triangles[0]) == [0, 1, 2]
    assert list(triangles[1]) == [1, 2, 3]


def test_embed_roads_lowers_only_near_road():
    # 20x20 Heightmap, 1m/Zelle, überall 100m hoch
    size = 20
    heights = np.full((size, size), 100.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    # Eine flache Straße bei Z=95 (5m unter natürlichem Terrain), Fläche x=[5,15], y=[5,15]
    road_vertices = np.array(
        [[5, 5, 95], [15, 5, 95], [5, 15, 95], [15, 15, 95]], dtype=np.float64
    )
    road_triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    margin = 0.1
    result = embed_roads_into_heightmap(
        heights, origin_x, origin_y, square_size, road_vertices, road_triangles, margin
    )

    # Zellen unter der Straße müssen auf ~95 - 0.1 = 94.9 abgesenkt sein
    assert np.isclose(result[10, 10], 95.0 - margin, atol=0.5)

    # Zellen weit weg von der Straße müssen unverändert bei 100 bleiben
    assert result[1, 1] == 100.0
    assert result[18, 18] == 100.0

    # Original-Array darf nicht verändert worden sein (Funktion gibt Kopie zurück)
    assert heights[10, 10] == 100.0


def test_embed_roads_never_raises_terrain():
    # Straße LIEGT HÖHER als natürliches Terrain -> Terrain darf NICHT angehoben werden
    size = 10
    heights = np.full((size, size), 50.0)
    origin_x, origin_y, square_size = 0.0, 0.0, 1.0

    road_vertices = np.array(
        [[2, 2, 200], [8, 2, 200], [2, 8, 200], [8, 8, 200]], dtype=np.float64
    )
    road_triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    result = embed_roads_into_heightmap(
        heights, origin_x, origin_y, square_size, road_vertices, road_triangles, margin=0.1
    )

    assert np.all(result <= 50.0)


if __name__ == "__main__":
    test_road_mesh_to_arrays()
    print("[OK] test_road_mesh_to_arrays")
    test_embed_roads_lowers_only_near_road()
    print("[OK] test_embed_roads_lowers_only_near_road")
    test_embed_roads_never_raises_terrain()
    print("[OK] test_embed_roads_never_raises_terrain")
    print("Alle Tests bestanden.")
```

- [ ] **Step 2: Test ausführen, Fehlschlag bestätigen**

Run: `python tests/terrain/test_road_embedding.py`
Expected: `ModuleNotFoundError: No module named 'world_to_beamng.terrain.road_embedding'`

- [ ] **Step 3: `road_embedding.py` implementieren**

Erstelle `world_to_beamng/terrain/road_embedding.py`:

```python
"""
Senkt das Terrain-Heightmap-Array entlang von Straßen ab, damit das
(unveränderte) Straßen-/Böschungs-Mesh sauber eingebettet liegt, statt zu
schweben oder das Terrain zu durchstechen (Spec Abschnitt 4).

Kern-Idee: für jede Rasterzelle nahe einer Straße wird die Höhe der
Straßen-/Böschungs-Mesh-Oberfläche an exakt dieser XY-Position abgefragt
(baryzentrische Interpolation im jeweiligen Dreieck) und das Terrain auf
diesen Wert minus Sicherheitsabstand abgesenkt - nie angehoben.
"""

from typing import Dict, List, Tuple

import numpy as np


def road_mesh_to_arrays(
    road_mesh_data: List[Dict], all_vertices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wandelt die strukturierte Road-Mesh-Ausgabe von RoadMeshBuilder.build() in
    ein einfaches (vertices, triangles)-Paar für embed_roads_into_heightmap() um.

    Args:
        road_mesh_data: Liste von {"vertices": [i0, i1, i2], ...} Dicts
                        (road_mesh[0] aus TerrainWorkflow.process_tile())
        all_vertices: (N, 3) Array aller Mesh-Vertex-Positionen
                      (vertex_manager.get_array())

    Returns:
        (all_vertices, triangles) - triangles ist ein (K, 3) int Array
    """
    if not road_mesh_data:
        return all_vertices, np.empty((0, 3), dtype=np.int64)
    triangles = np.array([face["vertices"] for face in road_mesh_data], dtype=np.int64)
    return all_vertices, triangles


def embed_roads_into_heightmap(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    road_vertices: np.ndarray,
    road_triangles: np.ndarray,
    margin: float,
) -> np.ndarray:
    """
    Senkt heights dort ab, wo das Straßenmesh liegt. Verändert heights NICHT
    in-place, gibt eine neue Kopie zurück.

    Args:
        heights: (size, size) float Array
        origin_x, origin_y: Welt-Koordinaten der Zelle [*, 0] bzw. [0, *]
        square_size: Meter pro Rasterzelle
        road_vertices: (M, 3) Array aller Straßen-Vertex-Positionen (x, y, z)
        road_triangles: (K, 3) Array von Vertex-Indizes (in road_vertices) pro Dreieck
        margin: Sicherheitsabstand in Metern (config.ROAD_EMBED_MARGIN)

    Returns:
        Neues (size, size) float Array
    """
    result = heights.copy()
    size_y, size_x = heights.shape

    for tri in road_triangles:
        p0 = road_vertices[tri[0]]
        p1 = road_vertices[tri[1]]
        p2 = road_vertices[tri[2]]
        _embed_triangle(result, origin_x, origin_y, square_size, p0, p1, p2, margin, size_x, size_y)

    return result


def _embed_triangle(
    heights: np.ndarray,
    origin_x: float,
    origin_y: float,
    square_size: float,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    margin: float,
    size_x: int,
    size_y: int,
) -> None:
    """Senkt alle Rasterzellen ab, die (in 2D-Draufsicht) innerhalb des Dreiecks liegen."""
    min_x = min(p0[0], p1[0], p2[0])
    max_x = max(p0[0], p1[0], p2[0])
    min_y = min(p0[1], p1[1], p2[1])
    max_y = max(p0[1], p1[1], p2[1])

    col_start = max(0, int(np.floor((min_x - origin_x) / square_size)))
    col_end = min(size_x - 1, int(np.ceil((max_x - origin_x) / square_size)))
    row_start = max(0, int(np.floor((min_y - origin_y) / square_size)))
    row_end = min(size_y - 1, int(np.ceil((max_y - origin_y) / square_size)))

    if col_start > col_end or row_start > row_end:
        return

    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(denom) < 1e-9:
        return  # entartetes Dreieck (Fläche ~0)

    cols = np.arange(col_start, col_end + 1)
    rows = np.arange(row_start, row_end + 1)
    cell_x = origin_x + cols * square_size
    cell_y = origin_y + rows * square_size
    grid_x, grid_y = np.meshgrid(cell_x, cell_y)  # shape (len(rows), len(cols))

    w0 = ((p1[1] - p2[1]) * (grid_x - p2[0]) + (p2[0] - p1[0]) * (grid_y - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (grid_x - p2[0]) + (p0[0] - p2[0]) * (grid_y - p2[1])) / denom
    w2 = 1.0 - w0 - w1

    inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not np.any(inside):
        return

    interpolated_z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
    target = interpolated_z - margin

    sub = heights[row_start : row_end + 1, col_start : col_end + 1]
    np.minimum(sub, np.where(inside, target, sub), out=sub)
```

- [ ] **Step 4: Test erneut ausführen**

Run: `python tests/terrain/test_road_embedding.py`
Expected: `Alle Tests bestanden.`

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/terrain/road_embedding.py tests/terrain/test_road_embedding.py
git commit -m "feat: Add road embedding - lowers terrain heightmap under road mesh"
```

---

## Task 6: `terrain/terrain_materials.py` — Layer-Map & Landnutzung

**Files:**
- Create: `world_to_beamng/terrain/terrain_materials.py`
- Test: `tests/terrain/test_terrain_materials.py`

**Interfaces:**
- Consumes: `data/osm_to_beamng.json["landuse_mappings"]` (siehe Task 1), OSM-Polygone mit `osm_tags`-Dict (gleiche Konvention wie überall sonst in diesem Projekt, z.B. `world_to_beamng/workflow/terrain_workflow.py:220`: `osm_tags = road.get("osm_tags", {})`).
- Produces:
  - `ACTIVE_LANDUSE_CATEGORIES: Set[str]` = `{"forest", "meadow", "farmland"}`
  - `get_landuse_category(osm_tags: Dict, landuse_mappings: Dict) -> Optional[str]`
  - `build_photo_fallback_layer(size: int, origin_x: float, origin_y: float, square_size: float, tile_size: float) -> Tuple[np.ndarray, List[str]]`
  - `paint_landuse_materials(layer_map: np.ndarray, material_names: List[str], size: int, origin_x: float, origin_y: float, square_size: float, landuse_polygons: List[Dict], landuse_mappings: Dict) -> Tuple[np.ndarray, List[str]]` — `landuse_polygons` ist `List[{"osm_tags": Dict, "geometry": shapely.geometry.Polygon}]` in lokalen Koordinaten
  - `build_terrain_material_entries(material_names: List[str], photo_tile_names: List[str], landuse_mappings: Dict, level_name: str, tile_size: float) -> Dict[str, Dict]`

- [ ] **Step 1: Fehlschlagenden Test schreiben**

Erstelle `tests/terrain/test_terrain_materials.py`:

```python
"""Tests für world_to_beamng.terrain.terrain_materials."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from shapely.geometry import Polygon

from world_to_beamng.terrain.terrain_materials import (
    get_landuse_category,
    build_photo_fallback_layer,
    paint_landuse_materials,
    build_terrain_material_entries,
)

LANDUSE_MAPPINGS_FIXTURE = {
    "forest": {"priority": 10, "internal_name": "mat_forest", "baseColorMap": "a/forest_b.png"},
    "meadow": {"priority": 4, "internal_name": "mat_grass", "baseColorMap": "a/grass_b.png"},
    "farmland": {"priority": 5, "internal_name": "mat_dirt", "baseColorMap": "a/dirt_b.png"},
    "water": {"priority": 15, "internal_name": "mat_water", "baseColorMap": "a/water_b.png"},
}


def test_get_landuse_category_matches_active_only():
    assert get_landuse_category({"landuse": "forest"}, LANDUSE_MAPPINGS_FIXTURE) == "forest"
    assert get_landuse_category({"natural": "wood"}, LANDUSE_MAPPINGS_FIXTURE) is None  # nicht in Fixture
    # "water" ist in landuse_mappings, aber NICHT in ACTIVE_LANDUSE_CATEGORIES -> None
    assert get_landuse_category({"natural": "water"}, LANDUSE_MAPPINGS_FIXTURE) is None
    assert get_landuse_category({}, LANDUSE_MAPPINGS_FIXTURE) is None


def test_photo_fallback_layer_one_material_per_tile():
    # 10x10 Raster, 1m/Zelle, tile_size=5m -> 2x2 Kacheln im Bereich
    layer_map, names = build_photo_fallback_layer(
        size=10, origin_x=0.0, origin_y=0.0, square_size=1.0, tile_size=5.0
    )
    assert layer_map.shape == (10, 10)
    assert len(names) == 4  # 2x2 Kacheln
    assert all(n.startswith("tile_") for n in names)
    # Zelle (0,0) und Zelle (9,9) müssen unterschiedliche Kachel-Materialien haben
    assert layer_map[0, 0] != layer_map[9, 9]


def test_paint_landuse_overwrites_photo_fallback():
    size = 20
    layer_map, names = build_photo_fallback_layer(
        size=size, origin_x=0.0, origin_y=0.0, square_size=1.0, tile_size=100.0
    )
    assert len(names) == 1  # ein Foto-Tile deckt alles ab

    forest_polygon = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    landuse_polygons = [{"osm_tags": {"landuse": "forest"}, "geometry": forest_polygon}]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    assert "mat_forest" in new_names
    forest_index = new_names.index("mat_forest")

    # Zelle innerhalb des Wald-Polygons muss jetzt das Wald-Material haben
    assert new_layer_map[10, 10] == forest_index
    # Zelle außerhalb muss beim Foto-Fallback bleiben
    assert new_layer_map[0, 0] == names.index(names[0])
    assert new_layer_map[0, 0] != forest_index


def test_paint_landuse_priority_resolves_overlap():
    size = 20
    layer_map = np.zeros((size, size), dtype=np.uint8)
    names = ["tile_0_0"]

    # Zwei überlappende Polygone: farmland (priority=5) und forest (priority=10)
    farmland_poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])  # deckt alles ab
    forest_poly = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])  # kleinerer Ausschnitt

    landuse_polygons = [
        {"osm_tags": {"landuse": "farmland"}, "geometry": farmland_poly},
        {"osm_tags": {"landuse": "forest"}, "geometry": forest_poly},
    ]

    new_layer_map, new_names = paint_landuse_materials(
        layer_map, names, size, 0.0, 0.0, 1.0, landuse_polygons, LANDUSE_MAPPINGS_FIXTURE
    )

    forest_index = new_names.index("mat_forest")
    farmland_index = new_names.index("mat_dirt")

    # Im Überlappungsbereich gewinnt forest (höhere priority)
    assert new_layer_map[10, 10] == forest_index
    # Außerhalb des Wald-Polygons, aber innerhalb des Farmland-Polygons: farmland
    assert new_layer_map[1, 1] == farmland_index


def test_build_terrain_material_entries():
    material_names = ["tile_0_0", "mat_forest"]
    photo_tile_names = ["tile_0_0"]

    entries = build_terrain_material_entries(
        material_names, photo_tile_names, LANDUSE_MAPPINGS_FIXTURE, "world_to_beamng", 500.0
    )

    assert "tile_0_0" in entries
    assert entries["tile_0_0"]["class"] == "TerrainMaterial"
    assert "levels/world_to_beamng" in entries["tile_0_0"]["baseColorBaseTex"]

    assert "mat_forest" in entries
    assert entries["mat_forest"]["baseColorBaseTex"] == "a/forest_b.png"


if __name__ == "__main__":
    test_get_landuse_category_matches_active_only()
    print("[OK] test_get_landuse_category_matches_active_only")
    test_photo_fallback_layer_one_material_per_tile()
    print("[OK] test_photo_fallback_layer_one_material_per_tile")
    test_paint_landuse_overwrites_photo_fallback()
    print("[OK] test_paint_landuse_overwrites_photo_fallback")
    test_paint_landuse_priority_resolves_overlap()
    print("[OK] test_paint_landuse_priority_resolves_overlap")
    test_build_terrain_material_entries()
    print("[OK] test_build_terrain_material_entries")
    print("Alle Tests bestanden.")
```

- [ ] **Step 2: Test ausführen, Fehlschlag bestätigen**

Run: `python tests/terrain/test_terrain_materials.py`
Expected: `ModuleNotFoundError: No module named 'world_to_beamng.terrain.terrain_materials'`

- [ ] **Step 3: `terrain_materials.py` implementieren**

Erstelle `world_to_beamng/terrain/terrain_materials.py`:

```python
"""
Baut die .ter-Layer-Map (Material-Index pro Rasterzelle) aus zwei Quellen:

1. Luftbild-Fallback: pro 500m-Kachel (config.TILE_SIZE) ein eigenes Foto-
   Material, mit denselben Dateinamen ("tile_<x>_<y>"), die
   io/dae.py:create_terrain_materials_json() bereits für den bisherigen
   Mesh-Ansatz erzeugt hat.
2. OSM-Landnutzung: Polygone aus data/osm_to_beamng.json["landuse_mappings"]
   werden priorisiert in die Layer-Map gebrannt (Spec Abschnitt 6).
"""

from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from affine import Affine
from rasterio.features import rasterize

# Bewusst konservativer Startsatz (siehe Spec Abschnitt 6/Global Constraints).
# water/industrial/commercial/residential/vineyard/greenhouse_horticulture/
# orchard bleiben trotz vorhandenem landuse_mappings-Eintrag Foto-Fallback,
# bis ihre Texturpfade verifiziert und bewusst aktiviert werden.
ACTIVE_LANDUSE_CATEGORIES = {"forest", "meadow", "farmland"}

EMPTY_RASTER_VALUE = 255


def get_landuse_category(osm_tags: Dict, landuse_mappings: Dict) -> Optional[str]:
    """
    Ermittelt die landuse_mappings-Kategorie für ein OSM-Element.

    landuse_mappings ist flach nach Kategorienamen organisiert (z.B. "forest",
    "meadow") - der Kategoriename IST der OSM-Tag-Wert. Prüft landuse-, dann
    natural-, dann leisure-Tag.

    Returns:
        Kategoriename oder None, falls kein aktiver Treffer
    """
    for tag_key in ("landuse", "natural", "leisure"):
        value = osm_tags.get(tag_key)
        if value in landuse_mappings and value in ACTIVE_LANDUSE_CATEGORIES:
            return value
    return None


def build_photo_fallback_layer(
    size: int,
    origin_x: float,
    origin_y: float,
    square_size: float,
    tile_size: float,
) -> Tuple[np.ndarray, List[str]]:
    """
    Baut die Basis-Layer-Map: pro tile_size-Kachel (config.TILE_SIZE, Default
    500m) ein eigenes Foto-Material, benannt wie die bestehenden
    Terrain-Tile-Texturen ("tile_<x>_<y>").

    Returns:
        (layer_map, material_names) - layer_map ist (size, size) uint8,
        material_names[i] ist der Materialname für layer_map-Wert i
    """
    layer_map = np.zeros((size, size), dtype=np.uint8)
    material_names: List[str] = []
    tile_index: Dict[Tuple[int, int], int] = {}

    for row in range(size):
        world_y = origin_y + row * square_size
        tile_y = int(np.floor(world_y / tile_size)) * int(tile_size)
        for col in range(size):
            world_x = origin_x + col * square_size
            tile_x = int(np.floor(world_x / tile_size)) * int(tile_size)

            key = (tile_x, tile_y)
            if key not in tile_index:
                if len(material_names) >= 254:
                    raise ValueError(
                        "Mehr als 254 Foto-Kacheln im Terrain-Bereich - "
                        "TERRAIN_SQUARE_SIZE oder TILE_SIZE erhöhen"
                    )
                tile_index[key] = len(material_names)
                material_names.append(f"tile_{tile_x}_{tile_y}")

            layer_map[row, col] = tile_index[key]

    return layer_map, material_names


def paint_landuse_materials(
    layer_map: np.ndarray,
    material_names: List[str],
    size: int,
    origin_x: float,
    origin_y: float,
    square_size: float,
    landuse_polygons: List[Dict],
    landuse_mappings: Dict,
) -> Tuple[np.ndarray, List[str]]:
    """
    Brennt OSM-Landnutzungs-Polygone in die Layer-Map, priorisiert nach
    landuse_mappings[category]["priority"] (höhere Priorität gewinnt bei
    Überlappung, siehe Spec Abschnitt 6/8).

    Args:
        layer_map: (size, size) uint8, wird NICHT verändert (Kopie wird zurückgegeben)
        material_names: bisherige Materialliste (Foto-Fallback-Namen)
        landuse_polygons: Liste von {"osm_tags": Dict, "geometry": shapely.Polygon}
                          in lokalen (Grid-)Koordinaten
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]

    Returns:
        (neue layer_map, erweiterte material_names)
    """
    result = layer_map.copy()
    names = list(material_names)
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(square_size, square_size)

    scored = []
    for poly in landuse_polygons:
        category = get_landuse_category(poly["osm_tags"], landuse_mappings)
        if category is None:
            continue
        category_data = landuse_mappings[category]
        scored.append((category_data.get("priority", 0), poly["geometry"], category_data["internal_name"]))

    # Aufsteigend nach Priorität sortieren -> hohe Priorität wird zuletzt (obenauf) gebrannt
    scored.sort(key=lambda item: item[0])

    for _priority, geometry, internal_name in scored:
        if internal_name not in names:
            if len(names) >= 254:
                raise ValueError("Mehr als 254 Materialien - Landnutzungs-Kategorien reduzieren")
            names.append(internal_name)
        material_index = names.index(internal_name)

        mask = rasterize(
            [(geometry, material_index)],
            out_shape=(size, size),
            transform=transform,
            fill=EMPTY_RASTER_VALUE,
            dtype="uint8",
        )
        hit = mask != EMPTY_RASTER_VALUE
        result[hit] = mask[hit]

    return result, names


def build_terrain_material_entries(
    material_names: List[str],
    photo_tile_names: List[str],
    landuse_mappings: Dict,
    level_name: str,
    tile_size: float,
) -> Dict[str, Dict]:
    """
    Baut TerrainMaterial-JSON-Einträge für materials.json (Schema verifiziert
    gegen BeamNGs eigenes template-Level, siehe Task 7).

    Args:
        material_names: alle Layer-Map-Materialnamen in Index-Reihenfolge
        photo_tile_names: Teilmenge von material_names, die Foto-Kacheln sind
                          (Namen wie "tile_<x>_<y>")
        landuse_mappings: data/osm_to_beamng.json["landuse_mappings"]
        level_name, tile_size: für den Foto-Textur-Pfad

    Returns:
        {material_name: {...TerrainMaterial JSON...}}
    """
    entries: Dict[str, Dict] = {}
    landuse_by_internal_name = {v["internal_name"]: v for v in landuse_mappings.values()}
    photo_tile_set = set(photo_tile_names)

    for name in material_names:
        if name in photo_tile_set:
            entries[name] = {
                "internalName": name,
                "class": "TerrainMaterial",
                "persistentId": str(uuid4()),
                "baseColorBaseTex": f"/levels/{level_name}/art/shapes/textures/{name}.dds",
                "baseColorBaseTexSize": tile_size,
            }
            continue

        category_data = landuse_by_internal_name.get(name)
        if category_data is None:
            continue

        entry = {
            "internalName": name,
            "class": "TerrainMaterial",
            "persistentId": str(uuid4()),
            "baseColorBaseTex": category_data["baseColorMap"],
            "baseColorBaseTexSize": 4.0,
        }
        if category_data.get("normalMap"):
            entry["normalBaseTex"] = category_data["normalMap"]
        entries[name] = entry

    return entries
```

- [ ] **Step 4: Test erneut ausführen**

Run: `python tests/terrain/test_terrain_materials.py`
Expected: `Alle Tests bestanden.`

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/terrain/terrain_materials.py tests/terrain/test_terrain_materials.py
git commit -m "feat: Add terrain layer-map builder with OSM landuse material painting"
```

---

## Task 7: `ItemManager.add_terrain_block()`

**Files:**
- Modify: `world_to_beamng/managers/item_manager.py`
- Test: `tests/test_item_manager_terrain_block.py`

**Interfaces:**
- Consumes: `ItemManager.add_item()` (bestehende Methode, siehe `world_to_beamng/managers/item_manager.py:199`).
- Produces: `ItemManager.add_terrain_block(name: str, terrain_filename: str, material_texture_set: str, max_height: float, z_min: float, origin_x: float, origin_y: float, overwrite: bool = False) -> str`

JSON-Schema verifiziert gegen BeamNGs eigenes `template`-Level
(`content/levels/template.zip`, `levels/template/main/MissionGroup/level_objects/terrain/items.level.json`):

```json
{
  "name": "theTerrain",
  "class": "TerrainBlock",
  "persistentId": "a500d9a8-fb60-4a81-b438-f9f9593e8f7c",
  "position": [-512, -512, 100],
  "materialTextureSet": "templateTerrainMaterialTextureSet",
  "maxHeight": 120,
  "terrainFile": "/levels/template/theTerrain.ter"
}
```

- [ ] **Step 1: Fehlschlagenden Test schreiben**

Erstelle `tests/test_item_manager_terrain_block.py`:

```python
"""Tests für ItemManager.add_terrain_block()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.managers.item_manager import ItemManager


def test_add_terrain_block_creates_correct_item(tmp_path):
    ItemManager.reset_instance()
    items = ItemManager.get_instance(tmp_path)

    items.add_terrain_block(
        name="theTerrain",
        terrain_filename="world_to_beamng.ter",
        material_texture_set="world_to_beamngTerrainMaterialTextureSet",
        max_height=574.0,
        z_min=263.0,
        origin_x=-1024.0,
        origin_y=-1024.0,
    )

    item = items.items["theTerrain"]
    assert item["class"] == "TerrainBlock"
    assert item["position"] == [-1024.0, -1024.0, 263.0]
    assert item["maxHeight"] == 574.0
    assert item["materialTextureSet"] == "world_to_beamngTerrainMaterialTextureSet"
    assert item["terrainFile"] == "/levels/world_to_beamng/world_to_beamng.ter"
    ItemManager.reset_instance()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_add_terrain_block_creates_correct_item(Path(tmp))
        print("[OK] test_add_terrain_block_creates_correct_item")
        print("Alle Tests bestanden.")
```

- [ ] **Step 2: Test ausführen, Fehlschlag bestätigen**

Run: `python tests/test_item_manager_terrain_block.py`
Expected: `AttributeError: 'ItemManager' object has no attribute 'add_terrain_block'`

- [ ] **Step 3: `add_terrain_block()` implementieren**

In `world_to_beamng/managers/item_manager.py`, füge nach der bestehenden `add_terrain()`-Methode (endet mit `return name` nach Zeile 284) folgende neue Methode ein:

```python
    def add_terrain_block(
        self,
        name: str,
        terrain_filename: str,
        material_texture_set: str,
        max_height: float,
        z_min: float,
        origin_x: float,
        origin_y: float,
        overwrite: bool = False,
    ) -> str:
        """
        Registriert das native BeamNG-Terrain (TerrainBlock, .ter-Datei).

        JSON-Schema verifiziert gegen BeamNGs eigenes template-Level
        (content/levels/template.zip).

        Args:
            name: Item-Name (üblich: "theTerrain")
            terrain_filename: Dateiname der .ter-Datei (z.B. "world_to_beamng.ter"),
                              relativ zum Level-Root abgelegt
            material_texture_set: Name des TerrainMaterialTextureSet
            max_height: Höhenbereich in Metern (config.TERRAIN_MAX_HEIGHT_BUFFER
                       + tatsächliche Elevation-Spanne)
            z_min: absolute Welthöhe (Meter), die Heightmap-Wert 0 entspricht
            origin_x, origin_y: Welt-Koordinaten der Terrain-Ecke [0, 0]
            overwrite: Überschreibe existierendes Item

        Returns:
            Item-Name
        """
        from .. import config

        self.add_item(
            name,
            item_class="TerrainBlock",
            position=(origin_x, origin_y, z_min),
            overwrite=overwrite,
            materialTextureSet=material_texture_set,
            maxHeight=max_height,
            terrainFile=f"/levels/{config.LEVEL_NAME}/{terrain_filename}",
        )
        return name
```

- [ ] **Step 4: Test erneut ausführen**

Run: `python tests/test_item_manager_terrain_block.py`
Expected: `Alle Tests bestanden.`

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/managers/item_manager.py tests/test_item_manager_terrain_block.py
git commit -m "feat: Add ItemManager.add_terrain_block() for native TerrainBlock registration"
```

---

## Task 8: `TerrainMaterial`-Registrierung im `MaterialManager`

**Files:**
- Modify: `world_to_beamng/managers/material_manager.py`
- Test: `tests/test_material_manager_terrain.py`

**Interfaces:**
- Consumes: Ausgabe von `terrain_materials.build_terrain_material_entries()` (Task 6).
- Produces: `MaterialManager.add_terrain_materials(entries: Dict[str, Dict]) -> None` — schreibt die Einträge in `self.materials` (dieselbe Sammlung, die später als `materials.json` exportiert wird — siehe existierendes Muster in `world_to_beamng/workflow/terrain_workflow.py:582-586`: `self.materials.materials[mat_name] = mat_data`).

- [ ] **Step 1: Fehlschlagenden Test schreiben**

Erstelle `tests/test_material_manager_terrain.py`:

```python
"""Tests für MaterialManager.add_terrain_materials()."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.managers.material_manager import MaterialManager


def test_add_terrain_materials_registers_entries(tmp_path):
    MaterialManager.reset_instance()
    materials = MaterialManager.get_instance(tmp_path)

    entries = {
        "mat_forest": {
            "internalName": "mat_forest",
            "class": "TerrainMaterial",
            "persistentId": "abc-123",
            "baseColorBaseTex": "a/forest_b.png",
            "baseColorBaseTexSize": 4.0,
        }
    }

    materials.add_terrain_materials(entries)

    assert "mat_forest" in materials.materials
    assert materials.materials["mat_forest"]["class"] == "TerrainMaterial"
    MaterialManager.reset_instance()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_add_terrain_materials_registers_entries(Path(tmp))
        print("[OK] test_add_terrain_materials_registers_entries")
        print("Alle Tests bestanden.")
```

- [ ] **Step 2: Test ausführen, Fehlschlag bestätigen**

Run: `python tests/test_material_manager_terrain.py`
Expected: `AttributeError: 'MaterialManager' object has no attribute 'add_terrain_materials'`

- [ ] **Step 3: `add_terrain_materials()` implementieren**

**Verifiziert:** `MaterialManager` hält seine Material-Sammlung in `self.materials` (Dict) — bestätigt in `world_to_beamng/managers/material_manager.py:428` (`self.materials = materials_dict if isinstance(materials_dict, dict) else {}`) und konsistent mit dem bestehenden Aufrufmuster in `terrain_workflow.py` (`self.materials.materials[mat_name] = mat_data`, wobei das äußere `.materials` der `MaterialManager`-Instanz-Alias in `TerrainWorkflow` ist, das innere `.materials` das hier gemeinte Dict).

In `world_to_beamng/managers/material_manager.py`, füge innerhalb der `MaterialManager`-Klasse (z.B. direkt nach `__init__`, das bei Zeile 35 beginnt) folgende Methode ein:

```python
    def add_terrain_materials(self, entries: Dict[str, Dict]) -> None:
        """
        Registriert TerrainMaterial-Einträge (aus
        terrain.terrain_materials.build_terrain_material_entries()) für den
        späteren materials.json-Export.

        Args:
            entries: {material_name: {...TerrainMaterial JSON...}}
        """
        for mat_name, mat_data in entries.items():
            self.materials[mat_name] = mat_data
```

- [ ] **Step 4: Test erneut ausführen**

Run: `python tests/test_material_manager_terrain.py`
Expected: `Alle Tests bestanden.`

- [ ] **Step 5: Commit**

```bash
git add world_to_beamng/managers/material_manager.py tests/test_material_manager_terrain.py
git commit -m "feat: Add MaterialManager.add_terrain_materials() for TerrainMaterial entries"
```

---

## Task 9: Integration in `terrain_workflow.py`

Der größte Task: `process_tile()` und `export_tile()` werden umgebaut, um den neuen Heightmap-Pfad statt `TerrainMeshBuilder`/`classify_grid_vertices` zu nutzen.

**Files:**
- Modify: `world_to_beamng/workflow/terrain_workflow.py`

**Interfaces:**
- Consumes: alle in Task 3-8 neu geschaffenen Funktionen/Methoden.
- Produces: `TerrainWorkflow.process_tile()` gibt weiterhin ein Dict zurück, jetzt mit zusätzlichen Keys `"heightmap"`, `"terrain_size"`, `"terrain_origin_x"`, `"terrain_origin_y"`, `"z_min"`, `"max_height"`, `"layer_map"`, `"terrain_material_names"`, `"photo_tile_names"` statt `"terrain_mesh"`. `export_tile()` schreibt zusätzlich die `.ter`-Datei und registriert den `TerrainBlock`.

- [ ] **Step 1: `process_tile()` umbauen — Vertex-Klassifizierung und Terrain-Mesh-Aufbau entfernen**

In `world_to_beamng/workflow/terrain_workflow.py`, entferne den Import und Aufruf von `classify_grid_vertices` (Zeilen 69 und 263-265):

Entferne:
```python
        from ..geometry.vertices import classify_grid_vertices
```
(aus dem Import-Block, Zeile 69)

Entferne:
```python
        # 9. Vertex-Klassifizierung (nutzt road_slope_polygons_2d)
        grid_points, grid_elevations, nx, ny = grid
        vertex_states = classify_grid_vertices(grid_points, grid_elevations, road_slope_polygons_2d)
```

und ersetze durch:

```python
        # 9. Grid-Dimensionen extrahieren (Vertex-Klassifizierung entfällt -
        # das Terrain wird nicht mehr trianguliert, siehe Task 9)
        grid_points, grid_elevations, nx, ny = grid
```

- [ ] **Step 2: `TerrainMeshBuilder`-Aufruf durch Heightmap-Pipeline ersetzen**

Ersetze den kompletten Block "11. Terrain Mesh (mit Builder)" (von `from ..builders import TerrainMeshBuilder` bis zum Ende von `.build()`) durch:

```python
        # 11. Terrain-Heightmap statt Mesh-Triangulierung (siehe Spec:
        # docs/superpowers/specs/2026-09-17-terrain-heightmap-migration-design.md)
        from ..terrain.heightmap import build_heightmap
        from ..terrain.road_embedding import embed_roads_into_heightmap, road_mesh_to_arrays
        from ..terrain.terrain_materials import build_photo_fallback_layer, paint_landuse_materials

        heightmap_result = build_heightmap(
            grid_points, grid_elevations, nx, ny, config.TERRAIN_SQUARE_SIZE
        )
        heights = heightmap_result["heights"]
        terrain_size = heightmap_result["size"]
        terrain_origin_x = heightmap_result["origin_x"]
        terrain_origin_y = heightmap_result["origin_y"]

        # Straßen-Einbettung: Terrain unter dem (unveränderten) Straßen-/
        # Böschungsmesh knapp absenken
        all_vertices = np.array(vertex_manager.get_array())
        road_mesh_data_for_embedding = road_mesh[0]
        road_vertices, road_triangles = road_mesh_to_arrays(road_mesh_data_for_embedding, all_vertices)
        heights = embed_roads_into_heightmap(
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            road_vertices,
            road_triangles,
            config.ROAD_EMBED_MARGIN,
        )

        # Layer-Map: Foto-Fallback pro Tile, dann OSM-Landnutzung obenauf
        layer_map, photo_tile_names = build_photo_fallback_layer(
            terrain_size, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, config.TILE_SIZE
        )

        from shapely.geometry import shape as shapely_shape
        from pyproj import Transformer
        from ..geometry.coordinates import transformer_to_wgs84

        # osm_data enthält an dieser Stelle noch RAW Overpass-Geometrie (lat/lon,
        # {"lat":.., "lon":..} pro Punkt) - dieselbe Situation, die
        # ForestWorkflow._transform_osm_to_local() für Wald-Polygone löst.
        # Für Landnutzungs-Polygone hier dieselbe WGS84->UTM->lokal-Transformation.
        transformer_utm = Transformer.from_proj(
            transformer_to_wgs84.target_crs,  # WGS84
            transformer_to_wgs84.source_crs,  # UTM
        )
        offset_x, offset_y = global_offset[0], global_offset[1]

        landuse_polygons = []
        for element in osm_data:
            tags = element.get("tags", {})
            if not tags:
                continue
            geometry = element.get("geometry")
            if not geometry or len(geometry) < 3:
                continue
            try:
                coords_2d = []
                for pt in geometry:
                    if not isinstance(pt, dict) or "lat" not in pt or "lon" not in pt:
                        continue
                    utm_x, utm_y = transformer_utm.transform(pt["lon"], pt["lat"])
                    coords_2d.append((utm_x - offset_x, utm_y - offset_y))
                if len(coords_2d) < 3:
                    continue
                polygon = shapely_shape({"type": "Polygon", "coordinates": [coords_2d]})
                if not polygon.is_valid or polygon.is_empty:
                    continue
            except Exception:
                continue
            landuse_polygons.append({"osm_tags": tags, "geometry": polygon})

        layer_map, terrain_material_names = paint_landuse_materials(
            layer_map,
            photo_tile_names,
            terrain_size,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            landuse_polygons,
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
        )

        z_min = float(heights.min())
        z_max = float(heights.max())
        max_height = (z_max - z_min) + config.TERRAIN_MAX_HEIGHT_BUFFER
```

**Verifiziert:** `OSMMapper.__init__` (`world_to_beamng/osm/osm_mapper.py:10-14`) lädt das JSON in `self.config` (nicht `osm_config`) — z.B. `self.forest_mappings = self.config.get("forest_mappings", {})` nutzt exakt dasselbe Muster, das hier für `landuse_mappings` übernommen wird.

- [ ] **Step 3: `process_tile()`-Rückgabewert anpassen**

Ersetze im `return`-Dict am Ende von `process_tile()`:

```python
            "terrain_mesh": terrain_mesh,
```

durch:

```python
            "heightmap": heights,
            "terrain_size": terrain_size,
            "terrain_origin_x": terrain_origin_x,
            "terrain_origin_y": terrain_origin_y,
            "z_min": z_min,
            "max_height": max_height,
            "layer_map": layer_map,
            "terrain_material_names": terrain_material_names,
            "photo_tile_names": photo_tile_names,
```

- [ ] **Step 4: `export_tile()` umbauen — Terrain-DAE-Export entfernen, `.ter`-Export hinzufügen**

`export_tile()` exportiert aktuell Terrain- UND Straßen-Faces gemeinsam als DAE-Tiles (`terrain_faces = terrain_mesh["faces"]`, `mesh_obj = terrain_mesh.get("mesh_obj")`, dann `slice_mesh_into_tiles(...)` über `all_faces` inkl. Terrain). Das wird aufgeteilt: Straßen bleiben DAE-Export (unverändert in ihrer Geometrie), Terrain wird `.ter`.

Ersetze die Zeilen ab `# Extrahiere Daten` bis `terrain_faces = terrain_mesh["faces"]` / `mesh_obj = terrain_mesh.get("mesh_obj")` / `vertex_normals = terrain_mesh.get("vertex_normals")`:

Alt:
```python
        road_mesh_tuple = mesh_data["road_mesh"]
        terrain_mesh = mesh_data["terrain_mesh"]
        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        vertex_manager = mesh_data["vertex_manager"]

        # Entpacke strukturierte Road-Daten
        # Format: [{'vertices': [v0,v1,v2], 'road_id': id, 'uvs': {...}}, ...]
        road_mesh_data = road_mesh_tuple[0]

        # Konvertiere zurück in die beiden Arrays für diese Funktion (zur Kompatibilität)
        all_road_faces = [rd["vertices"] for rd in road_mesh_data]
        road_face_to_idx = [rd["road_id"] for rd in road_mesh_data]

        # Entpacke terrain_mesh
        terrain_faces = terrain_mesh["faces"]
        mesh_obj = terrain_mesh.get("mesh_obj")  # NEU: Hole Mesh-Objekt mit face_uvs
        vertex_normals = terrain_mesh.get("vertex_normals")
```

Neu:
```python
        road_mesh_tuple = mesh_data["road_mesh"]
        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        vertex_manager = mesh_data["vertex_manager"]

        # Entpacke strukturierte Road-Daten
        # Format: [{'vertices': [v0,v1,v2], 'road_id': id, 'uvs': {...}}, ...]
        road_mesh_data = road_mesh_tuple[0]

        # Konvertiere zurück in die beiden Arrays für diese Funktion (zur Kompatibilität)
        all_road_faces = [rd["vertices"] for rd in road_mesh_data]
        road_face_to_idx = [rd["road_id"] for rd in road_mesh_data]
```

Danach, in demselben `all_faces`/`materials_per_face`-Aufbau weiter unten (der Block ab `# Kombiniere alle Faces mit Materials`), müssen alle Referenzen auf `terrain_faces`/`mesh_obj`/`vertex_normals` entfernt werden — `slice_mesh_into_tiles` wird nur noch mit den Straßen-Faces aufgerufen. Ersetze:

Alt:
```python
        all_faces = []
        materials_per_face = []

        # Iteriere über ALLE Faces in mesh_obj (enthält schon Roads mit korrekten Materialien!)
        for face_idx, face in enumerate(terrain_faces):
            all_faces.append(face)
            # Hole Material aus mesh_obj.face_props
            if mesh_obj and hasattr(mesh_obj, "face_props") and face_idx in mesh_obj.face_props:
                mat_name = mesh_obj.face_props[face_idx].get("material", "terrain")
                materials_per_face.append(mat_name)

                # Füge zu unique_materials hinzu falls Road-Material
                if mat_name != "terrain":
                    # Suche Properties in road_material_map
                    for r_id, (r_mat, r_props) in road_material_map.items():
                        if r_mat == mat_name:
                            unique_materials[mat_name] = r_props
                            break
            else:
                materials_per_face.append("terrain")

        # Hole alle Vertices vom VertexManager
        all_vertices = np.array(vertex_manager.get_array())

        # Slice in Tiles (übergebe mesh_obj für indexed UV-System!)
        tiles_dict = slice_mesh_into_tiles(
            vertices=all_vertices,
            faces=all_faces,
            materials_per_face=materials_per_face,
            tile_size=config.TILE_SIZE,
            vertex_normals=vertex_normals,
            mesh_obj=mesh_obj,  # Nutze indexed UV-System (uvs + uv_indices)
        )
```

Neu:
```python
        all_faces = []
        materials_per_face = []

        for face_data in road_mesh_data:
            all_faces.append(face_data["vertices"])
            mat_name = None
            r_id = face_data.get("road_id")
            if r_id in road_material_map:
                mat_name = road_material_map[r_id][0]
            materials_per_face.append(mat_name or "road_default")

        # Hole alle Vertices vom VertexManager
        all_vertices = np.array(vertex_manager.get_array())

        # slice_mesh_into_tiles() erwartet für hochwertige Straßen-Texturierung
        # ein mesh_obj mit .uv_indices/.uvs (siehe mesh/tile_slicer.py:330-360:
        # "if mesh_obj and hasattr(mesh_obj, 'uv_indices') and original_face_idx
        # in mesh_obj.uv_indices"). Vorher kamen diese UVs aus dem kombinierten
        # Terrain+Road mesh_obj von TerrainMeshBuilder; die UV-Rohdaten selbst
        # stammen aber unverändert aus RoadMeshBuilder (road_mesh_data[i]["uvs"]),
        # nicht aus TerrainMeshBuilder. Ein minimaler Adapter reicht, um exakt
        # dieselbe Straßen-Textur-Qualität wie vor der Migration zu erhalten
        # (statt auf die gröbere Tile-planare Fallback-UV zurückzufallen, die
        # slice_mesh_into_tiles sonst für Faces ohne mesh_obj-Treffer nutzt -
        # siehe tile_slicer.py:361-369).
        class _RoadUVAdapter:
            """Minimaler mesh_obj-Ersatz: stellt nur die Road-UVs aus
            road_mesh_data bereit, im von slice_mesh_into_tiles erwarteten
            Format (uv_indices: {face_idx: [i0,i1,i2]}, uvs: [(u,v), ...])."""

            def __init__(self, road_mesh_data, faces):
                self.uvs = []
                self.uv_indices = {}
                uv_lookup = {}

                for face_idx, face_data in enumerate(road_mesh_data):
                    face_vertices = faces[face_idx]
                    per_vertex_uv = face_data.get("uvs", {})
                    indices = []
                    for vertex_idx in face_vertices:
                        uv = per_vertex_uv.get(vertex_idx, (0.0, 0.0))
                        if uv not in uv_lookup:
                            uv_lookup[uv] = len(self.uvs)
                            self.uvs.append(uv)
                        indices.append(uv_lookup[uv])
                    self.uv_indices[face_idx] = indices

        road_uv_adapter = _RoadUVAdapter(road_mesh_data, all_faces)

        # Slice in Tiles (nur noch Straßen-Faces - Terrain ist jetzt .ter, kein Mesh mehr)
        tiles_dict = slice_mesh_into_tiles(
            vertices=all_vertices,
            faces=all_faces,
            materials_per_face=materials_per_face,
            tile_size=config.TILE_SIZE,
            vertex_normals=None,
            mesh_obj=road_uv_adapter,
        )
```

- [ ] **Step 5: `.ter`-Export und `TerrainBlock`-Registrierung ergänzen**

Nach dem bestehenden Export-Block (nach `dae_files = export_separate_tile_daes(...)`), vor `# Generiere und füge Materials hinzu`, ergänze:

```python
        # Terrain als .ter exportieren (natives BeamNG-Heightmap statt Mesh)
        from ..terrain.ter_writer import write_ter, encode_heights_to_u16
        from ..terrain.terrain_materials import build_terrain_material_entries

        heights = mesh_data["heightmap"]
        z_min = mesh_data["z_min"]
        max_height = mesh_data["max_height"]
        layer_map = mesh_data["layer_map"]
        terrain_material_names = mesh_data["terrain_material_names"]
        photo_tile_names = mesh_data["photo_tile_names"]

        heightmap_u16 = encode_heights_to_u16(heights, z_min, max_height)
        ter_filename = f"{config.LEVEL_NAME}.ter"
        ter_path = config.BEAMNG_DIR / ter_filename
        write_ter(ter_path, heightmap_u16, layer_map.astype("uint8"), terrain_material_names)
        logger.info(f"  [OK] Terrain exportiert: {ter_filename} ({mesh_data['terrain_size']}x{mesh_data['terrain_size']})")

        terrain_material_entries = build_terrain_material_entries(
            terrain_material_names,
            photo_tile_names,
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            config.LEVEL_NAME,
            config.TILE_SIZE,
        )
        self.materials.add_terrain_materials(terrain_material_entries)

        self.items.add_terrain_block(
            name="theTerrain",
            terrain_filename=ter_filename,
            material_texture_set=f"{config.LEVEL_NAME}TerrainMaterialTextureSet",
            max_height=max_height,
            z_min=z_min,
            origin_x=mesh_data["terrain_origin_x"],
            origin_y=mesh_data["terrain_origin_y"],
            overwrite=True,
        )
```

- [ ] **Step 6: `add_terrain()`-Aufrufe für Terrain-Tiles entfernen**

Entferne den kompletten Block "Erstelle TSStatic-Items für JEDES Tile" (der `self.items.add_terrain(...)`-Aufruf für DAE-Terrain-Tiles) — der bezog sich auf die alten Terrain-DAE-Tiles, die es nicht mehr gibt. Falls `dae_files` weiterhin für Straßen-DAEs existiert, muss geprüft werden, ob dafür ein anderer Registrierungs-Call nötig ist (Straßen-DAEs brauchen weiterhin ein `TSStatic`-Item, analog zu `add_terrain()`, aber mit anderem Namensschema z.B. `road_tile_<coords>` statt `terrain_tile_<coords>`).

- [ ] **Step 7: Vollständige Datei nochmal lesen und auf Konsistenz prüfen**

Run: `python -c "from world_to_beamng.workflow.terrain_workflow import TerrainWorkflow; print('Import OK')"`
Expected: `Import OK` (keine Syntax-/Importfehler)

- [ ] **Step 8: Commit**

```bash
git add world_to_beamng/workflow/terrain_workflow.py
git commit -m "feat: Integrate native .terrain heightmap export into TerrainWorkflow"
```

---

## Task 10: Obsoleten Code entfernen

Diese Module werden ausschließlich von `terrain_workflow.py` konsumiert (verifiziert per Grep über das gesamte Projekt in der Planungsphase — `horizon_workflow.py` nutzt nur `mesh/stitch_boundary.py`, das NICHT gelöscht wird).

**Files:**
- Delete: `world_to_beamng/mesh/stitch_local.py`
- Delete: `world_to_beamng/mesh/stitch_gaps.py`
- Delete: `world_to_beamng/mesh/stitch_terrain_roads.py`
- Delete: `world_to_beamng/mesh/fill_all_mesh_holes.py`
- Modify: `world_to_beamng/builders/mesh_builders.py` (TerrainMeshBuilder-Klasse entfernen)
- Modify: `world_to_beamng/builders/__init__.py` (Export entfernen)
- Modify: `world_to_beamng/geometry/vertices.py` (classify_grid_vertices entfernen)

- [ ] **Step 1: Vor dem Löschen erneut verifizieren, dass keine anderen Konsumenten existieren**

Run: `grep -rln "stitch_local\|stitch_gaps\|stitch_terrain_roads\|fill_all_mesh_holes\|TerrainMeshBuilder\|classify_grid_vertices" --include="*.py" world_to_beamng/ tools/`
Expected: nur noch Treffer in den Dateien, die in diesem Task selbst gelöscht/geändert werden (`builders/mesh_builders.py`, `builders/__init__.py`, `geometry/vertices.py`, und ggf. `mesh/stitch_boundary.py` falls es intern eine dieser Funktionen importiert — dann VOR dem Löschen prüfen und ggf. inline kopieren statt zu löschen).

- [ ] **Step 2: Module löschen**

```bash
git rm world_to_beamng/mesh/stitch_local.py
git rm world_to_beamng/mesh/stitch_gaps.py
git rm world_to_beamng/mesh/stitch_terrain_roads.py
git rm world_to_beamng/mesh/fill_all_mesh_holes.py
```

- [ ] **Step 3: `TerrainMeshBuilder` aus `builders/mesh_builders.py` entfernen**

Lies `world_to_beamng/builders/mesh_builders.py`, finde die `class TerrainMeshBuilder:`-Definition (von `class TerrainMeshBuilder` bis zur nächsten `class`-Definition oder Dateiende) und entferne sie vollständig. `RoadMeshBuilder` und `GridBuilder` in derselben Datei bleiben unverändert.

- [ ] **Step 4: Export in `builders/__init__.py` entfernen**

Lies `world_to_beamng/builders/__init__.py` und entferne `TerrainMeshBuilder` aus Import und `__all__` (falls vorhanden), behalte `RoadMeshBuilder` und `GridBuilder`.

- [ ] **Step 5: `classify_grid_vertices` aus `geometry/vertices.py` entfernen**

Lies `world_to_beamng/geometry/vertices.py`, finde `def classify_grid_vertices(...)` und entferne die Funktion vollständig (samt zugehöriger reiner Hilfsfunktionen, die ausschließlich von ihr aufgerufen werden — vor dem Löschen prüfen mit `grep -n "def \|classify_grid_vertices"` in derselben Datei, ob andere Funktionen in der Datei noch anderweitig gebraucht werden).

- [ ] **Step 6: Verifizieren, dass die Pipeline weiterhin importierbar ist**

Run: `python -c "from world_to_beamng.export.beamng_exporter import BeamNGExporter; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 7: Commit**

```bash
git add -A world_to_beamng/builders/ world_to_beamng/geometry/vertices.py
git commit -m "refactor: Remove obsolete terrain-mesh-stitching code (replaced by native .terrain)"
```

---

## Task 11: `.ter`-Struktur-Validator (unabhängige QA ohne BeamNG)

**Files:**
- Create: `tools/validate_ter.py`

**Interfaces:**
- Consumes: `world_to_beamng.terrain.ter_writer.read_ter`
- Produces: CLI-Script, aufrufbar als `python tools/validate_ter.py <pfad-zur-.ter-datei>`

- [ ] **Step 1: Script implementieren**

Erstelle `tools/validate_ter.py`:

```python
#!/usr/bin/env python3
"""
Validiert eine .ter-Datei strukturell (Format, Wertebereiche), OHNE BeamNG
zu starten. Findet Formatfehler in Sekunden statt nach jedem Test einen
BeamNG-Ladevorgang abzuwarten (siehe Spec Abschnitt 9).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.terrain.ter_writer import read_ter, VALID_SIZES


def validate_ter(path: Path) -> bool:
    print(f"[INFO] Validiere {path}")
    heightmap, layer_map, material_names = read_ter(path)

    ok = True

    size = heightmap.shape[0]
    if size not in VALID_SIZES:
        print(f"[FEHLER] Größe {size} ist keine gültige Zweierpotenz (128-8192)")
        ok = False
    else:
        print(f"[OK] Größe: {size}x{size}")

    if heightmap.shape != layer_map.shape:
        print(f"[FEHLER] heightmap shape {heightmap.shape} != layer_map shape {layer_map.shape}")
        ok = False
    else:
        print(f"[OK] heightmap/layer_map Shapes stimmen überein")

    max_material_index = layer_map[layer_map != 255].max() if (layer_map != 255).any() else -1
    if max_material_index >= len(material_names):
        print(
            f"[FEHLER] layer_map referenziert Material-Index {max_material_index}, "
            f"aber nur {len(material_names)} Materialien vorhanden"
        )
        ok = False
    else:
        print(f"[OK] Alle layer_map-Indizes (max {max_material_index}) haben ein Material ({len(material_names)} total)")

    hole_fraction = (layer_map == 255).mean()
    print(f"[INFO] Hole-Anteil (Wert 255): {hole_fraction:.1%}")

    print(f"[INFO] Höhenwerte (u16 roh): min={heightmap.min()}, max={heightmap.max()}")
    print(f"[INFO] Materialien ({len(material_names)}): {material_names}")

    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/validate_ter.py <pfad-zur-.ter-datei>")
        sys.exit(1)

    success = validate_ter(Path(sys.argv[1]))
    print("\n[✓] VALIDIERUNG BESTANDEN" if success else "\n[!] VALIDIERUNG FEHLGESCHLAGEN")
    sys.exit(0 if success else 1)
```

- [ ] **Step 2: Script testen (nach Task 9 gegen eine echte exportierte Datei)**

Run: `python tools/validate_ter.py "C:\Users\johan\AppData\Local\BeamNG\BeamNG.drive\current\levels\world_to_beamng\world_to_beamng.ter"`
Expected: `[✓] VALIDIERUNG BESTANDEN` (erst ausführbar, nachdem Task 9 einen echten Export produziert hat — siehe Task 12)

- [ ] **Step 3: Commit**

```bash
git add tools/validate_ter.py
git commit -m "feat: Add standalone .ter structural validator (no BeamNG required)"
```

---

## Task 12: Vollständiger Export-Lauf & visuelle Verifikation

Pflicht-Abschlussschritt vor "fertig" (siehe Spec Abschnitt 9) — kann nicht automatisiert werden, erfordert BeamNG.

**Files:** keine (Verifikationsschritt)

- [ ] **Step 1: Alle Unit-Tests laufen lassen**

Run:
```bash
python tests/terrain/test_ter_writer.py
python tests/terrain/test_heightmap.py
python tests/terrain/test_road_embedding.py
python tests/terrain/test_terrain_materials.py
python tests/test_item_manager_terrain_block.py
python tests/test_material_manager_terrain.py
```
Expected: jedes Script gibt `Alle Tests bestanden.` aus.

- [ ] **Step 2: Vollständigen Export laufen lassen**

Run: `python world_to_beamng.py`
Expected: Export läuft ohne Exception durch; Log zeigt `[OK] Terrain exportiert: world_to_beamng.ter (...)`.

- [ ] **Step 3: `.ter`-Struktur validieren**

Run: `python tools/validate_ter.py "<BEAMNG_DIR>\world_to_beamng.ter"` (Pfad aus `config.BEAMNG_DIR`)
Expected: `[✓] VALIDIERUNG BESTANDEN`

- [ ] **Step 4: In BeamNG laden und visuell prüfen**

Level in BeamNG laden. Prüfen:
- Terrain lädt ohne Fehler im `beamng.log` (`grep -i "terrain\|error" beamng.log`)
- Boden zeigt Textur (nicht "no Texture" / gelb)
- Wald-/Wiesen-/Acker-Flächen sind an OSM-Landnutzung erkennbar unterschiedlich texturiert
- Straßen liegen sichtbar auf dem Terrain, kein Schweben, kein Durchstechen
- An mindestens einer bekannten Hangsituation aus dem Testgebiet: Straßen-Querschnitt bleibt horizontal, Böschung zur Umgebung ist sauber (kein Spalt, keine Stufe)
- Heightmap-Orientierung stimmt (Gelände nicht spiegelverkehrt/gedreht — falls doch: siehe Hinweis in `terrain/heightmap.py`, Zeilen-Reihenfolge in `build_heightmap()` ggf. umkehren)

- [ ] **Step 5: Regression-Baseline anlegen**

Da sich das Ausgabeformat komplett ändert (Mesh → Heightmap), gibt es keine alten Golden-Files zum Vergleich (Spec Abschnitt 9). Nach erfolgreicher visueller Verifikation in Step 4, lege einen Baseline-Snapshot für künftige Regressionsprüfungen an:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from world_to_beamng.terrain.ter_writer import read_ter
from world_to_beamng import config

ter_path = config.BEAMNG_DIR / f'{config.LEVEL_NAME}.ter'
heightmap, layer_map, material_names = read_ter(ter_path)

import json
baseline = {
    'terrain_size': int(heightmap.shape[0]),
    'height_min_u16': int(heightmap.min()),
    'height_max_u16': int(heightmap.max()),
    'material_count': len(material_names),
    'material_names': sorted(material_names),
    'hole_fraction': float((layer_map == 255).mean()),
}
Path('tests/terrain/baseline_snapshot.json').write_text(json.dumps(baseline, indent=2))
print('Baseline gespeichert:', baseline)
"
```

```bash
git add tests/terrain/baseline_snapshot.json
git commit -m "test: Add terrain export baseline snapshot for regression tracking"
```

- [ ] **Step 6: Ergebnis dokumentieren**

Bei Erfolg: Plan als abgeschlossen markieren (alle Checkboxen), kurze Zusammenfassung an den Nutzer.
Bei Problemen: systematic-debugging-Skill nutzen, konkreten Fehlerbefund festhalten, ggf. einzelne Tasks nachbessern.
