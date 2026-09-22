[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

# 🗺️ World-to-BeamNG

🇬🇧 English version (this file) · 🇩🇪 [Deutsche Version](README_DE.md)

Turns open geodata into a **playable BeamNG.drive level**: terrain, aerial photo, roads, forest, vineyards,
streams and ponds, buildings and a horizon. Alpha status, under development.

## 🎯 What is generated

| Component | Source |
|---|---|
| **Terrain** (native `.terrain`, 1 m grid) with aerial photo texture | DGM1 + DOP20 |
| **Roads** as BeamNG `DecalRoad`, embedded into the terrain | OpenStreetMap + DGM1 |
| **Forest, single trees, vineyards, ground cover** | OpenStreetMap |
| **Streams and ponds** (`River`, `WaterBlock`) | OpenStreetMap + DGM1 |
| **Rubble stone walls** (50 cm thick, stone slabs on top, following the terrain; next to a road they stand on its centerline height) for `barrier=wall`/`retaining_wall` with a `height` tag | OpenStreetMap + DGM1 |
| **Buildings** from LOD2: plastered walls in white/beige/occasionally red, windows, doors, basement windows, roofs with overhang, gravel on flat roofs, church towers with a tower clock | LOD2 (+ OpenStreetMap for churches) |
| **Horizon** up to 50 km (optional) | DGM30 + satellite image |

## 📋 Requirements

| | |
|---|---|
| Operating system | **Windows 10/11** (BeamNG paths under `%LOCALAPPDATA%`, `texconv.exe`) |
| BeamNG.drive | installed and **started at least once** (this creates the user folder) |
| Python | **3.11 or newer**, tested with 3.13 |
| Internet | for OpenStreetMap (Overpass API) and the one-time download of `texconv.exe` |
| Area | Any region with a georeferenced GeoTIFF DEM and orthophoto: CRS and extent are detected automatically from the file content, no fixed file naming or tile size required. Plain ASCII-XYZ point clouds in a ZIP (the LGL Baden-Württemberg format) are recognised the same way, by content, not by name (see "Using data from other regions" below). Buildings (LoD2/CityGML) remain Baden-Württemberg-specific and must be disabled (`LOD2_ENABLED = False`) elsewhere. |
| Disk space | about 250 MB of raw data per 2×2 km tile (see below), plus cache and result |

## 🚀 Quick start

```powershell
# 1. Get the repository
git clone https://github.com/Tachy/World-to-BeamNG.git
cd World-to-BeamNG

# 2. Virtual environment, packages and texconv.exe
python -m venv .venv
.\.venv\Scripts\python.exe setup_project.py

# 3. Put the base data into data/ (see "Base data")
# 4. Set SPAWN_POINT in world_to_beamng/config.py to your own area (see "Configuration")

# 5. One time: take over assets from the BeamNG installation
.\.venv\Scripts\python.exe tools\generate_forest_assets.py
.\.venv\Scripts\python.exe tools\vendor_shared_textures.py

# 6. Generate the level
.\.venv\Scripts\python.exe world_to_beamng.py
```

Then start BeamNG.drive and choose the level **"World to BeamNG"**. The level folder is created automatically in the
BeamNG user folder (`%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\levels\world_to_beamng`), no path has to be set.

`setup_project.py` installs `requirements.txt` into the Python that runs it (here the `.venv`) and downloads
`texconv.exe` (Microsoft DirectXTex) to `bin/`. The program needs `texconv.exe` for all DDS textures.
The tests additionally need `pytest` (`.\.venv\Scripts\pip install pytest`).

## 📦 Base data

The data is **not in the repository** (about 1 GB per 4×4 km). It comes from the
**Landesamt für Geoinformation und Landentwicklung Baden-Württemberg (LGL)** open geodata portal
(<https://opengeodata.lgl-bw.de>) and is put **unchanged** as ZIP files into the folders under `data/`. You have to
create the folders yourself because they are not in Git.

### The area follows from the DGM1 tiles

All products come in **2×2 km tiles**. The file name contains the coordinate of the south-west corner in kilometres
(UTM 32, ETRS89): `…_32_399_5296_…` is easting 399 000 m, northing 5 296 000 m. The program reads **all**
DGM1 ZIPs it finds and processes exactly that area. Aerial photos and buildings must exist for the same tiles.
Example for a 4×4 km area: `399`/`401` × `5296`/`5298`.

### Required

| Folder | Content | File name | Size per tile |
|---|---|---|---|
| `data/DGM1/` | Digital terrain model, 1 m (ZIP with XYZ points) | `dgm1_32_<x>_<y>_2_bw.zip` | approx. 14 MB |
| `data/DOP20/` | Digital orthophotos, 20 cm, RGB (ZIP with TIF + TFW) | `dop20rgb_32_<x>_<y>_2_bw.zip` | approx. 230 MB |

The file name only matters for the LGL BW format above. Any other georeferenced GeoTIFF DEM (loose file or inside a
ZIP) works too, under any file name, and is always resampled to `GRID_SPACING` regardless of its native resolution;
the same applies to any georeferenced orthophoto (embedded GeoTIFF tags or a `.tfw` world file) under `data/DOP20/`.
See "Using data from other regions" below.

Without DGM1 the export aborts ("Keine DGM1-Kacheln gefunden" – no DGM1/GeoTIFF tiles found). If the aerial photo is
missing, the export reports an error in the log.

### Optional

| Folder | Content | File name | If it is missing |
|---|---|---|---|
| `data/LOD2/` | 3D building models LoD2 (ZIP with CityGML) | `LoD2_32_<x>_<y>_2_bw.zip` | no buildings (`LOD2_ENABLED`) |
| `data/DGM30/` | 30 m elevation model as GeoTIFF: **Copernicus DEM GLO-30**, download it yourself (see below). Several `*.tif` files may be in the folder. | any, e.g. `Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif` | the horizon is skipped |
| `data/DOP300/` | Satellite image for the horizon texture: **one** georeferenced RGB GeoTIFF in the resolved source CRS (default UTM 32N/EPSG:25832, or the CRS auto-detected from the DGM1 GeoTIFFs) covering ±50 km around the centre of the area. Any resolution, it is scaled to 8192×8192. | `horizon_temp.tif` (name in `config.SENTINEL2_FILE`) | horizon without texture |

**Downloading DGM30:** The program does not download it. It uses the **Copernicus DEM GLO-30** (30 m, worldwide,
free of charge). The easiest way without an account is the public AWS bucket `copernicus-dem-30m`
(region eu-central-1, Cloud-Optimized GeoTIFFs, about 43 MB per tile). One tile covers 1° × 1°, the name contains its
south-west corner:

```
https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/Copernicus_DSM_COG_10_N47_00_E007_00_DEM/Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif
                                                                                  └ 47° N ┘ └ 7° E ┘
```

You need **all tiles that touch the horizon area** (±50 km around the centre of the area, about ±0.7° in longitude and
±0.45° in latitude). For an area at 47.8° N / 7.7° E these are `N47` and `N48`, each with `E007` and `E008` (four
tiles, about 170 MB in total). Simply put the files into `data/DGM30/`; the program combines them and clips them to the
horizon area. If a tile is missing, the export reports in which compass direction the data ends, and the horizon is
shorter there. With the AWS command-line tool it also works without an account:
`aws s3 cp --no-sign-request s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N47_00_E007_00_DEM/Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif data/DGM30/`.
Alternatives are the Copernicus Data Space Ecosystem (<https://dataspace.copernicus.eu>) and OpenTopography
(<https://opentopography.org>).

**Horizon image:** There is no automatic download for the satellite image, but there is a tool that creates it from
any georeferenced RGB image (e.g. a Sentinel-2 export in Web Mercator or WGS84). It cuts out exactly the horizon area
(±50 km around the centre of the area, which follows from the DGM1 tiles), reprojects it to the resolved source CRS
and writes `data/DOP300/horizon_temp.tif`:

```powershell
.\.venv\Scripts\python.exe tools\make_horizon_image.py C:\path\to\satellite_image.tif
```

If the source image covers only part of the area, the tool warns; the rest stays black. An image in a different
coordinate system in the same folder is not used by the export, only the file `horizon_temp.tif`.

Finished example layout:

```
World-to-BeamNG/
└── data/
    ├── DGM1/    dgm1_32_399_5296_2_bw.zip   dgm1_32_399_5298_2_bw.zip   …
    ├── DOP20/   dop20rgb_32_399_5296_2_bw.zip   …
    ├── LOD2/    LoD2_32_399_5296_2_bw.zip   …
    ├── DGM30/   dgm30_copernicus.tif
    └── DOP300/  horizon_temp.tif
```

### Using data from other regions

`data/DGM1/` and `data/DOP20/` accept any georeferenced GeoTIFF (single-band elevation, RGB orthophoto), loose or
inside a ZIP, under any file name — the coordinate system and the covered area are read from the file itself, not
guessed from a naming scheme. To use data from a region other than Baden-Württemberg:

- Put the DEM GeoTIFF(s) into `data/DGM1/` and the orthophoto GeoTIFF(s) into `data/DOP20/`. The elevation model is
  always resampled to `config.GRID_SPACING` (default 1 m), whatever its native resolution; the orthophotos are
  composited to `config.PHOTO_TILE_SIZE_M`-sized tiles (default 2000 m) at `TERRAIN_BASE_TEX_PIXEL_SIZE` regardless
  of how many source files or what native tiling the source portal uses.
- The source CRS is auto-detected from the GeoTIFFs; `config.SOURCE_CRS_EPSG` is only the fallback for elevation data
  without an embedded CRS (plain ASCII-XYZ point clouds, e.g. the LGL BW format) and otherwise ignored. If the
  orthophoto's CRS differs from the elevation data's CRS, it is reprojected automatically.
- Set `LOD2_ENABLED = False` in `config.py` — buildings (LoD2/CityGML) remain specific to the Baden-Württemberg
  CityGML 1.0 schema.
- All DGM1 GeoTIFFs must share the same CRS (mixing different elevation CRS is not supported); DGM30/horizon data
  stays worldwide-capable as before (see "Optional" above).

Tested end-to-end with real-world data outside Baden-Württemberg: swissALTI3D (0.5 m DEM) and SwissImage DOP10
(0.1 m orthophoto), both EPSG:2056 (CH1903+/LV95), loose GeoTIFFs with no fixed tile size.

### What the program fetches itself

- **OpenStreetMap** (roads, forest, water, land use, churches) through the Overpass API with fallback servers. The
  responses are then stored in `cache/`.
- **Elevations for terrain, roads and water** come from DGM1, those of the buildings from LOD2; there is no download
  for them. Only the horizon needs DGM30 (see above).
- **Plaster and window textures** of the buildings are generated by the program itself. Textures that are created once
  (roof gravel, and the rubble stone wall from a photo) live in `data/textures/` in the repository, one folder per
  texture plus `manifest.json`; the export only converts them to DDS. `textures/registry.py` lists which textures the
  export needs and checks them first: missing procedural ones (gravel) are created once, a missing photo texture
  (rubble stone wall) **aborts the export** with the command to create it. New ones: `tools\make_seamless_texture.py` (photo)
  or `tools\generate_gravel_texture.py`.

### From the BeamNG installation

The installation path is read from `%LOCALAPPDATA%\BeamNG\BeamNG.drive.ini`. BeamNG's own content that must not be put
into the repository is taken from there:

| What | How | If it is missing |
|---|---|---|
| Tree models and `managedItemData.json` | once, `tools\generate_forest_assets.py` | no forest (warning in the log) |
| Standard textures (roads, roof tiles) | once, `tools\vendor_shared_textures.py` | BeamNG shows "no Texture" |
| Grape vines from the `italy` level | automatically during the export | vineyards without vines |

The scripts can be repeated. Run them again after a BeamNG update or if the level folder has been deleted.

## ⚙️ Configuration

All settings are in `world_to_beamng/config.py`.

| Setting | Meaning |
|---|---|
| **`SPAWN_POINT`** | Start position as `(latitude, longitude)` in degrees. Must be inside your own area. |
| `LOD2_ENABLED`, `FORESTS_ENABLED`, `VINEYARDS_ENABLED`, `WATER_ENABLED`, `GROUND_COVER_ENABLED`, `PHASE5_ENABLED` | switch individual components on and off (`PHASE5_ENABLED` is the horizon) |
| `BEAMNG_DIR` | Target folder of the level; derived from `%LOCALAPPDATA%`, only change it in special cases |
| `GRID_SPACING` | Terrain resolution in metres (default 1.0); GeoTIFF elevation data is always resampled to this, whatever its native resolution |
| `TERRAIN_BASE_TEX_PIXEL_SIZE` | Size of the aerial photo per tile |
| `PHOTO_TILE_SIZE_M` | Tile size (metres) of the aerial-photo/material grid (default 2000), independent of the source data's own tiling |
| `SOURCE_CRS_EPSG` | Fallback source CRS (default 25832) for elevation data without an embedded CRS (plain ASCII-XYZ); ignored for GeoTIFF sources, whose CRS is auto-detected |
| `ENV_DATE`, `ENV_CLOCK_TIME` | Date and time of day for the position of the sun |

## ⏱️ Process and duration

A run reads the DGM1 tiles, downloads the OSM data (the first time), builds the aerial photo and the terrain, and
writes roads, forest, water, buildings and horizon into the level folder. For 4×4 km a run with filled caches takes
about one minute. The first run takes longer because OSM is downloaded and the caches are built. The caches
(`cache/`) become invalid automatically when the data changes. If results look strange, deleting `cache/` helps.

## 🐛 Troubleshooting

The program's messages are in German; they are quoted as they appear, followed by a translation.

| Message / symptom | Cause and solution |
|---|---|
| `Keine DGM1-Kacheln gefunden` (no DGM1 tiles found) | `data/DGM1/` is missing, empty, or contains no readable ZIPs/GeoTIFFs (a GeoTIFF without a coordinate system is skipped with a warning) |
| `texconv.exe nicht gefunden` (texconv.exe not found) | `setup_project.py` has not run; or put the file manually at `bin\texconv.exe` |
| `BeamNG.drive.ini nicht gefunden` (BeamNG.drive.ini not found) | BeamNG.drive has never been started |
| `managedItemData.json nicht gefunden` (not found) | run `tools\generate_forest_assets.py` once |
| Roads or roofs with "no Texture" | run `tools\vendor_shared_textures.py` once |
| Level does not appear in BeamNG | check whether `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\levels\world_to_beamng` was created; otherwise adjust `BEAMNG_DIR` in `config.py` |
| `DGM30-Dateien decken die Horizont-Fläche … nicht ab` (DGM30 files do not cover the horizon area) | put the tiles for the compass direction named in the message into `data/DGM30/` (see above) |
| `Keine DGM30-Dateien` (no DGM30 files) | `data/DGM30/` is empty; the horizon is skipped otherwise |
| Horizon without texture or shifted | use `tools\make_horizon_image.py`; the file must show exactly the horizon area in the resolved source CRS |
| Crash or error while loading the level | check `C:\Users\<NAME>\AppData\Local\BeamNG\BeamNG.drive\current\beamng.log` for `\|E\|` lines |
| OSM timeout | the program tries fallback servers; start again, successful responses are cached |

## 🧪 Tests

```powershell
.\.venv\Scripts\pip install pytest
.\.venv\Scripts\python.exe -m pytest tests
```

## 🏗️ Project structure

```
world_to_beamng.py        entry point
setup_project.py          install packages and texconv.exe
world_to_beamng/
├── config.py             all settings
├── export/               level export (BeamNGExporter)
├── workflow/             terrain, building, forest and horizon workflow
├── terrain/              elevation model, road embedding, aerial photo, horizon
├── osm/                  OpenStreetMap download and evaluation
├── io/                   LOD2 reading, aerial photo, caches
├── facade/               buildings: plaster walls, windows, roofs, church towers
├── forest/               forest, vineyards, ground cover
├── walls/                rubble stone walls (OSM barrier=wall with height), stone slabs on top
├── textures/             texture library (data/textures) and tools to create tiling textures
├── managers/             materials, level objects, DAE export
├── builders/, core/      mesh builders, cache management
└── geometry/, mesh/, analysis/, utils/
data/                     configuration JSONs (in the repository) and base data (not in the repository)
tools/                    helper scripts (take over assets, create the horizon image, checks, viewer)
tests/                    pytest
docs/                     technical documentation (e.g. MATERIAL_TEMPLATES.md)
```

## 📄 License and attribution

The code is under the **MIT License** (see [LICENSE](LICENSE)). For the data the following applies:

- **LGL Baden-Württemberg** (DGM1, DOP20, LoD2): **Data licence Germany – Attribution – Version 2.0**
  (dl-de/by-2.0, <https://www.govdata.de/dl-de/by-2-0>). The source note reads:
  *"Datenquelle: LGL, www.lgl-bw.de, dl-de/by-2-0"* (data source), plus a note that the data has been modified.
  Anyone who passes on generated levels has to state this. The licence texts are also contained in the ZIP files.
- **OpenStreetMap**: © OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright).
- **Copernicus DEM GLO-30** (horizon): free to use under the terms of the Copernicus licence, see
  <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM>.
- **BeamNG content** (textures, trees) remains the property of BeamNG and is only copied from your own installation
  into your own level. It does not belong in the repository.

## 🤝 Contributing

Contributions are welcome: fork, create a branch, commit your changes (prefixes `feat:`, `fix:`, `perf:` …), open a
pull request. Please report bugs and wishes as an [issue](https://github.com/Tachy/World-to-BeamNG/issues).

## 🙏 Acknowledgements

**OpenStreetMap** and the **Overpass API**, the **LGL Baden-Württemberg** for the open geodata, **BeamNG**, and the
communities of **Shapely**, **NumPy**, **SciPy** and **Rasterio**.
