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

# 4. Generate the level
.\.venv\Scripts\python.exe world_to_beamng.py
```

Then start BeamNG.drive and choose the level **"World to BeamNG"**. The level folder is created automatically in the
BeamNG user folder (`%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\levels\world_to_beamng`), no path has to be set.

`setup_project.py` installs `requirements.txt` into the Python that runs it (here the `.venv`) and downloads
`texconv.exe` (Microsoft DirectXTex) to `bin/`. The program needs `texconv.exe` for all DDS textures; if it is
missing, the export downloads it itself on first use.
The tests additionally need `pytest` (`.\.venv\Scripts\pip install pytest`).

## 📦 Base data

The data is **not in the repository** (about 1 GB per 4×4 km). It comes from the
**Landesamt für Geoinformation und Landentwicklung Baden-Württemberg (LGL)** open geodata portal
(<https://opengeodata.lgl-bw.de>) and is put **unchanged** as ZIP files into the folders under `data/`. The three
folders (`data/height/`, `data/satellite/`, `data/buildings/`) already exist in the repository, each with a short
`README.md` describing what belongs there; only their contents are ignored by Git.

### The area follows from the DGM1 tiles

All products come in **2×2 km tiles**. The file name contains the coordinate of the south-west corner in kilometres
(UTM 32, ETRS89): `…_32_399_5296_…` is easting 399 000 m, northing 5 296 000 m. The program reads **all**
DGM1 ZIPs it finds and processes exactly that area. Aerial photos and buildings must exist for the same tiles.
Example for a 4×4 km area: `399`/`401` × `5296`/`5298`.

### Required

| Folder | Content | File name | Size per tile |
|---|---|---|---|
| `data/height/` | Digital terrain model, 1 m (ZIP with XYZ points) | `dgm1_32_<x>_<y>_2_bw.zip` | approx. 14 MB |
| `data/satellite/` | Digital orthophotos, 20 cm, RGB (ZIP with TIF + TFW) | `dop20rgb_32_<x>_<y>_2_bw.zip` | approx. 230 MB |

The file name only matters for the LGL BW format above. Any other georeferenced GeoTIFF DEM (loose file or inside a
ZIP) works too, under any file name, and is always resampled to `GRID_SPACING` regardless of its native resolution;
the same applies to any georeferenced orthophoto (embedded GeoTIFF tags or a `.tfw` world file) under `data/satellite/`.
See "Using data from other regions" below.

Without DGM1 the export aborts ("no DGM1 tiles found"). If the aerial photo is
missing, the export reports an error in the log.

### Optional

| Folder | Content | File name | If it is missing |
|---|---|---|---|
| `data/buildings/` | 3D building models LoD2 (ZIP with CityGML) | `LoD2_32_<x>_<y>_2_bw.zip` | no buildings (`LOD2_ENABLED`) |

Finished example layout:

```
World-to-BeamNG/
└── data/
    ├── height/     dgm1_32_399_5296_2_bw.zip   dgm1_32_399_5298_2_bw.zip   …
    ├── satellite/  dop20rgb_32_399_5296_2_bw.zip   …
    └── buildings/  LoD2_32_399_5296_2_bw.zip   …
```

### Horizon (optional, `PHASE5_ENABLED`)

The horizon up to 50 km needs two things — 30 m elevation (**Copernicus DEM GLO-30**) and a satellite image
(**Sentinel-2 cloudless** via the EOX WMS) — and both are fetched **fully automatically** on the first run
(`config.DGM30_AUTO_DOWNLOAD` / `config.EOX_AUTO_DOWNLOAD`, both `True` by default). Nothing to download or place by
hand; both live under `cache/` (not `data/`) because the program manages them entirely on its own:

- `cache/dgm30/` — the raw Copernicus DEM GLO-30 tiles (`Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif` etc., one per
  1°×1° tile, about 43 MB each). If a tile is missing (e.g. no internet), the export reports in which compass
  direction the data ends, and the horizon is shorter there.
- `cache/horizon_source/` — the raw Sentinel-2 mosaic before cropping, and `cache/horizon_texture/` — the final,
  cropped texture actually used. Both are keyed by area (+ target size/resampling for the texture), so switching the
  source region (`data/height/` etc.) and back never mixes up the two areas' images.

All three are ordinary caches: safe to delete any time, rebuilt automatically on the next run. If you ever need a
tile that the automatic download couldn't get (e.g. offline use), you can still place a Copernicus DEM GLO-30 GeoTIFF
into `cache/dgm30/` by hand — the public AWS bucket `copernicus-dem-30m` (region eu-central-1) has one, named by its
south-west corner:

```
https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/Copernicus_DSM_COG_10_N47_00_E007_00_DEM/Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif
                                                                                  └ 47° N ┘ └ 7° E ┘
```
(`aws s3 cp --no-sign-request s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N47_00_E007_00_DEM/Copernicus_DSM_COG_10_N47_00_E007_00_DEM.tif cache/dgm30/`
also works without an account.) The horizon satellite image has no such manual fallback — it is Sentinel-2 cloudless
only.

The Sentinel-2 cloudless imagery is © **EOX IT Services GmbH** (<https://cloudless.eox.at>), licensed under
**CC BY-NC-SA 4.0** (non-commercial use, attribution + share-alike required) — see "License and attribution" below
for the exact required attribution text.

### Using data from other regions

`data/height/` and `data/satellite/` accept any georeferenced GeoTIFF (single-band elevation, RGB orthophoto), loose or
inside a ZIP, under any file name — the coordinate system and the covered area are read from the file itself, not
guessed from a naming scheme. To use data from a region other than Baden-Württemberg:

- Put the DEM GeoTIFF(s) into `data/height/` and the orthophoto GeoTIFF(s) into `data/satellite/`. The elevation model is
  always resampled to `config.GRID_SPACING` (default 1 m), whatever its native resolution; the orthophotos are
  composited to `config.PHOTO_TILE_SIZE_M`-sized tiles (default 2000 m) at `TERRAIN_BASE_TEX_PIXEL_SIZE` regardless
  of how many source files or what native tiling the source portal uses.
- The source CRS is auto-detected from the GeoTIFFs; `config.SOURCE_CRS_EPSG` is only the fallback for elevation data
  without an embedded CRS (plain ASCII-XYZ point clouds, e.g. the LGL BW format) and otherwise ignored. If the
  orthophoto's CRS differs from the elevation data's CRS, it is reprojected automatically.
- Set `LOD2_ENABLED = False` in `config.py` — buildings (LoD2/CityGML) remain specific to the Baden-Württemberg
  CityGML 1.0 schema.
- Elevation GeoTIFFs in different CRSs can be mixed: the area is processed in the CRS of most tiles, the others are
  reprojected automatically. The horizon's DGM30/satellite auto-download stays worldwide-capable (see "Horizon" above).

Tested end-to-end with real-world data outside Baden-Württemberg: swissALTI3D (0.5 m DEM) and SwissImage DOP10
(0.1 m orthophoto), both EPSG:2056 (CH1903+/LV95), loose GeoTIFFs with no fixed tile size.

### What the program fetches itself

- **OpenStreetMap** (roads, forest, water, land use, churches) through the Overpass API with fallback servers. The
  responses are then stored in `cache/`.
- **Elevations for terrain, roads and water** come from DGM1, those of the buildings from LOD2; there is no download
  for them. Only the horizon's DGM30 + satellite image are fetched automatically (see "Horizon" above).
- **Plaster and window textures** of the buildings are generated by the program itself. Textures that are created once
  (roof gravel, and the rubble stone wall from a photo) live in `data/textures/` in the repository, one folder per
  texture plus `manifest.json`; the export only converts them to DDS. `textures/registry.py` lists which textures the
  export needs and checks them first: missing procedural ones (gravel) are created once, a missing photo texture
  (rubble stone wall) **aborts the export** with the command to create it. New photo textures:
  `tools\make_seamless_texture.py`.

### From the BeamNG installation

The installation path is read from `%LOCALAPPDATA%\BeamNG\BeamNG.drive.ini`. BeamNG's own content that must not be put
into the repository is taken from there:

| What | How | If it is missing |
|---|---|---|
| Tree models (`east_coast_usa`) and `managedItemData.json` | automatically during the export | no forest (warning in the log) |
| Standard textures (roads, roof tiles, terrain detail) | automatically during the export | BeamNG shows "no Texture" |
| Grape vines from the `italy` level | automatically during the export | vineyards without vines |

Everything is copied only when it is missing or the BeamNG content changed (e.g. after an update), so later exports
skip this step. The forest type templates in `data/osm_to_beamng.json` can be regenerated from the tree models with
`tools\generate_forest_types.py` (only needed when the tree selection should change).

## ⚙️ Configuration

All settings are in `world_to_beamng/config.py`.

| Setting | Meaning |
|---|---|
| `LOD2_ENABLED`, `FORESTS_ENABLED`, `VINEYARDS_ENABLED`, `WATER_ENABLED`, `GROUND_COVER_ENABLED`, `PHASE5_ENABLED` | switch individual components on and off (`PHASE5_ENABLED` is the horizon) |
| `BEAMNG_DIR` | Target folder of the level; derived from `%LOCALAPPDATA%`, only change it in special cases |
| `GRID_SPACING` | Terrain resolution in metres (default 1.0); GeoTIFF elevation data is always resampled to this, whatever its native resolution |
| `TERRAIN_BASE_TEX_PIXEL_SIZE` | Size of the aerial photo per tile |
| `PHOTO_TILE_SIZE_M` | Tile size (metres) of the aerial-photo/material grid (default 2000), independent of the source data's own tiling |
| `SOURCE_CRS_EPSG` | Fallback source CRS (default 25832) for elevation data without an embedded CRS (plain ASCII-XYZ); ignored for GeoTIFF sources, whose CRS is auto-detected |
| `SUN_REFERENCE_LATLON` | Approximate `(latitude, longitude)` used only for the sun's position (date/time of day) - **not** the vehicle spawn point, which is placed automatically (see below) |
| `ENV_DATE`, `ENV_CLOCK_TIME` | Date and time of day for the position of the sun |
| `DGM30_AUTO_DOWNLOAD`, `EOX_AUTO_DOWNLOAD` | automatically fetch the DGM30 tiles / the Sentinel-2 horizon image on the first run (default `True` each); `False` disables that source and degrades to the existing skip behaviour (horizon skipped / horizon without texture) |
| `DGM30_S3_BUCKET`, `DGM30_S3_REGION`, `DGM30_FETCH_MAX_RETRIES`, `DGM30_FETCH_TIMEOUT_S`, `DGM30_NOT_FOUND_CACHE_TTL_DAYS` | tuning for the Copernicus DEM GLO-30 auto-download (bucket/region, retry/timeout, how long a confirmed-missing tile — e.g. open sea — is remembered before retrying) |
| `EOX_WMS_URL`, `EOX_WMS_LAYER`, `EOX_WMS_VERSION`, `EOX_WMS_FORMAT`, `EOX_MAX_REQUEST_PX`, `EOX_TARGET_RESOLUTION_M`, `EOX_MOSAIC_MAX_PX`, `EOX_FETCH_MARGIN_FACTOR`, `EOX_FETCH_MAX_RETRIES`, `EOX_FETCH_TIMEOUT_S`, `EOX_KEEP_RAW_MOSAIC`, `EOX_MOSAIC_CACHE_DIR`, `EOX_TEXTURE_CACHE_DIR` | tuning for the EOX Sentinel-2 cloudless WMS auto-download (endpoint/layer/version, per-request tile size limit, target resolution, mosaic size cap, raw-mosaic and final-texture caches) |

See "Horizon" above for the `cache/dgm30/`, `cache/horizon_source/` and `cache/horizon_texture/` caches these
settings tune — all three are safe to delete any time and get rebuilt automatically.

**Vehicle spawn:** the car spawns automatically on whichever road is closest to the centre of the exported area,
facing one of the two directions along that road (arbitrary) — nothing to configure.

## ⏱️ Process and duration

A run reads the DGM1 tiles, downloads the OSM data (the first time), builds the aerial photo and the terrain, and
writes roads, forest, water, buildings and horizon into the level folder. For 4×4 km a run with filled caches takes
about one minute. The first run takes longer because OSM is downloaded and the caches are built. The caches
(`cache/`) become invalid automatically when the data changes. If results look strange, deleting `cache/` helps.

## 🐛 Troubleshooting

| Message / symptom | Cause and solution |
|---|---|
| `no DGM1 tiles found` | `data/height/` is missing, empty, or contains no readable ZIPs/GeoTIFFs (a GeoTIFF without a coordinate system is skipped with a warning) |
| `texconv.exe not found … download failed` | no internet at the first DDS conversion; download `texconv.exe` manually into `bin\` |
| `BeamNG.drive.ini not found` | BeamNG.drive has never been started |
| `Tree assets not available …` / no forest | BeamNG installation not found (`BeamNG.drive.ini`) or `east_coast_usa.zip` missing in `content/levels` |
| Roads or roofs with "no Texture" | check the warning `… stock texture(s) not found in the BeamNG content zips` in the log |
| Level does not appear in BeamNG | check whether `%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\levels\world_to_beamng` was created; otherwise adjust `BEAMNG_DIR` in `config.py` |
| `The DGM30 files do not cover the horizon area …` | a tile the auto-download couldn't get (e.g. no internet for that request) — retries automatically on the next run, or place the tile for the named compass direction into `cache/dgm30/` by hand (see "Horizon" above) |
| `No DGM30 files (*.tif) in …` | `cache/dgm30/` is empty and the auto-download hasn't run yet or failed entirely; the horizon is skipped otherwise |
| `DGM30 auto-download failed` / Sentinel-2 auto-download not working, no internet | the auto-download logs a warning and falls back to the existing skip behaviour (horizon skipped / horizon without texture) — it never aborts the whole export; it retries on the next run once the network is back |
| Horizon without texture or shifted | delete `cache/horizon_texture/` (and `cache/horizon_source/` if the raw mosaic itself looks wrong) to force a rebuild on the next run |
| Crash or error while loading the level | check `C:\Users\<NAME>\AppData\Local\BeamNG\BeamNG.drive\current\beamng.log` for `\|E\|` lines |
| OSM timeout | the program tries fallback servers; start again, successful responses are cached |

## 🧪 Tests

```powershell
.\.venv\Scripts\pip install pytest
.\.venv\Scripts\python.exe -m pytest tests
```

## 🔍 Level viewer

A developer viewer shows the exported level outside BeamNG (terrain with aerial photo, roads and markings, bridges,
tunnels, water, forest, tunnel zones, spawn points, horizon, debug network). By default it shows the terrain at
full resolution with the full aerial photo tiles (loads in a few seconds, needs about 4 GB of RAM):

```powershell
.\.venv\Scripts\python.exe -m tools.level_viewer                # level from config.BEAMNG_DIR
.\.venv\Scripts\python.exe -m tools.level_viewer --terrain-step 4 --minimap-photo   # faster, less memory
.\.venv\Scripts\python.exe -m tools.level_viewer --help         # all options (layers, terrain step, screenshot)
```

Mouse: left drag turns around the vertical axis and tilts the view (the terrain never rolls), shift + left drag or
the middle button pans, the wheel or right drag zooms. **Double-click** an object to see its name, class, material,
position and the terrain height below it; the camera flies there (40 m away). `space` frames the whole selected
object, `Esc` clears it, `f` flies to the point under the cursor, `Up`/`Down` zoom, `r` resets the camera, `v` shows
an oblique overview, `+`/`-` make lines and points thicker/thinner. `i` shows or hides this reference in the window.

Keys: `g` terrain, `x` photo/elevation colors, `a` roads, `m` markings, `o` water, `b` structures, `h` horizon,
`c` forest, `z` zones + spawns, `n` labels, `d` debug network, `l` reload, `i` help, `q` quit. Window, camera and
layers are stored in `tools/level_viewer.cfg`.

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
tools/                    helper scripts (take over assets, create the horizon image, checks, level_viewer/)
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
- **Sentinel-2 cloudless (s2cloudless)** (horizon background image): © EOX IT Services GmbH,
  <https://cloudless.eox.at>, licensed under CC BY-NC-SA 4.0 (non-commercial use, attribution + share-alike
  required) — see <https://cloudless.eox.at/documentation/license>. Required attribution: "EOxCloudless
  https://cloudless.eox.at by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2025)". Commercial
  use requires a separate EOX Commercial Attribution-RestrictedUse licence.
- **BeamNG content** (textures, trees) remains the property of BeamNG and is only copied from your own installation
  into your own level. It does not belong in the repository.

## 🤝 Contributing

Contributions are welcome: fork, create a branch, commit your changes (prefixes `feat:`, `fix:`, `perf:` …), open a
pull request. Please report bugs and wishes as an [issue](https://github.com/Tachy/World-to-BeamNG/issues).

### Versions and releases

The commit prefixes drive the version number ([Conventional Commits](https://www.conventionalcommits.org/)): `feat:` raises
the minor number, `fix:` and `perf:` the patch number, `feat!:` / `BREAKING CHANGE:` the major number (below 1.0 the
minor number). A GitHub Action ([release-please](https://github.com/googleapis/release-please)) keeps a release pull
request up to date on `main`; merging it creates the tag (`vX.Y.Z`), the GitHub release with the changelog
([CHANGELOG.md](CHANGELOG.md)) and updates `__version__`. The version is shown when an export finishes and in the level
viewer.

## 🙏 Acknowledgements

**OpenStreetMap** and the **Overpass API**, the **LGL Baden-Württemberg** for the open geodata, **BeamNG**, and the
communities of **Shapely**, **NumPy**, **SciPy** and **Rasterio**.
