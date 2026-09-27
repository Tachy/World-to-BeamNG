# data/satellite

Aerial/satellite images (RGB orthophotos) for the terrain texture. **Required** - without data here the
terrain only gets a green fill color instead of a photo texture, which is not a usable result.

- LGL Baden-Württemberg: ZIP files with TIF + TFW, named `dop20rgb_32_<x>_<y>_2_bw.zip`.
- Hessen: JPG + JGW, e.g. `dop20_32_<x>_<y>_1_he.jpg` with `dop20_32_<x>_<y>_1_he.jgw`.
- Other regions: one or more georeferenced orthophotos (TIF, JPG or PNG with embedded GeoTIFF tags or an
  accompanying world file: `.tfw`, `.jgw`, `.pgw`, `.wld`), any file name, loose or inside a ZIP. A world file has
  no coordinate system: the elevation data's CRS is assumed.

Must cover the same area as `data/height`.
