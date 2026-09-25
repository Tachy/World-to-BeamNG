# data/satellite

Aerial/satellite images (RGB orthophotos) for the terrain texture. **Required** - without data here the
terrain only gets a green fill color instead of a photo texture, which is not a usable result.

- LGL Baden-Württemberg: ZIP files with TIF + TFW, named `dop20rgb_32_<x>_<y>_2_bw.zip`.
- Other regions: one or more georeferenced orthophotos (embedded GeoTIFF tag or accompanying `.tfw`
  file), any file name, loose or inside a ZIP.

Must cover the same area as `data/height`.
