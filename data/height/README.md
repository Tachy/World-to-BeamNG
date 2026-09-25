# data/height

Elevation data (digital terrain model, 1 m resolution). **Required** - without data here the export
aborts immediately ("no DGM1 tiles found").

- LGL Baden-Württemberg: ZIP files with an ASCII XYZ point cloud, named `dgm1_32_<x>_<y>_2_bw.zip`.
- Other regions: one or more georeferenced GeoTIFF DEMs, any file name, loose or inside a ZIP.

The export area is derived from the tiles placed here.
