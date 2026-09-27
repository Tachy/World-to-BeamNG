# data/buildings

3D building models. **Optional** - controlled by `config.LOD2_ENABLED`. Without data here (or with
`LOD2_ENABLED = False`) the export simply contains no buildings; it does not abort.

The format is recognised from the file contents, not from the file name:

- LGL Baden-Württemberg: ZIP files with CityGML 1.0 (LoD2), e.g. `LoD2_32_<x>_<y>_2_bw.zip`.
- swisstopo swissBUILDINGS3D 2.0: ASCII DXF, loose or in ZIP files, e.g.
  `swissbuildings3d_2_2023-05_1251-24_2056_5728.dxf.zip`. The layers (object types) decide what is built, see
  `config.SWISSBUILDINGS_LAYER_KINDS`.

If the source already models the roof overhangs (swissBUILDINGS3D), they are taken as they are and no overhang is
computed for the whole import; otherwise (LoD2 from Baden-Württemberg) it is computed.
