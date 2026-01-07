#!/usr/bin/env python
from world_to_beamng.utils.tile_scanner import scan_lgl_tiles

tiles = scan_lgl_tiles('data/DGM1')
print(f"\nGefundene Tiles: {len(tiles)}")
for t in tiles:
    print(f"  {t['filename']}")
