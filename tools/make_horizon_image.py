"""
Erzeugt das Horizont-Bild (data/DOP300/horizon_temp.tif) aus einem beliebigen georeferenzierten Satellitenbild.

Der Horizont braucht ein GeoTIFF in UTM 32N (EPSG:25832), das GENAU die Fläche ±config.HORIZON_HALF_SIZE_M um die
Gebietsmitte zeigt. Dieses Werkzeug schneidet sie aus dem Quellbild aus (z. B. ein Sentinel-2-RGB-Export in Web-Mercator
oder WGS84) und projiziert sie um. Die Gebietsmitte ergibt sich wie beim Export aus den DGM1-Kacheln in data/DGM1/.

Aufruf:
    python tools/make_horizon_image.py <quellbild.tif>
    python tools/make_horizon_image.py <quellbild.tif> --ausgabe data/DOP300/horizon_temp.tif --groesse 8192
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from world_to_beamng import config
from world_to_beamng.geometry import coordinates
from world_to_beamng.terrain.horizon_image import build_horizon_image, horizon_area
from world_to_beamng.utils.tile_scanner import compute_global_center, resolve_source_crs_epsg, scan_elevation_tiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Horizont-Bild aus einem georeferenzierten RGB-Bild erzeugen")
    parser.add_argument("quelle", type=Path, help="georeferenziertes RGB-Bild (GeoTIFF, beliebiges Koordinatensystem)")
    parser.add_argument("--ausgabe", type=Path, default=config.DOP300_DATA_DIR / config.SENTINEL2_FILE, help="Zieldatei")
    parser.add_argument("--groesse", type=int, default=config.HORIZON_IMAGE_SIZE_PX, help="Kantenlänge in Pixeln")
    parser.add_argument("--resampling", default="bilinear", help="bilinear, cubic, average, ... (rasterio)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tiles = scan_elevation_tiles(dgm_dir=config.HEIGHT_DATA_DIR)
    if not tiles:
        print(f"[!] Keine DGM1-Kacheln in {config.HEIGHT_DATA_DIR} - ohne sie ist die Gebietsmitte unbekannt.")
        return 1
    coordinates.set_source_crs(resolve_source_crs_epsg(tiles))
    if not args.quelle.is_file():
        print(f"[!] Quellbild nicht gefunden: {args.quelle}")
        return 1
    if args.ausgabe.resolve() == args.quelle.resolve():
        print("[!] Quelle und Ausgabe sind dieselbe Datei.")
        return 1

    center = compute_global_center(tiles)
    area = horizon_area(center)
    print(f"Gebietsmitte (UTM 32N): {center[0]:.0f} / {center[1]:.0f}")
    print(f"Horizont-Fläche: X {area[0]:.0f}..{area[1]:.0f}, Y {area[2]:.0f}..{area[3]:.0f}")

    try:
        coverage = build_horizon_image(args.quelle, args.ausgabe, area, args.groesse, args.resampling)
    except ValueError as error:
        print(f"[!] {error}")
        return 1

    print(f"[OK] {args.ausgabe} geschrieben ({args.groesse}x{args.groesse}, Quellbild deckt {coverage:.0%} ab)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
