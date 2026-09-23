"""
WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator

Refactored Version mit modularer Architektur.
Main Entry Point für die Anwendung.

Benötigte Pakete:
  pip install requests numpy scipy pyproj pyvista shapely rtree rich
"""

import sys
import time

# UTF-8 Encoding für Windows Console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from world_to_beamng import config
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()
from world_to_beamng.export import BeamNGExporter
from world_to_beamng.textures.registry import MissingTexturesError
from world_to_beamng.geometry import coordinates
from world_to_beamng.progress import Pipeline
from world_to_beamng.utils.tile_scanner import scan_elevation_tiles, compute_global_center, resolve_source_crs_epsg


def main():
    """Hauptfunktion - verwendet neue BeamNGExporter API."""

    start_time = time.time()

    pipeline = Pipeline()
    exporter = BeamNGExporter(pipeline)

    with pipeline.task("Vorbereitung") as task:
        tiles = scan_elevation_tiles(dgm_dir=config.HEIGHT_DATA_DIR)

        if not tiles:
            task.fail("keine DGM1-Kacheln gefunden")
            return

        # Quell-CRS auflösen (aus GeoTIFF-Kacheln automatisch erkannt, sonst config.SOURCE_CRS_EPSG) -
        # MUSS vor jeder weiteren Koordinatentransformation gesetzt werden (OSM-BBox, LoD2, Horizont, ...)
        source_epsg = resolve_source_crs_epsg(tiles)
        coordinates.set_source_crs(source_epsg)

        global_center = compute_global_center(tiles)
        # 3-Tupel: (x, y, z) - z ist der Mittelwert der Höhen oder 0
        global_offset = (global_center[0], global_center[1], global_center[2] if len(global_center) > 2 else 0.0)

        task.done(f"{len(tiles)} Tiles, EPSG:{source_epsg}, Offset {global_offset}")

    # Export durchführen
    try:
        stats = exporter.export_complete_level(
            tiles=tiles,
            global_offset=global_offset,
            include_buildings=config.LOD2_ENABLED,
            include_horizon=config.PHASE5_ENABLED,
        )
    except MissingTexturesError as error:  # Foto-Textur fehlt: klare Meldung statt Traceback, Exit-Code 1
        logger.error(f"\n[!] Export abgebrochen:\n{error}")
        sys.exit(1)

    # Statistiken
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("EXPORT ABGESCHLOSSEN")
    logger.info(f"{'='*60}")
    logger.info(f"Tiles verarbeitet: {stats['tiles_processed']}")
    logger.info(f"Tiles fehlgeschlagen: {stats['tiles_failed']}")
    logger.info(f"Gebäude exportiert: {stats['buildings_exported']}")
    logger.info(f"Horizon exportiert: {'Ja' if stats['horizon_exported'] else 'Nein'}")
    logger.info(f"Gesamtzeit: {elapsed:.1f}s")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
