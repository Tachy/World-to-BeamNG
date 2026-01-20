"""
WORLD-TO-BEAMNG - OSM zu BeamNG Straßen-Generator

Refactored Version mit modularer Architektur.
Main Entry Point für die Anwendung.

Benötigte Pakete:
  pip install requests numpy scipy pyproj pyvista shapely rtree
"""

import sys
import time

# UTF-8 Encoding für Windows Console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from world_to_beamng import config
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()
from world_to_beamng.export import BeamNGExporter
from world_to_beamng.utils.tile_scanner import scan_lgl_tiles, compute_global_center


def main():
    """Hauptfunktion - verwendet neue BeamNGExporter API."""

    start_time = time.time()

    # 1. Konfiguration ist bereits global geladen
    # Anpassungen können direkt gemacht werden:
    # config.GRID_SPACING = 1.5
    # config.ROAD_CLIP_MARGIN = -30.0

    # 2. Exporter initialisieren (ohne Config-Parameter!)
    exporter = BeamNGExporter()

    # 3. Tiles scannen
    logger.info("\n" + "=" * 60)
    logger.info("WORLD-TO-BEAMNG - BeamNG Level Export")
    logger.info("=" * 60)

    tiles = scan_lgl_tiles(dgm1_dir=config.HEIGHT_DATA_DIR)

    if not tiles:
        logger.error("[!] Keine DGM1-Kacheln gefunden - Abbruch")
        return

    # 4. Globalen Offset berechnen
    global_center = compute_global_center(tiles)
    # 3-Tupel: (x, y, z) - z ist der Mittelwert der Höhen oder 0
    global_offset = (global_center[0], global_center[1], global_center[2] if len(global_center) > 2 else 0.0)

    logger.info(f"Gefundene Tiles: {len(tiles)}")
    logger.info(f"Global Offset: {global_offset}")

    # 5. Export durchführen
    stats = exporter.export_complete_level(
        tiles=tiles,
        global_offset=global_offset,
        include_buildings=config.LOD2_ENABLED,
        include_horizon=config.PHASE5_ENABLED,
    )

    # 6. Statistiken
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
