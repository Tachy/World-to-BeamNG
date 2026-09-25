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
from world_to_beamng.progress import Pipeline, console
from world_to_beamng.utils.tile_scanner import scan_elevation_tiles, compute_global_center, resolve_source_crs_epsg


def main():
    """Hauptfunktion - verwendet neue BeamNGExporter API."""

    start_time = time.time()

    pipeline = Pipeline()
    exporter = BeamNGExporter(pipeline)

    with pipeline.task("Preparation") as task:
        tiles = scan_elevation_tiles(dgm_dir=config.HEIGHT_DATA_DIR)

        if not tiles:
            task.fail("no DGM1 tiles found")
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
    except MissingTexturesError:
        # Die volle Fehlermeldung steht bereits in der "✗ Texturen - ..."-Zeile der Hauptaufgabe
        # (siehe PipelineTask.__exit__ in progress.py) - hier nicht nochmal ausgeben.
        logger.error("Export aborted - see the error message above.")
        sys.exit(1)

    # Statistiken - über console.print() statt logger, damit die Box nicht durch RichHandlers
    # Level-Spalte verrutscht (mehrzeilige logger.info()-Aufrufe wurden dort falsch eingerückt).
    elapsed = time.time() - start_time
    console.print()
    console.print("[bold]" + "=" * 60 + "[/bold]")
    console.print("[bold]EXPORT FINISHED[/bold]")
    console.print("[bold]" + "=" * 60 + "[/bold]")
    console.print(f"Tiles processed: {stats['tiles_processed']}")
    console.print(f"Tiles failed: {stats['tiles_failed']}")
    console.print(f"Buildings exported: {stats['buildings_exported']}")
    console.print(f"Horizon exported: {'yes' if stats['horizon_exported'] else 'no'}")
    console.print(f"Total time: {elapsed:.1f}s")
    console.print("[bold]" + "=" * 60 + "[/bold]")


if __name__ == "__main__":
    main()
