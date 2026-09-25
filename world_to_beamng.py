"""
WORLD-TO-BEAMNG - OSM to BeamNG road generator

Refactored version with a modular architecture.
Main entry point for the application.

Required packages:
  pip install requests numpy scipy pyproj pyvista shapely rtree rich
"""

import sys
import time

# UTF-8 encoding for the Windows console
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
    """Main function - uses the new BeamNGExporter API."""

    start_time = time.time()

    pipeline = Pipeline()
    exporter = BeamNGExporter(pipeline)

    with pipeline.task("Preparation") as task:
        tiles = scan_elevation_tiles(dgm_dir=config.HEIGHT_DATA_DIR)

        if not tiles:
            task.fail("no DGM1 tiles found")
            return

        # Resolve the source CRS (detected automatically from the GeoTIFF tiles, otherwise config.SOURCE_CRS_EPSG) -
        # MUST be set before any further coordinate transformation (OSM bbox, LoD2, horizon, ...)
        source_epsg = resolve_source_crs_epsg(tiles)
        coordinates.set_source_crs(source_epsg)

        global_center = compute_global_center(tiles)
        # 3-tuple: (x, y, z) - z is the mean of the heights or 0
        global_offset = (global_center[0], global_center[1], global_center[2] if len(global_center) > 2 else 0.0)

        task.done(f"{len(tiles)} Tiles, EPSG:{source_epsg}, Offset {global_offset}")

    # Run the export
    try:
        stats = exporter.export_complete_level(
            tiles=tiles,
            global_offset=global_offset,
            include_buildings=config.LOD2_ENABLED,
            include_horizon=config.PHASE5_ENABLED,
        )
    except MissingTexturesError:
        # The full error message is already in the "✗ Textures - ..." line of the main task
        # (see PipelineTask.__exit__ in progress.py) - do not print it again here.
        logger.error("Export aborted - see the error message above.")
        sys.exit(1)

    # Statistics - via console.print() instead of logger, so the box is not shifted by the RichHandler's
    # level column (multi-line logger.info() calls were indented incorrectly there).
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
