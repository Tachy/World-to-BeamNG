"""
Beispiel: Verwendung der neuen BeamNGExporter API.

Zeigt wie die neue Architektur verwendet wird.
"""

from world_to_beamng.core import Config
from world_to_beamng.export import BeamNGExporter
from world_to_beamng.utils.tile_scanner import scan_lgl_tiles, compute_global_center


def main():
    # 1. Konfiguration erstellen
    config = Config(
        beamng_dir="C:/Users/johan/AppData/Local/BeamNG.drive/0.36/levels/World_to_BeamNG",
        work_dir=".",
        level_name="World_to_BeamNG",
    )

    # Optional: Mesh-Parameter anpassen
    config.mesh.grid_spacing = 2.0
    config.mesh.road_width = 7.0

    # 2. Exporter initialisieren
    exporter = BeamNGExporter(config)

    # Optional: Cache zurücksetzen
    # exporter.clear_cache()

    # 3. Tiles scannen
    tiles = scan_lgl_tiles(dgm1_dir="data/DGM1")

    if not tiles:
        print("Keine Tiles gefunden!")
        return

    # 4. Globalen Offset berechnen
    global_center = compute_global_center(tiles)
    global_offset = (global_center[0], global_center[1])

    print(f"Gefundene Tiles: {len(tiles)}")
    print(f"Global Offset: {global_offset}")

    # 5. Export durchführen
    stats = exporter.export_complete_level(
        tiles=tiles, global_offset=global_offset, include_buildings=True, include_horizon=True
    )

    # 6. Statistiken
    print(f"\n{'='*60}")
    print("EXPORT ABGESCHLOSSEN")
    print(f"{'='*60}")
    print(f"Tiles verarbeitet: {stats['tiles_processed']}")
    print(f"Tiles fehlgeschlagen: {stats['tiles_failed']}")
    print(f"Gebäude exportiert: {stats['buildings_exported']}")
    print(f"Horizon exportiert: {stats['horizon_exported']}")
    print(f"{'='*60}")


def example_single_tile():
    """Beispiel: Einzelnes Tile exportieren."""
    config = Config(beamng_dir="C:/BeamNG/levels/Test", work_dir=".")

    exporter = BeamNGExporter(config)

    # Tile-Metadaten (normalerweise von scan_lgl_tiles)
    tile = {"name": "Test-Tile", "filepath": "data/DGM1/dgm1_32494_5396.zip", "tile_x": 0, "tile_y": 0}

    dae_path = exporter.export_single_tile(tile=tile, global_offset=(3249000, 5396000), tile_x=0, tile_y=0)

    if dae_path:
        print(f"Exportiert: {dae_path}")
    else:
        print("Export fehlgeschlagen!")


def example_terrain_only():
    """Beispiel: Nur Terrain (ohne Buildings/Horizon)."""
    config = Config(beamng_dir="C:/BeamNG/levels/TerrainOnly")
    exporter = BeamNGExporter(config)

    tiles = scan_lgl_tiles("data/DGM1")
    global_offset = compute_global_center(tiles)

    count = exporter.export_terrain_only(tiles=tiles, global_offset=(global_offset[0], global_offset[1]))

    print(f"{count} Tiles exportiert")


if __name__ == "__main__":
    main()
