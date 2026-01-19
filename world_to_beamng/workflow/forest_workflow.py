"""
Forest Workflow
===============

Orchestriert Wald-Generierung pro Tile (nach Asset-Scanning durch BeamNGExporter):
1. Tile-Initialisierung (OSM-Polygon-Normalisierung)
2. Pro Tile:
   - Poisson-Disk-Sampling → Tree-Positionen
   - Bilineare Interpolation → Tree-Höhen
   - Forest-Instance-Generierung (Rotation + Scale)
3. Forest.json-Finalisierung nach Tile-Loop
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from world_to_beamng.forest.forest_normalizer import ForestNormalizer
from world_to_beamng.forest.forest_point_generator import ForestPointGenerator
from world_to_beamng.forest.forest_height_calculator import ForestHeightCalculator
from world_to_beamng.forest.forest_instance_generator import ForestInstanceGenerator
from world_to_beamng.forest.forest_json_writer import ForestJSONWriter

logger = logging.getLogger(__name__)


class ForestWorkflow:
    """Orchestriert Tile-basierte Wald-Generierung."""

    def __init__(self, config):
        """
        Initialize workflow.

        Args:
            config: Configuration module
        """
        self.config = config

        # Forest Normalizer (wird in initialize_tiling() initialisiert)
        self.normalizer = None
        self.forest_config = {}

        # Point Generator (Poisson-Disk-Sampling)
        self.point_generator = ForestPointGenerator(min_distance=5.0, max_attempts=30)

        # Height Calculator (Bilineare Interpolation)
        self.height_calculator = ForestHeightCalculator()

        # Instance Generator (Rotation + Scale + Type-Selection)
        # Wird in set_forest_config() mit registered_trees initialisiert!
        self.instance_generator = None

        # JSON Writer (wird in set_forest_config initialisiert)
        self.json_writer = None

        # Sammle Tree-Instances über alle Tiles (wird in process_tile() gefüllt)
        self.all_tree_instances = []

    def set_forest_config(self, forest_config: Dict, osm_mapper, registered_trees: Optional[Dict] = None):
        """
        Setze Forest-Konfiguration vor Tile-Loop.

        Args:
            forest_config: Dict mit "forest_types" + "forest_mappings"
            osm_mapper: OSMMapper-Instance
            registered_trees: Optional - verfügbare Baumarten

        Raises:
            ValueError: Wenn registered_trees leer ist
        """
        if not registered_trees:
            raise ValueError("registered_trees darf nicht leer sein!")

        self.forest_config = forest_config
        self.normalizer = ForestNormalizer(forest_config, osm_mapper)

        # Initialisiere InstanceGenerator mit registered_trees
        self.instance_generator = ForestInstanceGenerator(registered_trees)

        from .. import config

        output_dir = config.BEAMNG_DIR / "main"
        self.json_writer = ForestJSONWriter(output_dir)

    def process_tile(
        self,
        tile_bounds: Tuple[float, float, float, float],
        tile_name: str = "unknown",
        elevation_data=None,
        height_grid_info: Optional[Dict] = None,
        height_hash: Optional[str] = None,
        global_offset: Optional[Tuple[float, float]] = None,
    ) -> Dict:
        """
        PHASE 1b: Verarbeite Wälder für ein 2×2km Tile.

        MUSS nach set_forest_config() aufgerufen werden!
        Wird für JEDES Tile aufgerufen.

        Schritte:
        1. Normalisiere OSM-Waldpolygone auf Tile-Grenzen
        2. Generiere Tree-Punkte (Poisson-Disk-Sampling)
        3. Interpoliere Höhen aus Elevation-Grid
        4. Generiere Instances (Type, Rotation, Scale)

        Args:
            tile_bounds: (x_min, y_min, x_max, y_max) in lokalen Koordinaten
            tile_name: Optional - Name des Tiles für Logging
            elevation_data: Optional - numpy array mit Höhendaten
            height_grid_info: Optional - Dict mit "origin", "spacing", "elevations"
            height_hash: Optional - Hash für Cache-Konsistenz (vom Terrain-Workflow)
            global_offset: Optional - (utm_x_origin, utm_y_origin) für WGS84-Transformation
                          WICHTIG: Muss der UTM-Ursprung sein, nicht der Tile-Zentroid!

        Returns:
            {
                "status": "success" | "no_forests" | "error",
                "tile_name": str,
                "tile_bounds": (x_min, y_min, x_max, y_max),
                "tree_count": int,
                "forests_count": int,
                "tree_instances": [
                    {
                        "type": "oak",
                        "pos": [x, y, z],
                        "rot": [rx, ry, rz, rw],
                        "scale": 1.15
                    },
                    ...
                ],
                "error": Optional[str]
            }
        """
        try:
            print(f"\n[Forest Phase 1b] Starte für {tile_name} (bounds: {tile_bounds})")

            # Initialisiere osm_data
            osm_data = None

            # Prüfe ob set_forest_config() aufgerufen wurde
            if not self.normalizer or not self.instance_generator:
                print(f"[Forest ERROR] set_forest_config() nicht aufgerufen!")
                return {
                    "status": "error",
                    "tile_name": tile_name,
                    "tile_bounds": tile_bounds,
                    "tree_count": 0,
                    "forests_count": 0,
                    "tree_instances": [],
                    "error": "set_forest_config() not called",
                }

            # Phase 1b: Normalisierung (mit bereits geladenen OSM-Daten)
            if not osm_data:
                print(f"  [→] Lade OSM-Daten aus Cache...")
                from ..osm.downloader import get_osm_data
                from ..geometry.coordinates import transformer_to_wgs84

                # Konvertiere lokale Bounds zurück zu UTM (einfach + offset)
                # global_offset kann (x, y) oder (x, y, z) sein - wir brauchen nur (x, y)
                if global_offset:
                    ox, oy = global_offset[0], global_offset[1]
                else:
                    ox, oy = 0, 0
                utm_x_min = tile_bounds[0] + ox
                utm_y_min = tile_bounds[1] + oy
                utm_x_max = tile_bounds[2] + ox
                utm_y_max = tile_bounds[3] + oy

                # Konvertiere UTM zu lat/lon für BBox (Overpass Query braucht lat/lon)
                lat_min, lon_min = transformer_to_wgs84.transform(utm_x_min, utm_y_min)
                lat_max, lon_max = transformer_to_wgs84.transform(utm_x_max, utm_y_max)

                # Overpass BBox: (lat_min, lon_min, lat_max, lon_max)
                bbox_tuple = (lat_min, lon_min, lat_max, lon_max)

                # Nutze height_hash für Cache-Konsistenz (wie Terrain-Workflow)
                osm_data = get_osm_data(bbox_tuple, height_hash=height_hash)
                print(f"  [→] {len(osm_data) if osm_data else 0} OSM-Elemente geladen")

            if not osm_data:
                print(f"  [→] Keine OSM-Daten verfügbar")
                return {
                    "status": "no_forests",
                    "tile_name": tile_name,
                    "tile_bounds": tile_bounds,
                    "tree_count": 0,
                    "forests_count": 0,
                    "tree_instances": [],
                    "error": None,
                }

            print(f"  [→] Normalisiere OSM-Waldpolygone...")

            # Berechne local_offset für Koordinaten-Transformation
            # global_offset kann (x, y) oder (x, y, z) sein - wir brauchen nur (x, y)
            if global_offset:
                ox, oy = global_offset[0], global_offset[1]
            else:
                ox, oy = 0, 0

            # Nutze den echten global_offset für Waldtransformation
            forest_local_offset = (ox, oy)

            normalized = self.normalizer.normalize_tile(
                tile_bounds, tile_name, osm_data=osm_data, local_offset=forest_local_offset
            )
            print(f"  [Forest] Normalisierung: {normalized.get('status')} - {normalized.get('forest_count')} Wälder")

            # DEBUG: Speichere Dump wenn forest_count = 0
            if normalized.get("forest_count", 0) == 0:
                import json
                from pathlib import Path

                dump_file = Path(f"cache/forest_debug_{tile_name}.json")
                dump_data = {
                    "tile": tile_name,
                    "status": normalized["status"],
                    "error": normalized.get("error"),
                    "osm_count": len(osm_data) if osm_data else 0,
                    "forest_count": normalized.get("forest_count"),
                    "tile_bounds": tile_bounds,
                    "global_offset": (ox, oy),
                    "forest_local_offset": forest_local_offset,
                }
                with open(dump_file, "w") as f:
                    json.dump(dump_data, f, indent=2)
                print(f"  [DEBUG] Dump geschrieben: {dump_file}")

            if normalized["status"] != "success" or normalized["forest_count"] == 0:
                print(f"  [Forest] Keine Wälder gefunden: {normalized.get('error', 'unbekannter Fehler')}")
                return {
                    "status": "no_forests" if normalized["status"] == "success" else "error",
                    "tile_name": tile_name,
                    "tile_bounds": tile_bounds,
                    "tree_count": 0,
                    "forests_count": 0,
                    "tree_instances": [],
                    "error": normalized.get("error"),
                }

            forests = normalized["forests"]
            print(f"  [→] {len(forests)} Waldpolygone zu bearbeiten")

            # Phase 2: Point Generation (Poisson-Disk-Sampling)
            print(f"  [→] Generiere Tree-Positionen (Poisson-Disk)...")
            forest_properties = {
                ft: self.normalizer.get_forest_properties(ft)
                for ft in self.forest_config.get("forest_types", {}).keys()
            }

            forest_points = self.point_generator.generate_points_for_forests(
                forests=forests, forest_properties=forest_properties
            )

            total_points = sum(len(pts) for pts in forest_points.values())
            print(f"  [→] {total_points} Baumpositionen generiert")

            # Phase 3: Height Interpolation (Bilineare Interpolation)
            print(f"  [→] Interpoliere Höhen...")
            forest_points_3d = self.height_calculator.calculate_heights_for_forest_points(
                forest_points=forest_points,
                height_points=elevation_data,
                height_elevations=height_grid_info.get("elevations") if height_grid_info else None,
                grid_info=height_grid_info,
            )

            print(f"  [→] Höhen für {total_points} Punkte interpoliert")

            # Phase 4: Instance Generation (Type, Rotation, Scale)
            print(f"  [→] Generiere Baum-Instanzen...")
            tree_instances = self.instance_generator.generate_instances_for_forests(
                forest_points_3d=forest_points_3d,
                forests=forests,
                forest_properties_map={
                    ft: self.normalizer.get_forest_properties(ft)
                    for ft in self.forest_config.get("forest_types", {}).keys()
                },
            )

            # Sammle Instances für finalen Export
            self.all_tree_instances.extend(tree_instances)

            print(f"  [✓] {len(tree_instances)} Baum-Instanzen generiert für {tile_name}")

            result = {
                "status": "success",
                "tile_name": tile_name,
                "tile_bounds": tile_bounds,
                "tree_count": len(tree_instances),
                "forests_count": len(forests),
                "tree_instances": tree_instances,
                "error": None,
            }

            return result

        except Exception as e:
            print(f"[Forest ERROR] Exception in process_tile: {e}")
            import traceback

            traceback.print_exc()
            logger.error(f"Fehler beim Forest-Processing für {tile_name}: {e}", exc_info=True)
            return {
                "status": "error",
                "tile_name": tile_name,
                "tile_bounds": tile_bounds,
                "tree_count": 0,
                "forests_count": 0,
                "tree_instances": [],
                "error": str(e),
            }

    def finalize_forest_export(self) -> Dict:
        """
        FINALISIERUNG (nach Tile-Loop): Schreibe forest.json.

        Sammelt alle Tree-Instances aus process_tile() und schreibt forest.json.

        MUSS NACH dem Tile-Loop aufgerufen werden!

        Returns:
            {
                "status": "success" | "no_forests" | "error",
                "total_trees": int,
                "forest_json_path": str,
                "statistics": Dict,
                "error": Optional[str]
            }
        """
        try:
            print(f"[Forest] Finalisiere Export ({len(self.all_tree_instances)} Instanzen)...")

            # Prüfe ob Instanzen vorhanden
            if not self.all_tree_instances:
                print("[Forest] Keine Baum-Instanzen generiert, überspringe forest.json")
                return {
                    "status": "no_forests",
                    "total_trees": 0,
                    "forest_json_path": "",
                    "statistics": {},
                    "error": None,
                }

            # Prüfe ob JSON Writer initialisiert
            if not self.json_writer:
                print("[Forest ERROR] ForestJSONWriter nicht initialisiert!")
                return {
                    "status": "error",
                    "total_trees": 0,
                    "forest_json_path": "",
                    "statistics": {},
                    "error": "ForestJSONWriter not initialized",
                }

            # Schreibe forest.json
            write_result = self.json_writer.write_forest_json(
                tree_instances=self.all_tree_instances, filename="forest.json"
            )

            if write_result["status"] != "success":
                return {
                    "status": "error",
                    "total_trees": len(self.all_tree_instances),
                    "forest_json_path": "",
                    "statistics": {},
                    "error": write_result.get("error"),
                }

            # Statistiken
            statistics = self.json_writer.get_statistics(self.all_tree_instances)

            print(f"[✓] Forest-Export abgeschlossen:")
            print(f"  - Gesamt Bäume: {statistics['total_trees']}")
            print(f"  - Baumarten: {len(statistics['types'])}")
            for tree_type, count in sorted(statistics["types"].items()):
                print(f"    • {tree_type}: {count}")
            print(f"  - Durchschn. Scale: {statistics['avg_scale']:.2f}")
            print(f"  - Höhenbereich: {statistics['min_height']:.1f}m - {statistics['max_height']:.1f}m")

            return {
                "status": "success",
                "total_trees": len(self.all_tree_instances),
                "forest_json_path": write_result["filepath"],
                "statistics": statistics,
                "error": None,
            }

        except Exception as e:
            print(f"[Forest ERROR] Forest-Finalisierung: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "total_trees": 0, "forest_json_path": "", "statistics": {}, "error": str(e)}
