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

from world_to_beamng.logging_config import LoggerConfig
from world_to_beamng.forest.forest_normalizer import ForestNormalizer
from world_to_beamng.forest.forest_point_generator import ForestPointGenerator
from world_to_beamng.forest.forest_height_calculator import ForestHeightCalculator
from world_to_beamng.forest.forest_instance_generator import ForestInstanceGenerator
from world_to_beamng.forest.forest_json_writer import ForestJSONWriter

logger = LoggerConfig.get_logger()


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

    def _transform_osm_to_local(self, osm_data, global_offset: Tuple[float, float]):
        """
        ZENTRALE OSM-TRANSFORMATION: Transformiert ALLE OSM-Geometrien einmalig zu lokalen Koordinaten.

        Transformiert alle 'geometry'-Felder von WGS84 (lat/lon) zu lokalen Koordinaten.
        Nach diesem Aufruf sind ALLE Geometrien in lokalen Koordinaten!

        Unterstützt multiple Formate:
        - {"lat": ..., "lon": ...} (Overpass-Format)
        - [lat, lon] oder [lon, lat] (Liste/Tuple-Format)

        Args:
            osm_data: Liste von OSM-Elementen mit 'geometry' in WGS84
            global_offset: (utm_x_origin, utm_y_origin)

        Returns:
            OSM-Daten mit transformierten Geometrien (in-place Modifikation)
        """
        if not osm_data:
            return osm_data

        from ..geometry.coordinates import transformer_to_wgs84
        from pyproj import Transformer

        # Inverse Transformer: WGS84 → UTM
        transformer_utm = Transformer.from_proj(
            transformer_to_wgs84.target_crs,  # WGS84
            transformer_to_wgs84.source_crs,  # UTM
        )

        ox, oy = global_offset[0], global_offset[1]

        for element in osm_data:
            if "geometry" not in element:
                continue

            geometry = element["geometry"]
            if not isinstance(geometry, list):
                continue

            # Transformiere jedes Geometrie-Punkt
            transformed_geometry = []
            for point in geometry:
                lat = None
                lon = None

                # Format 1: {"lat": ..., "lon": ...}
                if isinstance(point, dict) and "lat" in point and "lon" in point:
                    lat = point["lat"]
                    lon = point["lon"]

                # Format 2: [lat, lon] oder [lon, lat] oder (lat, lon) oder (lon, lat)
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    # Heuristik: Wenn Wert in [-180, 180] → lon, wenn in [-90, 90] → lat
                    val1, val2 = point[0], point[1]
                    if -90 <= val1 <= 90 and -180 <= val2 <= 180:
                        lat, lon = val1, val2  # [lat, lon]
                    elif -180 <= val1 <= 180 and -90 <= val2 <= 90:
                        lon, lat = val1, val2  # [lon, lat]
                    else:
                        continue

                if lat is None or lon is None:
                    continue

                # WGS84 → UTM → lokal
                utm_x, utm_y = transformer_utm.transform(lon, lat)
                local_x = utm_x - ox
                local_y = utm_y - oy

                # Ersetze lat/lon durch x/y
                transformed_geometry.append({"x": local_x, "y": local_y})

            # Ersetze geometry in-place
            element["geometry"] = transformed_geometry

        return osm_data

    def _create_road_buffer(self, osm_data, road_margin: float = None):
        """
        Erstellt einen gepufferten Road-Buffer aus OSM-Daten.

        VORAUSSETZUNG: osm_data MUSS bereits in lokalen Koordinaten vorliegen!

        Args:
            osm_data: OSM-Elemente mit 'geometry' in LOKALEN Koordinaten (x, y)
            road_margin: Puffer um Straßen (in Metern). Wenn None, wird config.FOREST_ROAD_MARGIN verwendet

        Returns:
            shapely.geometry.Polygon (gepufferte Vereinigung aller Straßen) oder None
        """
        if road_margin is None:
            road_margin = self.config.FOREST_ROAD_MARGIN

        if not osm_data:
            return None

        from ..osm.parser import extract_roads_from_osm
        from shapely.geometry import LineString
        from shapely.ops import unary_union

        roads = extract_roads_from_osm(osm_data)

        if not roads:
            logger.debug(f"  [Forest] Keine Straßen gefunden für Road Buffer")
            return None

        # Konvertiere Straßen-Ways zu LineStrings (Koordinaten MÜSSEN lokal sein!)
        road_lines = []

        for road in roads:
            if "geometry" not in road or len(road["geometry"]) < 2:
                continue

            # Geometrie MUSS in lokalen Koordinaten sein (x, y)
            coords_local = [(pt["x"], pt["y"]) for pt in road["geometry"] if "x" in pt and "y" in pt]

            if len(coords_local) >= 2:
                road_lines.append(LineString(coords_local))

        if not road_lines:
            logger.debug(f"  [Forest] Keine validen Road Lines erstellt")
            return None

        # Vereinige alle Straßen und erstelle Puffer
        if len(road_lines) == 1:
            road_union = road_lines[0]
        else:
            road_union = unary_union(road_lines)

        # Erstelle gepufferte Polygon
        road_buffer = road_union.buffer(road_margin)

        logger.debug(
            f"  [Forest] Road Buffer erstellt: {len(roads)} Straßen, {len(road_lines)} Lines, Margin={road_margin}m, Buffer-Area={road_buffer.area:.0f}m²"
        )

        return road_buffer

        # except Exception as e:
        #     import traceback
        #     logger.warning(f"  [Forest] Fehler beim Erstellen von Road Buffer: {e}")
        #     logger.debug(f"  [Forest] Stack Trace: {traceback.format_exc()}")
        #     return None

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
            logger.info(f"\n[Forest Phase 1b] Starte für {tile_name} (bounds: {tile_bounds})")

            # Initialisiere osm_data
            osm_data = None

            # Prüfe ob set_forest_config() aufgerufen wurde
            if not self.normalizer or not self.instance_generator:
                logger.error(f"[Forest ERROR] set_forest_config() nicht aufgerufen!")
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
                logger.info(f"  [→] Lade OSM-Daten aus Cache...")
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
                logger.info(f"  [→] {len(osm_data) if osm_data else 0} OSM-Elemente geladen")

            if not osm_data:
                logger.warning(f"  [→] Keine OSM-Daten verfügbar")
                return {
                    "status": "no_forests",
                    "tile_name": tile_name,
                    "tile_bounds": tile_bounds,
                    "tree_count": 0,
                    "forests_count": 0,
                    "tree_instances": [],
                    "error": None,
                }

            logger.info(f"  [→] Normalisiere OSM-Waldpolygone...")

            # Berechne local_offset für Koordinaten-Transformation
            # global_offset kann (x, y) oder (x, y, z) sein - wir brauchen nur (x, y)
            if global_offset:
                ox, oy = global_offset[0], global_offset[1]
            else:
                ox, oy = 0, 0

            # ZENTRALE TRANSFORMATION: Konvertiere ALLE OSM-Geometrien einmalig zu lokalen Koordinaten
            logger.info(f"  [→] Transformiere OSM-Daten zu lokalen Koordinaten...")
            osm_data = self._transform_osm_to_local(osm_data, (ox, oy))

            # Ab jetzt: ALLE Geometrien in osm_data sind in lokalen Koordinaten!
            # WGS84 (lat/lon) existiert nicht mehr - nur noch lokale (x, y)!

            # Nutze den echten global_offset für Waldtransformation
            forest_local_offset = (ox, oy)

            normalized = self.normalizer.normalize_tile(
                tile_bounds, tile_name, osm_data=osm_data, local_offset=forest_local_offset
            )
            logger.info(
                f"  [Forest] Normalisierung: {normalized.get('status')} - {normalized.get('forest_count')} Wälder"
            )

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
                logger.debug(f"  [DEBUG] Dump geschrieben: {dump_file}")

            if normalized["status"] != "success" or normalized["forest_count"] == 0:
                logger.error(f"  [Forest] Keine Wälder gefunden: {normalized.get('error', 'unbekannter Fehler')}")
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
            logger.info(f"  [→] {len(forests)} Waldpolygone zu bearbeiten")

            # Phase 2: Point Generation (Poisson-Disk-Sampling)
            logger.info(f"  [→] Generiere Tree-Positionen (Poisson-Disk)...")

            # Erstelle Road Buffer (OSM-Daten bereits in lokalen Koordinaten!)
            road_buffer = self._create_road_buffer(osm_data)
            if road_buffer:
                logger.info(
                    f"  [Forest] Road Buffer erstellt - Bounds: {road_buffer.bounds}, Area: {road_buffer.area:.0f}m²"
                )
            else:
                logger.info(f"  [Forest] Road Buffer ist None!")
            self.point_generator.set_road_buffer(road_buffer)

            forest_properties = {
                ft: self.normalizer.get_forest_properties(ft)
                for ft in self.forest_config.get("forest_type_templates", {}).keys()
            }

            forest_points = self.point_generator.generate_points_for_forests(
                forests=forests, forest_properties=forest_properties
            )

            total_points = sum(len(pts) for pts in forest_points.values())
            logger.info(f"  [→] {total_points} Baumpositionen generiert")

            # Phase 3: Height Interpolation (Bilineare Interpolation)
            logger.info(f"  [→] Interpoliere Höhen...")
            forest_points_3d = self.height_calculator.calculate_heights_for_forest_points(
                forest_points=forest_points,
                height_points=elevation_data,
                height_elevations=height_grid_info.get("elevations") if height_grid_info else None,
                grid_info=height_grid_info,
            )

            logger.info(f"  [→] Höhen für {total_points} Punkte interpoliert")

            # Phase 4: Instance Generation (Type, Rotation, Scale)
            logger.info(f"  [→] Generiere Baum-Instanzen...")
            tree_instances = self.instance_generator.generate_instances_for_forests(
                forest_points_3d=forest_points_3d,
                forests=forests,
                forest_properties_map={
                    ft: self.normalizer.get_forest_properties(ft)
                    for ft in self.forest_config.get("forest_type_templates", {}).keys()
                },
            )

            # Sammle Instances für finalen Export
            self.all_tree_instances.extend(tree_instances)

            logger.info(f"  [✓] {len(tree_instances)} Baum-Instanzen generiert für {tile_name}")

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
            logger.info(f"[Forest ERROR] Exception in process_tile: {e}")
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
        FINALISIERUNG (nach Tile-Loop): Schreibe forest.forest4.json.

        Sammelt alle Tree-Instances aus process_tile() und schreibt forest.forest4.json.

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
            logger.info(f"[Forest] Finalisiere Export ({len(self.all_tree_instances)} Instanzen)...")

            # Prüfe ob Instanzen vorhanden
            if not self.all_tree_instances:
                logger.warning("[Forest] Keine Baum-Instanzen generiert, überspringe forest.forest4.json")
                return {
                    "status": "no_forests",
                    "total_trees": 0,
                    "forest_json_path": "",
                    "statistics": {},
                    "error": None,
                }

            # Prüfe ob JSON Writer initialisiert
            if not self.json_writer:
                logger.info("[Forest ERROR] ForestJSONWriter nicht initialisiert!")
                return {
                    "status": "error",
                    "total_trees": 0,
                    "forest_json_path": "",
                    "statistics": {},
                    "error": "ForestJSONWriter not initialized",
                }

            # Schreibe forest.forest4.json
            write_result = self.json_writer.write_forest_json(
                tree_instances=self.all_tree_instances, filename="forest.forest4.json"
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

            logger.info(f"[✓] Forest-Export abgeschlossen:")
            logger.info(f"  - Gesamt Bäume: {statistics['total_trees']}")
            logger.info(f"  - Baumarten: {len(statistics['types'])}")
            for tree_type, count in sorted(statistics["types"].items()):
                logger.info(f"    • {tree_type}: {count}")
            logger.info(f"  - Durchschn. Scale: {statistics['avg_scale']:.2f}")
            logger.info(f"  - Höhenbereich: {statistics['min_height']:.1f}m - {statistics['max_height']:.1f}m")

            return {
                "status": "success",
                "total_trees": len(self.all_tree_instances),
                "forest_json_path": write_result["filepath"],
                "statistics": statistics,
                "error": None,
            }

        except Exception as e:
            logger.info(f"[Forest ERROR] Forest-Finalisierung: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "total_trees": 0, "forest_json_path": "", "statistics": {}, "error": str(e)}
