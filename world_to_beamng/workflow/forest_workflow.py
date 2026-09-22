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

Die fertigen Baum-Instanzen (Schritt 2, der teuerste Teil bei zehntausenden Bäumen) werden gecacht
- siehe _forest_cache_key()/_load_cached_tree_instances()/_save_cached_tree_instances() - Poisson-
Disk-Sampling und Rotation sind sonst bei jedem Lauf unterschiedlich (kein fester Seed), der Cache
macht wiederholte Läufe über dasselbe Gebiet also nebenbei auch deterministisch.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from world_to_beamng.logging_config import LoggerConfig
from world_to_beamng.forest.forest_normalizer import ForestNormalizer
from world_to_beamng.forest.forest_point_generator import ForestPointGenerator
from world_to_beamng.forest.forest_height_calculator import ForestHeightCalculator
from world_to_beamng.forest.forest_instance_generator import ForestInstanceGenerator
from world_to_beamng.forest.forest_json_writer import ForestJSONWriter
from world_to_beamng.forest.tree_footprints import TrunkFitter, load_trunk_feet

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

        # Stammfüße je Baumtyp (wird in set_forest_config() gefüllt)
        self.trunk_feet = {}

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

        # BeamNG erwartet *.forest4.json Platzierungsdateien im Level-Unterordner "forest/" (nicht "main/")
        output_dir = config.BEAMNG_DIR / "forest"
        self.json_writer = ForestJSONWriter(output_dir)

        # Stammfüße der Baumtypen, die in Wäldern vorkommen (aus dem Kollisionsmodell der .dae): Gruppen-Assets haben
        # Stämme bis ~9 m neben dem Ursprung, die Ausschlusszonen und der Boden müssen für jeden Stamm gelten
        used_types = {
            name
            for template in (forest_config.get("forest_type_templates") or {}).values()
            for name in template.get("preferred_trees", {})
        }
        # dae_path ist relativ zum BeamNG-Benutzerordner ("current"), der über "levels/<level>" liegt
        self.trunk_feet = load_trunk_feet(
            {name: info for name, info in registered_trees.items() if name in used_types}, config.BEAMNG_DIR.parent.parent
        )

    def _forest_cache_key(self, tile_bounds, global_offset, height_hash) -> Optional[str]:
        """
        Cache-Schlüssel für die fertigen Baum-Instanzen einer Fläche (Poisson-Disk-Sampling +
        Höhen-Interpolation + Rotation/Scale/Trunk-Fitting - der teuerste Teil von process_tile()).

        None ohne height_hash (kein Cache möglich - wie bei den anderen Caches dieser Pipeline).
        Bewusst grob (wie tile_hash/height_hash überall sonst in dieser Pipeline): eine Änderung
        an FOREST_*/Straßen-/Terrain-Konfigurationskonstanten wird NICHT automatisch erkannt -
        siehe README-Troubleshooting ("cache/ löschen, wenn Ergebnisse seltsam aussehen").
        """
        if not height_hash:
            return None

        def _file_sig(path) -> str:
            p = Path(path)
            if not p.is_file():
                return "missing"
            st = p.stat()
            return f"{st.st_size}:{int(st.st_mtime)}"

        ox, oy = (global_offset[0], global_offset[1]) if global_offset else (0.0, 0.0)
        managed_item_data = self.config.BEAMNG_DIR / "art" / "forest" / "managedItemData.json"
        signature = "|".join(
            [
                str(height_hash),
                ",".join(f"{v:.2f}" for v in tile_bounds),
                f"{ox:.2f}_{oy:.2f}",
                _file_sig("data/osm_to_beamng.json"),
                _file_sig(managed_item_data),
            ]
        )
        return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]

    def _forest_cache_path(self, cache_key: str) -> Path:
        return self.config.CACHE_DIR / f"forest_instances_{cache_key}.json"

    def _load_cached_tree_instances(self, cache_key: Optional[str]):
        """(tree_instances, forests_count) aus dem Cache, oder None (kein Treffer/kein Cache-Key)."""
        if cache_key is None:
            return None
        path = self._forest_cache_path(cache_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["tree_instances"], data["forests_count"]
        except (OSError, ValueError, KeyError):
            return None

    def _save_cached_tree_instances(self, cache_key: Optional[str], tree_instances, forests_count: int) -> None:
        if cache_key is None:
            return
        self.config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = self._forest_cache_path(cache_key)
        path.write_text(
            json.dumps({"tree_instances": tree_instances, "forests_count": forests_count}), encoding="utf-8"
        )

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

    def _create_road_surface_exclusion(self, road_slope_polygons_2d, margin: float):
        """
        Gepufferte Vereinigung der tatsächlich eingebetteten Straßenflächen (geglättet, mit echter Breite).

        Der OSM-Linienpuffer kennt weder die Fahrbahnbreite noch die Glättung der Mittellinie; die
        Straßenpolygone entsprechen dem, was BeamNG als DecalRoad auf das Terrain projiziert.

        Args:
            road_slope_polygons_2d: bereits vereinigte Straßenfläche (shapely-Geometrie, siehe
                geometry.road_surfaces.union_road_surfaces) oder eine Liste von Dicts mit "road_polygon"
                ((M, 2) Array, lokale Koordinaten)
            margin: Abstand zur Fahrbahnkante in Metern

        Returns:
            shapely-Geometrie oder None
        """
        from ..geometry.road_surfaces import union_road_surfaces

        if hasattr(road_slope_polygons_2d, "geom_type"):
            surface = road_slope_polygons_2d
        else:
            surface = union_road_surfaces(road_slope_polygons_2d)
        # Erst vereinigen, dann einmal puffern (Minkowski-Summe: gleiches Ergebnis wie Pufferung je Polygon)
        return surface.buffer(margin) if surface is not None and not surface.is_empty else None

    def _create_building_buffer(self, osm_data, margin: float = None):
        """
        Gepufferte Vereinigung aller OSM-Gebäudegrundrisse: dort stehen keine Bäume/Büsche.

        Wichtig für Gärten und Wohngebiete, deren Polygone die Häuser umschließen.

        VORAUSSETZUNG: osm_data liegt bereits in lokalen Koordinaten vor.

        Returns:
            shapely-Geometrie oder None (keine Gebäude)
        """
        if margin is None:
            margin = self.config.FOREST_BUILDING_MARGIN

        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        shapes = []
        for element in osm_data or []:
            if element.get("type") != "way" or "building" not in (element.get("tags") or {}):
                continue
            geometry = element.get("geometry") or []
            coords = [(pt["x"], pt["y"]) for pt in geometry if isinstance(pt, dict) and "x" in pt and "y" in pt]
            if len(coords) < 4:
                continue
            polygon = Polygon(coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                shapes.append(polygon.buffer(margin))

        return unary_union(shapes) if shapes else None

    def _create_row_exclusion(self, osm_data, building_buffer, surface_exclusion=None):
        """
        Ausschluss für Baumreihen: Gebäude und Straßen mit dem KLEINEREN Puffer FOREST_ROW_ROAD_MARGIN
        (Alleen stehen wenige Meter neben der Straße, nicht auf der Fahrbahn).

        Returns:
            shapely-Geometrie oder None
        """
        from shapely.ops import unary_union

        road_buffer = self._create_road_buffer(osm_data, road_margin=self.config.FOREST_ROW_ROAD_MARGIN) if osm_data else None
        parts = [g for g in (road_buffer, building_buffer, surface_exclusion) if g is not None]
        return unary_union(parts) if parts else None

    def _single_tree_points(self, osm_data, tile_bounds, global_offset, exclusion=None):
        """
        Positionen einzelner Bäume (OSM-Punkte mit natural=tree) innerhalb des Tiles.

        Die Punkte tragen noch lat/lon (nur "geometry"-Listen wurden transformiert). Punkte in
        `exclusion` (Straßen, Gebäude) werden verworfen.

        Returns:
            Liste lokaler (x, y)
        """
        from shapely import intersects_xy

        from ..osm.landuse_polygons import make_local_transform

        to_local = make_local_transform(global_offset)
        x_min, y_min, x_max, y_max = tile_bounds
        points = []
        for element in osm_data or []:
            if element.get("type") != "node" or (element.get("tags") or {}).get("natural") != "tree":
                continue
            if "lat" not in element or "lon" not in element:
                continue
            x, y = to_local([{"lat": element["lat"], "lon": element["lon"]}])[0]
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue
            if exclusion is not None and intersects_xy(exclusion, x, y):
                continue
            points.append((x, y))
        return points

    def process_tile(
        self,
        tile_bounds: Tuple[float, float, float, float],
        tile_name: str = "unknown",
        elevation_data=None,
        height_grid_info: Optional[Dict] = None,
        height_hash: Optional[str] = None,
        global_offset: Optional[Tuple[float, float]] = None,
        height_at=None,
        road_surfaces=None,
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
            height_at: Optional - Höhenabfrage (x, y) -> z der FERTIGEN Terrain-Heightmap (nach Straßen-Einbettung).
                       Ohne sie fallen die Höhen auf die rohen DGM1-Punkte (Nearest-Neighbor) zurück.
            road_surfaces: Optional - vereinigte eingebettete Straßenfläche (shapely, lokal) oder Liste von Dicts
                           mit "road_polygon"; dort und in FOREST_ROAD_SURFACE_MARGIN Umgebung stehen keine Bäume

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
                        "rotationMatrix": [r00, r01, r02, r10, r11, r12, r20, r21, r22],
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

            # Cache: Poisson-Disk-Sampling + Höhen-Interpolation + Instanz-Generierung sind der
            # teuerste Teil hier unten (zehntausende Bäume) - bei unverändertem Gebiet/Höhendaten/
            # Config direkt die fertigen Baum-Instanzen wiederverwenden (siehe _forest_cache_key()).
            cache_key = self._forest_cache_key(tile_bounds, global_offset, height_hash)
            cached = self._load_cached_tree_instances(cache_key)
            if cached is not None:
                tree_instances, forests_count = cached
                self.all_tree_instances.extend(tree_instances)
                logger.info(f"  [OK] Forest-Cache gefunden: {len(tree_instances)} Baum-Instanzen (bereits berechnet)")
                return {
                    "status": "success",
                    "tile_name": tile_name,
                    "tile_bounds": tile_bounds,
                    "tree_count": len(tree_instances),
                    "forests_count": forests_count,
                    "tree_instances": tree_instances,
                    "error": None,
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
            # Bäume/Büsche dürfen weder auf Straßen noch in/an Gebäuden stehen (Gärten, Wohngebiete)
            building_buffer = self._create_building_buffer(osm_data)
            if building_buffer is not None:
                logger.info(f"  [Forest] Gebäude-Puffer erstellt - Fläche: {building_buffer.area:.0f}m²")
                from shapely.ops import unary_union

                exclusion = unary_union([road_buffer, building_buffer]) if road_buffer else building_buffer
            else:
                exclusion = road_buffer
            # Tatsächlich eingebettete (geglättete, echt breite) Straßenflächen zusätzlich zum rohen OSM-Linienpuffer
            surface_exclusion = self._create_road_surface_exclusion(road_surfaces, self.config.FOREST_ROAD_SURFACE_MARGIN)
            if surface_exclusion is not None:
                from shapely.ops import unary_union

                exclusion = unary_union([exclusion, surface_exclusion]) if exclusion is not None else surface_exclusion
            row_surface_exclusion = self._create_road_surface_exclusion(road_surfaces, self.config.FOREST_ROW_SURFACE_MARGIN)
            row_exclusion = self._create_row_exclusion(osm_data, building_buffer, row_surface_exclusion)
            self.point_generator.set_road_buffer(exclusion)
            self.point_generator.set_row_exclusion(row_exclusion)

            forest_properties = {
                ft: self.normalizer.get_forest_properties(ft)
                for ft in self.forest_config.get("forest_type_templates", {}).keys()
            }

            forest_points = self.point_generator.generate_points_for_forests(
                forests=forests, forest_properties=forest_properties
            )

            # Einzelbäume (OSM natural=tree als Punkt) als eigener synthetischer "Wald"-Eintrag
            single_type = self.forest_config.get("forest_mappings", {}).get("single_trees", {}).get("forest_type")
            if single_type and single_type in forest_properties:
                singles = self._single_tree_points(osm_data, tile_bounds, (ox, oy), exclusion)
                if singles:
                    forests.append({"type": single_type, "geometry": None, "osm_tags": {"natural": "tree"}})
                    forest_points[len(forests) - 1] = singles
                    logger.info(f"  [Forest] {len(singles)} Einzelbäume (natural=tree)")

            total_points = sum(len(pts) for pts in forest_points.values())
            logger.info(f"  [→] {total_points} Baumpositionen generiert")

            # Phase 3: Height Interpolation (Bilineare Interpolation)
            logger.info(f"  [→] Interpoliere Höhen...")
            forest_points_3d = self.height_calculator.calculate_heights_for_forest_points(
                forest_points=forest_points,
                height_points=elevation_data,
                height_elevations=height_grid_info.get("elevations") if height_grid_info else None,
                grid_info=height_grid_info,
                height_at=height_at,
            )

            logger.info(f"  [→] Höhen für {total_points} Punkte interpoliert")

            # Phase 4: Instance Generation (Type, Rotation, Scale)
            logger.info(f"  [→] Generiere Baum-Instanzen...")
            # Die Ursprünge halten die Abstände ein; die Stämme von Gruppen-Assets (bis ~9 m daneben) müssen es auch,
            # und sie dürfen nicht in der Luft hängen. Dieselben Zonen und Abstände wie oben, nur pro Stamm geprüft.
            fitter = TrunkFitter(
                self.trunk_feet,
                exclusion=exclusion,
                row_exclusion=row_exclusion,
                height_at=height_at,
                max_float=self.config.FOREST_TRUNK_MAX_FLOAT,
                max_sink=self.config.FOREST_TRUNK_MAX_SINK,
            )
            tree_instances = self.instance_generator.generate_instances_for_forests(
                forest_points_3d=forest_points_3d,
                forests=forests,
                forest_properties_map={
                    ft: self.normalizer.get_forest_properties(ft)
                    for ft in self.forest_config.get("forest_type_templates", {}).keys()
                },
                fitter=fitter,
            )

            # Sammle Instances für finalen Export
            self.all_tree_instances.extend(tree_instances)
            self._save_cached_tree_instances(cache_key, tree_instances, len(forests))

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

    def add_instances(self, instances: List[Dict]) -> int:
        """
        Fügt zusätzliche Forest-Instanzen hinzu (z.B. Weinberg-Reben), die nicht aus
        Waldpolygonen stammen. Sie werden in finalize_forest_export() zusammen mit den
        Bäumen in forest.forest4.json geschrieben.

        Args:
            instances: Instanzen im forest4-Format (type, pos, rotationMatrix, scale)

        Returns:
            Anzahl der hinzugefügten Instanzen
        """
        self.all_tree_instances.extend(instances)
        return len(instances)

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
