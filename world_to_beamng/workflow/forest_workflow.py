"""
Forest Workflow
===============

Orchestrates forest generation per tile (after asset scanning by BeamNGExporter):
1. Tile initialization (OSM polygon normalization)
2. Per tile:
   - Poisson disk sampling → tree positions
   - Bilinear interpolation → tree heights
   - Forest instance generation (rotation + scale)
3. Forest.json finalization after the tile loop

The finished tree instances (step 2, the most expensive part with tens of thousands of trees) are cached
- see _forest_cache_key()/_load_cached_tree_instances()/_save_cached_tree_instances() - Poisson
disk sampling and rotation otherwise differ on every run (no fixed seed), so the cache also makes
repeated runs over the same area deterministic as a side effect.
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
    """Orchestrates tile-based forest generation."""

    def __init__(self, config):
        """
        Initialize workflow.

        Args:
            config: Configuration module
        """
        self.config = config

        # Forest Normalizer (initialized in initialize_tiling())
        self.normalizer = None
        self.forest_config = {}

        # Point Generator (Poisson disk sampling)
        self.point_generator = ForestPointGenerator(min_distance=5.0, max_attempts=30)

        # Height Calculator (bilinear interpolation)
        self.height_calculator = ForestHeightCalculator()

        # Instance Generator (rotation + scale + type selection)
        # Initialized in set_forest_config() with registered_trees!
        self.instance_generator = None

        # JSON Writer (initialized in set_forest_config)
        self.json_writer = None

        # Collect tree instances across all tiles (filled in process_tile())
        self.all_tree_instances = []

        # Trunk feet per tree type (filled in set_forest_config())
        self.trunk_feet = {}

    def set_forest_config(self, forest_config: Dict, osm_mapper, registered_trees: Optional[Dict] = None):
        """
        Set the forest configuration before the tile loop.

        Args:
            forest_config: dict with "forest_types" + "forest_mappings"
            osm_mapper: OSMMapper instance
            registered_trees: optional - available tree species

        Raises:
            ValueError: if registered_trees is empty
        """
        if not registered_trees:
            raise ValueError("registered_trees must not be empty!")

        self.forest_config = forest_config
        self.normalizer = ForestNormalizer(forest_config, osm_mapper)

        # Initialize the InstanceGenerator with registered_trees
        self.instance_generator = ForestInstanceGenerator(registered_trees)

        from .. import config

        # BeamNG expects *.forest4.json placement files in the level subfolder "forest/" (not "main/")
        output_dir = config.BEAMNG_DIR / "forest"
        self.json_writer = ForestJSONWriter(output_dir)

        # Trunk feet of the tree types that occur in forests (from the collision model of the .dae): group assets have
        # trunks up to ~9 m beside the origin, the exclusion zones and the ground must hold for every trunk
        used_types = {
            name
            for template in (forest_config.get("forest_type_templates") or {}).values()
            for name in template.get("preferred_trees", {})
        }
        # dae_path is relative to the BeamNG user folder ("current"), which sits above "levels/<level>"
        self.trunk_feet = load_trunk_feet(
            {name: info for name, info in registered_trees.items() if name in used_types}, config.BEAMNG_DIR.parent.parent
        )

    def _forest_cache_key(self, tile_bounds, global_offset, height_hash) -> Optional[str]:
        """
        Cache key for the finished tree instances of an area (Poisson disk sampling +
        height interpolation + rotation/scale/trunk fitting - the most expensive part of process_tile()).

        None without height_hash (no cache possible - as with the other caches of this pipeline).
        Deliberately coarse (like tile_hash/height_hash everywhere else in this pipeline): a change
        to FOREST_*/road/terrain configuration constants is NOT detected automatically -
        see the README troubleshooting ("delete cache/ when results look odd").
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
        # height_hash stays visible in the key (like osm_all_<height_hash>.json,
        # grid_v3_grid_<height_hash>_... and dgm30_horizon_<tile_hash>_... elsewhere in
        # this pipeline) - makes related cache files of one run recognizable; only the
        # remaining, additional inputs here are combined into a suffix hash.
        signature = "|".join(
            [
                ",".join(f"{v:.2f}" for v in tile_bounds),
                f"{ox:.2f}_{oy:.2f}",
                _file_sig("data/osm_to_beamng.json"),
                _file_sig(managed_item_data),
            ]
        )
        suffix = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
        return f"{height_hash}_{suffix}"

    def _forest_cache_path(self, cache_key: str) -> Path:
        return self.config.CACHE_DIR / f"forest_instances_{cache_key}.json"

    def _load_cached_tree_instances(self, cache_key: Optional[str]):
        """(tree_instances, forests_count) from the cache, or None (no hit/no cache key)."""
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
        CENTRAL OSM TRANSFORMATION: transforms ALL OSM geometries to local coordinates once.

        Transforms all 'geometry' fields from WGS84 (lat/lon) to local coordinates.
        After this call ALL geometries are in local coordinates!

        Supports multiple formats:
        - {"lat": ..., "lon": ...} (Overpass format)
        - [lat, lon] or [lon, lat] (list/tuple format)

        Args:
            osm_data: list of OSM elements with 'geometry' in WGS84
            global_offset: (utm_x_origin, utm_y_origin)

        Returns:
            OSM data with transformed geometries (in-place modification)
        """
        if not osm_data:
            return osm_data

        from ..geometry.coordinates import transformer_to_wgs84
        from pyproj import Transformer

        # Inverse transformer: WGS84 → UTM
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

            # Transform each geometry point
            transformed_geometry = []
            for point in geometry:
                lat = None
                lon = None

                # Format 1: {"lat": ..., "lon": ...}
                if isinstance(point, dict) and "lat" in point and "lon" in point:
                    lat = point["lat"]
                    lon = point["lon"]

                # Format 2: [lat, lon] or [lon, lat] or (lat, lon) or (lon, lat)
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    # Heuristic: if the value is in [-180, 180] → lon, if in [-90, 90] → lat
                    val1, val2 = point[0], point[1]
                    if -90 <= val1 <= 90 and -180 <= val2 <= 180:
                        lat, lon = val1, val2  # [lat, lon]
                    elif -180 <= val1 <= 180 and -90 <= val2 <= 90:
                        lon, lat = val1, val2  # [lon, lat]
                    else:
                        continue

                if lat is None or lon is None:
                    continue

                # WGS84 → UTM → local
                utm_x, utm_y = transformer_utm.transform(lon, lat)
                local_x = utm_x - ox
                local_y = utm_y - oy

                # Replace lat/lon with x/y
                transformed_geometry.append({"x": local_x, "y": local_y})

            # Replace geometry in place
            element["geometry"] = transformed_geometry

        return osm_data

    def _create_road_buffer(self, osm_data, road_margin: float = None):
        """
        Creates a buffered road buffer from OSM data.

        PREREQUISITE: osm_data MUST already be in local coordinates!

        Args:
            osm_data: OSM elements with 'geometry' in LOCAL coordinates (x, y)
            road_margin: buffer around roads (in meters). If None, config.FOREST_ROAD_MARGIN is used

        Returns:
            shapely.geometry.Polygon (buffered union of all roads) or None
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
            logger.debug(f"  [Forest] No roads found for the road buffer")
            return None

        # Convert road ways to LineStrings (coordinates MUST be local!)
        road_lines = []

        for road in roads:
            if "geometry" not in road or len(road["geometry"]) < 2:
                continue

            # Geometry MUST be in local coordinates (x, y)
            coords_local = [(pt["x"], pt["y"]) for pt in road["geometry"] if "x" in pt and "y" in pt]

            if len(coords_local) >= 2:
                road_lines.append(LineString(coords_local))

        if not road_lines:
            logger.debug(f"  [Forest] No valid road lines created")
            return None

        # Union all roads and create the buffer
        if len(road_lines) == 1:
            road_union = road_lines[0]
        else:
            road_union = unary_union(road_lines)

        # Create the buffered polygon
        road_buffer = road_union.buffer(road_margin)

        logger.debug(
            f"  [Forest] Road buffer created: {len(roads)} roads, {len(road_lines)} lines, margin={road_margin}m, buffer area={road_buffer.area:.0f}m²"
        )

        return road_buffer

        # except Exception as e:
        #     import traceback
        #     logger.warning(f"  [Forest] Error creating road buffer: {e}")
        #     logger.debug(f"  [Forest] Stack Trace: {traceback.format_exc()}")
        #     return None

    def _create_road_surface_exclusion(self, road_slope_polygons_2d, margin: float):
        """
        Buffered union of the road surfaces actually embedded (smoothed, with true width).

        The OSM line buffer knows neither the carriageway width nor the smoothing of the centerline; the
        road polygons correspond to what BeamNG projects onto the terrain as a DecalRoad.

        Args:
            road_slope_polygons_2d: already unioned road surface (shapely geometry, see
                geometry.road_surfaces.union_road_surfaces) or a list of dicts with "road_polygon"
                ((M, 2) array, local coordinates)
            margin: distance to the road edge in meters

        Returns:
            shapely geometry or None
        """
        from ..geometry.road_surfaces import union_road_surfaces

        if hasattr(road_slope_polygons_2d, "geom_type"):
            surface = road_slope_polygons_2d
        else:
            surface = union_road_surfaces(road_slope_polygons_2d)
        # Union first, then buffer once (Minkowski sum: same result as buffering each polygon)
        return surface.buffer(margin) if surface is not None and not surface.is_empty else None

    def _create_building_buffer(self, osm_data, margin: float = None):
        """
        Buffered union of all OSM building footprints: no trees/bushes stand there.

        Important for gardens and residential areas whose polygons enclose the houses.

        PREREQUISITE: osm_data is already in local coordinates.

        Returns:
            shapely geometry or None (no buildings)
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
        Exclusion for tree rows: buildings and roads with the SMALLER buffer FOREST_ROW_ROAD_MARGIN
        (avenues stand a few meters beside the road, not on the carriageway).

        Returns:
            shapely geometry or None
        """
        from shapely.ops import unary_union

        road_buffer = self._create_road_buffer(osm_data, road_margin=self.config.FOREST_ROW_ROAD_MARGIN) if osm_data else None
        parts = [g for g in (road_buffer, building_buffer, surface_exclusion) if g is not None]
        return unary_union(parts) if parts else None

    def _single_tree_points(self, osm_data, tile_bounds, global_offset, exclusion=None):
        """
        Positions of individual trees (OSM points with natural=tree) within the tile.

        The points still carry lat/lon (only "geometry" lists were transformed). Points in
        `exclusion` (roads, buildings) are discarded.

        Returns:
            list of local (x, y)
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
        PHASE 1b: Process forests for a 2×2 km tile.

        MUST be called after set_forest_config()!
        Is called for EVERY tile.

        Steps:
        1. Normalize OSM forest polygons to tile borders
        2. Generate tree points (Poisson disk sampling)
        3. Interpolate heights from the elevation grid
        4. Generate instances (type, rotation, scale)

        Args:
            tile_bounds: (x_min, y_min, x_max, y_max) in local coordinates
            tile_name: optional - name of the tile for logging
            elevation_data: optional - numpy array with elevation data
            height_grid_info: optional - dict with "origin", "spacing", "elevations"
            height_hash: optional - hash for cache consistency (from the terrain workflow)
            global_offset: optional - (utm_x_origin, utm_y_origin) for the WGS84 transformation
                          IMPORTANT: must be the UTM origin, not the tile centroid!
            height_at: optional - height query (x, y) -> z of the FINISHED terrain heightmap (after road embedding).
                       Without it the heights fall back to the raw DGM1 points (nearest neighbor).
            road_surfaces: optional - unioned embedded road surface (shapely, local) or list of dicts
                           with "road_polygon"; no trees stand there or within FOREST_ROAD_SURFACE_MARGIN of it

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
            logger.debug(f"\n[Forest Phase 1b] Starting for {tile_name} (bounds: {tile_bounds})")

            # Initialize osm_data
            osm_data = None

            # Check whether set_forest_config() was called
            if not self.normalizer or not self.instance_generator:
                logger.error(f"[Forest ERROR] set_forest_config() not called!")
                return {
                    "status": "error",
                    "tile_name": tile_name,
                    "tile_bounds": tile_bounds,
                    "tree_count": 0,
                    "forests_count": 0,
                    "tree_instances": [],
                    "error": "set_forest_config() not called",
                }

            # Cache: Poisson disk sampling + height interpolation + instance generation are the
            # most expensive part below (tens of thousands of trees) - with unchanged area/elevation data/
            # config directly reuse the finished tree instances (see _forest_cache_key()).
            cache_key = self._forest_cache_key(tile_bounds, global_offset, height_hash)
            cached = self._load_cached_tree_instances(cache_key)
            if cached is not None:
                tree_instances, forests_count = cached
                self.all_tree_instances.extend(tree_instances)
                logger.info(f"  [OK] Forest cache found: {len(tree_instances)} tree instances (already computed)")
                return {
                    "status": "success",
                    "tile_name": tile_name,
                    "tile_bounds": tile_bounds,
                    "tree_count": len(tree_instances),
                    "forests_count": forests_count,
                    "tree_instances": tree_instances,
                    "error": None,
                }

            # Phase 1b: Normalization (with already loaded OSM data)
            if not osm_data:
                logger.debug(f"  [→] Loading OSM data from cache...")
                from ..osm.downloader import get_osm_data
                from ..geometry.coordinates import transformer_to_wgs84

                # Convert local bounds back to UTM (simply + offset)
                # global_offset can be (x, y) or (x, y, z) - we only need (x, y)
                if global_offset:
                    ox, oy = global_offset[0], global_offset[1]
                else:
                    ox, oy = 0, 0
                utm_x_min = tile_bounds[0] + ox
                utm_y_min = tile_bounds[1] + oy
                utm_x_max = tile_bounds[2] + ox
                utm_y_max = tile_bounds[3] + oy

                # Convert UTM to lat/lon for the BBox (the Overpass query needs lat/lon)
                lat_min, lon_min = transformer_to_wgs84.transform(utm_x_min, utm_y_min)
                lat_max, lon_max = transformer_to_wgs84.transform(utm_x_max, utm_y_max)

                # Overpass BBox: (lat_min, lon_min, lat_max, lon_max)
                bbox_tuple = (lat_min, lon_min, lat_max, lon_max)

                # Use height_hash for cache consistency (like the terrain workflow)
                osm_data = get_osm_data(bbox_tuple, height_hash=height_hash)
                logger.debug(f"  [→] {len(osm_data) if osm_data else 0} OSM elements loaded")

            if not osm_data:
                logger.warning(f"  [→] No OSM data available")
                return {
                    "status": "no_forests",
                    "tile_name": tile_name,
                    "tile_bounds": tile_bounds,
                    "tree_count": 0,
                    "forests_count": 0,
                    "tree_instances": [],
                    "error": None,
                }

            logger.debug(f"  [→] Normalizing OSM forest polygons...")

            # Compute local_offset for the coordinate transformation
            # global_offset can be (x, y) or (x, y, z) - we only need (x, y)
            if global_offset:
                ox, oy = global_offset[0], global_offset[1]
            else:
                ox, oy = 0, 0

            # CENTRAL TRANSFORMATION: convert ALL OSM geometries to local coordinates once
            logger.debug(f"  [→] Transforming OSM data to local coordinates...")
            osm_data = self._transform_osm_to_local(osm_data, (ox, oy))

            # From now on: ALL geometries in osm_data are in local coordinates!
            # WGS84 (lat/lon) no longer exists - only local (x, y)!

            # Use the real global_offset for the forest transformation
            forest_local_offset = (ox, oy)

            normalized = self.normalizer.normalize_tile(
                tile_bounds, tile_name, osm_data=osm_data, local_offset=forest_local_offset
            )
            logger.debug(
                f"  [Forest] Normalization: {normalized.get('status')} - {normalized.get('forest_count')} forests"
            )

            # DEBUG: save a dump if forest_count = 0
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
                logger.debug(f"  [DEBUG] Dump written: {dump_file}")

            if normalized["status"] != "success" or normalized["forest_count"] == 0:
                logger.error(f"  [Forest] No forests found: {normalized.get('error', 'unknown error')}")
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
            logger.debug(f"  [→] {len(forests)} forest polygons to process")

            # Phase 2: Point generation (Poisson disk sampling)
            logger.debug(f"  [→] Generating tree positions (Poisson disk)...")

            # Create the road buffer (OSM data already in local coordinates!)
            road_buffer = self._create_road_buffer(osm_data)
            if road_buffer:
                logger.debug(
                    f"  [Forest] Road buffer created - bounds: {road_buffer.bounds}, area: {road_buffer.area:.0f}m²"
                )
            else:
                logger.debug(f"  [Forest] Road buffer is None!")
            # Trees/bushes must stand neither on roads nor in/at buildings (gardens, residential areas)
            building_buffer = self._create_building_buffer(osm_data)
            if building_buffer is not None:
                logger.debug(f"  [Forest] Building buffer created - area: {building_buffer.area:.0f}m²")
                from shapely.ops import unary_union

                exclusion = unary_union([road_buffer, building_buffer]) if road_buffer else building_buffer
            else:
                exclusion = road_buffer
            # Actually embedded (smoothed, true-width) road surfaces in addition to the raw OSM line buffer
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

            # Individual trees (OSM natural=tree as a point) as their own synthetic "forest" entry
            single_type = self.forest_config.get("forest_mappings", {}).get("single_trees", {}).get("forest_type")
            if single_type and single_type in forest_properties:
                singles = self._single_tree_points(osm_data, tile_bounds, (ox, oy), exclusion)
                if singles:
                    forests.append({"type": single_type, "geometry": None, "osm_tags": {"natural": "tree"}})
                    forest_points[len(forests) - 1] = singles
                    logger.debug(f"  [Forest] {len(singles)} single trees (natural=tree)")

            total_points = sum(len(pts) for pts in forest_points.values())
            logger.debug(f"  [→] {total_points} tree positions generated")

            # Phase 3: Height interpolation (bilinear interpolation)
            logger.debug(f"  [→] Interpolating heights...")
            forest_points_3d = self.height_calculator.calculate_heights_for_forest_points(
                forest_points=forest_points,
                height_points=elevation_data,
                height_elevations=height_grid_info.get("elevations") if height_grid_info else None,
                grid_info=height_grid_info,
                height_at=height_at,
            )

            logger.debug(f"  [→] Heights interpolated for {total_points} points")

            # Phase 4: Instance generation (type, rotation, scale)
            logger.debug(f"  [→] Generating tree instances...")
            # The origins keep the distances; the trunks of group assets (up to ~9 m beside them) must do so too,
            # and they must not hang in the air. Same zones and distances as above, only checked per trunk.
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

            # Collect instances for the final export
            self.all_tree_instances.extend(tree_instances)
            self._save_cached_tree_instances(cache_key, tree_instances, len(forests))

            logger.debug(f"  [✓] {len(tree_instances)} tree instances generated for {tile_name}")

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
            logger.error(f"Error in forest processing for {tile_name}: {e}", exc_info=True)
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
        Adds additional forest instances (e.g. vineyard vines) that do not come from
        forest polygons. They are written to forest.forest4.json in finalize_forest_export() together with the
        trees.

        Args:
            instances: instances in forest4 format (type, pos, rotationMatrix, scale)

        Returns:
            Number of added instances
        """
        self.all_tree_instances.extend(instances)
        return len(instances)

    def finalize_forest_export(self) -> Dict:
        """
        FINALIZATION (after the tile loop): write forest.forest4.json.

        Collects all tree instances from process_tile() and writes forest.forest4.json.

        MUST be called AFTER the tile loop!

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
            logger.debug(f"[Forest] Finalizing export ({len(self.all_tree_instances)} instances)...")

            # Check whether instances exist
            if not self.all_tree_instances:
                logger.warning("[Forest] No tree instances generated, skipping forest.forest4.json")
                return {
                    "status": "no_forests",
                    "total_trees": 0,
                    "forest_json_path": "",
                    "statistics": {},
                    "error": None,
                }

            # Check whether the JSON writer is initialized
            if not self.json_writer:
                logger.info("[Forest ERROR] ForestJSONWriter not initialized!")
                return {
                    "status": "error",
                    "total_trees": 0,
                    "forest_json_path": "",
                    "statistics": {},
                    "error": "ForestJSONWriter not initialized",
                }

            # Write forest.forest4.json
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

            # Statistics
            statistics = self.json_writer.get_statistics(self.all_tree_instances)

            logger.info(f"[✓] Forest export finished:")
            logger.info(f"  - Total trees: {statistics['total_trees']}")
            logger.info(f"  - Tree species: {len(statistics['types'])}")
            for tree_type, count in sorted(statistics["types"].items()):
                logger.info(f"    • {tree_type}: {count}")
            logger.info(f"  - Avg. scale: {statistics['avg_scale']:.2f}")
            logger.info(f"  - Height range: {statistics['min_height']:.1f}m - {statistics['max_height']:.1f}m")

            return {
                "status": "success",
                "total_trees": len(self.all_tree_instances),
                "forest_json_path": write_result["filepath"],
                "statistics": statistics,
                "error": None,
            }

        except Exception as e:
            logger.info(f"[Forest ERROR] Forest finalization: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "total_trees": 0, "forest_json_path": "", "statistics": {}, "error": str(e)}
