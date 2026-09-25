"""
Forest normalization: clip OSM forest polygons to tile boundaries.

Per tile (2x2 km):
- Extract relevant OSM forest polygons
- Clip to tile boundaries
- Assign the forest type based on configuration
"""

from typing import Dict, List, Optional, Tuple
from shapely.geometry import box, Polygon

from ..osm.osm_mapper import OSMMapper
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()


class ForestNormalizer:
    """
    Normalizes OSM forest polygons for generation per tile.

    Each OSM forest polygon is trimmed to tile boundaries and provided with
    a forest_type.
    """

    def __init__(self, forest_config: Dict, osm_mapper: OSMMapper):
        """
        Args:
            forest_config: dict from osm_to_beamng.json["forest_type_templates"] + ["forest_mappings"]
            osm_mapper: OSMMapper instance with loaded OSM data
        """
        self.forest_config = forest_config
        self.osm_mapper = osm_mapper
        self.forest_types = forest_config.get("forest_type_templates", {})

        # Fallback: get forest_mappings from osm_mapper if not in forest_config
        if "forest_mappings" in forest_config:
            self.forest_mappings = forest_config.get("forest_mappings", {})
        else:
            self.forest_mappings = osm_mapper.forest_mappings

        logger.info(
            f"✓ ForestNormalizer initialized ({len(self.forest_types)} forest types, {len(self.forest_mappings)} mappings)"
        )

    def normalize_tile(
        self,
        tile_bounds: Tuple[float, float, float, float],
        tile_name: str = "unknown",
        osm_data: Optional[List[Dict]] = None,
        local_offset: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Normalize OSM forests for a single tile.

        Includes all OSM forest polygons that overlap the tile,
        clips them to tile boundaries, and assigns a forest_type.

        Args:
            tile_bounds: tuple (x_min, y_min, x_max, y_max) in local coordinates
            tile_name: optional - name of the tile for logging
            osm_data: optional - raw OSM data (list of elements with tags, geometry in lat/lon)
            local_offset: optional - (offset_x, offset_y) for coordinate transformation

        Returns:
            Dict with format:
            {
                "status": "success" | "error",
                "tile_bounds": (x_min, y_min, x_max, y_max),
                "tile_name": str,
                "forests": [
                    {
                        "type": "deciduous_dense",
                        "geometry": Polygon (in local coordinates!),
                        "bounds": (x_min, y_min, x_max, y_max),
                        "osm_tags": {...},
                        "properties": {...}
                    },
                    ...
                ],
                "forest_count": int,
                "error": Optional[str]
            }
        """
        try:
            x_min, y_min, x_max, y_max = tile_bounds
            tile_box = box(x_min, y_min, x_max, y_max)

            result = {
                "status": "success",
                "tile_bounds": tile_bounds,
                "tile_name": tile_name,
                "forests": [],
                "forest_count": 0,
                "error": None,
            }

            # Extract forest polygons from raw OSM data (WGS84)
            osm_forests = self._extract_forests_from_osm(osm_data) if osm_data else []
            if not osm_forests:
                logger.info(f"  [→] No forests in {tile_name}")
                return result

            logger.info(f"  [→] Checking {len(osm_forests)} OSM forest polygons...")

            # Now: all geometries are in local coordinates (already transformed in the workflow!)
            # Iterate over all OSM forest polygons
            for osm_forest in osm_forests:
                geom = osm_forest.get("geometry")  # In LOCAL coordinates
                tags = osm_forest.get("tags", {})

                if not geom or geom.is_empty:
                    continue

                # Check overlap with the tile (both in local coordinates!)
                if not geom.intersects(tile_box):
                    continue

                # IMPORTANT: keep the WHOLE polygon unclipped!
                # Point generation later checks per point whether it lies in the tile.
                # This avoids forest loss at tile edges (e.g. Black Forest across several tiles)

                # Determine the forest type based on OSM tags; clearings (inner rings) get
                # the type configured in forest_mappings["clearings"] (e.g. low deciduous forest)
                forest_type = self._map_to_forest_type(tags)
                if osm_forest.get("is_clearing"):
                    clearings = self.forest_mappings.get("clearings", {})
                    only_for = clearings.get("only_for")
                    if only_for is not None and self._base_forest_mapping(tags)[1] not in only_for:
                        continue  # a hole in a residential area etc. is not a forest clearing: do not plant
                    forest_type = clearings.get("forest_type", forest_type)
                if not forest_type:
                    logger.debug(f"    [i] Forest polygon mapped to no forest_type: {tags}")
                    continue

                # Create the forest entry with the UNCLIPPED polygon
                forest_entry = {
                    "type": forest_type,
                    "geometry": geom,  # UNCLIPPED! May contain points outside the tile
                    "bounds": tuple(geom.bounds),  # (x_min, y_min, x_max, y_max) in local - whole polygon
                    "tile_box": tile_box,  # For point filtering later!
                    "osm_tags": tags,
                    "properties": {
                        "name": tags.get("name", "unnamed"),
                        "area": geom.area,  # In m² - of the WHOLE polygon
                        "perimeter": geom.length,  # Of the WHOLE polygon
                    },
                }

                result["forests"].append(forest_entry)
                logger.debug(
                    f"    ✓ Forest polygon (unclipped): {forest_type} " f"({geom.area:.0f} m², " f"{geom.geom_type})"
                )

            result["forest_count"] = len(result["forests"])
            if result["forest_count"] > 0:
                logger.info(f"  [✓] Tile {tile_name}: {result['forest_count']} forest polygons normalized")
            else:
                logger.debug(f"  [i] Tile {tile_name}: no forests after normalization")

            return result

        except Exception as e:
            logger.error(f"Error normalizing tile {tile_name}: {e}", exc_info=True)
            return {
                "status": "error",
                "tile_bounds": tile_bounds,
                "tile_name": tile_name,
                "forests": [],
                "forest_count": 0,
                "error": str(e),
            }

    def _extract_forests_from_osm(self, osm_data: List[Dict]) -> List[Dict]:
        """
        Extract forest polygons from raw OSM data.

        Searches for:
        - landuse=forest
        - landuse=wood
        - natural=wood
        - natural=forest

        Handles:
        - Simple ways with forest tags
        - Multipolygon relations (type=multipolygon) with forest tags

        IMPORTANT: expects LOCAL coordinates {x, y}!
        (The central transformation in ForestWorkflow._transform_osm_to_local() happens BEFORE!)

        Args:
            osm_data: OSM elements with tags and geometry (Overpass format, already transformed!)

        Returns:
            List of dicts with "geometry" (Shapely Polygon), "tags"
        """
        from shapely.geometry import Polygon, LineString
        from shapely.ops import unary_union

        forests = []

        # Build index: way_id → way_element (for multipolygon assembly)
        ways_by_id = {}
        for element in osm_data:
            if element.get("type") == "way":
                ways_by_id[element.get("id")] = element

        for element in osm_data:
            # Safety check: element must be a dict
            if not isinstance(element, dict):
                logger.error(f"  [!] Element is not a dict: {type(element)}")
                continue

            tags = element.get("tags", {})
            if not isinstance(tags, dict):
                logger.error(f"  [!] Tags are not a dict: {type(tags)}")
                continue

            # Use OSMMapper to check whether it is a forest
            if not self.osm_mapper.is_forest(tags):
                continue

            element_type = element.get("type", "way")

            # === CASE 1: Relation (Multipolygon) ===
            if element_type == "relation" and tags.get("type") == "multipolygon":
                try:
                    # Forest without clearings + clearings (inner rings) separately
                    geom, clearings = self._build_multipolygon_from_members(element, ways_by_id)
                    if geom and not geom.is_empty:
                        forests.append(
                            {"geometry": geom, "tags": tags, "osm_id": element.get("id"), "type": "relation"}
                        )
                    if clearings is not None:
                        forests.append(
                            {
                                "geometry": clearings,
                                "tags": tags,
                                "osm_id": element.get("id"),
                                "type": "relation",
                                "is_clearing": True,
                            }
                        )
                except Exception as e:
                    logger.debug(f"  [!] Error in multipolygon assembly: {e}")
                    continue

            # === CASE 2: Way (simple polygon) ===
            else:
                # Extract geometry (ALREADY in local coordinates!)
                geom_data = element.get("geometry")
                if not geom_data:
                    continue

                try:
                    # Geometry MUST already be transformed: {x, y}
                    # (The central transformation in ForestWorkflow._transform_osm_to_local() happened BEFORE!)
                    if isinstance(geom_data, list) and len(geom_data) > 0:
                        if isinstance(geom_data[0], dict) and "x" in geom_data[0] and "y" in geom_data[0]:
                            # Local coordinates - CORRECT!
                            coords = [(pt["x"], pt["y"]) for pt in geom_data]
                            if self._is_row_type(tags):
                                # Tree row (natural=tree_row): a LINE, not a polygon - the trees later stand
                                # along the line at row_spacing intervals
                                if len(coords) >= 2:
                                    forests.append(
                                        {
                                            "geometry": LineString(coords),
                                            "tags": tags,
                                            "osm_id": element.get("id"),
                                            "type": "way",
                                        }
                                    )
                                continue
                            if len(coords) >= 3:  # a polygon needs at least 3 points
                                geom = Polygon(coords)

                                if geom.is_valid:
                                    forests.append(
                                        {"geometry": geom, "tags": tags, "osm_id": element.get("id"), "type": "way"}
                                    )
                except Exception as e:
                    logger.debug(f"  [!] Error parsing forest geometry: {e}")
                    continue

        return forests

    def _build_multipolygon_from_members(self, relation: Dict, ways_by_id: Dict):
        """
        Build forest and clearings of a multipolygon from its member ways.

        The segments of a role are assembled into closed rings (OSM splits long rings into
        several ways - closing each one individually would yield wrong areas).
        Inner rings are clearings: they are subtracted from the forest and returned separately.

        IMPORTANT: expects local coordinates {x, y}!
        (The central transformation in ForestWorkflow._transform_osm_to_local() happens BEFORE)

        Args:
            relation: relation element with members
            ways_by_id: index way_id → way_element

        Returns:
            (forest geometry without clearings, clearing geometry or None) - or (None, None)
        """
        from shapely.geometry import LineString
        from shapely.ops import polygonize, unary_union

        def role_polygons(roles):
            lines = []
            for member in relation.get("members", []):
                if member.get("type") != "way" or member.get("role", "") not in roles:
                    continue
                way = ways_by_id.get(member.get("ref"))
                geom_data = way.get("geometry") if way else None
                if isinstance(geom_data, list) and geom_data and isinstance(geom_data[0], dict) and "x" in geom_data[0]:
                    coords = [(pt["x"], pt["y"]) for pt in geom_data]
                    if len(coords) >= 2:
                        lines.append(LineString(coords))
            return list(polygonize(unary_union(lines))) if lines else []

        try:
            outer_polygons = role_polygons(("outer", ""))
            if not outer_polygons:
                return None, None
            outer = unary_union(outer_polygons)
            inner_polygons = role_polygons(("inner",))
            if not inner_polygons:
                return (outer if outer.is_valid else outer.buffer(0)), None

            clearings = unary_union(inner_polygons).intersection(outer)
            forest = outer.difference(clearings)
            forest = forest if forest.is_valid else forest.buffer(0)
            return (None if forest.is_empty else forest), (None if clearings.is_empty else clearings)
        except Exception as e:
            logger.debug(f"Error in multipolygon assembly: {e}")
            return None, None

    def _is_row_type(self, osm_tags: Dict) -> bool:
        """True if the forest type of the tags is a tree row (template with row_spacing): line instead of area."""
        base_type, _ = self._base_forest_mapping(osm_tags)
        return bool(base_type and self.forest_types.get(base_type, {}).get("row_spacing"))

    def _base_forest_mapping(self, osm_tags: Dict) -> Tuple[Optional[str], Optional[str]]:
        """(base forest type, triggering tag "key=value") from landuse/natural/leisure or (None, None)."""
        for tag_key in ["landuse", "natural", "leisure"]:
            values = self.forest_mappings.get(tag_key)
            if not values:
                continue
            tag_value = osm_tags.get(tag_key)
            if tag_value and tag_value in values:
                return values[tag_value], f"{tag_key}={tag_value}"
        return None, None

    def _map_to_forest_type(self, osm_tags: Dict) -> Optional[str]:
        """
        Map OSM tags to forest_type.

        Follows the logic from forest_mappings:
        1. Base type from landuse/natural/leisure
        2. tag_overrides (e.g. trees=conifer) refine the base type - but only if the
           base tag is listed in "tag_overrides_only_for" (e.g. "landuse=forest"). Otherwise a
           natural=wood with a conifer tag could be redirected to a tall forest type. If the
           key is missing, the overrides apply to all (backward compatible).
        3. Fallback: None

        Args:
            osm_tags: dict with OSM tags

        Returns:
            forest_type string or None
        """
        mappings = self.forest_mappings
        base_type, base_tag = self._base_forest_mapping(osm_tags)

        # 2. Tag overrides (only for allowed base tags)
        overrides = mappings.get("tag_overrides")
        scope = mappings.get("tag_overrides_only_for")
        if overrides and (scope is None or base_tag in scope):
            for override_key, override_value in overrides.items():
                key, value = override_key.split("=", 1)
                if osm_tags.get(key) == value:
                    return override_value

        # 3. Base type (or None)
        return base_type

    def get_forest_properties(self, forest_type: str) -> Dict:
        """
        Get the properties of a forest type.

        Args:
            forest_type: name of the forest type (e.g. "deciduous_dense")

        Returns:
            Dict with tree_density, tree_distribution, average_height, etc.
        """
        return self.forest_types.get(forest_type, {})
