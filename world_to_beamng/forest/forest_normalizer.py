"""
Forest-Normalisierung: Clippe OSM-Waldpolygone auf Tile-Grenzen.

Pro Tile (2x2 km):
- Extrahiere relevante OSM-Waldpolygone
- Clippe auf Tile-Grenzen
- Ordne Forest-Type basierend auf Konfiguration zu
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from shapely.geometry import box, Polygon, MultiPolygon

from .. import config
from ..osm.osm_mapper import OSMMapper

logger = logging.getLogger(__name__)


class ForestNormalizer:
    """
    Normalisiert OSM-Waldpolygone für die Generierung pro Tile.

    Jedes OSM-Waldpolygon wird auf Tile-Grenzen gekürzt und mit
    einem forest_type versehen.
    """

    def __init__(self, forest_config: Dict, osm_mapper: OSMMapper):
        """
        Args:
            forest_config: Dict aus osm_to_beamng.json["forest_types"] + ["forest_mappings"]
            osm_mapper: OSMMapper-Instance mit geladenen OSM-Daten
        """
        self.forest_config = forest_config
        self.osm_mapper = osm_mapper
        self.forest_types = forest_config.get("forest_types", {})

        # Fallback: Hole forest_mappings aus osm_mapper wenn nicht in forest_config
        if "forest_mappings" in forest_config:
            self.forest_mappings = forest_config.get("forest_mappings", {})
        else:
            self.forest_mappings = osm_mapper.forest_mappings

        logger.info(
            f"✓ ForestNormalizer initialisiert ({len(self.forest_types)} Waldtypen, {len(self.forest_mappings)} Mappings)"
        )

    def normalize_tile(
        self,
        tile_bounds: Tuple[float, float, float, float],
        tile_name: str = "unknown",
        osm_data: Optional[List[Dict]] = None,
        local_offset: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Normalisiere OSM-Wälder für ein einzelnes Tile.

        Blendet alle OSM-Waldpolygone ein, die sich mit dem Tile überschneiden,
        clippt sie auf Tile-Grenzen, und ordnet einen forest_type zu.

        Args:
            tile_bounds: Tuple (x_min, y_min, x_max, y_max) in lokalen Koordinaten
            tile_name: Optional - Name des Tiles für Logging
            osm_data: Optional - OSM-Rohdaten (Liste von Elements mit tags, geometry in lat/lon)
            local_offset: Optional - (offset_x, offset_y) für Koordinaten-Transformation

        Returns:
            Dict mit Format:
            {
                "status": "success" | "error",
                "tile_bounds": (x_min, y_min, x_max, y_max),
                "tile_name": str,
                "forests": [
                    {
                        "type": "deciduous_dense",
                        "geometry": Polygon (in lokalen Koordinaten!),
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

            # Extrahiere Waldpolygone aus OSM-Rohdaten (WGS84)
            osm_forests = self._extract_forests_from_osm(osm_data) if osm_data else []
            if not osm_forests:
                print(f"  [→] Keine Wälder in {tile_name}")
                return result

            print(f"  [→] Prüfe {len(osm_forests)} OSM-Waldpolygone...")

            # Jetzt: Alle Geometrien sind in lokalen Koordinaten (bereits transformiert in workflow!)
            # Iteriere über alle OSM-Waldpolygone
            for osm_forest in osm_forests:
                geom = osm_forest.get("geometry")  # In LOKALEN Koordinaten
                tags = osm_forest.get("tags", {})

                if not geom or geom.is_empty:
                    continue

                # Prüfe Überschneidung mit Tile (beide in lokalen Koordinaten!)
                if not geom.intersects(tile_box):
                    continue

                # WICHTIG: Behalte das GANZE Polygon ungeclippt!
                # Die Punkt-Generierung prüft später pro Punkt, ob er im Tile liegt.
                # Damit vermeiden wir Wald-Verlust an Tile-Rändern (z.B. Schwarzwald über mehrere Tiles)

                # Bestimme Forest-Type basierend auf OSM-Tags
                forest_type = self._map_to_forest_type(tags)
                if not forest_type:
                    logger.debug(f"    [i] Waldpolygon gemappt zu keinem forest_type: {tags}")
                    continue

                # Erstelle Forest-Eintrag mit UNGECLIPPTEM Polygon
                forest_entry = {
                    "type": forest_type,
                    "geometry": geom,  # UNGECLIPPT! Enthält ggf. Punkte außerhalb des Tiles
                    "bounds": tuple(geom.bounds),  # (x_min, y_min, x_max, y_max) in lokal - Gesamt-Polygon
                    "tile_box": tile_box,  # Für Punkt-Filterung später!
                    "osm_tags": tags,
                    "properties": {
                        "name": tags.get("name", "unnamed"),
                        "area": geom.area,  # In m² - des GANZEN Polygons
                        "perimeter": geom.length,  # Des GANZEN Polygons
                    },
                }

                result["forests"].append(forest_entry)
                logger.debug(
                    f"    ✓ Waldpolygon (ungeclippt): {forest_type} " f"({geom.area:.0f} m², " f"{geom.geom_type})"
                )

            result["forest_count"] = len(result["forests"])
            if result["forest_count"] > 0:
                logger.info(f"  [✓] Tile {tile_name}: {result['forest_count']} Waldpolygone normalisiert")
            else:
                logger.debug(f"  [i] Tile {tile_name}: Keine Wälder nach Normalisierung")

            return result

        except Exception as e:
            logger.error(f"Fehler bei Normalisierung von Tile {tile_name}: {e}", exc_info=True)
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
        Extrahiere Waldpolygone aus OSM-Rohdaten.

        Sucht nach:
        - landuse=forest
        - landuse=wood
        - natural=wood
        - natural=forest

        Verarbeitet:
        - Einfache Ways mit Waldtags
        - Multipolygon-Relations (type=multipolygon) mit Waldtags

        WICHTIG: Erwartet LOKALE Koordinaten {x, y}!
        (Zentrale Transformation in ForestWorkflow._transform_osm_to_local() erfolgt VORHER!)

        Args:
            osm_data: OSM-Elements mit tags und geometry (Overpass-Format, bereits transformiert!)

        Returns:
            Liste von Dicts mit "geometry" (Shapely Polygon), "tags"
        """
        from shapely.geometry import Polygon, MultiPolygon, LineString
        from shapely.ops import unary_union

        forests = []

        # Erstelle Index: way_id → way_element (für Multipolygon-Assembly)
        ways_by_id = {}
        for element in osm_data:
            if element.get("type") == "way":
                ways_by_id[element.get("id")] = element

        for element in osm_data:
            # Sicherheitscheck: element muss ein Dict sein
            if not isinstance(element, dict):
                print(f"  [!] Element ist kein Dict: {type(element)}")
                continue

            tags = element.get("tags", {})
            if not isinstance(tags, dict):
                print(f"  [!] Tags sind kein Dict: {type(tags)}")
                continue

            # Prüfe mit OSMMapper ob es ein Wald ist
            if not self.osm_mapper.is_forest(tags):
                continue

            element_type = element.get("type", "way")

            # === CASE 1: Relation (Multipolygon) ===
            if element_type == "relation" and tags.get("type") == "multipolygon":
                try:
                    # Versuche Geometrie aus members zusammenzusetzen
                    geom = self._build_multipolygon_from_members(element, ways_by_id)
                    if geom and not geom.is_empty:
                        forests.append(
                            {"geometry": geom, "tags": tags, "osm_id": element.get("id"), "type": "relation"}
                        )
                except Exception as e:
                    logger.debug(f"  [!] Fehler beim Multipolygon-Assembly: {e}")
                    continue

            # === CASE 2: Way (einfaches Polygon) ===
            else:
                # Extrahiere Geometrie (BEREITS in lokalen Koordinaten!)
                geom_data = element.get("geometry")
                if not geom_data:
                    continue

                try:
                    # Geometrie MUSS bereits transformiert sein: {x, y}
                    # (Zentrale Transformation in ForestWorkflow._transform_osm_to_local() erfolgte VORHER!)
                    if isinstance(geom_data, list) and len(geom_data) > 0:
                        if isinstance(geom_data[0], dict) and "x" in geom_data[0] and "y" in geom_data[0]:
                            # Lokale Koordinaten - CORRECT!
                            coords = [(pt["x"], pt["y"]) for pt in geom_data]
                            if len(coords) >= 3:  # Polygon benötigt mind. 3 Punkte
                                geom = Polygon(coords)

                                if geom.is_valid:
                                    forests.append(
                                        {"geometry": geom, "tags": tags, "osm_id": element.get("id"), "type": "way"}
                                    )
                except Exception as e:
                    logger.debug(f"  [!] Fehler beim Parsing von Waldgeometrie: {e}")
                    continue

        return forests

    def _build_multipolygon_from_members(self, relation: Dict, ways_by_id: Dict):
        """
        Versuche Geometrie eines Multipolygon aus seinen Member-Ways zu bauen.

        WICHTIG: Erwartet lokale Koordinaten {x, y}!
        (Zentrale Transformation in ForestWorkflow._transform_osm_to_local() erfolgt VORHER)

        Args:
            relation: Relation-Element mit members
            ways_by_id: Index way_id → way_element

        Returns:
            Shapely Polygon/MultiPolygon oder None
        """
        from shapely.geometry import Polygon, LineString, MultiPolygon

        members = relation.get("members", [])
        if not members:
            return None

        # Sammle outer ways (diese ergeben die Außenkante)
        outer_coords_list = []

        for member in members:
            if member.get("type") != "way":
                continue

            role = member.get("role", "")
            if role != "outer":  # Wir interessieren uns nur für outer
                continue

            way_id = member.get("ref")

            if way_id not in ways_by_id:
                # Way nicht im Cache
                continue

            way = ways_by_id[way_id]
            geom_data = way.get("geometry")

            if not geom_data:
                continue

            # Parse Way-Geometrie (MUSS bereits in lokalen Koordinaten {x, y} sein!)
            coords = None
            if isinstance(geom_data, list) and len(geom_data) > 0:
                if isinstance(geom_data[0], dict) and "x" in geom_data[0] and "y" in geom_data[0]:
                    # Lokale Koordinaten - CORRECT!
                    coords = [(pt["x"], pt["y"]) for pt in geom_data]

            if coords and len(coords) >= 2:
                outer_coords_list.append(coords)

        if not outer_coords_list:
            return None

        try:
            # Wenn wir nur einen outer haben, nimm ihn direkt
            if len(outer_coords_list) == 1:
                coords = outer_coords_list[0]
                if len(coords) >= 3:
                    # Stelle sicher dass Polygon geschlossen ist
                    if coords[0] != coords[-1]:
                        coords = coords + [coords[0]]

                    poly = Polygon(coords)
                    return poly if poly.is_valid else None

            # Mehrere outer ways - versuche sie zu merger
            else:
                # Baue Ring aus jedem set of coords
                rings = []
                for coords in outer_coords_list:
                    if len(coords) >= 3:
                        if coords[0] != coords[-1]:
                            coords = coords + [coords[0]]
                        ring_poly = Polygon(coords)
                        if ring_poly.is_valid:
                            rings.append(ring_poly)

                if not rings:
                    return None
                elif len(rings) == 1:
                    return rings[0]
                else:
                    # Mehrere Polygone - als MultiPolygon zurückgeben
                    return MultiPolygon(rings)

        except Exception as e:
            logger.debug(f"Fehler beim Multipolygon-Assembly: {e}")
            return None

    def _map_to_forest_type(self, osm_tags: Dict) -> Optional[str]:
        """
        Mappe OSM-Tags zu forest_type.

        Folgt der Logik aus forest_mappings:
        1. Prüfe tag_overrides (z.B. trees=conifer)
        2. Prüfe landuse/natural/leisure Tags
        3. Fallback: None

        Args:
            osm_tags: Dict mit OSM-Tags

        Returns:
            forest_type-String oder None
        """
        mappings = self.forest_mappings

        # 1. Tag-Overrides (höchste Priorität)
        if "tag_overrides" in mappings:
            for override_key, override_value in mappings["tag_overrides"].items():
                if osm_tags.get(override_key.split("=")[0]) == override_key.split("=")[1]:
                    return override_value

        # 2. Landuse/Natural/Leisure
        for tag_key in ["landuse", "natural", "leisure"]:
            if tag_key not in mappings:
                continue

            tag_value = osm_tags.get(tag_key)
            if tag_value and tag_value in mappings[tag_key]:
                return mappings[tag_key][tag_value]

        # 3. Fallback
        return None

    def get_forest_properties(self, forest_type: str) -> Dict:
        """
        Hole die Eigenschaften eines Waldtyps.

        Args:
            forest_type: Name des Waldtyps (z.B. "deciduous_dense")

        Returns:
            Dict mit tree_density, tree_distribution, average_height, etc.
        """
        return self.forest_types.get(forest_type, {})

    def _transform_forests_wgs84_to_local(
        self, osm_forests: List[Dict], local_offset: Tuple[float, float], transformer_to_wgs84
    ) -> List[Dict]:
        """
        ZENTRALE NORMALISIERUNG: Transformiere ALLE Waldpolygone von WGS84 (lat/lon) zu lokalen Koordinaten.

        Diese Funktion ist der einzelne Punkt, wo alle Koordinaten normalisiert werden.
        Nach diesem Aufruf sind ALLE Waldgeometrien in lokalen Koordinaten!

        Args:
            osm_forests: Liste von Wäldern mit geometry in WGS84 (lon, lat)
            local_offset: (offset_x, offset_y) in UTM
            transformer_to_wgs84: pyproj Transformer von UTM zu WGS84

        Returns:
            Liste von Wäldern mit geometry in lokalen Koordinaten (oder leere Liste bei Fehler)
        """
        from pyproj import Transformer

        try:
            # Erstelle inverse Transformer: WGS84 → UTM
            transformer_utm = Transformer.from_proj(
                transformer_to_wgs84.target_crs,  # WGS84 (target von to_wgs84)
                transformer_to_wgs84.source_crs,  # UTM (source von to_wgs84)
            )
        except Exception as e:
            print(f"  [!] Fehler beim Erstellen des inversen Transformers: {e}")
            return []

        transformed_forests = []
        ox, oy = local_offset

        for forest in osm_forests:
            geom_wgs84 = forest.get("geometry")
            if not geom_wgs84 or geom_wgs84.is_empty:
                continue

            try:
                # Sammle alle Polygone (für MultiPolygon: ALLE Polygone, nicht nur das größte!)
                polygons_to_process = []

                if geom_wgs84.geom_type == "Polygon":
                    polygons_to_process = [geom_wgs84]
                elif geom_wgs84.geom_type == "MultiPolygon":
                    # WICHTIG: MultiPolygon → alle einzelnen Polygone verarbeiten!
                    # (z.B. Wald mit mehreren Inseln oder Lichtungen)
                    polygons_to_process = list(geom_wgs84.geoms)
                else:
                    continue

                # Transformiere JEDES Polygon einzeln
                for poly_wgs84 in polygons_to_process:
                    if not poly_wgs84 or poly_wgs84.is_empty:
                        continue

                    coords_wgs84 = list(poly_wgs84.exterior.coords)

                    # Transformiere jeden Punkt (lon, lat) → (utm_x, utm_y) → (local_x, local_y)
                    local_coords = []
                    for lon, lat in coords_wgs84:
                        # transformer_utm.transform(lon, lat) → (utm_x, utm_y)
                        # WICHTIG: pyproj erwartet (x, y) = (lon, lat) für WGS84, gibt (x, y) = (utm_easting, utm_northing)
                        utm_x, utm_y = transformer_utm.transform(lon, lat)
                        local_x = utm_x - ox
                        local_y = utm_y - oy
                        local_coords.append((local_x, local_y))

                    # Erstelle transformiertes Polygon
                    if len(local_coords) >= 3:
                        geom_local = Polygon(local_coords)
                        if geom_local.is_valid:
                            # Erstelle neuen Forest-Eintrag mit transformierter Geometrie
                            forest_transformed = forest.copy()
                            forest_transformed["geometry"] = geom_local
                            transformed_forests.append(forest_transformed)

            except Exception as e:
                logger.debug(f"  [!] Fehler bei Transformation eines Waldpolygons: {e}")
                continue

        return transformed_forests
