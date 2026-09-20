"""
Forest-Normalisierung: Clippe OSM-Waldpolygone auf Tile-Grenzen.

Pro Tile (2x2 km):
- Extrahiere relevante OSM-Waldpolygone
- Clippe auf Tile-Grenzen
- Ordne Forest-Type basierend auf Konfiguration zu
"""

from typing import Dict, List, Optional, Tuple
from shapely.geometry import box, Polygon

from ..osm.osm_mapper import OSMMapper
from world_to_beamng.logging_config import LoggerConfig

logger = LoggerConfig.get_logger()


class ForestNormalizer:
    """
    Normalisiert OSM-Waldpolygone für die Generierung pro Tile.

    Jedes OSM-Waldpolygon wird auf Tile-Grenzen gekürzt und mit
    einem forest_type versehen.
    """

    def __init__(self, forest_config: Dict, osm_mapper: OSMMapper):
        """
        Args:
            forest_config: Dict aus osm_to_beamng.json["forest_type_templates"] + ["forest_mappings"]
            osm_mapper: OSMMapper-Instance mit geladenen OSM-Daten
        """
        self.forest_config = forest_config
        self.osm_mapper = osm_mapper
        self.forest_types = forest_config.get("forest_type_templates", {})

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
                logger.info(f"  [→] Keine Wälder in {tile_name}")
                return result

            logger.info(f"  [→] Prüfe {len(osm_forests)} OSM-Waldpolygone...")

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

                # Bestimme Forest-Type basierend auf OSM-Tags; Lichtungen (innere Ringe) bekommen
                # den in forest_mappings["clearings"] konfigurierten Typ (z.B. niedriger Laubwald)
                forest_type = self._map_to_forest_type(tags)
                if osm_forest.get("is_clearing"):
                    clearings = self.forest_mappings.get("clearings", {})
                    only_for = clearings.get("only_for")
                    if only_for is not None and self._base_forest_mapping(tags)[1] not in only_for:
                        continue  # Loch in einem Wohngebiet o.ä. ist keine Waldlichtung: nicht bepflanzen
                    forest_type = clearings.get("forest_type", forest_type)
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
        from shapely.geometry import Polygon, LineString
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
                logger.error(f"  [!] Element ist kein Dict: {type(element)}")
                continue

            tags = element.get("tags", {})
            if not isinstance(tags, dict):
                logger.error(f"  [!] Tags sind kein Dict: {type(tags)}")
                continue

            # Prüfe mit OSMMapper ob es ein Wald ist
            if not self.osm_mapper.is_forest(tags):
                continue

            element_type = element.get("type", "way")

            # === CASE 1: Relation (Multipolygon) ===
            if element_type == "relation" and tags.get("type") == "multipolygon":
                try:
                    # Wald ohne Lichtungen + Lichtungen (innere Ringe) getrennt
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
                            if self._is_row_type(tags):
                                # Baumreihe (natural=tree_row): eine LINIE, kein Polygon - die Bäume stehen
                                # später im Abstand row_spacing entlang der Linie
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
        Baue Wald und Lichtungen eines Multipolygons aus seinen Member-Ways.

        Die Teilstücke einer Rolle werden zu geschlossenen Ringen zusammengesetzt (OSM zerlegt
        lange Ringe in mehrere Ways - jeden einzeln zu schließen ergäbe falsche Flächen).
        Innere Ringe sind Lichtungen: sie werden vom Wald abgezogen und separat geliefert.

        WICHTIG: Erwartet lokale Koordinaten {x, y}!
        (Zentrale Transformation in ForestWorkflow._transform_osm_to_local() erfolgt VORHER)

        Args:
            relation: Relation-Element mit members
            ways_by_id: Index way_id → way_element

        Returns:
            (Wald-Geometrie ohne Lichtungen, Lichtungs-Geometrie oder None) - oder (None, None)
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
            logger.debug(f"Fehler beim Multipolygon-Assembly: {e}")
            return None, None

    def _is_row_type(self, osm_tags: Dict) -> bool:
        """True, wenn der Waldtyp der Tags eine Baumreihe ist (Vorlage mit row_spacing): Linie statt Fläche."""
        base_type, _ = self._base_forest_mapping(osm_tags)
        return bool(base_type and self.forest_types.get(base_type, {}).get("row_spacing"))

    def _base_forest_mapping(self, osm_tags: Dict) -> Tuple[Optional[str], Optional[str]]:
        """(Basis-Waldtyp, auslösender Tag "key=value") aus landuse/natural/leisure oder (None, None)."""
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
        Mappe OSM-Tags zu forest_type.

        Folgt der Logik aus forest_mappings:
        1. Basis-Typ aus landuse/natural/leisure
        2. tag_overrides (z.B. trees=conifer) verfeinern den Basis-Typ - aber nur, wenn der
           Basis-Tag in "tag_overrides_only_for" steht (z.B. "landuse=forest"). Sonst könnte ein
           natural=wood mit Nadelbaum-Tag in einen hohen Waldtyp umgeleitet werden. Fehlt der
           Schlüssel, gelten die Overrides für alle (abwärtskompatibel).
        3. Fallback: None

        Args:
            osm_tags: Dict mit OSM-Tags

        Returns:
            forest_type-String oder None
        """
        mappings = self.forest_mappings
        base_type, base_tag = self._base_forest_mapping(osm_tags)

        # 2. Tag-Overrides (nur für erlaubte Basis-Tags)
        overrides = mappings.get("tag_overrides")
        scope = mappings.get("tag_overrides_only_for")
        if overrides and (scope is None or base_tag in scope):
            for override_key, override_value in overrides.items():
                key, value = override_key.split("=", 1)
                if osm_tags.get(key) == value:
                    return override_value

        # 3. Basis-Typ (oder None)
        return base_type

    def get_forest_properties(self, forest_type: str) -> Dict:
        """
        Hole die Eigenschaften eines Waldtyps.

        Args:
            forest_type: Name des Waldtyps (z.B. "deciduous_dense")

        Returns:
            Dict mit tree_density, tree_distribution, average_height, etc.
        """
        return self.forest_types.get(forest_type, {})
