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
        self.forest_mappings = forest_config.get("forest_mappings", {})

        logger.info(f"✓ ForestNormalizer initialisiert ({len(self.forest_types)} Waldtypen)")

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

            # ZENTRALE NORMALISIERUNG: Transformiere alle Waldpolygone zu lokalen Koordinaten
            if local_offset:
                from ..geometry.coordinates import transformer_to_wgs84

                osm_forests = self._transform_forests_wgs84_to_local(osm_forests, local_offset, transformer_to_wgs84)
                if not osm_forests:
                    print(f"  [→] Fehler bei Koordinaten-Transformation, keine Wälder")
                    return result

            # Jetzt: Alle Geometrien sind in lokalen Koordinaten!
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

    def _get_osm_forests(self) -> List[Dict]:
        """
        Extrahiere alle OSM-Waldpolygone aus geladenem OSM-Cache.

        Sucht nach:
        - landuse=forest
        - landuse=wood
        - natural=wood
        - natural=forest

        Returns:
            Liste von Dicts mit "geometry", "tags"
        """
        forests = []

        # OSM-Daten müssen über Tile-Kontext geladen werden (wird in normalize_tile() gemacht)
        # Hier haben wir nur Zugriff auf den OSMMapper (für Konfiguration)
        # Die echten OSM-Rohdaten müssen von außen übergeben werden!

        print("[Forest WARNING] _get_osm_forests() benötigt OSM-Rohdaten von außen!")
        return forests

    def _extract_forests_from_osm(self, osm_data: List[Dict]) -> List[Dict]:
        """
        Extrahiere Waldpolygone aus OSM-Rohdaten.

        Sucht nach:
        - landuse=forest
        - landuse=wood
        - natural=wood
        - natural=forest

        Args:
            osm_data: OSM-Elements mit tags und geometry (Overpass-Format!)

        Returns:
            Liste von Dicts mit "geometry" (Shapely Polygon), "tags"
        """
        from shapely.geometry import Polygon

        forests = []

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

            # Extrahiere Geometrie (Overpass-Format: Liste von {lat, lon} Dicts)
            geom_data = element.get("geometry")
            if not geom_data:
                continue

            try:
                # Konvertiere Overpass-Format zu Shapely Polygon
                # Overpass gibt [{'lat': x, 'lon': y}, ...] zurück
                # Shapely erwartet [(lon, lat), ...] (x, y) !

                if isinstance(geom_data, list) and len(geom_data) > 0:
                    if isinstance(geom_data[0], dict) and "lat" in geom_data[0]:
                        # Overpass-Format: Liste von {lat, lon} Dicts
                        coords = [(pt["lon"], pt["lat"]) for pt in geom_data]
                        if len(coords) >= 3:  # Polygon benötigt mind. 3 Punkte
                            geom = Polygon(coords)

                            if geom.is_valid:
                                forests.append({"geometry": geom, "tags": tags, "osm_id": element.get("id")})
                    elif isinstance(geom_data[0], (list, tuple)):
                        # GeoJSON-Format: [[lon, lat], ...]
                        coords = [tuple(pt[:2]) for pt in geom_data]
                        if len(coords) >= 3:
                            geom = Polygon(coords)
                            if geom.is_valid:
                                forests.append({"geometry": geom, "tags": tags, "osm_id": element.get("id")})
            except Exception as e:
                print(f"  [!] Fehler beim Parsing von Waldgeometrie: {e}")
                continue

        return forests

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
                # Transformiere Polygon
                if geom_wgs84.geom_type == "Polygon":
                    coords_wgs84 = list(geom_wgs84.exterior.coords)
                elif geom_wgs84.geom_type == "MultiPolygon":
                    # Für MultiPolygon: nutze das größte Polygon
                    coords_wgs84 = list(max(geom_wgs84.geoms, key=lambda x: x.area).exterior.coords)
                else:
                    continue

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
