"""
Terrain-Export Workflow.

Orchestriert den kompletten Terrain-Export-Prozess.
"""

from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path
import logging

from .. import config
from ..core.cache_manager import CacheManager
from ..managers import MaterialManager, ItemManager, DAEExporter
from .tile_processor import TileProcessor

logger = logging.getLogger(__name__)

WATER_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "data" / "water_templates.json"


def make_height_sampler_for_water(heights, origin_x, origin_y):
    """Bilineare Höhenabfrage auf der fertigen Heightmap (wie für die Weinberg-Reben)."""
    from ..forest.vineyard_generator import make_height_sampler

    return make_height_sampler(heights, origin_x, origin_y, config.TERRAIN_SQUARE_SIZE)


WATER_BOUNDS_MARGIN = 2.0  # knapp innerhalb der echten Daten: dahinter ist das Terrain aufgefüllt


def water_bounds(grid_bounds_local):
    """(xmin, ymin, xmax, ymax) des Geländes mit echten Höhendaten, in dem Wasser entstehen darf."""
    x_min, x_max, y_min, y_max = grid_bounds_local
    m = WATER_BOUNDS_MARGIN
    return (x_min + m, y_min + m, x_max - m, y_max - m)


class TerrainWorkflow:
    """
    Orchestriert den Terrain-Export-Workflow.

    Verantwortlich für:
    - Mesh-Generierung
    - Straßen-Integration
    - DAE-Export
    - Material/Item-Management
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        dae_exporter: DAEExporter,
    ):
        self.cache = cache_manager
        self.materials = MaterialManager.get_instance()  # Singleton
        self.items = ItemManager.get_instance()  # Singleton
        self.dae = dae_exporter
        self.tile_processor = TileProcessor(cache_manager)

    def process_tile(
        self,
        tiles: List[Dict],
        global_offset: Tuple[float, float],
        bbox_margin: float = 50.0,
        buildings_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Verarbeite alle übergebenen Kacheln als EINE zusammenhängende Fläche.

        Die Höhendaten aller Kacheln werden zu einer einzigen Punktwolke
        kombiniert (setzt voraus, dass sie einen lückenlosen, rechteckigen
        Bereich bilden - Nutzer-Verantwortung, siehe utils.tile_scanner).
        Ab hier läuft die komplette restliche Verarbeitung (BBox, OSM-Abfrage,
        Grid, Straßen-Mesh, Junction-Erkennung, Böschung, Heightmap) EINMAL
        über die Gesamtfläche statt einmal pro Kachel - Clipping findet nur
        noch am Außenrand der Gesamtfläche statt, nicht mehr an den früheren
        Kachelgrenzen. Ein einzelnes Tile ist einfach der Spezialfall
        len(tiles) == 1 desselben Codepfads.

        Args:
            tiles: Liste von Tile-Metadaten (typischerweise alle DGM1-Kacheln
                eines Exports)
            global_offset: Globaler Offset (origin_x, origin_y)
            bbox_margin: BBox-Erweiterung in Metern
            buildings_data: Optional - LoD2 Gebäudedaten

        Returns:
            Dict mit Verarbeitungs-Ergebnissen
        """
        from ..osm.parser import calculate_bbox_from_height_data, extract_roads_from_osm
        from ..osm.downloader import get_osm_data
        from ..geometry.polygon import get_road_polygons, clip_road_polygons
        from ..geometry.junctions import (
            detect_junctions_in_centerlines,
            mark_junction_endpoints,
            split_roads_at_mid_junctions,
        )
        from ..io.cache import calculate_global_tiles_hash

        # 1. Höhendaten aller Kacheln zu einer Punktwolke kombinieren
        height_points, height_elevations = self.tile_processor.load_height_data_multi(tiles)
        if height_points is None:
            return {"status": "failed", "reason": "no_height_data"}

        # Kombinierter Hash über alle Kacheln - Cache-Identität für OSM/
        # Elevation/Grid (ersetzt den früheren Pro-Kachel-Dateihash)
        tile_hash = calculate_global_tiles_hash(tiles) if tiles else "unknown"

        # 2. BBox berechnen (VOR der lokalen Transformation!)
        # Berechne BBox mit Margin direkt in UTM (Meter), dann Transformation zu WGS84
        osm_bbox = calculate_bbox_from_height_data(height_points, margin=bbox_margin)

        # 3. Transformiere zu lokalen Koordinaten
        local_points, elevations = self.tile_processor.ensure_local_offset(
            global_offset, height_points, height_elevations
        )

        # 4. OSM-Daten laden (mit tile_hash für tile-spezifischen Cache)
        osm_data = get_osm_data(osm_bbox, height_hash=tile_hash)

        if not osm_data:
            logger.warning("  [!] Keine OSM-Daten")
            return {"status": "failed", "reason": "no_osm_data"}

        # 5. Straßen extrahieren
        roads = extract_roads_from_osm(osm_data)

        # 6. Road Polygons (konvertiert OSM-Daten zu coords)
        # WICHTIG: Übergebe LOKALE Koordinaten! Alle internen Berechnungen in lokal!

        road_polygons = get_road_polygons(roads, osm_bbox, local_points, elevations, global_offset, tile_hash=tile_hash)

        # 6a. Luftbilder: werden NICHT mehr hier pro Kachel verarbeitet - bei
        # mehreren Kacheln würde die Datei-Existenz-Prüfung ("gibt es schon
        # irgendeine .dds?") den Export für alle Kacheln außer der ersten
        # überspringen. Stattdessen ruft BeamNGExporter.export_complete_level()
        # process_aerial_images() einmalig für die Gesamt-BBox aller Kacheln auf,
        # bevor die Tile-Schleife beginnt.

        # 6b. LoD2-Gebäude laden (wenn aktiviert und noch nicht übergeben)
        if buildings_data is None and config.LOD2_ENABLED:
            from ..io.lod2 import cache_lod2_buildings, load_buildings_from_cache

            # Berechne Z-Min aus den Höhendaten für vollständige 3D-Normalisierung
            # WICHTIG: Gebäude-Z-Koordinaten NICHT normalisieren!
            # Das Terrain selbst hat absolute Höhen (263-580m), nicht normalisiert.
            # Die Gebäude in CityGML haben ebenfalls absolute Höhen über NN.
            # Daher: z_offset = 0 (keine Z-Normalisierung für Gebäude!)
            z_offset = 0.0
            # Erweitere global_offset zu 3D-Offset
            local_offset_3d = (global_offset[0], global_offset[1], z_offset)

            # Versuche normalisierte Gebäude direkt aus Cache zu laden
            # (sie wurden mit cache_lod2_buildings bereits normalisiert)
            buildings_cache_path = cache_lod2_buildings(
                lod2_dir=config.LOD2_DATA_DIR,
                bbox=osm_bbox,  # WGS84-BBox
                local_offset=local_offset_3d,  # 3D-Offset mit Z-Min!
                cache_dir=config.CACHE_DIR,
                height_hash=tile_hash,
            )
            if buildings_cache_path:
                buildings_data = load_buildings_from_cache(buildings_cache_path)
                if buildings_data:
                    logger.info(f"  [OK] {len(buildings_data)} normalisierte Gebäude aus Cache geladen")

            if not buildings_data:
                logger.info("  [i] Keine LoD2-Gebäude gefunden")

        # Kirchtürme: keine Fenster, dafür eine Turmuhr (Kirche aus OSM, Turmwände aus der Geometrie)
        if buildings_data:
            from ..facade.church_towers import ChurchTowerFinder
            from ..osm.landuse_polygons import make_local_transform

            towers = ChurchTowerFinder.from_osm(osm_data, make_local_transform(global_offset)).mark(buildings_data)
            logger.info(f"  [OK] {towers} Kirchen mit Turm erkannt (Turmuhr statt Fenster)")

        # Berechne Grid-Bounds aus lokalen Punkten für Clipping
        grid_bounds_local = (
            float(local_points[:, 0].min()),
            float(local_points[:, 0].max()),
            float(local_points[:, 1].min()),
            float(local_points[:, 1].max()),
        )

        # Verwende ROAD_CLIP_MARGIN aus Config (negativ = erweitern!)
        road_polygons = clip_road_polygons(road_polygons, grid_bounds_local, margin=config.ROAD_CLIP_MARGIN)

        # 7. Junction-Detection (benötigt road_polygons mit coords)
        # WICHTIG: Reihenfolge wie im alten Workflow: detect → split → mark
        junctions = detect_junctions_in_centerlines(road_polygons)
        road_polygons, junctions = split_roads_at_mid_junctions(road_polygons, junctions)  # ZUERST Split
        road_polygons = mark_junction_endpoints(road_polygons, junctions)  # DANN Mark

        # Wandle road_polygons in road_slope_polygons_2d um (für Klassifizierung)
        # WICHTIG: NACH Junction-Detection, damit die gesplitteten Straßen verwendet werden!
        # WICHTIG: Erzeuge tatsächliche Straßen-Polygone (Puffer um Centerline)
        from shapely.geometry import LineString
        from ..config import OSM_MAPPER
        from ..geometry.road_structures import classify_structure, extend_gallery_centerline_ends, split_by_structure_type
        from ..utils.debug_exporter import DebugNetworkExporter

        road_slope_polygons_2d = []
        debug_exporter = DebugNetworkExporter.get_instance()

        for road in road_polygons:
            coords = np.asarray(road.get("coords", []), dtype=float)
            if len(coords) < 2:
                continue

            # Berechne Straßenbreite aus OSM-Tags
            osm_tags = road.get("osm_tags", {})
            road_width = OSM_MAPPER.get_road_properties(osm_tags)["width"]

            # Erzeuge Polygon durch Pufferung der Centerline
            centerline_2d = coords[:, :2]
            try:
                line = LineString(centerline_2d)
                road_poly = line.buffer(road_width / 2.0, cap_style=2)  # cap_style=2 = flat
                road_polygon_2d = np.array(road_poly.exterior.coords[:-1])  # ohne Duplikat
            except Exception:
                # Fallback: verwende Centerline direkt
                road_polygon_2d = centerline_2d

            road_id = road.get("id")
            road_slope_polygons_2d.append(
                {
                    "road_id": road_id,  # Wichtig für Material-Mapping
                    "road_polygon": road_polygon_2d,
                    "trimmed_centerline": coords,
                    "osm_tags": osm_tags,
                    "structure_type": classify_structure(osm_tags),
                }
            )

            # Exportiere Road zur Debug-Visualisierung
            debug_exporter.add_line(
                coords,
                color=[0.0, 0.0, 1.0],
                width=2.0,
                label=f"Road_{road_id}",
            )

        # 8. Grid erstellen (mit Builder)
        from ..builders import GridBuilder

        grid_builder = GridBuilder()
        grid = (
            grid_builder.with_points(local_points)
            .with_elevations(elevations)
            .with_spacing(config.GRID_SPACING)
            .with_cache_key(f"grid_{tile_hash}")
            .build()
        )

        # 9. Grid-Dimensionen extrahieren (Vertex-Klassifizierung entfällt -
        # das Terrain wird nicht mehr trianguliert, siehe Task 9)
        grid_points, grid_elevations, nx, ny = grid

        # 10. Terrain-Heightmap statt Mesh-Triangulierung.
        # Straßen werden seit der DecalRoad-Umstellung nicht mehr als Mesh
        # gebaut (kein RoadMeshBuilder/Junction-Fan-Material-Mehrheitsvotum
        # mehr nötig) - siehe export_decal_roads().
        from ..config import OSM_MAPPER
        from ..terrain.heightmap import build_heightmap
        from ..terrain.road_embedding import (
            embed_roads_into_heightmap,
            build_road_embankment_profiles,
            apply_embankment_blend,
        )
        from ..tunnels.gallery_mesh import resolve_open_side
        from ..terrain.terrain_materials import (
            build_photo_fallback_layer,
            mark_padding_as_holes,
            mask_layer_map_with_photo,
            paint_landuse_materials,
        )

        heightmap_result = build_heightmap(
            grid_points, grid_elevations, nx, ny, config.TERRAIN_SQUARE_SIZE
        )
        heights = heightmap_result["heights"]
        terrain_size = heightmap_result["size"]
        terrain_origin_x = heightmap_result["origin_x"]
        terrain_origin_y = heightmap_result["origin_y"]

        # Böschung: Übergang von Straßenkante zur natürlichen Umgebung direkt
        # im Heightmap erzeugen (das Mesh generiert keine Böschungs-Geometrie mehr - siehe Spec Abschnitt 4b).
        # WICHTIG: muss auf den noch UNVERÄNDERTEN heights laufen, damit
        # "natürliche Höhe" wirklich natürlich ist (vor embed_roads_into_heightmap).
        # Brücken/Tunnel werden NICHT ins Terrain eingebettet und bekommen keine Böschung - siehe
        # Design-Spec Abschnitt 3 (das Gelände bleibt darunter/daneben vollständig natürlich). Galerien
        # dagegen WERDEN wie normale Straßen eingebettet (siehe unten) - kein separates Terrain-Loch mehr
        # nötig, seit Boden/Wand/Dach massive Quader sind (tunnels/gallery_mesh.py).
        surface_road_polygons, structure_road_polygons = split_by_structure_type(road_slope_polygons_2d)
        # Galerie-Enden ein Stück in den angrenzenden "surface"-Straßenabschnitt hinein verlängern -
        # zentral hier, VOR jeder Verwendung von structure_road_polygons (Terrain-Einbettung und Mesh
        # nutzen beide dieselbe trimmed_centerline), siehe extend_gallery_centerline_ends()-Docstring.
        structure_road_polygons = extend_gallery_centerline_ends(structure_road_polygons, config.GALLERY_CENTERLINE_EXTENSION)

        # Galerien wie normale Straßen einbetten (dieselben Böschungs-/Einbettungs-Parameter), aber mit
        # festen statt berechneten Böschungsbreiten auf beiden Seiten (slope_width_override, siehe
        # build_road_embankment_profiles()-Docstring und config.GALLERY_VALLEY_SLOPE_WIDTH): bergseits 0 m
        # (die massive Wand reicht ohnehin bis in den Hang), talseits ein kurzer fester Wert (das DGM zeigt
        # dort die reale Talseiten-Struktur statt echtem Naturgelände, eine berechnete Breite wäre
        # verrauscht/facettiert). Ohne avalanche_protector:left/right-Tag (kein zuverlässiger Fallback)
        # bleibt die Böschung auf beiden Seiten normal wie bei einer Oberflächenstraße.
        def _gallery_slope_override(osm_tags):
            open_side = resolve_open_side(osm_tags or {})
            if open_side == "left":
                return {"left": config.GALLERY_VALLEY_SLOPE_WIDTH, "right": 0.0}
            if open_side == "right":
                return {"right": config.GALLERY_VALLEY_SLOPE_WIDTH, "left": 0.0}
            return {}

        gallery_roads = [
            {**r, "slope_width_override": _gallery_slope_override(r.get("osm_tags"))}
            for r in structure_road_polygons
            if r.get("structure_type") == "gallery"
        ]
        embeddable_roads = surface_road_polygons + gallery_roads

        embankment_profiles = build_road_embankment_profiles(
            embeddable_roads,
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            OSM_MAPPER,
            config.SLOPE_ANGLE,
            config.MIN_SLOPE_WIDTH,
            max_slope_width=config.MAX_SLOPE_WIDTH,
        )
        heights = apply_embankment_blend(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, embankment_profiles)

        # Straßen-Einbettung: Terrain auf der reinen Fahrbahnfläche exakt auf
        # Centerline-Höhe setzen (Böschung ist bereits durch
        # apply_embankment_blend abgedeckt) - siehe road_embedding.py-
        # Moduldocstring für die DecalRoad-Begründung.
        heights = embed_roads_into_heightmap(
            heights,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            embeddable_roads,
        )

        # Brücken: Terrain, das innerhalb der Brückenbreite HÖHER als das Deck liegt, auf Deck-Niveau kappen
        # (nicht unbedingt setzen wie oben) - betrifft praktisch nur die Brücken-Enden (Auflager), wo die
        # Fahrbahn ins natürliche Gelände übergeht und das quer zur Fahrtrichtung nicht zwingend flach ist;
        # ohne Kappung könnte das Gelände dort stellenweise durchs (flache) Deck ragen. Der Talboden, den
        # die Brücke überspannt, bleibt unverändert sichtbar (deutlich unter Deck-Niveau, clamp_to_max
        # greift dort nicht).
        bridge_roads = [r for r in structure_road_polygons if r.get("structure_type") == "bridge"]
        if bridge_roads:
            heights = embed_roads_into_heightmap(
                heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, bridge_roads,
                clamp_to_max=True,
            )

        # Layer-Map: EIN Luftbild-Material für die gesamte Fläche, dann OSM-
        # Landnutzung obenauf (siehe build_photo_fallback_layer()).
        layer_map, photo_tile_names = build_photo_fallback_layer(terrain_size)

        from ..osm.landuse_polygons import build_landuse_polygons, make_local_transform

        # osm_data enthält an dieser Stelle noch RAW Overpass-Geometrie (lat/lon).
        # Landnutzungs-Polygone entstehen aus Ways UND Multipolygon-Relationen
        # (große Wald-/Weinberg-/Wohngebietsflächen sind in OSM meist Relationen).
        landuse_polygons = build_landuse_polygons(osm_data, make_local_transform(global_offset))

        # Teichmulden: innerhalb der Wasserflächen das Terrain tiefer legen (vor allem, was die Höhen weiterverwendet:
        # Bäche, Bäume, Reben). Der Wasserspiegel kommt aus dem natürlichen Rand, siehe _build_water().
        natural_heights = heights
        if config.WATER_ENABLED:
            from ..terrain.water import carve_pond_basins, select_pond_areas

            pond_areas = select_pond_areas(landuse_polygons, water_bounds(grid_bounds_local))
            if pond_areas:
                heights = carve_pond_basins(
                    heights,
                    terrain_origin_x,
                    terrain_origin_y,
                    config.TERRAIN_SQUARE_SIZE,
                    pond_areas,
                    depth=config.WATER_POND_BANK_DEPTH,
                    slope_deg=config.WATER_POND_BANK_SLOPE_DEG,
                )
                logger.info(
                    f"  [OK] Teichmulden: {len(pond_areas)} Wasserfläche(n), Terrain {config.WATER_POND_BANK_DEPTH * 100:.0f} cm tiefer "
                    f"(Böschung {config.WATER_POND_BANK_SLOPE_DEG:.0f} Grad)"
                )

        layer_map, terrain_material_names = paint_landuse_materials(
            layer_map,
            photo_tile_names,
            terrain_size,
            terrain_origin_x,
            terrain_origin_y,
            config.TERRAIN_SQUARE_SIZE,
            landuse_polygons,
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
        )

        # Straßen- und Gebäudeflächen: (1) Bodenbewuchs wächst auf dem Layer - dort geht es
        # zurück aufs Luftbild, sonst wächst Gras durch Decals und Häuser; (2) Ausschlusszone
        # für die Weinberg-Reben.
        from ..geometry.road_surfaces import union_road_surfaces

        # Alle Straßenflächen EINMAL vereinigt (vereinfacht): dient Maske, Reben-Ausschluss und dem Wald.
        # Oberflächenstraßen UND Galerien (jetzt wie normale Straßen ins Terrain eingebettet, siehe oben) -
        # nur Brücken/Tunnel bleiben außen vor, die sollen die Vegetation nicht blockieren, sonst bliebe
        # z.B. beim Tunnel ein kahler Streifen über dem ganzen Bergrücken.
        road_surface_union = union_road_surfaces(surface_road_polygons + gallery_roads)
        road_shapes = [road_surface_union] if road_surface_union is not None else []
        building_shapes = [
            p["geometry"]
            for p in build_landuse_polygons(osm_data, make_local_transform(global_offset), tag_keys=("building",))
        ]

        if config.GROUND_COVER_ENABLED:
            layer_map = mask_layer_map_with_photo(
                layer_map,
                terrain_size,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
                road_shapes,
                buffer=config.GROUND_COVER_ROAD_MARGIN,
            )
            layer_map = mask_layer_map_with_photo(
                layer_map,
                terrain_size,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
                building_shapes,
                buffer=config.GROUND_COVER_BUILDING_MARGIN,
            )

        # Überschussrand der Zweierpotenz-Heightmap (nur Extrapolation) als Hole: das sichtbare
        # Terrain endet exakt am Datenrand, den Streifen dahinter deckt der Horizont ab.
        # Zuletzt, damit Malen/Maskieren oben unverändert auf der vollen Layer-Map laufen.
        if config.TERRAIN_PADDING_AS_HOLES:
            layer_map = mark_padding_as_holes(layer_map, data_cols=nx, data_rows=ny)

        # Vier-Bilder-Modus: erst jetzt (Malen, Masken und Löcher sind fertig) wird die Layer-Map pro Kachel in
        # physische Materialien aufgeteilt - jede Kachel bekommt ihr eigenes Foto. Die Foto-Kachelung ist ein
        # FESTER Raster über die Gesamtfläche (config.PHOTO_TILE_SIZE_M), unabhängig von der Größe/Anzahl der
        # rohen Höhendaten-Kacheln (siehe terrain/photo_tiles.py-Moduldocstring) - export/beamng_exporter.py
        # baut denselben Raster aus denselben Eingaben (deterministisch, ohne dass Daten geteilt werden müssen).
        from ..terrain.photo_tiles import build_processing_tile_grid
        from ..utils.tile_scanner import compute_global_bbox

        processing_tiles = build_processing_tile_grid(compute_global_bbox(tiles), config.PHOTO_TILE_SIZE_M)

        photo_tiles = None
        if config.AERIAL_PHOTO_PER_TILE and len(processing_tiles) > 1:
            from ..terrain.photo_tiles import split_layers_by_tile

            photo_tiles = split_layers_by_tile(
                layer_map,
                terrain_material_names,
                processing_tiles,
                global_offset,
                terrain_origin_x,
                terrain_origin_y,
                config.TERRAIN_SQUARE_SIZE,
            )
            layer_map = photo_tiles["layer_map"]
            terrain_material_names = photo_tiles["material_names"]
            photo_tile_names = photo_tiles["photo_tile_names"]
            logger.info(
                f"  [OK] Vier-Bilder-Modus: {len(photo_tile_names)} Luftbilder, {len(terrain_material_names)} Terrain-Materialien"
            )

        # Weinberg-Reben (Forest-Items) entlang der Falllinie, auf der fertigen Heightmap
        vineyard_instances = []
        if config.VINEYARDS_ENABLED and config.FORESTS_ENABLED:
            from ..forest.vineyard_generator import build_exclusion_geometry, generate_vineyards, make_height_sampler
            from shapely.geometry import box

            vineyard_instances = generate_vineyards(
                landuse_polygons,
                config.OSM_MAPPER.config.get("landuse_mappings", {}),
                make_height_sampler(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE),
                exclusion=build_exclusion_geometry(road_shapes + building_shapes, config.VINEYARD_EXCLUSION_MARGIN),
                # Nur über echten Höhendaten: die OSM-Abfrage reicht über das Terrain hinaus,
                # und der Terrainrand ist aufgefüllt (dort gibt es keine echten Höhen)
                bounds=box(
                    grid_bounds_local[0] + config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[2] + config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[1] - config.VINEYARD_EXCLUSION_MARGIN,
                    grid_bounds_local[3] - config.VINEYARD_EXCLUSION_MARGIN,
                ),
            )
            logger.info(f"  [OK] {len(vineyard_instances)} Rebzeilen-Segmente generiert")

        # Echtes Wasser: Bäche als River-Splines, Wasserflächen als WaterBlocks (auf der fertigen Heightmap)
        water = {"rivers": [], "ponds": []}
        if config.WATER_ENABLED:
            water = self._build_water(
                osm_data,
                landuse_polygons,
                global_offset,
                make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y),
                grid_bounds_local,
                rim_height_at=make_height_sampler_for_water(natural_heights, terrain_origin_x, terrain_origin_y),
            )

        # Bruchsteinmauern (OSM barrier=wall mit height) auf der fertigen Heightmap
        wall_meshes = []
        if config.WALLS_ENABLED:
            wall_meshes, _ = self._build_wall_meshes(
                osm_data,
                global_offset,
                make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y),
                road_slope_polygons_2d,
            )

        # Brücken (Deck + Pfeiler) auf der fertigen Heightmap - siehe bridges/bridge_mesh.py
        bridge_meshes = []
        if config.BRIDGES_ENABLED:
            bridge_meshes = self._build_bridges(structure_road_polygons, heights, terrain_origin_x, terrain_origin_y)

        # Tunnel (Röhre + Portale) und Galerien (Dach + Stützen) auf der fertigen Heightmap - siehe tunnels/
        tunnel_meshes = []
        if config.TUNNELS_ENABLED:
            tunnel_meshes = self._build_tunnels(structure_road_polygons, heights, terrain_origin_x, terrain_origin_y)

        # POI-Kandidaten (Orte, große Parkplätze) für zusätzliche, in der Fahrzeugauswahl wählbare Spawn-
        # Punkte - siehe osm/poi_points.py und ItemManager._compute_poi_spawn_points(). Höhe auf der
        # FERTIGEN Heightmap abgetastet (die Positionen kommen als reine XY-Punkte aus OSM, nicht von
        # einer Straßen-Centerline).
        poi_points = self._collect_poi_points(osm_data, global_offset, heights, terrain_origin_x, terrain_origin_y, grid_bounds_local)

        z_min = float(heights.min())
        z_max = float(heights.max())
        max_height = (z_max - z_min) + config.TERRAIN_MAX_HEIGHT_BUFFER

        return {
            "status": "success",
            "heightmap": heights,
            "terrain_size": terrain_size,
            "terrain_origin_x": terrain_origin_x,
            "terrain_origin_y": terrain_origin_y,
            "z_min": z_min,
            "max_height": max_height,
            "layer_map": layer_map,
            "terrain_material_names": terrain_material_names,
            "photo_tile_names": photo_tile_names,
            # Vier-Bilder-Modus (sonst None): Kachel-Varianten der Schichten, ihr Foto und die Foto-Größe je Kachel
            "layer_variants": photo_tiles["layer_variants"] if photo_tiles else None,
            "variant_parents": photo_tiles["variant_parents"] if photo_tiles else None,
            "photo_extents": photo_tiles["photo_extents"] if photo_tiles else None,
            "poi_points": poi_points,  # Orte/große Parkplätze für ItemManager._compute_poi_spawn_points()
            "vineyard_instances": vineyard_instances,  # Forest-Items (grape_vine)
            "water": water,  # {"rivers": [...], "ponds": [...]} für export_water()
            "wall_meshes": wall_meshes,  # Mesh-Dicts der Bruchsteinmauern für export_walls()
            "bridge_meshes": bridge_meshes,  # Brücken-Mesh-Dicts für export_bridges()
            "tunnel_meshes": tunnel_meshes,  # Tunnel-/Galerie-Mesh-Dicts für export_tunnels()
            "grid": grid,
            "road_polygons": road_polygons,
            "road_slope_polygons_2d": road_slope_polygons_2d,  # Für DecalRoad-Export
            "structure_road_polygons": structure_road_polygons,  # Brücken/Tunnel/Galerien - für export_bridges()/export_tunnels()
            "road_surface_union": road_surface_union,  # vereinigte Straßenfläche für Ausschlusszonen (oder None)
            "grid_bounds_local": grid_bounds_local,
            "global_offset": global_offset,
            "buildings_data": buildings_data,  # Übergebe Gebäude-Daten
            "height_points": local_points,  # Für Spawn-Punkt-Berechnung
            "height_elevations": elevations,  # Für Spawn-Punkt-Berechnung
            "height_hash": tile_hash,  # Für Cache-Konsistenz in Forest-Workflow
        }

    def _build_water(self, osm_data, landuse_polygons, global_offset, height_at, grid_bounds_local, rim_height_at=None) -> Dict:
        """
        Berechnet Bach-Knoten und Teich-Blöcke (siehe terrain/water.py). Die Wasserhöhen werden aus der fertigen
        Heightmap abgeleitet; der Teichspiegel aus dem Rand der NATÜRLICHEN Heightmap (`rim_height_at`, ohne die
        Teichmulde - sonst läge er um die Böschung zu tief), sonst aus `height_at`.

        Returns:
            {"rivers": [{"name", "waterway", "nodes"}], "ponds": [{"name", "blocks"}]}
        """
        from shapely.ops import unary_union

        from ..osm.landuse_polygons import make_local_transform
        from ..terrain.water import (
            build_pond_blocks,
            build_river_nodes,
            clip_line_to_bounds,
            cut_line_by_area,
            select_pond_areas,
            select_waterways,
            split_nodes,
        )

        bounds = water_bounds(grid_bounds_local)

        # Teiche zuerst: die Bäche enden an ihrem Ufer
        ponds = []
        pond_areas = select_pond_areas(landuse_polygons, bounds)
        for geometry in pond_areas:
            blocks = build_pond_blocks(
                geometry,
                rim_height_at or height_at,
                depth=config.WATER_POND_DEPTH,
                cell=config.WATER_POND_CELL,
                margin=config.WATER_POND_MARGIN,
            )
            if blocks:
                ponds.append({"name": f"pond_{len(ponds)}", "blocks": blocks})
        pond_area = unary_union(pond_areas) if pond_areas else None

        rivers = []
        for way in select_waterways(osm_data, make_local_transform(global_offset), config.WATERWAY_WIDTHS):
            for clipped in clip_line_to_bounds(way["coords"], bounds):
                for part in cut_line_by_area(clipped, pond_area):
                    if len(part) < 2:
                        continue
                    nodes = build_river_nodes(
                        part,
                        height_at,
                        width=way["width"],
                        depth=config.WATER_RIVER_DEPTH,
                        spacing=config.WATER_NODE_SPACING,
                        lift=config.WATER_STREAM_LIFT,
                    )
                    for chunk in split_nodes(nodes, config.WATER_MAX_RIVER_NODES):
                        if len(chunk) >= 2:
                            rivers.append({"name": f"river_{len(rivers)}", "waterway": way["waterway"], "nodes": chunk})

        length = sum(sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(r["nodes"], r["nodes"][1:])) for r in rivers)
        logger.info(
            f"  [OK] Wasser: {len(rivers)} River-Objekt(e) ({length:.0f} m Bachlauf), "
            f"{len(ponds)} Wasserfläche(n) mit {sum(len(p['blocks']) for p in ponds)} WaterBlocks"
        )
        return {"rivers": rivers, "ponds": ponds}

    def _build_wall_meshes(self, osm_data, global_offset, height_at, road_polygons=None):
        """
        Bruchsteinmauern aus OSM (barrier=wall/retaining_wall, nur mit height-Tag), dem Gelände folgend (siehe
        walls/wall_mesh.py); höchstens config.WALL_ROAD_SNAP_M neben einer Straßen-Centerline auf deren Höhe (road_base.py).

        Args:
            road_polygons: Straßen-Dicts mit "trimmed_centerline" (siehe road_slope_polygons_2d)

        Returns:
            (Mesh-Dicts für den DAE-Export, Statistik {"built", "length", "without_height"})
        """
        from ..osm.landuse_polygons import make_local_transform
        from ..textures import library
        from ..walls.road_base import RoadBaseHeight, centerlines_from_roads
        from ..walls.wall_mesh import build_walls

        meshes, stats = build_walls(
            osm_data,
            make_local_transform(global_offset),
            height_at,
            config.WALL_MATERIAL_NAME,
            thickness=config.WALL_THICKNESS,
            sink=config.WALL_SINK,
            max_step=config.WALL_MAX_SEGMENT,
            tile_m=library.texture_tile_m(config.WALL_TEXTURE_NAME, config.WALL_TEXTURE_TILE_M),
            road_base_at=RoadBaseHeight(centerlines_from_roads(road_polygons or []), config.WALL_ROAD_SNAP_M),
            cap_thickness=config.WALL_CAP_THICKNESS,
            cap_overhang=config.WALL_CAP_OVERHANG,
            cap_plate_length=config.WALL_CAP_PLATE_LENGTH,
            cap_joint=config.WALL_CAP_JOINT,
        )
        logger.info(
            f"  [OK] Mauern: {stats['built']} Bruchsteinmauer(n) mit Höhenangabe ({stats['length']:.0f} m), "
            f"{stats['without_height']} ohne Höhenangabe übersprungen"
        )
        return meshes, stats

    def _build_bridges(self, structure_road_polygons: List[Dict], heights: np.ndarray, terrain_origin_x: float, terrain_origin_y: float) -> List[Dict]:
        """Brücken-Meshes (Deck + Pfeiler) für alle Straßen mit structure_type == "bridge" (siehe bridges/bridge_mesh.py)."""
        from ..bridges.bridge_mesh import build_bridges
        from ..terrain.road_embedding import sample_heightmap_bilinear

        def ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        bridges = [
            {
                "id": road["road_id"],
                "coords": road["trimmed_centerline"],
                "width": config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"],
                "deck_material": f"{config.OSM_MAPPER.get_road_properties(road.get('osm_tags', {})).get('internal_name', 'road_default')}_structure",
            }
            for road in structure_road_polygons
            if road.get("structure_type") == "bridge"
        ]
        return build_bridges(
            bridges,
            ground_at,
            pier_material=config.BRIDGE_MATERIAL_NAME,
            railing_material=config.BRIDGE_RAILING_MATERIAL_NAME,
            deck_thickness=config.BRIDGE_DECK_THICKNESS,
            pier_spacing=config.BRIDGE_PIER_SPACING,
            pier_size=config.BRIDGE_PIER_SIZE,
            min_pier_clearance=config.BRIDGE_MIN_PIER_CLEARANCE,
            curb_width=config.BRIDGE_CURB_WIDTH,
            curb_height=config.BRIDGE_CURB_HEIGHT,
            railing_height=config.BRIDGE_RAILING_HEIGHT,
            railing_post_spacing=config.BRIDGE_RAILING_POST_SPACING,
            railing_post_size=config.BRIDGE_RAILING_POST_SIZE,
        )

    def export_bridges(self, mesh_data: Dict) -> int:
        """
        Exportiert Brücken als EINE DAE (Deck + Pfeiler je Brücke) mit EINEM TSStatic und registriert Fahrbahn-
        und Beton-Material. Ohne Brücken werden Reste eines früheren Exports entfernt.

        Returns:
            Anzahl exportierter Brücken
        """
        bridges_dir = config.BEAMNG_DIR_SHAPES / "bridges"
        meshes = mesh_data.get("bridge_meshes") or []
        if not config.BRIDGES_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (bridges_dir / f"bridges{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        concrete = registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]
        self.materials.add_building_material(
            config.BRIDGE_MATERIAL_NAME,
            textures={**concrete, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )
        railing = registry.prepared_textures()[config.RAILING_TEXTURE_NAME]
        self.materials.add_building_material(
            config.BRIDGE_RAILING_MATERIAL_NAME,
            textures={**railing, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )

        unique_deck_materials: Dict[str, Dict] = {}
        for road in mesh_data.get("structure_road_polygons", []):
            if road.get("structure_type") != "bridge":
                continue
            props = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
            mat_name = f"{props.get('internal_name', 'road_default')}_structure"
            unique_deck_materials[mat_name] = props

        for mat_name, props in unique_deck_materials.items():
            self.materials.add_building_material(
                mat_name,
                textures=props.get("textures", {}),
                groundType=str(props.get("groundModelName", "asphalt")).upper(),
                materialTag0="RoadAndPath",
                materialTag1="beamng",
            )

        self.dae.export_multi_mesh(output_path=bridges_dir / "bridges.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "bridges",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "bridges" / "bridges.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.info(f"  [OK] {len(meshes)} Brücken exportiert (bridges.dae)")
        return len(meshes)

    def _build_tunnels(self, structure_road_polygons: List[Dict], heights: np.ndarray, terrain_origin_x: float, terrain_origin_y: float) -> List[Dict]:
        """Tunnel- (Röhre+Portale) und Galerie-Meshes (Dach+Stützen) für alle Straßen mit structure_type in
        ("tunnel", "gallery") - siehe tunnels/tunnel_mesh.py und tunnels/gallery_mesh.py."""
        from ..terrain.road_embedding import sample_heightmap_bilinear
        from ..tunnels.gallery_mesh import build_galleries
        from ..tunnels.tunnel_mesh import build_tunnels

        def ground_at(x, y):
            return sample_heightmap_bilinear(heights, terrain_origin_x, terrain_origin_y, config.TERRAIN_SQUARE_SIZE, np.column_stack([x, y]))

        def _items(structure_type):
            return [
                {
                    "id": road["road_id"],
                    "coords": road["trimmed_centerline"],
                    "width": config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))["width"],
                    "floor_material": f"{config.OSM_MAPPER.get_road_properties(road.get('osm_tags', {})).get('internal_name', 'road_default')}_structure",
                    "osm_tags": road.get("osm_tags", {}),
                }
                for road in structure_road_polygons
                if road.get("structure_type") == structure_type
            ]

        tunnel_meshes = build_tunnels(
            _items("tunnel"),
            ground_at,
            wall_material=config.TUNNEL_MATERIAL_NAME,
            frame_material=config.TUNNEL_MATERIAL_NAME,
            width_margin=config.TUNNEL_WIDTH_MARGIN,
            arc_segments=config.TUNNEL_ARC_SEGMENTS,
            segment_step=config.TUNNEL_SEGMENT_STEP,
            portal_slope_sample_dist=config.TUNNEL_PORTAL_SLOPE_SAMPLE_DIST,
            frame_margin=config.TUNNEL_PORTAL_FRAME_MARGIN,
        )
        gallery_meshes = build_galleries(
            _items("gallery"),
            ground_at,
            roof_material=config.TUNNEL_MATERIAL_NAME,
            height=config.GALLERY_HEIGHT,
            column_spacing=config.GALLERY_COLUMN_SPACING,
            roof_thickness=config.GALLERY_ROOF_THICKNESS,
            floor_thickness=config.GALLERY_FLOOR_THICKNESS,
            wall_thickness=config.GALLERY_WALL_THICKNESS,
            column_size=config.GALLERY_COLUMN_SIZE,
            curb_height=config.GALLERY_CURB_HEIGHT,
            curb_width=config.GALLERY_CURB_WIDTH,
        )
        return tunnel_meshes + gallery_meshes

    def _collect_poi_points(
        self,
        osm_data: List[Dict],
        global_offset: Tuple[float, float],
        heights: np.ndarray,
        terrain_origin_x: float,
        terrain_origin_y: float,
        grid_bounds_local: Tuple[float, float, float, float],
    ) -> List[Dict]:
        """POI-Kandidaten (Orte, große Parkplätze) samt Höhe auf der fertigen Heightmap - siehe
        osm/poi_points.py. Nur innerhalb der echten Terrainfläche (die OSM-Abfrage reicht über das
        Terrain hinaus, siehe generate_vineyards()-Aufrufer)."""
        if not config.POI_SPAWN_POINTS_ENABLED:
            return []

        from ..osm.landuse_polygons import make_local_transform
        from ..osm.poi_points import extract_parking_points, extract_place_points

        to_local = make_local_transform(global_offset)
        candidates = extract_place_points(osm_data, to_local) + extract_parking_points(
            osm_data, to_local, min_area_m2=config.POI_MIN_PARKING_AREA_M2
        )
        if not candidates:
            return []

        margin = config.POI_SPAWN_BOUNDS_MARGIN
        x_min, x_max, y_min, y_max = grid_bounds_local
        in_bounds = [
            c for c in candidates
            if x_min + margin <= c["position_xy"][0] <= x_max - margin
            and y_min + margin <= c["position_xy"][1] <= y_max - margin
        ]
        if not in_bounds:
            return []

        height_at = make_height_sampler_for_water(heights, terrain_origin_x, terrain_origin_y)
        xs = np.array([c["position_xy"][0] for c in in_bounds])
        ys = np.array([c["position_xy"][1] for c in in_bounds])
        zs = height_at(xs, ys)
        return [
            {**candidate, "position": [float(x), float(y), float(z)]}
            for candidate, x, y, z in zip(in_bounds, xs, ys, zs)
        ]

    def export_tunnels(self, mesh_data: Dict) -> int:
        """
        Exportiert Tunnel (Röhre + 2 Portal-Rahmen je Tunnel) und Galerien (Dach + Stützen) als EINE DAE mit
        EINEM TSStatic und registriert Fahrbahn- und Beton-Material. Ohne Tunnel/Galerien werden Reste eines
        früheren Exports entfernt.

        Returns:
            Anzahl exportierter Tunnel-/Portal-/Galerie-Meshes
        """
        tunnels_dir = config.BEAMNG_DIR_SHAPES / "tunnels"
        meshes = mesh_data.get("tunnel_meshes") or []
        if not config.TUNNELS_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (tunnels_dir / f"tunnels{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        concrete = registry.prepared_textures()[config.CONCRETE_TEXTURE_NAME]
        self.materials.add_building_material(
            config.TUNNEL_MATERIAL_NAME,
            textures={**concrete, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )

        unique_floor_materials: Dict[str, Dict] = {}
        for road in mesh_data.get("structure_road_polygons", []):
            if road.get("structure_type") not in ("tunnel", "gallery"):
                continue
            props = config.OSM_MAPPER.get_road_properties(road.get("osm_tags", {}))
            mat_name = f"{props.get('internal_name', 'road_default')}_structure"
            unique_floor_materials[mat_name] = props

        for mat_name, props in unique_floor_materials.items():
            self.materials.add_building_material(
                mat_name,
                textures=props.get("textures", {}),
                groundType=str(props.get("groundModelName", "asphalt")).upper(),
                materialTag0="RoadAndPath",
                materialTag1="beamng",
            )

        self.dae.export_multi_mesh(output_path=tunnels_dir / "tunnels.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "tunnels",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "tunnels" / "tunnels.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.info(f"  [OK] {len(meshes)} Tunnel-/Galerie-Mesh(e) exportiert (tunnels.dae)")
        return len(meshes)

    def _set_fog_height(self, heights: np.ndarray) -> None:
        """
        fogAtmosphereHeight (Höhe, ab der der Höhennebel ausdünnt) = höchster Terrainpunkt + Marge. Alle Original-Level
        setzen einen Wert in der Größenordnung ihrer Geländehöhe; ein fester Wert wäre für unser Gelände (hier 236-689 m
        absolut) falsch.
        """
        height = float(np.max(heights)) + float(config.ENV_FOG_HEIGHT_MARGIN)
        self.items.set_base_line_fields("theLevelInfo", fogAtmosphereHeight=round(height, 1))

    def export_water(self, mesh_data: Dict) -> int:
        """
        Registriert Bäche (`River`) und Teiche/Seen (`WaterBlock`) als BeamNG-Objekte. Die Render-
        Parameter stammen aus BeamNGs eigenem east_coast_usa-Level (data/water_templates.json,
        nur core-Texturen), siehe tools/extract_water_templates.py.

        Returns:
            Anzahl der erzeugten Wasser-Objekte
        """
        water = mesh_data.get("water") or {}
        if not config.WATER_ENABLED or not (water.get("rivers") or water.get("ponds")):
            return 0

        import copy

        templates = json.loads(WATER_TEMPLATES_PATH.read_text(encoding="utf-8"))
        count = 0

        for river in water.get("rivers", []):
            fields = copy.deepcopy(templates["stream"]["fields"])
            fields.pop("class", None)
            nodes = river["nodes"]
            self.items.add_item(
                river["name"],
                item_class="River",
                position=tuple(nodes[0][:3]),
                overwrite=True,
                nodes=nodes,
                **fields,
            )
            count += 1

        for pond in water.get("ponds", []):
            for index, block in enumerate(pond["blocks"]):
                fields = copy.deepcopy(templates["pond"]["fields"])
                fields.pop("class", None)
                fields["cubemap"] = config.WATER_POND_CUBEMAP
                # Das Wasser-Raster darf nicht größer als der Block sein (sonst warnt BeamNG und kürzt selbst)
                fields["gridElementSize"] = float(min(fields.get("gridElementSize", 5.0), block["scale"][0], block["scale"][1]))
                self.items.add_item(
                    f"{pond['name']}_{index}",
                    item_class="WaterBlock",
                    position=tuple(block["position"]),
                    scale=tuple(block["scale"]),
                    overwrite=True,
                    **fields,
                )
                count += 1

        logger.info(f"  [OK] {count} Wasser-Objekt(e) exportiert")
        return count

    def export_walls(self, mesh_data: Dict) -> int:
        """
        Exportiert die Bruchsteinmauern als EINE DAE (jede Mauer ein Node) mit EINEM TSStatic und registriert das
        Stein-Material. Ohne Mauern werden Reste eines früheren Exports entfernt.

        Returns:
            Anzahl exportierter Mauern
        """
        walls_dir = config.BEAMNG_DIR_SHAPES / "walls"
        meshes = mesh_data.get("wall_meshes") or []
        if not config.WALLS_ENABLED or not meshes:
            for suffix in (".dae", ".cdae"):
                (walls_dir / f"walls{suffix}").unlink(missing_ok=True)
            return 0

        from ..textures import registry

        hints = self.materials.get_templates().get("buildings", {}).get("wall", {}).get("material_hints", {})
        stone = registry.prepared_textures()[config.WALL_TEXTURE_NAME]  # fehlt das Foto, ist der Export längst abgebrochen
        self.materials.add_building_material(
            config.WALL_MATERIAL_NAME,
            textures={**stone, "useAnisotropic": True},
            groundType=hints.get("groundType", "concrete"),
            materialTag0=hints.get("materialTag0", "beamng"),
            materialTag1=hints.get("materialTag1", "Building"),
        )
        self.dae.export_multi_mesh(output_path=walls_dir / "walls.dae", meshes=meshes, with_uv=True)
        self.items.add_item(
            "walls",
            item_class="TSStatic",
            shape_name=str(config.RELATIVE_DIR_SHAPES / "walls" / "walls.dae"),
            position=(0, 0, 0),
            overwrite=True,
            collisionType="Visible Mesh Final",
        )
        logger.info(f"  [OK] {len(meshes)} Mauer(n) exportiert (walls.dae)")
        return len(meshes)

    def export_decal_roads(self, mesh_data: Dict) -> int:
        """
        Exportiert jede Straße als eigenes BeamNG `DecalRoad`-Item - ein
        Spline-Decal, das zur Laufzeit direkt auf die Terrain-Oberfläche
        projiziert wird (siehe road_embedding.py-Moduldocstring für die
        Begründung). Ersetzt das frühere Mesh-Bauwerk (prepare_road_export()/
        export_merged_roads()/RoadMeshBuilder/DAE-Export) komplett - kein
        Straßen-Mesh, keine Junction-Fan-Geometrie, keine
        Face-Material-Mehrheitsentscheidung mehr nötig: jede Straße bekommt
        ihr eigenes DecalRoad-Item mit ihrem eigenen (aus OSM abgeleiteten)
        Material, BeamNGs `autoJunction` verbindet angrenzende Straßen
        automatisch.

        Args:
            mesh_data: Ergebnis von process_tile() (braucht
                "road_slope_polygons_2d")

        Returns:
            Anzahl der erzeugten DecalRoad-Items
        """
        from ..config import OSM_MAPPER
        from ..geometry.polygon import drop_close_nodes

        road_slope_polygons_2d = mesh_data["road_slope_polygons_2d"]
        unique_materials: Dict[str, Dict] = {}
        count = 0

        for poly in road_slope_polygons_2d:
            if poly.get("structure_type", "surface") != "surface":
                continue
            road_id = poly.get("road_id")
            centerline = poly.get("trimmed_centerline")
            if road_id is None or centerline is None or len(centerline) < 2:
                continue

            # Entartete (Nulllängen-)Straßen überspringen: Clipping/Junction-
            # Split können vereinzelt einen "Rest" mit 2 identischen Punkten
            # hinterlassen. Ein DecalRoad mit Länge 0 ist ein degenerierter
            # Spline (im alten Mesh-Ansatz war das ein unsichtbares
            # Nulldreieck, hier würde es ein kaputtes Decal-Item erzeugen).
            xy_unique = {(round(float(x), 3), round(float(y), 3)) for x, y, _ in centerline}
            if len(xy_unique) < 2:
                continue

            props = OSM_MAPPER.get_road_properties(poly.get("osm_tags", {}))
            mat_name = props.get("internal_name", "road_default")
            unique_materials[mat_name] = props

            width = float(props.get("width", 4.0))
            nodes = [[float(x), float(y), float(z), width] for x, y, z in centerline]

            # Zu kurze Segmente entfernen: BeamNG zeichnet ein DecalRoad mit
            # einem zu kurzen Segment (z.B. 0,10 m vom Junction-Schnitt neben
            # einem Resample-Punkt) gar nicht - das ganze Stück fehlt dann.
            nodes = drop_close_nodes(nodes, config.DECAL_ROAD_MIN_NODE_SPACING)
            if len(nodes) < 2:
                continue

            # renderPriority aus dem vorhandenen "priority"-Feld ableiten
            # (surface_types in data/osm_to_beamng.json): an Kreuzungen
            # überlappen sich die (immer volle Breite habenden) Enden
            # mehrerer DecalRoad-Objekte - ohne explizite, konsistente
            # Zeichenreihenfolge sortiert BeamNG das beliebig, was an
            # Kreuzungen wie ein "Flickenteppich" aussieht. Höherwertige
            # Straßen (Asphalt) werden so immer über niedrigerwertigen
            # (Dirt/Concrete) gezeichnet.
            render_priority = int(props.get("priority", 0))

            self.items.add_decal_road(
                name=f"road_{road_id}",
                nodes=nodes,
                material=mat_name,
                drivability=props.get("drivability", 1.0),
                overwrite=True,
                autoLanes=True,
                autoJunction=True,
                improvedSpline=True,
                renderPriority=render_priority,
            )
            count += 1

        road_material_entries = [
            OSM_MAPPER.generate_materials_json_entry(mat_name, props) for mat_name, props in unique_materials.items()
        ]
        for mat_entry in road_material_entries:
            mat_name = mat_entry.pop("__name", None)
            if mat_name:
                self.materials.materials[mat_name] = mat_entry

        logger.info(f"  [OK] {count} DecalRoad-Item(s) exportiert ({len(unique_materials)} Materialien)")
        return count

    def export_ground_cover(
        self, layer_map: np.ndarray, terrain_material_names: List[str], layer_variants: Optional[Dict] = None
    ) -> int:
        """
        Registriert Bodenbewuchs (Gras, Blumen, Farn, Unkraut) als GroundCover-
        Objekte für jeden Terrain-Layer, der in der Layer-Map tatsächlich vorkommt,
        samt der Billboard-Materialien (gemeinsame BeamNG-Assets).

        Args:
            layer_map: fertige globale Layer-Map (Index in terrain_material_names)
            terrain_material_names: Layer-Namen in Index-Reihenfolge
            layer_variants: Vier-Bilder-Modus: Schicht -> Kachel-Varianten (Namen in terrain_material_names)

        Returns:
            Anzahl der erzeugten GroundCover-Objekte
        """
        if not config.GROUND_COVER_ENABLED:
            return 0

        from ..terrain.ground_cover import (
            build_billboard_material_entries,
            build_ground_cover_items,
            load_ground_cover_templates,
        )

        used_physical = [terrain_material_names[i] for i in np.unique(layer_map) if i < len(terrain_material_names)]
        # Im Vier-Bilder-Modus stehen in der Layer-Map nur Varianten (mat_grass_t0 ...): zurück auf die Schicht
        logical_of = {variant: layer for layer, variants in (layer_variants or {}).items() for variant in variants}
        used_layers = list(dict.fromkeys(logical_of.get(name, name) for name in used_physical))
        templates_data = load_ground_cover_templates()
        items = build_ground_cover_items(
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            used_layers,
            templates_data,
            max_elements=config.GROUND_COVER_MAX_ELEMENTS,
            max_radius=config.GROUND_COVER_MAX_RADIUS,
            layer_variants=layer_variants,
        )
        for item in items:
            fields = dict(item)
            self.items.add_ground_cover(fields.pop("name"), fields.pop("material"), fields.pop("Types"), **fields)

        self.materials.materials.update(build_billboard_material_entries(items, templates_data))
        logger.info(f"  [OK] {len(items)} GroundCover-Objekt(e) für {len(used_layers) - 1} Landnutzungs-Layer")
        return len(items)

    def export_merged_terrain(
        self,
        heights: np.ndarray,
        layer_map: np.ndarray,
        terrain_material_names: List[str],
        terrain_origin_x: float,
        terrain_origin_y: float,
        terrain_size: int,
        z_min: float,
        max_height: float,
        photo_tile_names: List[str],
        layer_variants: Optional[Dict] = None,
        variant_parents: Optional[Dict] = None,
        photo_extents: Optional[Dict] = None,
    ) -> None:
        """
        Schreibt EIN .ter und registriert TerrainBlock + TerrainMaterials.

        Args:
            heights, layer_map: fertige (bereits gepaddete) globale Arrays,
                shape (terrain_size, terrain_size)
            terrain_material_names: globale, deduplizierte Materialliste
                (Index entspricht layer_map-Werten)
            terrain_origin_x, terrain_origin_y: Welt-Koordinaten der Zelle [0, 0]
            z_min, max_height: siehe ter_writer.encode_heights_to_u16()
            photo_tile_names: Teilmenge von terrain_material_names, die auf
                das zusammengesetzte Luftbild verweisen (siehe
                terrain_materials.build_terrain_material_entries)
        """
        from ..terrain.ter_writer import write_ter, encode_heights_to_u16
        from ..terrain.terrain_materials import (
            build_terrain_material_entries,
            build_terrain_material_texture_set,
            ensure_flat_pbr_placeholders,
            ensure_landuse_detail_textures_sized,
            DETAIL_TEX_SIZE,
        )

        self._set_fog_height(heights)
        heightmap_u16 = encode_heights_to_u16(heights, z_min, max_height)
        ter_filename = f"{config.LEVEL_NAME}.ter"
        ter_path = config.BEAMNG_DIR / ter_filename
        write_ter(ter_path, heightmap_u16, layer_map.astype("uint8"), terrain_material_names)
        logger.info(f"  [OK] Terrain exportiert: {ter_filename} ({terrain_size}x{terrain_size})")

        placeholders = ensure_flat_pbr_placeholders(
            config.BEAMNG_DIR_TEXTURES, config.LEVEL_NAME, config.TERRAIN_BASE_TEX_PIXEL_SIZE
        )
        sized_landuse_mappings = ensure_landuse_detail_textures_sized(
            config.OSM_MAPPER.config.get("landuse_mappings", {}),
            DETAIL_TEX_SIZE,
            config.BEAMNG_DIR,
            config.BEAMNG_DIR_TEXTURES,
            config.LEVEL_NAME,
        )
        # photo_extent_size: der Wert, den wir BeamNG als baseColorBaseTexSize
        # für das Luftbild-Material mitteilen.
        #
        # DIAGNOSE 2026-09-18 (drei Messpunkte mit echtem Nutzer-Test, siehe
        # [[project_road_embed_slope_margin]]-Nachbar-Memory für den Kontext
        # dieser Session):
        #   - Grid @ 2m/Zelle: terrain_size=1024, TERRAIN_SQUARE_SIZE=2.0,
        #     physische Größe 2048m -> deklarierter Wert 1024 war korrekt.
        #   - Grid @ 1m/Zelle: terrain_size=2048, TERRAIN_SQUARE_SIZE=1.0,
        #     physische Größe UNVERÄNDERT 2048m -> derselbe Wert 1024 (aus der
        #     alten "physische_Größe * 0.5"-Formel) war jetzt FALSCH, korrekt
        #     ist 2048.
        # Die physische Kartengröße blieb in beiden Fällen identisch (2048m),
        # nur die Rasterauflösung hat sich geändert - trotzdem musste sich der
        # korrekte Wert mit der Rasterauflösung verdoppeln. Das heißt,
        # baseColorBaseTexSize wird von BeamNG offenbar in Heightmap-
        # Rasterzellen interpretiert, NICHT in Weltmetern (entgegen der
        # Doku-Formel "world_size = size * squareSize"): der korrekte Wert
        # ist schlicht terrain_size, die reine .ter-Rasterpunktzahl, komplett
        # unabhängig von TERRAIN_SQUARE_SIZE.
        photo_extent_size = float(terrain_size)
        terrain_material_entries = build_terrain_material_entries(
            terrain_material_names,
            photo_tile_names,
            sized_landuse_mappings,
            config.LEVEL_NAME,
            photo_extent_size,
            placeholders,
            variant_parents=variant_parents,
            photo_extents=photo_extents,
        )
        texture_set_name = f"{config.LEVEL_NAME}TerrainMaterialTextureSet"
        terrain_material_entries.update(
            build_terrain_material_texture_set(
                texture_set_name, base_tex_size=config.TERRAIN_BASE_TEX_PIXEL_SIZE
            )
        )
        self.materials.add_terrain_materials(terrain_material_entries)

        self.export_ground_cover(layer_map, terrain_material_names, layer_variants=layer_variants)

        self.items.add_terrain_block(
            name="theTerrain",
            terrain_filename=ter_filename,
            material_texture_set=texture_set_name,
            max_height=max_height,
            z_min=z_min,
            origin_x=terrain_origin_x,
            origin_y=terrain_origin_y,
            square_size=config.TERRAIN_SQUARE_SIZE,
            overwrite=True,
        )

    def export_tile(self, tile_x: int, tile_y: int, mesh_data: Dict) -> int:
        """
        Exportiere EINE einzelne Kachel komplett (DecalRoad-Straßen + eigenes .ter).

        Convenience-Wrapper um export_decal_roads()/export_merged_terrain()
        für Aufrufer, die nur eine einzelne Kachel exportieren
        (export_single_tile()/export_terrain_only()).

        Args:
            tile_x, tile_y: unbenutzt, nur für Aufrufer-Kompatibilität
            mesh_data: Mesh-Daten aus process_tile()

        Returns:
            Anzahl der erzeugten DecalRoad-Items
        """
        road_count = self.export_decal_roads(mesh_data)
        self.export_water(mesh_data)
        self.export_walls(mesh_data)
        self.export_bridges(mesh_data)
        self.export_tunnels(mesh_data)
        self.export_merged_terrain(
            heights=mesh_data["heightmap"],
            layer_map=mesh_data["layer_map"],
            terrain_material_names=list(mesh_data["terrain_material_names"]),
            terrain_origin_x=mesh_data["terrain_origin_x"],
            terrain_origin_y=mesh_data["terrain_origin_y"],
            terrain_size=mesh_data["terrain_size"],
            z_min=mesh_data["z_min"],
            max_height=mesh_data["max_height"],
            photo_tile_names=mesh_data["photo_tile_names"],
            layer_variants=mesh_data.get("layer_variants"),
            variant_parents=mesh_data.get("variant_parents"),
            photo_extents=mesh_data.get("photo_extents"),
        )
        return road_count
