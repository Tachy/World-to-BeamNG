"""
Zentrale Konfiguration fuer World-to-BeamNG.
"""

import logging
import os
from pathlib import Path, PurePosixPath

from .osm.osm_mapper import OSMMapper
from .logging_config import LoggerConfig

LEVEL_NAME = "world_to_beamng"  # Name des BeamNG Levels (muss mit BEAMNG_DIR übereinstimmen)

# OSM Mapper Singleton (lädt data/osm_to_beamng.json)
OSM_MAPPER = OSMMapper(config_path=Path("data/osm_to_beamng.json"))

# Ungefährer Referenzpunkt NUR für die Sonnenstand-Berechnung (managers/environment.py, siehe
# ENV_DATE/ENV_CLOCK_TIME unten) - NICHT der Fahrzeug-Spawn-Punkt. Der wird automatisch berechnet
# (Gebietsmitte, auf die nächstgelegene Straße gelegt, siehe managers/item_manager.py::
# _compute_vehicle_spawn()) und braucht keine Einstellung mehr. Die Abweichung dieses Referenz-
# punkts vom tatsächlichen Gebiet macht für den Sonnenwinkel praktisch keinen Unterschied.
SUN_REFERENCE_LATLON = (47.842840, 7.684767)

# BeamNG-Benutzerordner der aktuellen Version (legt BeamNG beim ersten Start an):
# %LOCALAPPDATA%\BeamNG\BeamNG.drive\current. Das Level entsteht darin unter levels/<LEVEL_NAME>.
BEAMNG_USER_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "BeamNG" / "BeamNG.drive" / "current"
BEAMNG_DIR = BEAMNG_USER_DIR / "levels" / LEVEL_NAME
BEAMNG_DIR_SHAPES = BEAMNG_DIR / "art" / "shapes"
BEAMNG_DIR_TEXTURES = BEAMNG_DIR_SHAPES / "textures"
BEAMNG_DIR_BUILDINGS = BEAMNG_DIR_SHAPES / "buildings"

# Eingecheckte, einmalig erzeugte bzw. fotobasierte Texturen (Ordner je Textur + manifest.json), siehe textures/library.py
TEXTURE_LIBRARY_DIR = Path("data/textures")

# In-Game relative paths (MUST use forward slashes for BeamNG)
RELATIVE_DIR = PurePosixPath("levels") / LEVEL_NAME
RELATIVE_DIR_SHAPES = RELATIVE_DIR / "art" / "shapes"
RELATIVE_DIR_TEXTURES = RELATIVE_DIR_SHAPES / "textures"
RELATIVE_DIR_BUILDINGS = RELATIVE_DIR_SHAPES / "buildings"


# === BEAMNG LEVEL-STRUKTUR ===
ITEMS_JSON = Path("main") / "MissionGroup" / "items.level.json"  # Items im MissionGroup-Verzeichnis
MATERIALS_JSON = Path("main") / "materials.json"  # Enthält Material-Definitionen

# Ablaufsteuerung
LOD2_ENABLED = False  # LoD2-Gebäude verarbeiten
PHASE5_ENABLED = True  # Horizont-Layer aktivieren (erfordert DGM30 + DOP300 Daten)
FORESTS_ENABLED = True  # Wald-Export global aktivieren/deaktivieren

# Zusätzlich zum automatischen Standard-Spawn (nächste Straße zur Gebietsmitte) bekommt jede eindeutig
# benannte OSM-Straße (osm_tags["name"]) einen eigenen, in der Fahrzeugauswahl wählbaren Spawn-Punkt -
# siehe ItemManager._compute_named_spawn_points(). Tunnel/Galerien werden ausgeschlossen (ungeeigneter
# Spawn-Ort), Brücken bleiben erlaubt. Bei mehr benannten Straßen als das Limit gewinnen die längsten.
MAX_NAMED_SPAWN_POINTS = 20


# === MATERIAL-EINSTELLUNGEN ===
# Materialien verwenden IMMER Texturen (keine Farb-Fallbacks)

HORIZON_GRID_SPACING = 200  # Horizont-Grid Auflösung in Metern (200m)
HORIZON_HALF_SIZE_M = 50000  # Der Horizont reicht so weit von der Gebietsmitte in jede Richtung (also 100 x 100 km)
HORIZON_IMAGE_SIZE_PX = 8192  # Kantenlänge der Horizont-Textur; das Satellitenbild wird darauf skaliert
# Naht Terrain <-> Horizont (siehe terrain/horizon_seam.py): der Horizont hat ein exakt passendes Loch
# für den Terrain-Block, feinen Randring mit den Terrain-Randhöhen und sanften Höhenübergang.
TERRAIN_PADDING_AS_HOLES = True  # Aufgefüllten Heightmap-Rand (jenseits der Daten) als Hole - der Horizont deckt ihn ab
HORIZON_SEAM_STEP = 1.0  # Punktabstand des Randrings am Terrain-Loch in Metern (= TERRAIN_SQUARE_SIZE: Naht exakt, DGM1 unverändert)
HORIZON_BLEND_DISTANCE = 1000.0  # Länge des Höhenübergangs Terrain -> DGM30 in Metern
HORIZON_FLANGE_INSET = 5.0  # Breite des Flansches unter dem Terrain (verdeckt Restrisse), 0 = aus
HORIZON_FLANGE_SINK = 15.0  # Tiefe des Flansches unter der Terrainhöhe in Metern

# === SICHTWEITE / NEBEL (LevelInfo) ===
# Ohne visibleDistance nutzt BeamNG seinen Standard (~1 km) - alles dahinter wird geclippt.
# Original-Level: utah 5000, east_coast 12000, west_coast 15667, italy 25000. Der Horizont reicht
# +-50 km; der Nebel blendet die Kappung aus (bei 0.0002 sind nach 25 km nur noch ~0.7 % sichtbar).
# Größere Werte zeigen mehr vom Horizont, kosten aber Tiefenpräzision (nicht getestet über 25000).
LEVEL_VISIBLE_DISTANCE = 25000  # Sichtweite in Metern (LevelInfo.visibleDistance)
LEVEL_FOG_DENSITY = 0.0002  # Nebeldichte (LevelInfo.fogDensity), kleiner = klarere Fernsicht
# Licht/Wetter: Sonnenstand, Himmel und Wolken kommen aus BeamNGs eigenen Vorgaben (data/environment_defaults.json,
# managers/environment.py). Startzustand = "sonnig"; Uhrzeit, Wolken und Wetter sind im Spiel regelbar.
ENV_FOG_COLOR = [0.741176, 0.815686, 0.92549, 1.0]  # Dunstfarbe (hellblau, wie die meisten Original-Level; ohne wird der ferne Horizont grau)
ENV_FOG_HEIGHT_MARGIN = 50.0  # fogAtmosphereHeight = höchster Terrainpunkt + diese Marge in m (Originale: ca. Geländehöhe)
ENV_DATE = (2026, 6, 21)  # Datum für den Sonnenstand (Sommer: die DOP20-Luftbilder sind Sommeraufnahmen)
ENV_CLOCK_TIME = "11:00"  # Start-Uhrzeit "HH:MM" (Ortszeit; 12:00 = höchster Sonnenstand)

# === WASSER (echte BeamNG-Objekte: River für Bäche, WaterBlock für Teiche/Seen) ===
WATER_ENABLED = True
WATERWAY_WIDTHS = {"stream": 2.5, "river": 8.0, "canal": 5.0}  # Standardbreite je Art in m (Tag "width" hat Vorrang); Gräben nicht
WATER_RIVER_DEPTH = 1.0  # Tiefe des River-Volumens in m
WATER_NODE_SPACING = 10.0  # Abstand der River-Knoten in m
WATER_STREAM_LIFT = 0.2  # Wasserstand über dem Rinnenboden des DGM1 in m (Ufer liegen im Median 0,4 m höher)
WATER_MAX_RIVER_NODES = 40  # längere Bäche werden in mehrere River-Objekte geteilt
WATER_POND_MARGIN = 2.0  # so weit reichen die WaterBlocks über den Rand des OSM-Polygons hinaus (füllen das ganze Loch), in m
WATER_POND_DEPTH = 3.0  # Tiefe der WaterBlocks in m
WATER_POND_BANK_DEPTH = 0.5  # innerhalb des OSM-Wasserpolygons wird das Terrain so viel tiefer gelegt, in m
WATER_POND_BANK_SLOPE_DEG = 45.0  # Böschungswinkel der Mulde nach innen (45 Grad = 1 m tiefer je Meter nach innen)
WATER_POND_CELL = 6.0  # maximale Kantenlänge der Kacheln, mit denen Teiche gefüllt werden, in m
WATER_POND_CUBEMAP = "DefaultSkyCubemap"  # Engine-eigene Cubemap (die der Vorlage ist level-spezifisch und fehlte)

# === FOREST GENERATION PARAMETERS ===
FOREST_ROAD_MARGIN = 2.0  # Puffer um die OSM-Mittellinie (Rückfallebene; die Fahrbahnkante unten ist maßgeblich), in Metern
FOREST_BUILDING_MARGIN = 2.5  # Puffer um Gebäude: dort stehen keine Bäume/Büsche (in Metern)
FOREST_ROAD_SURFACE_MARGIN = 2.0  # Abstand zur Kante der eingebetteten Fahrbahn (gilt für jeden Stamm, auch von Gruppen-Bäumen)
FOREST_ROW_SURFACE_MARGIN = 1.0  # dasselbe für Baumreihen: Stämme nicht auf der Fahrbahn
FOREST_ROW_ROAD_MARGIN = 3.0  # Baumreihen (Alleen) stehen näher an Straßen als Wald: kleinerer Puffer (in Metern)
# Gruppen-Assets haben mehrere Stämme bis ~9 m neben dem Ursprung. Die Abstände oben gelten auch für jeden Stamm
# (forest/tree_footprints.py), nicht nur für den Ursprung.
FOREST_TRUNK_MAX_FLOAT = 0.5  # so weit darf ein Stammfuß nach dem Absenken über dem Boden stehen (in Metern)
FOREST_TRUNK_MAX_SINK = 1.0  # so weit darf ein Baum höchstens abgesenkt werden, sonst Typwechsel (in Metern)

# Die Böschung entsteht NICHT im Mesh: Straßen sind DecalRoads (workflow/terrain_workflow.py::export_decal_roads()), der
# Übergang zur Umgebung entsteht direkt im Terrain-Heightmap (terrain/road_embedding.py::apply_embankment_blend()).
# Minimale Boeschungsbreite (Meter) unabhängig von Hoehenunterschieden
MIN_SLOPE_WIDTH = 2
# Obergrenze der Böschungsbreite (Meter), unabhängig davon, wie groß der
# Höhenunterschied zwischen Straßenkante und natürlichem Terrain ist
# (verhindert unrealistisch breite Böschungskorridore bei Extremfällen).
MAX_SLOPE_WIDTH = 30.0
SLOPE_ANGLE = 45.0  # Neigungswinkel der Boeschung in Grad (45° = 1:1 Steigung)
# Vorab-Reduktion ueber groeberes Grid (Strategie 2). Fuer feineres Terrain z.B. 1.0 setzen.
GRID_SPACING = 1.0  # Abstand zwischen Grid-Punkten in Metern (native DGM1-Auflösung; 10.0 = grob)

# Fallback-CRS (EPSG-Code) fuer Hoehen-/Luftbilddaten ohne eigenes eingebettetes CRS (z.B. reine
# ASCII-XYZ-Punktwolken wie bei LGL Baden-Wuerttemberg). Bei GeoTIFF-Quellen wird das eingebettete
# CRS automatisch erkannt (geometry.coordinates.set_source_crs()) und hat Vorrang vor diesem Wert.
SOURCE_CRS_EPSG = 25832

# === NATIVES TERRAIN (.terrain-Heightmap) ===
# Meter pro Heightmap-Rasterzelle. = GRID_SPACING für Auflösungs-Parität zum
# bisherigen Mesh-Ansatz (siehe Spec Abschnitt 2, Anforderung 2).
TERRAIN_SQUARE_SIZE = GRID_SPACING
# Kein ROAD_EMBED_MARGIN/Gefälle-Kompensation mehr nötig (frühere, jetzt
# entfernte Konstanten): seit der Umstellung auf BeamNG `DecalRoad` (siehe
# terrain/road_embedding.py-Moduldocstring) gibt es keine zweite, separat
# kodierte Straßen-Oberfläche mehr, die getroffen werden müsste - das
# Terrain wird direkt exakt auf Straßen-Centerline-Höhe gesetzt.
# Puffer (Meter) über dem tatsächlichen Höhen-Max/-Min beim Berechnen von
# maxHeight für die .terrain-Datei (siehe Spec Abschnitt 8).
TERRAIN_MAX_HEIGHT_BUFFER = 50.0

# DEBUG / EXPORTS
DEBUG_EXPORTS = True  # Debug-Dumps (Netz, Grid) nur bei Bedarf aktivieren
DEBUG_VERBOSE = False  # Zusätzliche Konsolen-Logs

# === LANDNUTZUNG / BODENBEWUCHS ===
# Bodenbewuchs (Gras, Blumen, Farn ...) als BeamNG-GroundCover-Objekte je Terrain-Layer.
GROUND_COVER_ENABLED = True
GROUND_COVER_MAX_ELEMENTS = 100000  # Obergrenze gleichzeitig gezeichneter Elemente je Objekt (Leistung!)
GROUND_COVER_MAX_RADIUS = 100.0  # Obergrenze der Sichtweite in Metern je Objekt (BeamNG-Originale: 50-120)
# Unter Straßen (plus Schulter) und Gebäuden wird der Layer auf das Luftbild zurückgesetzt,
# damit dort kein Gras durch Decals/Häuser wächst.
GROUND_COVER_ROAD_MARGIN = 1.0
GROUND_COVER_BUILDING_MARGIN = 0.5
# Weinberg-Reben (Forest-Items) - benötigen FORESTS_ENABLED.
VINEYARDS_ENABLED = True
# Abstand der Reben zum Rand von Straßen/Gebäuden in Metern. Die Zeilen laufen sonst bis exakt an die
# Polygongrenze des Weinbergs und enden hier auf den Zentimeter an dieser Ausschlusszone.
VINEYARD_EXCLUSION_MARGIN = 2.0

# === LOGGING ===
LOGGING_FILE = None  # Path("logs/world_to_beamng.log")  # Optional; None = nur stdout
LOGGING_LEVEL = logging.DEBUG if DEBUG_VERBOSE else logging.INFO

# Initialisiere zentrale Logger-Instanz
LoggerConfig.get_instance(log_file=LOGGING_FILE, level=LOGGING_LEVEL, verbose=DEBUG_VERBOSE)

# === STRASSENGLÄTTUNG / OPTIONEN ===
ENABLE_ROAD_SMOOTHING = True  # False = Spline-Glättung komplett aus
SAMPLE_SPACING_FACTOR = 0.5  # Faktor für Segment-Spacing: road_width * SAMPLE_SPACING_FACTOR
ROAD_SMOOTH_ITERATIONS = 1  # Anzahl Smoothing-Iterationen (1-3; höher = glatter)
ROAD_SMOOTH_WEIGHT = 0.6  # Chaikin-Filter Gewicht (0.5-0.9; höher = weniger Glättung, 0.75 = mild)

# Mindestabstand zwischen zwei DecalRoad-Knoten in Metern. BeamNG zeichnet ein
# DecalRoad gar nicht, wenn es ein zu kurzes Segment hat (0,10 m -> unsichtbar,
# 0,32 m -> ok; verifiziert an der Eichgasse). Typischer Abstand nach dem
# Resampling ist ~0,8 m.
DECAL_ROAD_MIN_NODE_SPACING = 0.5

# === CLIPPING ===
ENABLE_ROAD_CLIPPING = True  # True = Clip + Segment-Unterteilung am Grid-Rand, False = Skip (Testbetrieb)
ROAD_CLIP_MARGIN = -20.0  # Clipping-Abstand vom Grid-Rand in Metern (Faces < 3m vom Rand werden entfernt)

# === TILE-EXPORT (DAE) ===
BUILDINGS_AS_ONE_OBJECT = True  # True: ALLE Gebäude in EINER DAE/EINEM Objekt auf der Gesamtfläche (wie die Straßen)
MAX_BUILDINGS_PER_SHAPE = 1500  # BeamNG verwirft ab 2048 Nodes je Shape alles Weitere (1 Node je Gebäude) -> aufteilen
TILE_SIZE = 500  # Größe pro DAE-Tile in Metern (nur bei BUILDINGS_AS_ONE_OBJECT = False)

# === MAUERN (OSM barrier=wall / retaining_wall, nur mit height-Tag) ===
WALLS_ENABLED = True
WALL_THICKNESS = 0.5  # Mauerdicke in Metern (Bruchsteinmauer)
WALL_ROAD_SNAP_M = 5.0  # Mauern höchstens so weit neben einer Straßen-Centerline nehmen deren Höhe als Basis (sonst Gelände), in Metern
WALL_CAP_THICKNESS = 0.05  # Abdeckplatten oben auf der Mauer: Dicke in Metern (0 = keine Platten); die Gesamthöhe bleibt die OSM-Höhe
WALL_CAP_OVERHANG = 0.04  # Überstand der Platten über den Mauerkörper (je Seite und an offenen Enden), in Metern
WALL_CAP_PLATE_LENGTH = 0.8  # mittlere Plattenlänge in Metern (Länge variiert um +-25 %)
WALL_CAP_JOINT = 0.01  # Fuge zwischen zwei Platten in Metern
WALL_SINK = 0.3  # so tief reicht die Unterkante unter den Boden (kein Spalt am Fuß), in Metern
WALL_MAX_SEGMENT = 1.0  # längste Teilstrecke, damit die Mauer dem Gelände folgt, in Metern
WALL_TEXTURE_TILE_M = 1.2  # Rückfall für die Kachelgröße der Bruchstein-Textur in Metern; maßgeblich ist tile_m im Manifest von data/textures
WALL_TEXTURE_NAME = "rubble_stone_wall"  # Name der Bruchstein-Textur in data/textures (Foto, tools/make_seamless_texture.py)
WALL_MATERIAL_NAME = "rubble_stone_wall"  # "wall" im Namen wählt das Wand-Template; Texturen: WALL_TEXTURE_NAME (textures/registry.py)

# Brücken (OSM highway=* mit bridge=*): generisches Beton-Deck mit dem Fahrbahnmaterial der Straße obenauf und
# Stützpfeilern zum natürlichen Gelände darunter (siehe Design-Spec docs/superpowers/specs/
# 2026-09-22-bridges-tunnels-design.md Abschnitt 4). Ersetzt für diese Straßen die normale Terrain-Einbettung
# und den DecalRoad-Export.
BRIDGES_ENABLED = True
BRIDGE_DECK_THICKNESS = 0.6  # Deck-Dicke in Metern
BRIDGE_PIER_SPACING = 25.0  # Abstand der Stützpfeiler entlang der Brücke, in Metern
BRIDGE_PIER_SIZE = 1.5  # Querschnitt der (quadratischen) Stützpfeiler, in Metern
BRIDGE_MIN_PIER_CLEARANCE = 1.0  # kein Pfeiler, wenn der Abstand Deck-Unterkante/Gelände kleiner ist, in Metern
BRIDGE_MATERIAL_NAME = "bridge_concrete"  # Pfeiler-/Bordstein-Material (Textur: CONCRETE_TEXTURE_NAME, siehe Task 10)

# Echtes Straßenbrücken-Querschnittsprofil: Betonrand (Bordstein) beidseits der Fahrbahn, darauf ein
# einfaches Geländer (Pfosten + durchlaufender Handlauf), statt einer randlosen Deckplatte.
BRIDGE_CURB_WIDTH = 0.25  # Breite des Bordsteins je Seite, in Metern (Fahrbahn wird entsprechend schmaler)
BRIDGE_CURB_HEIGHT = 0.15  # Höhe des Bordsteins über der Fahrbahn, in Metern
BRIDGE_RAILING_HEIGHT = 0.9  # Handlauf-Höhe über der Bordstein-Oberkante, in Metern
BRIDGE_RAILING_POST_SPACING = 2.0  # Pfosten-Abstand entlang der Brücke, in Metern
BRIDGE_RAILING_POST_SIZE = 0.08  # Querschnitt der (quadratischen) Pfosten und des Handlaufs, in Metern
BRIDGE_RAILING_MATERIAL_NAME = "bridge_railing"  # Textur: RAILING_TEXTURE_NAME

# Manche OSM-Brücken sind zu knapp bemessen und beginnen bereits mitten in der Hanglage statt auf
# Straßenniveau (die lineare Höheninterpolation zwischen den Way-Endpunkten ergibt dann eine unrealistisch
# steile Rampe). Fix: die Brücke wird in ihre Nachbarstraße hinein verlängert, bis dort wieder normales
# Gefälle herrscht (siehe geometry.polygon.extend_short_bridges_to_natural_grade).
BRIDGE_APPROACH_SLOPE_THRESHOLD = 0.10  # Gefälle, ab dem die Verlängerung stoppt (10 % ≈ 5,7°)
BRIDGE_APPROACH_MAX_EXTENSION = 40.0  # längstens so weit wird in die Nachbarstraße hinein verlängert, in Metern

# Tunnel (OSM highway=* mit tunnel=yes/culvert/building_passage) und Galerien (tunnel=avalanche_protector):
# Röhre bzw. talseitig offene Galerie entlang des linear interpolierten Höhenprofils (siehe Design-Spec
# Abschnitt 5/6). Ersetzt für diese Straßen die normale Terrain-Einbettung und den DecalRoad-Export.
TUNNELS_ENABLED = True  # deckt auch Galerien (tunnel=avalanche_protector) ab
TUNNEL_WIDTH_MARGIN = 1.5  # zusätzliche Breite über die Fahrbahnbreite hinaus, in Metern
TUNNEL_ARC_SEGMENTS = 12  # Diskretisierung des 240°-Kreisbogens (Radius/Kronenhöhe ergeben sich aus der Breite)
TUNNEL_SEGMENT_STEP = 10.0  # Extrusions-Schrittweite entlang der Achse, in Metern (grob, da geradlinig)
TUNNEL_PORTAL_SLOPE_SAMPLE_DIST = 5.0  # Abtastradius der Hangneigung an den Portalen, in Metern
TUNNEL_PORTAL_FRAME_MARGIN = 0.6  # Rahmenbreite um die Portalöffnung, in Metern
GALLERY_HEIGHT = 5.0  # lichte Höhe der (rechteckigen, nicht kreisrunden) Galerie, in Metern
GALLERY_COLUMN_SPACING = 6.0  # Stützenabstand auf der offenen Talseite, in Metern
GALLERY_ROOF_THICKNESS = 0.35  # Dachdicke, in Metern
GALLERY_COLUMN_SIZE = 0.4  # Querschnitt der (quadratischen) Stützen, in Metern

# Galerien bekommen (anders als Brücken/Tunnel) ihr eigenes Boden-/Wand-/Dach-Mesh auf echtem Straßenniveau,
# liegen aber - anders als ein tief im Berg liegender Tunnel - direkt am Hang: das unveränderte natürliche
# Gelände würde Durchfahrt, Eingang und die bergseitige Wand/Dachkante blockieren. Terrain-Hole (Layer 255)
# statt Einebnen, da die Galerie schon ein eigenes Boden-Mesh hat (sonst Z-Fighting) - siehe
# terrain_materials.mark_structure_roads_as_holes().
GALLERY_TERRAIN_HOLE_MARGIN = 1.0  # Puffer über die Fahrbahnbreite hinaus, in Metern (Seitenwand/Dachkante)
TUNNEL_MATERIAL_NAME = "tunnel_concrete"  # Wand-/Decke-/Rahmen-/Dach-Material (Textur: CONCRETE_TEXTURE_NAME)

# === LOD2-DACH-TEXTUR ===
# t_roof_slates_rounded_b.color.dds (256 px) zeigt 5 Biberschwanz-Ziegel nebeneinander je Wiederholung (6 Reihen
# übereinander). Die UVs sind metrisch: eine Wiederholung = ROOF_REPEAT_M Meter in der Dachebene, ein Ziegel ist
# damit immer ROOF_TILE_WIDTH_M breit, egal wie steil das Dach ist.
ROOF_TILE_WIDTH_M = 0.20
ROOF_TILES_PER_REPEAT = 5
ROOF_REPEAT_M = ROOF_TILE_WIDTH_M * ROOF_TILES_PER_REPEAT

# Schrägdächer ragen über Wände hinaus (die LOD2-Daten enden exakt an der Wand): an der Traufe ROOF_EAVE_OVERHANG_M,
# am Giebel (Ortgang) ROOF_VERGE_OVERHANG_M. Der Überstand ist ROOF_OVERHANG_THICKNESS_M dick modelliert (Stirnbrett
# und Untersicht).
ROOF_EAVE_OVERHANG_M = 0.6
ROOF_VERGE_OVERHANG_M = 0.3
ROOF_OVERHANG_THICKNESS_M = 0.10

# Flachdächer (Neigung bis FLAT_ROOF_MAX_SLOPE_DEG): Kiesfläche (kachelbare Textur, FLAT_ROOF_GRAVEL_REPEAT_M Meter je
# Wiederholung) mit umlaufendem Blechrand als echte Geometrie.
FLAT_ROOF_MAX_SLOPE_DEG = 5.0
FLAT_ROOF_GRAVEL_REPEAT_M = 2.0  # Kachelgröße der Kies-Textur; das Werkzeug generate_gravel_texture.py trägt sie ins Manifest ein
FLAT_ROOF_GRAVEL_TEXTURE = "roof_gravel"  # Name in data/textures
FLAT_ROOF_GRAVEL_TEXTURE_PX = 1024
FLAT_ROOF_EDGE_HEIGHT_M = 0.25
FLAT_ROOF_EDGE_THICKNESS_M = 0.05

# Prozedurale Beton-Textur für Brücken-Pfeiler/Bordsteine, Tunnel-Wände/Decke/Portale und Galerie-Dach/Stützen.
CONCRETE_TEXTURE_NAME = "tunnel_concrete"
CONCRETE_TEXTURE_TILE_M = 2.0
CONCRETE_TEXTURE_PX = 1024

# Prozedurale Stahl-Textur für Brücken-Geländer (Pfosten + Handlauf).
RAILING_TEXTURE_NAME = "bridge_railing"
RAILING_TEXTURE_TILE_M = 0.3
RAILING_TEXTURE_PX = 512

# === LOD2-WÄNDE (verputzt) ===
# Wand = EIN ungeschnittenes Polygon mit fugenloser, metrisch gekachelter Putztextur (FACADE_PLASTER_REPEAT_M Meter je
# Wiederholung); es gibt keine Zellen oder Fugen im Putz. Die Farbe ist je Gebäude fest (gewichtete Auswahl, siehe
# facade/facade_styles.py), jede Farbe ist ein eigenes Material auf derselben Normal-/Roughness-Textur.
# Fenster und Türen sind eigene, kleine Flächen (Sprites aus einem Atlas), die FACADE_WINDOW_OFFSET_M vor der Wand liegen.
FACADE_PLASTER_TEXTURE_PX = 1024
FACADE_PLASTER_REPEAT_M = 3.0
FACADE_STOREY_HEIGHT_M = 3.0  # Geschosshöhe; die Geschosse werden von der Traufe nach UNTEN gezählt
FACADE_BAY_WIDTH_M = 2.5  # Soll-Achsbreite; real wird die Wandbreite gleichmäßig auf ganze Achsen verteilt
FACADE_NARROW_WALL_M = 2.5  # schmalere Wände (Vorsprünge, Stirnseiten) bekommen keine Fenster
FACADE_WINDOW_OFFSET_M = 0.03  # Fenster liegen so weit vor der Wand (gegen Z-Fighting)
FACADE_WINDOW_SILL_M = 0.9  # Brüstungshöhe: Unterkante Fenster über dem Geschossboden
FACADE_WINDOW_ATLAS_PX_PER_M = 200  # Auflösung der Fenster-Sprites
FACADE_GUTTER_PX = 8  # Rand um jedes Sprite, in den die Kanten repliziert werden (gegen Bluten beim Filtern)
FACADE_MAX_MIP_LEVELS = 6  # längere Mip-Ketten lassen Nachbar-Sprites ineinander bluten
# Erhöhter Keller: bleibt unter den vollen Geschossen ein Rest von mindestens FACADE_BASEMENT_MIN_REMAINDER_M übrig,
# ist das ein Kellergeschoss; Kellerfenster erscheinen dort, wo es über dem Boden mindestens
# FACADE_BASEMENT_MIN_EXPOSED_M sichtbar ist, mit Unterkante FACADE_BASEMENT_SILL_M über dem Boden.
FACADE_BASEMENT_MIN_REMAINDER_M = 0.6
FACADE_BASEMENT_MIN_EXPOSED_M = 1.0
FACADE_BASEMENT_SILL_M = 0.3
FACADE_STOREY_ROUNDING = 0.2  # Anteil einer Geschosshöhe, um den das unterste Geschoss zu kurz sein darf
FACADE_DOOR_MAX_HEIGHT_ABOVE_BASE_M = 0.5  # Türen nur, wo der Geschossboden höchstens so hoch über dem Boden liegt
# Kirchtürme: keine Fenster, dafür eine Turmuhr an der Frontseite. Eine Kirche ist ein OSM-Polygon (building=church/
# cathedral/chapel, amenity=place_of_worship), das mindestens CHURCH_OVERLAP_MIN eines LOD2-Gebäudes überdeckt. In den
# LOD2-Daten sind Schiff und Turm EIN Gebäude; Turmwände sind die Wände, deren Oberkante mindestens
# CHURCH_TOWER_HEIGHT_FRACTION des Wegs vom Median der Wandoberkanten zur höchsten Wand erreicht (und die höchste
# Wand liegt mindestens CHURCH_TOWER_MIN_RISE_M über dem Median), dazu Wände in OSM-Glockenturm-Polygonen.
CHURCH_OVERLAP_MIN = 0.5
CHURCH_TOWER_HEIGHT_FRACTION = 0.6
CHURCH_TOWER_MIN_RISE_M = 4.0
CHURCH_CLOCK_MIN_WALL_M = 3.2  # schmalere Turmwände bekommen keine Uhr
CHURCH_CLOCK_BELOW_TOP_M = 3.5  # Uhrmitte höchstens so weit unter der Wandoberkante (sonst tiefer, wo die Wand breit genug ist)
CHURCH_CLOCK_MIN_HEIGHT_M = 6.0  # Uhrmitte mindestens so hoch über dem Wandfuß
# Grüner Kanal der Normalmaps: True = grün zeigt nach oben (OpenGL). Gemessen an t_roof_slates_rounded_nm.normal.dds
# (Korrelation von G mit dem vertikalen AO-Gradienten +0,61, von R mit dem horizontalen -0,80): BeamNG nutzt hier
# Grün-nach-oben.
TEXTURE_NORMAL_GREEN_UP = True

# Pixel-Kantenlänge des EINEN zusammengesetzten Luftbilds für die gesamte
# Fläche (io/aerial.py::process_aerial_images() - seit 2026-09-18 kein
# Foto-Material mehr pro 500m-Kachel, siehe dortigen Docstring). MUSS mit dem
# baseTexSize der TerrainMaterialTextureSet (terrain_workflow.py)
# übereinstimmen, sonst packt BeamNG eine falsch dimensionierte Textur in den
# Terrain-Material-Atlas (siehe Terrain-Material-Crash-Fix vom 2026-09-17).
#
# 8192 statt der BeamNG-"typisch"-Obergrenze 4096, weil unsere offizielle
# Doku-Recherche (2026-09-18, https://documentation.beamng.com/modding/
# levels/level_formats/terrain/) ausdrücklich höhere Werte erlaubt, "wenn die
# Basis eine einzigartige Gesamt-Terrain-Karte ist" - genau unser Fall (EIN
# Luftbild für die ganze Fläche statt vieler Kacheln). 8192 statt 16384
# (Stand 2026-09-18) auf Nutzerwunsch wegen Dateigröße (jede Basis-Textur
# muss laut BeamNG genau die baseTexSize haben; die Landnutzungs-Schichten
# nutzen dasselbe Luftbild als Basis, siehe build_terrain_material_entries()).
# 2048m-Kachel bei 8192px ≈ 0.25m/Pixel, nah an der nativen DOP20-Auflösung
# (0.2m/Pixel).
TERRAIN_BASE_TEX_PIXEL_SIZE = 8192


# Vier-Bilder-Modus: die Flaeche wird in PHOTO_TILE_SIZE_M-Kacheln aufgeteilt, jede bekommt ihr EIGENES Foto
# (TERRAIN_BASE_TEX_PIXEL_SIZE px für 2 km = 0,244 m/px statt EINES Gesamtfotos mit 0,5 m/px bei 4x4 km). Kosten:
# die Landnutzungs-Schichten und die GroundCover-Typen werden je Kachel geführt (siehe terrain/photo_tiles.py).
# False = ein Gesamtfoto (bei einer Kachel ohnehin immer ein Foto).
AERIAL_PHOTO_PER_TILE = True

# Kachelgroesse (Meter) des Vier-Bilder-Modus - unabhaengig von der Groesse/Anzahl der Rohdaten-Kacheln (die je nach
# Quelle z.B. 1 km statt 2 km gross sein koennen, siehe terrain/photo_tiles.py::build_processing_tile_grid()).
# Default 2000.0 entspricht dem bisherigen impliziten Verhalten bei LGL Baden-Wuerttemberg (ein Foto je 2x2-km-ZIP).
PHOTO_TILE_SIZE_M = 2000.0

# BigMap-Vorschaubild (info.json-Feld "minimap") aus den bereits gebauten Luftbild-PNGs
# (io/aerial.py::build_minimap_image()) - ohne dieses Feld bleibt BigMap nutzbar, zeigt aber nur einen
# leeren Hintergrund statt des Terrains.
MINIMAP_ENABLED = True
MINIMAP_PIXEL_SIZE = 2048  # Kantenlaenge der Minimap in Pixeln, unabhaengig von der tatsaechlichen Terrain-Groesse

# === VERZEICHNISSE ===
CACHE_DIR = Path("cache")  # Verzeichnis fuer Cache-Dateien
HEIGHT_DATA_DIR = Path("data/height")  # Verzeichnis mit Hoehendaten (DGM1)
AERIAL_DATA_DIR = Path("data/satellite")  # Verzeichnis mit Luftbildern (DOP20)
LOD2_DATA_DIR = Path("data/buildings")  # Verzeichnis mit 3D-Gebäudemodellen (LoD2/CityGML)
# 30m Hoehendaten fuer den Horizont: IMMER vollautomatisch von Copernicus geladen (dgm30_fetch.py),
# nie manuell abgelegt - gehoert deshalb unter cache/ (jederzeit sicher loeschbar/neu ladbar),
# nicht nach data/ (das bleibt fuer selbst mitgebrachte Rohdaten wie Hoehe/Luftbild/Gebaeude).
DGM30_CACHE_DIR = CACHE_DIR / "dgm30"

# === AUTOMATISCHER DOWNLOAD: HORIZONT-QUELLDATEN (DGM30 + SENTINEL-2) ===

DGM30_AUTO_DOWNLOAD = True
DGM30_S3_BUCKET = "copernicus-dem-30m"          # verifiziert im README (funktionierender Link)
DGM30_S3_REGION = "eu-central-1"
DGM30_FETCH_MAX_RETRIES = 3
DGM30_FETCH_TIMEOUT_S = 60
DGM30_NOT_FOUND_CACHE_TTL_DAYS = 30

EOX_AUTO_DOWNLOAD = True
EOX_WMS_URL = "https://tiles.maps.eox.at/wms"   # verifiziert per GetCapabilities am 2026-09-22
EOX_WMS_LAYER = "s2cloudless-2025_3857"         # verifiziert: Layer-Liste enthaelt s2cloudless-<jahr>_3857,
                                                 # aktuell bis 2025; bei kuenftiger Umsetzung ggf. neuestes
                                                 # verfuegbares Jahr aus GetCapabilities uebernehmen
EOX_WMS_VERSION = "1.1.1"                       # verifiziert: Server unterstuetzt nur 1.1.1, NICHT 1.3.0
                                                 # (Achsreihenfolge in EPSG:3857 bei 1.1.1 vs. 1.3.0 identisch,
                                                 # daher unkritisch fuer die BBOX-Berechnung)
EOX_WMS_FORMAT = "image/jpeg"                   # verifiziert verfuegbar
EOX_MAX_REQUEST_PX = 2048                       # GetCapabilities nennt kein MaxWidth/MaxHeight - konservativ
                                                 # belassen, bei 400/429 im echten Lauf senken
EOX_TARGET_RESOLUTION_M = 10.0
EOX_MOSAIC_MAX_PX = 12000
EOX_FETCH_MARGIN_FACTOR = 1.02
EOX_FETCH_MAX_RETRIES = 3
EOX_FETCH_TIMEOUT_S = 60
EOX_USER_AGENT = "World-to-BeamNG/1.0 (privates OSM-zu-BeamNG-Konvertierungstool)"
EOX_KEEP_RAW_MOSAIC = True
EOX_MOSAIC_CACHE_DIR = CACHE_DIR / "horizon_source"    # Rohmosaik (vor Zuschnitt), pro Gebiet+Layer gehasht
EOX_TEXTURE_CACHE_DIR = CACHE_DIR / "horizon_texture"  # fertige, zugeschnittene Textur, pro Gebiet+Groesse+
                                                        # Resampling+Layer gehasht - NICHT data/DOP300/, das
                                                        # bleibt der manuelle Override-Slot (siehe
                                                        # sentinel2_fetch.ensure_horizon_texture())
EOX_ATTRIBUTION_NOTICE = (
    "Horizont-Hintergrundbild: EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH "
    "(Contains modified Copernicus Sentinel data 2025). Lizenz: CC BY-NC-SA 4.0 "
    "(nicht-kommerzielle Nutzung, Attribution + ShareAlike Pflicht) - siehe "
    "https://cloudless.eox.at/documentation/license. Kommerzielle Nutzung erfordert eine separate "
    "EOX Commercial Attribution-RestrictedUse Lizenz."
)


# === OVERPASS API ENDPOINTS ===
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# Mehrere Overpass-Server lehnen Anfragen mit generischem "python-requests/x.x"
# User-Agent ab (406 Not Acceptable) bzw. drosseln sie eher (429 Too Many
# Requests). Community-Empfehlung aller drei Server oben: Projekt-Name + eine
# Kontaktmöglichkeit angeben. Bei Bedarf hier durch eigene Kontaktdaten
# (E-Mail/Projekt-URL) ergänzen - wird als Header an die Overpass-Server gesendet.
OVERPASS_USER_AGENT = "World-to-BeamNG/1.0 (privates OSM-zu-BeamNG-Konvertierungstool)"
