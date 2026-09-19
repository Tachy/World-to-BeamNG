"""
Zentrale Konfiguration fuer World-to-BeamNG.
"""

import logging
from pathlib import Path, PurePosixPath

from .osm.osm_mapper import OSMMapper
from .logging_config import LoggerConfig

LEVEL_NAME = "world_to_beamng"  # Name des BeamNG Levels (muss mit BEAMNG_DIR übereinstimmen)

# OSM Mapper Singleton (lädt data/osm_to_beamng.json)
OSM_MAPPER = OSMMapper(config_path=Path("data/osm_to_beamng.json"))

SPAWN_POINT = (47.842840, 7.684767)  # Standard-Spawn-Punkt (kann pro Level überschrieben werden)

# BEAMNG Zielordner (Anpassbar)
BEAMNG_DIR = Path("C:/Users/johan/AppData/Local/BeamNG/BeamNG.drive/current/levels/world_to_beamng")
BEAMNG_DIR_SHAPES = BEAMNG_DIR / "art" / "shapes"
BEAMNG_DIR_TEXTURES = BEAMNG_DIR_SHAPES / "textures"
BEAMNG_DIR_BUILDINGS = BEAMNG_DIR_SHAPES / "buildings"

# In-Game relative paths (MUST use forward slashes for BeamNG)
RELATIVE_DIR = PurePosixPath("levels") / LEVEL_NAME
RELATIVE_DIR_SHAPES = RELATIVE_DIR / "art" / "shapes"
RELATIVE_DIR_TEXTURES = RELATIVE_DIR_SHAPES / "textures"
RELATIVE_DIR_BUILDINGS = RELATIVE_DIR_SHAPES / "buildings"


# === BEAMNG LEVEL-STRUKTUR ===
ITEMS_JSON = Path("main") / "MissionGroup" / "items.level.json"  # Items im MissionGroup-Verzeichnis
MATERIALS_JSON = Path("main") / "materials.json"  # Enthält Material-Definitionen

# Ablaufsteuerung
LOD2_ENABLED = True  # LoD2-Gebäude verarbeiten
PHASE5_ENABLED = True  # Horizont-Layer aktivieren (erfordert DGM30 + DOP300 Daten)
HORIZON_BOUNDARY_STITCHING = False  # Stitching zwischen Terrain und Horizon aktivieren
FORESTS_ENABLED = True  # Wald-Export global aktivieren/deaktivieren


# === MATERIAL-EINSTELLUNGEN ===
# Materialien verwenden IMMER Texturen (keine Farb-Fallbacks)

# === OpenTopography API für Horizont ===
OPENTOPOGRAPHY_API_KEY = "9805a06e82a636afd885c07a2f2e1838"  # Registrierung: https://opentopography.org/
OPENTOPOGRAPHY_ENABLED = False  # Automatischer Download von DGM30 aktivieren
HORIZON_GRID_SPACING = 200  # Horizont-Grid Auflösung in Metern (200m)
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
WATER_POND_LIFT = 0.15  # Teichspiegel über dem unteren Viertel des Geländes im Polygon in m
WATER_POND_DEPTH = 3.0  # Tiefe der WaterBlocks in m
WATER_POND_CELL = 6.0  # maximale Kantenlänge der Kacheln, mit denen Teiche gefüllt werden, in m
WATER_POND_CUBEMAP = "DefaultSkyCubemap"  # Engine-eigene Cubemap (die der Vorlage ist level-spezifisch und fehlte)

# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0

# === FOREST GENERATION PARAMETERS ===
FOREST_ROAD_MARGIN = 5.0  # Puffer um Straßen zur Baum-Filterung (in Metern, links & rechts)
FOREST_BUILDING_MARGIN = 2.5  # Puffer um Gebäude: dort stehen keine Bäume/Büsche (in Metern)
FOREST_ROAD_SURFACE_MARGIN = 4.0  # Abstand zur Kante der eingebetteten Fahrbahn (Kronen großer Gruppen-Bäume ragen mehrere Meter aus)
FOREST_ROW_SURFACE_MARGIN = 1.0  # dasselbe für Baumreihen: Stämme nicht auf der Fahrbahn
FOREST_ROW_ROAD_MARGIN = 3.0  # Baumreihen (Alleen) stehen näher an Straßen als Wald: kleinerer Puffer (in Metern)
# Gruppen-Assets haben mehrere Stämme bis ~9 m neben dem Ursprung. Die Abstände oben gelten auch für jeden Stamm
# (forest/tree_footprints.py), nicht nur für den Ursprung.
FOREST_TRUNK_MAX_FLOAT = 0.5  # so weit darf ein Stammfuß nach dem Absenken über dem Boden stehen (in Metern)
FOREST_TRUNK_MAX_SINK = 1.0  # so weit darf ein Baum höchstens abgesenkt werden, sonst Typwechsel (in Metern)

# Böschungs-Geometrie entsteht NICHT im Mesh - Straßen selbst werden seit der
# DecalRoad-Umstellung überhaupt nicht mehr als Mesh exportiert (siehe
# workflow/terrain_workflow.py::export_decal_roads()). Der Übergang zur
# Umgebung entsteht direkt im Terrain-Heightmap, siehe
# terrain/road_embedding.py:apply_embankment_blend(). Dieser Flag bleibt
# dauerhaft False.
GENERATE_SLOPES = False
# Minimale Boeschungsbreite (Meter) unabhängig von Hoehenunterschieden
MIN_SLOPE_WIDTH = 2
# Obergrenze der Böschungsbreite (Meter), unabhängig davon, wie groß der
# Höhenunterschied zwischen Straßenkante und natürlichem Terrain ist
# (verhindert unrealistisch breite Böschungskorridore bei Extremfällen).
MAX_SLOPE_WIDTH = 30.0
SLOPE_ANGLE = 45.0  # Neigungswinkel der Boeschung in Grad (45° = 1:1 Steigung)
# Vorab-Reduktion ueber groeberes Grid (Strategie 2). Fuer feineres Terrain z.B. 1.0 setzen.
GRID_SPACING = 1.0  # Abstand zwischen Grid-Punkten in Metern (native DGM1-Auflösung; 10.0 = grob)
TERRAIN_REDUCTION = 0  # Decimation bleibt aus; steuern wir ueber GRID_SPACING

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
LOGGING_ENABLED = True
LOGGING_FILE = None  # Path("logs/world_to_beamng.log")  # Optional; None = nur stdout
LOGGING_LEVEL = logging.DEBUG if DEBUG_VERBOSE else logging.INFO

# Initialisiere zentrale Logger-Instanz
LoggerConfig.get_instance(log_file=LOGGING_FILE, level=LOGGING_LEVEL, verbose=DEBUG_VERBOSE)

# === STRASSENGLÄTTUNG / OPTIONEN ===
ENABLE_ROAD_SMOOTHING = True  # False = Spline-Glättung komplett aus
ROAD_SMOOTH_ANGLE_THRESHOLD = 10.0  # Winkel in Grad - ab diesem Wert werden Kurven unterteilt
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
CLIP_ROAD_FACES_AT_BOUNDS = True  # True = Entferne Straßen-Dreiecke, die komplett außerhalb der Grid-Bounds liegen

# === TILE-EXPORT (DAE) ===
BUILDINGS_AS_ONE_OBJECT = True  # True: ALLE Gebäude in EINER DAE/EINEM Objekt auf der Gesamtfläche (wie die Straßen)
MAX_BUILDINGS_PER_SHAPE = 1500  # BeamNG verwirft ab 2048 Nodes je Shape alles Weitere (1 Node je Gebäude) -> aufteilen
TILE_SIZE = 500  # Größe pro DAE-Tile in Metern (nur bei BUILDINGS_AS_ONE_OBJECT = False)

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


# Vier-Bilder-Modus: bei mehreren DGM1-Kacheln bekommt jede Kachel ihr EIGENES Foto (TERRAIN_BASE_TEX_PIXEL_SIZE px
# für 2 km = 0,244 m/px statt EINES Gesamtfotos mit 0,5 m/px bei 4x4 km). Kosten: die Landnutzungs-Schichten und die
# GroundCover-Typen werden je Kachel geführt (siehe terrain/photo_tiles.py). False = ein Gesamtfoto (bei einer
# Kachel ohnehin immer ein Foto).
AERIAL_PHOTO_PER_TILE = True

# === VERZEICHNISSE ===
CACHE_DIR = Path("cache")  # Verzeichnis fuer Cache-Dateien
HEIGHT_DATA_DIR = Path("data/DGM1")  # Verzeichnis mit Hoehendaten
LOD2_DATA_DIR = Path("data/LOD2")  # Verzeichnis mit 3D-Gebäudemodellen (CityGML)
DGM30_DATA_DIR = Path("data/DGM30")  # Verzeichnis mit 30m Höhendaten für Horizont
DOP300_DATA_DIR = Path("data/DOP300")  # Verzeichnis mit Sentinel-2 RGB Bildern


# === MULTIPROCESSING ===
# WARNUNG: Unter Windows kann Multiprocessing hängen bleiben!
# Bei Problemen: False setzen
USE_MULTIPROCESSING = True  # False = Single-Thread (langsamer, aber stabil)
NUM_WORKERS = 4  # None = Automatisch (alle CPU-Kerne), oder Anzahl (z.B. 4)
# Hoehenabfrage: "kdtree" (schnell, NN) oder "interpolator" (NearestNDInterpolator)
HEIGHT_LOOKUP_MODE = "kdtree"
# Maximale Strassen pro Batch im Multiprocessing
MAX_ROADS_PER_BATCH = 500

# === GLOBALE ZUSTANDSVARIABLEN (werden in main() initialisiert) ===
# WICHTIG: Nur echte GLOBALE Parameter hier! Keine Tile-spezifischen Werte!
LOCAL_OFFSET = None  # Globaler Offset fuer lokale Koordinaten (zentral für alle Tiles)
GRID_BOUNDS_LOCAL = None  # Grid Bounds in lokalen Koordinaten (wird pro Tile überschrieben)

# === OVERPASS API ENDPOINTS ===
OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# Mehrere Overpass-Server lehnen Anfragen mit generischem "python-requests/x.x"
# User-Agent ab (406 Not Acceptable) bzw. drosseln sie eher (429 Too Many
# Requests). Community-Empfehlung aller drei Server oben: Projekt-Name + eine
# Kontaktmöglichkeit angeben. Bei Bedarf hier durch eigene Kontaktdaten
# (E-Mail/Projekt-URL) ergänzen - wird als Header an die Overpass-Server gesendet.
OVERPASS_USER_AGENT = "World-to-BeamNG/1.0 (privates OSM-zu-BeamNG-Konvertierungstool)"
