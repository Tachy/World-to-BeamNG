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
LOD2_ENABLED = False  # LoD2-Gebäude verarbeiten
PHASE5_ENABLED = False  # Horizont-Layer aktivieren (erfordert DGM30 + DOP300 Daten)
HORIZON_BOUNDARY_STITCHING = False  # Stitching zwischen Terrain und Horizon aktivieren
FORESTS_ENABLED = True  # Wald-Export global aktivieren/deaktivieren


# === MATERIAL-EINSTELLUNGEN ===
# Materialien verwenden IMMER Texturen (keine Farb-Fallbacks)

# === OpenTopography API für Horizont ===
OPENTOPOGRAPHY_API_KEY = "9805a06e82a636afd885c07a2f2e1838"  # Registrierung: https://opentopography.org/
OPENTOPOGRAPHY_ENABLED = False  # Automatischer Download von DGM30 aktivieren
HORIZON_GRID_SPACING = 200  # Horizont-Grid Auflösung in Metern (200m)

# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0

# === FOREST GENERATION PARAMETERS ===
FOREST_ROAD_MARGIN = 5.0  # Puffer um Straßen zur Baum-Filterung (in Metern, links & rechts)

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
TILE_SIZE = 500  # Größe pro DAE-Tile in Metern

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
# (Stand 2026-09-18) auf Nutzerwunsch wegen Dateigröße (jede Landnutzungs-
# Textur muss laut BeamNG auf dieselbe baseTexSize hochskaliert werden, siehe
# ensure_landuse_base_textures_sized() - bei 16384 wurden das >1.6GB).
# 2048m-Kachel bei 8192px ≈ 0.25m/Pixel, nah an der nativen DOP20-Auflösung
# (0.2m/Pixel).
TERRAIN_BASE_TEX_PIXEL_SIZE = 8192


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
