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


# === MATERIAL-EINSTELLUNGEN ===
# Materialien verwenden IMMER Texturen (keine Farb-Fallbacks)

# === MESH-HOLE-FILLING ===
FILL_ALL_MESH_HOLES = False  # Schließe ALLE Boundary-Holes (äußer + Inseln)
FILL_HOLES_MAX_EDGE_LENGTH = 100.0  # Warnung bei Edge-Länge > X Metern

# === OpenTopography API für Horizont ===
OPENTOPOGRAPHY_API_KEY = "9805a06e82a636afd885c07a2f2e1838"  # Registrierung: https://opentopography.org/
OPENTOPOGRAPHY_ENABLED = False  # Automatischer Download von DGM30 aktivieren
HORIZON_GRID_SPACING = 200  # Horizont-Grid Auflösung in Metern (200m)

# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0
# Winkel-Schwelle für dynamischen Junction-Buffer (Grad). Unterhalb dieses Winkels wird ein winkelabhängiger Buffer aktiviert.
# Buffer = half_width / sin(angle/2) - half_width (asymmetrisch pro Straße)
JUNCTION_STOP_ANGLE_THRESHOLD = 90.0
# Buffer-Abstand beim Stoppen vor Junctions (Meter)
JUNCTION_STOP_BUFFER = 5.0

# === FOREST GENERATION PARAMETERS ===
FOREST_ROAD_MARGIN = 5.0  # Puffer um Straßen zur Baum-Filterung (in Metern, links & rechts)

# Böschungs-Generierung (vorübergehend deaktiviert bis Remeshing stabil)
GENERATE_SLOPES = False
# Minimale Boeschungsbreite (Meter) unabhängig von Hoehenunterschieden
MIN_SLOPE_WIDTH = 2
SLOPE_ANGLE = 45.0  # Neigungswinkel der Boeschung in Grad (45° = 1:1 Steigung)
# Vorab-Reduktion ueber groeberes Grid (Strategie 2). Fuer feineres Terrain z.B. 1.0 setzen.
GRID_SPACING = 2.0  # Abstand zwischen Grid-Punkten in Metern (1.0 = sehr fein, 10.0 = grob)
TERRAIN_REDUCTION = 0  # Decimation bleibt aus; steuern wir ueber GRID_SPACING

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

# === CLIPPING ===
ENABLE_ROAD_CLIPPING = True  # True = Clip + Segment-Unterteilung am Grid-Rand, False = Skip (Testbetrieb)
ROAD_CLIP_MARGIN = -20.0  # Clipping-Abstand vom Grid-Rand in Metern (Faces < 3m vom Rand werden entfernt)
CLIP_ROAD_FACES_AT_BOUNDS = True  # True = Entferne Straßen-Dreiecke, die komplett außerhalb der Grid-Bounds liegen

# === TILE-EXPORT (DAE) ===
TILE_SIZE = 500  # Größe pro DAE-Tile in Metern


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
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
