"""
Zentrale Konfiguration fuer World-to-BeamNG.
"""

from .osm.osm_mapper import OSMMapper

# OSM Mapper Singleton (lädt data/osm_to_beamng.json)
OSM_MAPPER = OSMMapper(config_path="data/osm_to_beamng.json")

SPAWN_POINT = (47.842840, 7.684767)  # Standard-Spawn-Punkt (kann pro Level überschrieben werden)

# BEAMNG Zielordner (Anpassbar)
BEAMNG_DIR = "C:\\Users\\johan\\AppData\\Local\\BeamNG\\BeamNG.drive\\current\\levels\\world_to_beamng"
BEAMNG_DIR_SHAPES = BEAMNG_DIR + "\\art\\shapes"
BEAMNG_DIR_TEXTURES = BEAMNG_DIR_SHAPES + "\\textures"
BEAMNG_DIR_BUILDINGS = BEAMNG_DIR_SHAPES + "\\buildings"

RELATIVE_DIR = "levels/world_to_beamng/"
RELATIVE_DIR_SHAPES = RELATIVE_DIR + "art/shapes/"  # Mit levels/world_to_beamng/ Prefix für BeamNG!
RELATIVE_DIR_TEXTURES = RELATIVE_DIR_SHAPES + "textures/"
RELATIVE_DIR_BUILDINGS = RELATIVE_DIR_SHAPES + "buildings/"

# === BEAMNG LEVEL-STRUKTUR ===
ITEMS_JSON = "main\\items.level.json"  # Enthält nur MissionGroup (BeamNG lädt dann automatisch main/MissionGroup/items.level.json)
MATERIALS_JSON = "main\\materials.json"  # Enthält Material-Definitionen

# === MATERIAL-EINSTELLUNGEN ===
# Materialien verwenden IMMER Texturen (keine Farb-Fallbacks)

# === OpenTopography API ===
OPENTOPOGRAPHY_API_KEY = "9805a06e82a636afd885c07a2f2e1838"  # Registrierung: https://opentopography.org/
OPENTOPOGRAPHY_ENABLED = False  # Automatischer Download von DGM30 aktivieren

# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0
# Winkel-Schwelle für dynamischen Junction-Buffer (Grad). Unterhalb dieses Winkels wird ein winkelabhängiger Buffer aktiviert.
# Buffer = half_width / sin(angle/2) - half_width (asymmetrisch pro Straße)
JUNCTION_STOP_ANGLE_THRESHOLD = 90.0
# Buffer-Abstand beim Stoppen vor Junctions (Meter)
JUNCTION_STOP_BUFFER = 5.0
# Böschungs-Generierung (vorübergehend deaktiviert bis Remeshing stabil)
GENERATE_SLOPES = False
# Minimale Boeschungsbreite (Meter) unabhängig von Hoehenunterschieden
MIN_SLOPE_WIDTH = 2
# Loch-Check schaltbar: False = kein Check/Export, True = Check + immer Export
HOLE_CHECK_ENABLED = False
SLOPE_ANGLE = 45.0  # Neigungswinkel der Boeschung in Grad (45° = 1:1 Steigung)
# Vorab-Reduktion ueber groeberes Grid (Strategie 2). Fuer feineres Terrain z.B. 1.0 setzen.
GRID_SPACING = 2.0  # Abstand zwischen Grid-Punkten in Metern (1.0 = sehr fein, 10.0 = grob)
TERRAIN_REDUCTION = 0  # Decimation bleibt aus; steuern wir ueber GRID_SPACING
LEVEL_NAME = "world_to_beamng"  # Name des BeamNG Levels (muss mit BEAMNG_DIR übereinstimmen)

# DEBUG / EXPORTS
DEBUG_EXPORTS = True  # Debug-Dumps (Netz, Grid) nur bei Bedarf aktivieren
DEBUG_VERBOSE = False  # Zusätzliche Konsolen-Logs
SKIP_PHASES_2_TO_4_IF_MAIN_ITEMS_EXISTS = True  # Wenn main.items.json existiert, Phase 2-4 überspringen

# === STRASSENGLÄTTUNG / OPTIONEN ===
ENABLE_ROAD_SMOOTHING = True  # False = Spline-Glättung komplett aus
ROAD_SMOOTH_ANGLE_THRESHOLD = 10.0  # Winkel in Grad - ab diesem Wert werden Kurven unterteilt
SAMPLE_SPACING_FACTOR = 0.5  # Faktor für Segment-Spacing: road_width * SAMPLE_SPACING_FACTOR
# (Alte feste Werte für Referenz: bei 7m Straßen war 2.5m → jetzt dynamisch via Faktor 0.5)
ROAD_SMOOTH_TENSION = 0.05  # Spline-Glättungsfaktor (0.0 = eng an Originalpunkten, 1.0 = sehr glatt)
ROAD_SMOOTH_MAX_DIR_CHANGE_DEG = 0.0  # Optional: maximale Richtungsänderung pro Segment (Grad); 0 = aus

# === CLIPPING ===
ENABLE_ROAD_CLIPPING = True  # True = Clip + Segment-Unterteilung am Grid-Rand, False = Skip (Testbetrieb)
ROAD_CLIP_MARGIN = -20.0  # Clipping-Abstand vom Grid-Rand in Metern (Faces < 3m vom Rand werden entfernt)
CLIP_ROAD_FACES_AT_BOUNDS = True  # True = Entferne Straßen-Dreiecke, die komplett außerhalb der Grid-Bounds liegen

# === TILE-EXPORT (DAE) ===
TILE_SIZE = 500  # Größe pro DAE-Tile in Metern
MATERIAL_TYPES = ["terrain", "road"]  # Verfügbare Materialien (später erweiterbar)


# === VERZEICHNISSE ===
CACHE_DIR = "cache"  # Verzeichnis fuer Cache-Dateien
HEIGHT_DATA_DIR = "data/DGM1"  # Verzeichnis mit Hoehendaten
LOD2_DATA_DIR = "data/LOD2"  # Verzeichnis mit 3D-Gebäudemodellen (CityGML)
DGM30_DATA_DIR = "data/DGM30"  # Verzeichnis mit 30m Höhendaten für Horizont
DOP300_DATA_DIR = "data/DOP300"  # Verzeichnis mit Sentinel-2 RGB Bildern

# === GEBÄUDE (LoD2) ===
LOD2_ENABLED = False  # LoD2-Gebäude verarbeiten
LOD2_SNAP_TO_TERRAIN = True  # Gebäude auf Terrain ausrichten
LOD2_FOUNDATION_EXTRUDE = 0.5  # Meter: Wände nach unten verlängern für Fundament

# === PHASE 5: HORIZONT-LAYER ===
PHASE5_ENABLED = False  # Horizont-Layer aktivieren (erfordert DGM30 + DOP300 Daten)
HORIZON_BBOX_BUFFER = 50000  # Buffer um Kerngebiet in Metern (50km)
HORIZON_GRID_SPACING = 1000  # Horizont-Grid Auflösung in Metern (1km)

# === MULTIPROCESSING ===
# WARNUNG: Unter Windows kann Multiprocessing hängen bleiben!
# Bei Problemen: False setzen
USE_MULTIPROCESSING = True  # False = Single-Thread (langsamer, aber stabil)
NUM_WORKERS = 4  # None = Automatisch (alle CPU-Kerne), oder Anzahl (z.B. 4)
# Hoehenabfrage: "kdtree" (schnell, NN) oder "interpolator" (NearestNDInterpolator)
HEIGHT_LOOKUP_MODE = "kdtree"
# Maximale Strassen pro Batch im Multiprocessing
MAX_ROADS_PER_BATCH = 500

# === PHASE-SKIPPING ===
SKIP_PHASES_2_TO_4_IF_ITEMS_EXISTS = True  # Wenn items.json existiert, Phase 2-4 überspringen

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
