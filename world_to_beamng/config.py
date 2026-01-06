"""
Zentrale Konfiguration fuer World-to-BeamNG.
"""

from .osm.osm_mapper import OSMMapper

# OSM Mapper Singleton (lädt data/osm_to_beamng.json)
OSM_MAPPER = OSMMapper(config_path="data/osm_to_beamng.json")

# BEAMNG Zielordner (Anpassbar)
BEAMNG_DIR = "C:\\Users\\johan\\AppData\\Local\\BeamNG.drive\\0.36\\levels\\World_to_BeamNG"
BEAMNG_DIR_SHAPES = BEAMNG_DIR + "\\art\\shapes"
BEAMNG_DIR_TEXTURES = BEAMNG_DIR_SHAPES + "\\textures"
BEAMNG_DIR_BUILDINGS = BEAMNG_DIR_SHAPES + "\\buildings"

# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0
# Winkel-Schwelle für dynamischen Junction-Buffer (Grad). Unterhalb dieses Winkels wird ein winkelabhängiger Buffer aktiviert.
# Buffer = half_width / sin(angle/2) - half_width (asymmetrisch pro Straße)
JUNCTION_STOP_ANGLE_THRESHOLD = 90.0
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
LEVEL_NAME = "World_to_BeamNG"  # Name des BeamNG Levels (muss mit BEAMNG_DIR übereinstimmen)

# DEBUG / EXPORTS
DEBUG_EXPORTS = True  # Debug-Dumps (Netz, Grid) nur bei Bedarf aktivieren
DEBUG_VERBOSE = False  # Zusätzliche Konsolen-Logs

# === STRASSENGLÄTTUNG / OPTIONEN ===
ENABLE_ROAD_SMOOTHING = False  # False = Spline-Glättung komplett aus
ROAD_SMOOTH_ANGLE_THRESHOLD = 10.0  # Winkel in Grad - ab diesem Wert werden Kurven unterteilt
SAMPLE_SPACING_FACTOR = 0.5  # Faktor für Segment-Spacing: road_width * SAMPLE_SPACING_FACTOR
# (Alte feste Werte für Referenz: bei 7m Straßen war 2.5m → jetzt dynamisch via Faktor 0.5)
ROAD_SMOOTH_TENSION = 0.05  # Spline-Glättungsfaktor (0.0 = eng an Originalpunkten, 1.0 = sehr glatt)

# === CLIPPING ===
ROAD_CLIP_MARGIN = 10.0  # Clipping-Abstand vom Grid-Rand in Metern (Faces < 3m vom Rand werden entfernt)

# === TILE-EXPORT (DAE) ===
TILE_SIZE = 500  # Größe pro DAE-Tile in Metern
MATERIAL_TYPES = ["terrain", "road"]  # Verfügbare Materialien (später erweiterbar)


# === VERZEICHNISSE ===
CACHE_DIR = "cache"  # Verzeichnis fuer Cache-Dateien
HEIGHT_DATA_DIR = "data/DGM1"  # Verzeichnis mit Hoehendaten
LOD2_DATA_DIR = "data/LOD2"  # Verzeichnis mit 3D-Gebäudemodellen (CityGML)

# === GEBÄUDE (LoD2) ===
LOD2_ENABLED = True  # LoD2-Gebäude verarbeiten
LOD2_WALL_COLOR = (1.0, 1.0, 1.0)  # Weiß (RGB 0-1)
LOD2_ROOF_COLOR = (0.8, 0.3, 0.2)  # Ziegelrot
LOD2_SNAP_TO_TERRAIN = True  # Gebäude auf Terrain ausrichten
LOD2_FOUNDATION_EXTRUDE = 0.5  # Meter: Wände nach unten verlängern für Fundament

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
BBOX = None
LOCAL_OFFSET = None  # Globaler Offset fuer lokale Koordinaten
GRID_BOUNDS_LOCAL = None  # Grid Bounds in lokalen Koordinaten
HEIGHT_HASH = None  # Height-Daten Hash (wird in main() gesetzt)

# === OVERPASS API ENDPOINTS ===
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
