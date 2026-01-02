"""
Zentrale Konfiguration fuer World-to-BeamNG.
"""

# BEAMNG Zielordner (Anpassbar)
BEAMNG_DIR = (
    "C:\\Users\\johan\\AppData\\Local\\BeamNG.drive\\0.36\\levels\\World_to_BeamNG"
)
BEAMNG_DIR_SHAPES = BEAMNG_DIR + "\\art\\shapes"
BEAMNG_DIR_TEXTURES = BEAMNG_DIR_SHAPES + "\\textures"
# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0
# Böschungs-Generierung (vorübergehend deaktiviert bis Remeshing stabil)
GENERATE_SLOPES = False
# Minimale Boeschungsbreite (Meter) unabhängig von Hoehenunterschieden
MIN_SLOPE_WIDTH = 2
# Loch-Check schaltbar: False = kein Check/Export, True = Check + immer Export
HOLE_CHECK_ENABLED = False
# Exportpfad fuer offene Kanten als OBJ (mit MTL), wenn HOLE_CHECK_ENABLED=True
BOUNDARY_EDGES_EXPORT = "boundary_edges.obj"
SLOPE_ANGLE = 45.0  # Neigungswinkel der Boeschung in Grad (45° = 1:1 Steigung)
# Vorab-Reduktion ueber groeberes Grid (Strategie 2). Fuer feineres Terrain z.B. 1.0 setzen.
GRID_SPACING = (
    2.0  # Abstand zwischen Grid-Punkten in Metern (1.0 = sehr fein, 10.0 = grob)
)
TERRAIN_REDUCTION = 0  # Decimation bleibt aus; steuern wir ueber GRID_SPACING
LEVEL_NAME = "osm_generated_map"

# === STRASSENGLÄTTUNG / OPTIONEN ===
ENABLE_ROAD_SMOOTHING = True  # False = Spline-Glättung komplett aus
ENABLE_ROAD_EDGE_SNAPPING = True  # False = Rand-Snap an Kreuzungen aus
ROAD_SMOOTH_ANGLE_THRESHOLD = (
    10.0  # Winkel in Grad - ab diesem Wert werden Kurven unterteilt
)
ROAD_SMOOTH_MAX_SEGMENT = 5.0  # Maximale Segmentlänge in Metern
ROAD_SMOOTH_MIN_SEGMENT = 1.0  # Minimale Segmentlänge in Metern
ROAD_SMOOTH_TENSION = (
    0.0  # Spline-Glättungsfaktor (0.0 = eng an Originalpunkten, 1.0 = sehr glatt)
)

# === CENTERLINE-SAMPLING / SUCHE ===
CENTERLINE_SAMPLE_SPACING = (
    10.0  # Abstand zwischen Sample-Punkten entlang der Centerline (m)
)
CENTERLINE_SEARCH_RADIUS = 10.0  # Suchradius um Centerline-Punkte (m)

# === JUNCTION REMESHING ===
JUNCTION_REMESH_RADIUS = 18.0  # Suchradius für Junction-Remesh in Metern

# === CLIPPING ===
ROAD_CLIP_MARGIN = 10.0  # Clipping-Abstand vom Grid-Rand in Metern (Faces < 3m vom Rand werden entfernt)

# === TILE-EXPORT (DAE) ===
TILE_SIZE = 500  # Größe pro DAE-Tile in Metern
MATERIAL_TYPES = ["terrain", "road"]  # Verfügbare Materialien (später erweiterbar)


# === VERZEICHNISSE ===
CACHE_DIR = "cache"  # Verzeichnis fuer Cache-Dateien
HEIGHT_DATA_DIR = "height-data"  # Verzeichnis mit Hoehendaten

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

# === OVERPASS API ENDPOINTS ===
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
