"""
Zentrale Konfiguration für World-to-BeamNG.
"""

# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0
# Minimale Böschungsbreite (Meter) unabhängig von Höhenunterschieden
MIN_SLOPE_WIDTH = 2
# Loch-Check schaltbar: False = kein Check/Export, True = Check + immer Export
HOLE_CHECK_ENABLED = True
# Exportpfad für offene Kanten als OBJ (mit MTL), wenn HOLE_CHECK_ENABLED=True
BOUNDARY_EDGES_EXPORT = "boundary_edges.obj"
SLOPE_ANGLE = 45.0  # Neigungswinkel der Böschung in Grad (45° = 1:1 Steigung)
# Vorab-Reduktion über gröberes Grid (Strategie 2). Für feineres Terrain z.B. 1.0 setzen.
GRID_SPACING = (
    2.0  # Abstand zwischen Grid-Punkten in Metern (1.0 = sehr fein, 10.0 = grob)
)
TERRAIN_REDUCTION = 0  # Decimation bleibt aus; steuern wir über GRID_SPACING
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


# === VERZEICHNISSE ===
CACHE_DIR = "cache"  # Verzeichnis für Cache-Dateien
HEIGHT_DATA_DIR = "height-data"  # Verzeichnis mit Höhendaten

# === MULTIPROCESSING ===
# WARNUNG: Unter Windows kann Multiprocessing hängen bleiben!
# Bei Problemen: False setzen
USE_MULTIPROCESSING = True  # False = Single-Thread (langsamer, aber stabil)
NUM_WORKERS = 4  # None = Automatisch (alle CPU-Kerne), oder Anzahl (z.B. 4)
# Höhenabfrage: "kdtree" (schnell, NN) oder "interpolator" (NearestNDInterpolator)
HEIGHT_LOOKUP_MODE = "kdtree"
# Maximale Straßen pro Batch im Multiprocessing
MAX_ROADS_PER_BATCH = 500

# === GLOBALE ZUSTANDSVARIABLEN (werden in main() initialisiert) ===
BBOX = None
LOCAL_OFFSET = None  # Globaler Offset für lokale Koordinaten
GRID_BOUNDS_UTM = None  # Grid Bounds in UTM

# === OVERPASS API ENDPOINTS ===
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
