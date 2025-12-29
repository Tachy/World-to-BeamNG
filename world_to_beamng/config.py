"""
Zentrale Konfiguration für World-to-BeamNG.
"""

# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0
SLOPE_ANGLE = 45.0  # Neigungswinkel der Böschung in Grad (45° = 1:1 Steigung)
GRID_SPACING = 3.0  # Abstand zwischen Grid-Punkten in Metern (1.0 = hohe Auflösung, 10.0 = niedrige Auflösung)
TERRAIN_REDUCTION = 0  # PyVista Decimation als Dezimalwert (0.70 = 70% Reduktion)
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
USE_MULTIPROCESSING = False  # False = Single-Thread (langsamer, aber stabil)
NUM_WORKERS = 2  # None = Automatisch (alle CPU-Kerne), oder Anzahl (z.B. 4)

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
