"""
Zentrale Konfiguration für World-to-BeamNG.
"""

# === MESH-PARAMETER ===
ROAD_WIDTH = 7.0
SLOPE_ANGLE = 45.0  # Neigungswinkel der Böschung in Grad (45° = 1:1 Steigung)
GRID_SPACING = 10.0  # Abstand zwischen Grid-Punkten in Metern (1.0 = hohe Auflösung, 10.0 = niedrige Auflösung)
TERRAIN_REDUCTION = 0.0  # PyVista Decimation als Dezimalwert (0.70 = 70% Reduktion)
LEVEL_NAME = "osm_generated_map"

# === VERZEICHNISSE ===
CACHE_DIR = "cache"  # Verzeichnis für Cache-Dateien
HEIGHT_DATA_DIR = "height-data"  # Verzeichnis mit Höhendaten

# === MULTIPROCESSING ===
# WARNUNG: Unter Windows kann Multiprocessing hängen bleiben!
# Bei Problemen: False setzen
USE_MULTIPROCESSING = False  # False = Single-Thread (langsamer, aber stabil)
NUM_WORKERS = 8  # None = Automatisch (alle CPU-Kerne), oder Anzahl (z.B. 4)

# === GLOBALE ZUSTANDSVARIABLEN (werden in main() initialisiert) ===
BBOX = None
LOCAL_OFFSET = None  # Globaler Offset für lokale Koordinaten
GRID_BOUNDS_UTM = None  # Grid Bounds in UTM

# === TERRAIN-FACE-LÖSCH-INFO ===
terrain_face_types_to_delete = None
terrain_vertices_original = None
terrain_faces_original = None

# === OVERPASS API ENDPOINTS ===
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
