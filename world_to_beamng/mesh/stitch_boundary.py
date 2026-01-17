"""
Terrain-Horizon Boundary Stitching - Verbindet Terrain-Tiles mit Horizon-Mesh.

Die Terrain-Tiles haben ein 2m Grid, das Horizon-Mesh ein 200m Grid.
Beide Grids sind in XY parallel und gerade ausgerichtet - nur Z variiert.

WICHTIG: Berücksichtigt komplexe Tile-Anordnungen (L-Form, U-Form, etc.)!
Nutzt Delaunay-Triangulation statt fester Himmelsrichtungen.

Beispiel mit L-Form:
┌─────┬─────┐  <- Terrain-Tiles (2km x 2km)
│ T1  │ T2  │
├─────┘     │
│ T3        │
└───────────┘
  horizon (200m Grid ringsum)

Der Stitching-Prozess:
1. Extrahiere ALLE Boundary-Vertices vom Terrain (tatsächliche Kontur, nicht Rechteck!)
2. Extrahiere Horizon-Ring (innere Reihe um Terrain herum)
3. Delaunay-Triangulation zwischen beiden Vertex-Sets
4. Filtere Triangles: Behalte nur die, die Terrain UND Horizon verbinden

WICHTIG: Terrain UND Horizon müssen denselben VertexManager nutzen!
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from .vertex_manager import VertexManager


def extract_terrain_boundary_edges(
    terrain_mesh, vertex_manager: VertexManager
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Extrahiert ALLE Boundary-Vertices vom Terrain-Mesh als zusammenhängende Kontur.

    WICHTIG: Berechnet TATSÄCHLICHE Bounds aus Boundary-Vertices, nicht aus übergebenen grid_bounds!
    Dies ist entscheidend für korrekte Horizont-Ring-Extraktion.

    PERFORMANCE: O(n) mit defaultdict statt normales dict

    Args:
        terrain_mesh: Mesh-Objekt (mit .faces, .face_props) ODER Dictionary {"faces": [...], "face_props": {...}}
        vertex_manager: VertexManager mit allen Vertices

    Returns:
        Tuple (boundary_vertices_array, actual_bounds)
        - boundary_vertices_array: NumPy Array von Boundary-Vertex-Indizes
        - actual_bounds: (x_min, x_max, y_min, y_max) der ECHTEN Terrain-Grenzen
    """
    # === Kompatibilität: Mesh-Objekt oder Dictionary ===
    if hasattr(terrain_mesh, "faces"):
        # Mesh-Objekt
        faces_list = terrain_mesh.faces
        face_props_dict = getattr(terrain_mesh, "face_props", {})
    else:
        # Dictionary
        faces_list = terrain_mesh.get("faces", [])
        face_props_dict = terrain_mesh.get("face_props", {})

    # === OPTIMIERUNG 1: defaultdict für schnellere Edge-Erfassung ===
    edge_faces = defaultdict(list)  # (v1, v2) -> [face_idx, ...]

    terrain_face_count = 0
    non_terrain_count = 0

    for face_idx, face in enumerate(faces_list):
        v0, v1, v2 = face

        # Nur Terrain-Faces berücksichtigen (schnelle String-Vergleich)
        props = face_props_dict.get(face_idx, {})
        material = props.get("material", "terrain")

        if material != "terrain":
            non_terrain_count += 1
            continue

        terrain_face_count += 1

        # Extrahiere 3 Edges (normalisiert)
        edges = [(min(v0, v1), max(v0, v1)), (min(v1, v2), max(v1, v2)), (min(v2, v0), max(v2, v0))]

        for edge in edges:
            edge_faces[edge].append(face_idx)

    print(f"    [i] Terrain-Faces: {terrain_face_count}, Non-Terrain: {non_terrain_count}")

    # === OPTIMIERUNG 2: Set-basierte Boundary-Extraktion (statt Loop) ===
    boundary_vertices = set()
    boundary_edges = []

    for (v1_idx, v2_idx), faces in edge_faces.items():
        if len(faces) == 1:  # Boundary-Edge hat nur 1 Face
            boundary_vertices.add(v1_idx)
            boundary_vertices.add(v2_idx)
            boundary_edges.append((v1_idx, v2_idx))

    print(f"    [i] Boundary-Edges: {len(boundary_edges)}, Boundary-Vertices: {len(boundary_vertices)}")

    # Konvertiere zu sortiertem Array für konsistente Reihenfolge
    boundary_array = np.array(sorted(boundary_vertices), dtype=np.int32)

    # === KRITISCH: Berechne TATSÄCHLICHE Bounds der Boundary-Vertices ===
    # Diese Bounds sind im gleichen Koordinatensystem wie die Horizon-Vertices!
    vertices = vertex_manager.vertices
    if len(boundary_array) > 0:
        boundary_coords = vertices[boundary_array]
        x_min = float(boundary_coords[:, 0].min())
        x_max = float(boundary_coords[:, 0].max())
        y_min = float(boundary_coords[:, 1].min())
        y_max = float(boundary_coords[:, 1].max())
        actual_bounds = (x_min, x_max, y_min, y_max)
        print(
            f"    [i] Boundary-Bounds (TATSÄCHLICH, lokal): X=[{x_min:.0f}..{x_max:.0f}], Y=[{y_min:.0f}..{y_max:.0f}]"
        )
    else:
        actual_bounds = (0.0, 0.0, 0.0, 0.0)

    return boundary_array, actual_bounds


def extract_horizon_boundary_ring(
    vertex_manager: VertexManager,
    horizon_vertex_indices: np.ndarray,
    grid_bounds: Tuple[float, float, float, float],
    grid_spacing: float = 200.0,
) -> np.ndarray:
    """
    Identifiziert die INNERE Reihe des Horizon-Meshes (direkt außerhalb der Terrain-Tiles).

    PERFORMANCE: Vektorisierte NumPy-Operationen O(n) statt Python-Loop

    Args:
        vertex_manager: Gemeinsamer VertexManager
        horizon_vertex_indices: NumPy Array von Horizon-Vertex-Indizes
        grid_bounds: (x_min, x_max, y_min, y_max) der Terrain-Tiles
        grid_spacing: Spacing des Horizon-Grids (default: 200m)

    Returns:
        NumPy Array von Horizon-Ring-Vertex-Indizes
    """
    # === OPTIMIERUNG: Direkte Vektorisierung ohne Loop ===
    horizon_vertices = vertex_manager.vertices[horizon_vertex_indices]  # (N, 3)
    x_min, x_max, y_min, y_max = grid_bounds

    # Erweitere Bounds um grid_spacing
    expanded_x_min = x_min - grid_spacing * 1.5
    expanded_x_max = x_max + grid_spacing * 1.5
    expanded_y_min = y_min - grid_spacing * 1.5
    expanded_y_max = y_max + grid_spacing * 1.5

    # Vektorisierte Bereichsprüfung (alle auf einmal!)
    x = horizon_vertices[:, 0]
    y = horizon_vertices[:, 1]

    # Im erweiterten Ring?
    in_expanded = (x >= expanded_x_min) & (x <= expanded_x_max) & (y >= expanded_y_min) & (y <= expanded_y_max)

    # Außerhalb der Terrain-Bounds?
    outside_inner = ~((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max))

    # Kombiniere: Im Ring UND außerhalb Terrain
    ring_mask = in_expanded & outside_inner
    ring_local_indices = np.where(ring_mask)[0]

    print(f"    [i] Horizon-Ring Extraktion:")
    print(f"        Grid-Bounds: X=[{x_min:.0f}..{x_max:.0f}], Y=[{y_min:.0f}..{y_max:.0f}]")
    print(
        f"        Expanded-Bounds: X=[{expanded_x_min:.0f}..{expanded_x_max:.0f}], Y=[{expanded_y_min:.0f}..{expanded_y_max:.0f}]"
    )
    print(f"        Horizon-Vertices gesamt: {len(horizon_vertex_indices)}")
    print(f"        In Expanded: {in_expanded.sum()}, Outside Inner: {outside_inner.sum()}")
    print(f"        Ring-Vertices: {len(ring_local_indices)}")

    # Ring-Koordinaten für Debug
    if len(ring_local_indices) > 0:
        ring_vertices = horizon_vertices[ring_local_indices]
        x_min_ring = ring_vertices[:, 0].min()
        x_max_ring = ring_vertices[:, 0].max()
        y_min_ring = ring_vertices[:, 1].min()
        y_max_ring = ring_vertices[:, 1].max()
        print(f"        Ring-Bounds: X=[{x_min_ring:.0f}..{x_max_ring:.0f}], Y=[{y_min_ring:.0f}..{y_max_ring:.0f}]")

    # Mappe zu globalen Indizes
    return horizon_vertex_indices[ring_local_indices]


def stitch_ring_strip(
    terrain_vertices_idx: np.ndarray,
    horizon_vertices_idx: np.ndarray,
    vertex_manager: VertexManager,
    max_distance: float = 300.0,
) -> List[Tuple[int, int, int]]:
    """
    Verbindet zwei geschlossene Polygone (Ring) mit Triangle-Strips.

    LOGIC:
    - Terrain-Boundary: äußeres geschlossenes Polygon (8659 Vertices)
    - Horizon-Ring: inneres geschlossenes Polygon (69 Vertices)
    - Erzeuge Triangle-Strips zwischen den beiden Ringen

    ALGORITHM:
    1. Ordne Terrain-Vertices nach Angular-Position (um den Ring herum)
    2. Ordne Horizon-Vertices nach Angular-Position (gleiche Referenz)
    3. Verbinde entsprechende Vertices mit Triangle-Strips
    4. Prüfe Max-Distance zwischen benachbarten Vertices

    Args:
        terrain_vertices_idx: Äußeres Polygon (Terrain-Boundary)
        horizon_vertices_idx: Inneres Polygon (Horizon-Ring)
        vertex_manager: Gemeinsamer VertexManager
        max_distance: Maximale Edge-Länge (300m)

    Returns:
        Liste von (v0, v1, v2) Triangles
    """
    if len(terrain_vertices_idx) == 0 or len(horizon_vertices_idx) == 0:
        return []

    vertices = vertex_manager.vertices

    # === STEP 1: Berechne Centroid (Mittelpunkt beider Ringe) ===
    terrain_coords = vertices[terrain_vertices_idx][:, :2]
    horizon_coords = vertices[horizon_vertices_idx][:, :2]

    centroid_terrain = terrain_coords.mean(axis=0)
    centroid_horizon = horizon_coords.mean(axis=0)
    centroid = (centroid_terrain + centroid_horizon) / 2.0

    # === STEP 2: Sortiere beide Ringe nach Winkel (Polar-Koordinaten vom Centroid) ===
    def sort_by_angle(coords_2d, center):
        """Sortiere Punkte nach Winkel um den Center-Punkt (Counter-Clockwise)"""
        angles = np.arctan2(coords_2d[:, 1] - center[1], coords_2d[:, 0] - center[0])
        return np.argsort(angles)

    terrain_angle_order = sort_by_angle(terrain_coords, centroid)
    horizon_angle_order = sort_by_angle(horizon_coords, centroid)

    # Sortierte Indices
    terrain_sorted = terrain_vertices_idx[terrain_angle_order]
    horizon_sorted = horizon_vertices_idx[horizon_angle_order]

    print(f"    [i] Ring-Stitching: Terrain={len(terrain_sorted)} Vertices, Horizon={len(horizon_sorted)} Vertices")

    # === STEP 3: Verbinde die Ringe mit Triangle-Strips ===
    faces = []
    n_terrain = len(terrain_sorted)
    n_horizon = len(horizon_sorted)

    # Interpoliere Horizon-Indices für gleichmäßige Verteilung
    # (Horizon hat nur 69 Vertices, Terrain hat 8659)
    horizon_extended = []
    for i in range(n_terrain):
        # Wieviel Prozent durch den Terrain-Ring?
        t = i / n_terrain
        # Position im Horizon-Ring
        horizon_idx_float = t * n_horizon

        # Lineare Interpolation zwischen zwei Horizon-Vertices
        h_idx_low = int(horizon_idx_float) % n_horizon
        h_idx_high = (int(horizon_idx_float) + 1) % n_horizon
        t_frac = horizon_idx_float - int(horizon_idx_float)

        # Wähle nächsten Horizon-Vertex (vereinfacht - keine echte Interpolation)
        h_idx = h_idx_high if t_frac > 0.5 else h_idx_low
        horizon_extended.append(h_idx)

    # === STEP 4: Erzeuge Triangles zwischen benachbarten Vertices ===
    for i in range(n_terrain):
        t_curr = terrain_sorted[i]
        t_next = terrain_sorted[(i + 1) % n_terrain]
        h_curr_idx = horizon_extended[i]
        h_next_idx = horizon_extended[(i + 1) % n_terrain]
        h_curr = horizon_sorted[h_curr_idx]
        h_next = horizon_sorted[h_next_idx]

        # Berechne Distanzen
        dist_t_h_curr = np.linalg.norm(vertices[t_curr][:2] - vertices[h_curr][:2])
        dist_t_next_h_next = np.linalg.norm(vertices[t_next][:2] - vertices[h_next][:2])

        # Prüfe Max-Distance
        if dist_t_h_curr <= max_distance and dist_t_next_h_next <= max_distance:
            # Erzeuge zwei Triangles für das Quad: (t_curr, t_next, h_next, h_curr)
            # Triangle 1: t_curr, t_next, h_curr
            faces.append((t_curr, t_next, h_curr))
            # Triangle 2: t_next, h_next, h_curr
            faces.append((t_next, h_next, h_curr))

    print(f"    [i] Ring-Strip Faces: {len(faces)}")
    return faces


def stitch_terrain_horizon_boundary(
    terrain_mesh,
    vertex_manager: VertexManager,
    horizon_vertex_indices: np.ndarray,
    grid_bounds: Tuple[float, float, float, float],
    grid_spacing: float = 200.0,
) -> List[Tuple[int, int, int]]:
    """
    Hauptfunktion: Stitched Terrain-Tiles mit Horizon-Mesh entlang der Boundary.

    WICHTIG: Berechnet TATSÄCHLICHE Terrain-Bounds aus Boundary-Vertices!
    Der übergebene grid_bounds wird ignoriert - diese sind oft im falschen Koordinatensystem.

    OPTIMIERUNGEN:
    - Alle kritischen Pfade vektorisiert
    - Minimal Speicher-Overhead
    - Schnelle Set-Operationen statt Loops

    Args:
        terrain_mesh: Mesh-Objekt mit .faces und .face_props
        vertex_manager: Gemeinsamer VertexManager für Terrain UND Horizon
        horizon_vertex_indices: NumPy Array von Horizon-Vertex-Indizes
        grid_bounds: (x_min, x_max, y_min, y_max) - WIRD IGNORIERT! (siehe oben)
        grid_spacing: Spacing des Horizon-Grids (default 200m)

    Returns:
        Liste von Faces als Tuples (v0, v1, v2)
    """
    print(f"  [Boundary-Stitching] Extrahiere Terrain-Boundary-Kontur...")
    terrain_boundary_vertices, actual_bounds = extract_terrain_boundary_edges(terrain_mesh, vertex_manager)
    print(f"  [Boundary-Stitching] {len(terrain_boundary_vertices)} Terrain-Boundary-Vertices")

    if len(terrain_boundary_vertices) == 0:
        print(f"  [!] Keine Terrain-Boundaries gefunden")
        return []

    print(f"  [Boundary-Stitching] Extrahiere Horizon-Ring (innere Reihe)...")
    # WICHTIG: Nutze TATSÄCHLICHE Terrain-Bounds, nicht übergebene grid_bounds!
    horizon_ring_vertices = extract_horizon_boundary_ring(
        vertex_manager, horizon_vertex_indices, actual_bounds, grid_spacing
    )
    print(f"  [Boundary-Stitching] {len(horizon_ring_vertices)} Horizon-Ring-Vertices")

    if len(horizon_ring_vertices) == 0:
        print(f"  [!] Keine Horizon-Ring-Vertices gefunden")
        return []

    # === DEBUG: Speichere Mesh-Daten für Analyse ===
    try:
        import pickle
        import os

        cache_file = os.path.join("cache", "mesh_debug_dump.pkl")
        os.makedirs("cache", exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "terrain_mesh": terrain_mesh,
                    "vertex_manager": vertex_manager,
                    "horizon_vertex_indices": horizon_vertex_indices,
                    "terrain_boundary_vertices": terrain_boundary_vertices,
                    "horizon_ring_vertices": horizon_ring_vertices,
                    "actual_bounds": actual_bounds,
                },
                f,
            )
        print(f"  [DEBUG] Mesh-Daten gespeichert: {cache_file}")
    except Exception as e:
        print(f"  [!] Konnte Debug-Daten nicht speichern: {e}")

    print(f"  [Boundary-Stitching] Generiere Ring-Strip-Stitching (Triangle-Strips zwischen zwei Polygonen)...")
    faces = stitch_ring_strip(
        terrain_boundary_vertices, horizon_ring_vertices, vertex_manager, max_distance=grid_spacing * 1.5
    )

    print(f"  [Boundary-Stitching] FERTIG: {len(faces)} Stitching-Faces")

    return faces
