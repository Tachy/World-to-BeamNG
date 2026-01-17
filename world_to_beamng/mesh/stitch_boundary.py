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

    print(f"    [i] Boundary-Edges gesamt: {len(boundary_edges)}, Boundary-Vertices gesamt: {len(boundary_vertices)}")

    # === KRITISCH: Berechne TATSÄCHLICHE Bounds der Boundary-Vertices ===
    vertices = vertex_manager.vertices
    if len(boundary_vertices) > 0:
        boundary_coords = vertices[np.array(list(boundary_vertices))]
        x_min = float(boundary_coords[:, 0].min())
        x_max = float(boundary_coords[:, 0].max())
        y_min = float(boundary_coords[:, 1].min())
        y_max = float(boundary_coords[:, 1].max())
        actual_bounds = (x_min, x_max, y_min, y_max)
    else:
        return np.array([], dtype=np.int32), (0.0, 0.0, 0.0, 0.0)

    # === FILTERUNG: Nur Vertices die EXAKT auf den 4 Tile-Kanten liegen ===
    # Dies entspricht dem Ansatz aus stitch_terrain_roads
    # Ignoriere innenliegende Vertices!
    boundary_rim_vertices = set()
    tolerance = 0.1  # 10cm - für Float-Gleichheit

    for v_idx in boundary_vertices:
        v = vertices[v_idx]
        x, y = v[0], v[1]

        # EXAKT auf einer der 4 Seiten?
        on_rim = False
        if abs(y - y_max) < tolerance:  # NORTH
            on_rim = True
        elif abs(y - y_min) < tolerance:  # SOUTH
            on_rim = True
        elif abs(x - x_max) < tolerance:  # EAST
            on_rim = True
        elif abs(x - x_min) < tolerance:  # WEST
            on_rim = True

        if on_rim:
            boundary_rim_vertices.add(v_idx)

    boundary_array = np.array(sorted(boundary_rim_vertices), dtype=np.int32)
    print(f"    [i] Boundary-Rim-Vertices (nur Kanten): {len(boundary_array)}")
    print(f"    [i] Boundary-Bounds (TATSÄCHLICH): X=[{x_min:.0f}..{x_max:.0f}], Y=[{y_min:.0f}..{y_max:.0f}]")

    return boundary_array, actual_bounds


def extract_horizon_boundary_ring(
    horizon_vertex_manager: VertexManager,
    horizon_mesh,
    grid_bounds: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Extrahiert die INNERE Boundary des Horizon-Meshes (Rand des Loches in der Mitte).

    STRATEGIE: Edge-basierte Boundary-Extraktion (wie bei extract_terrain_boundary_edges)
    Das Horizon-Mesh hat ein Loch in der Mitte (wo das Terrain ist).
    Wir extrahieren ALLE Boundary-Vertices und filtern dann die innere Boundary
    (näher am Terrain) von der äußeren Boundary (äußerer Rand des Horizon-Meshes).

    Args:
        horizon_vertex_manager: VertexManager des Horizon-Meshes (separater VM!)
        horizon_mesh: Horizon-Mesh-Objekt (mit .faces)
        grid_bounds: (x_min, x_max, y_min, y_max) der Terrain-Tiles (für Filterung)

    Returns:
        NumPy Array von Horizon-Ring-Vertex-Indizes im horizon_vertex_manager
    """
    # === Kompatibilität: Mesh-Objekt oder Dictionary ===
    if hasattr(horizon_mesh, "faces"):
        faces_list = horizon_mesh.faces
    else:
        faces_list = horizon_mesh.get("faces", [])

    # === Edge-basierte Boundary-Extraktion (wie bei Terrain) ===
    edge_faces = defaultdict(list)

    for face_idx, face in enumerate(faces_list):
        v0, v1, v2 = face
        edges = [(min(v0, v1), max(v0, v1)), (min(v1, v2), max(v1, v2)), (min(v2, v0), max(v2, v0))]
        for edge in edges:
            edge_faces[edge].append(face_idx)

    # Boundary-Edges: Nur 1 Face pro Edge
    boundary_vertices = set()
    for (v1_idx, v2_idx), faces in edge_faces.items():
        if len(faces) == 1:
            boundary_vertices.add(v1_idx)
            boundary_vertices.add(v2_idx)

    print(f"    [i] Horizon-Mesh Boundary-Vertices gesamt: {len(boundary_vertices)}")

    # === Filtere innere vs. äußere Boundary ===
    # Innere Boundary: Vertices nahe am Terrain (innerhalb von 2000m Abstand)
    # Äußere Boundary: Vertices weit weg vom Terrain (außerhalb von 2000m Abstand)
    vertices = horizon_vertex_manager.vertices
    x_min, x_max, y_min, y_max = grid_bounds
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    inner_boundary = []
    outer_boundary = []

    for v_idx in boundary_vertices:
        v = vertices[v_idx]
        dist_to_center = np.sqrt((v[0] - center_x) ** 2 + (v[1] - center_y) ** 2)

        # Heuristik: Innere Boundary liegt näher am Center als 2000m
        # (das Terrain ist 2km x 2km, also max 1414m vom Center)
        if dist_to_center < 2000.0:
            inner_boundary.append(v_idx)
        else:
            outer_boundary.append(v_idx)

    inner_boundary_array = np.array(sorted(inner_boundary), dtype=np.int32)

    print(f"    [i] Horizon-Ring Extraktion:")
    print(f"        Grid-Bounds: X=[{x_min:.0f}..{x_max:.0f}], Y=[{y_min:.0f}..{y_max:.0f}]")
    print(f"        Innere Boundary-Vertices: {len(inner_boundary)}")
    print(f"        Äußere Boundary-Vertices: {len(outer_boundary)}")

    if len(inner_boundary_array) > 0:
        ring_vertices = vertices[inner_boundary_array]
        x_min_ring = ring_vertices[:, 0].min()
        x_max_ring = ring_vertices[:, 0].max()
        y_min_ring = ring_vertices[:, 1].min()
        y_max_ring = ring_vertices[:, 1].max()
        print(f"        Ring-Bounds: X=[{x_min_ring:.0f}..{x_max_ring:.0f}], Y=[{y_min_ring:.0f}..{y_max_ring:.0f}]")

    return inner_boundary_array


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

    # === STEP 3 & 4: Erzeuge saubere Triangle-Strips zwischen den Ringen ===
    # Für jeden Terrain-Vertex: Verbinde zu nächstem Horizon-Vertex
    # WICHTIG: Saubere topologische Struktur, keine wilden Sprünge!

    n_terrain = len(terrain_sorted)
    n_horizon = len(horizon_sorted)
    faces = []

    for i in range(n_terrain):
        t_curr = terrain_sorted[i]
        t_next = terrain_sorted[(i + 1) % n_terrain]

        # Finde NÄCHSTEN Horizon-Vertex zu t_curr
        # (Nicht interpolieren, sondern topologisch konsistent)
        t_curr_pos = vertices[t_curr][:2]
        distances_to_horizon = np.linalg.norm(vertices[horizon_sorted][:, :2] - t_curr_pos, axis=1)
        h_curr_idx = np.argmin(distances_to_horizon)
        h_curr = horizon_sorted[h_curr_idx]

        # Nächster Horizon-Vertex (ringsum)
        h_next = horizon_sorted[(h_curr_idx + 1) % n_horizon]

        # Prüfe Distanzen
        dist_t_curr_h_curr = np.linalg.norm(vertices[t_curr][:2] - vertices[h_curr][:2])
        dist_t_next_h_curr = np.linalg.norm(vertices[t_next][:2] - vertices[h_curr][:2])

        # Erzeuge NUR EIN DREIECK pro Terrain-Vertex (t_curr, t_next, h_curr)
        # Dies erzeugt einen sauberen Streifen ohne wilde Sprünge
        if dist_t_curr_h_curr <= max_distance and dist_t_next_h_curr <= max_distance:
            faces.append((t_curr, t_next, h_curr))

    print(f"    [i] Ring-Strip Faces: {len(faces)}")
    return faces


def stitch_terrain_horizon_boundary(
    terrain_mesh,
    terrain_vertex_manager: VertexManager,
    horizon_mesh,
    horizon_vertex_manager: VertexManager,
    grid_bounds: Tuple[float, float, float, float],
    grid_spacing: float = 200.0,
) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """
    Stitched Terrain-Tiles mit Horizon-Mesh entlang der Boundary.

    SAUBERE ARCHITEKTUR:
    - terrain_mesh: Unverändert! (bleibt in eigenem terrain_vertex_manager)
    - horizon_mesh: Wird erweitert mit neuen Vertices + Faces
    - Terrain-Ring-Vertices werden in den HORIZON-VM kopiert
    - Stitching-Faces arbeiten mit Horizon-VM Indizes

    WICHTIG: Berechnet TATSÄCHLICHE Terrain-Bounds aus Boundary-Vertices!
    Der übergebene grid_bounds wird ignoriert - diese sind oft im falschen Koordinatensystem.

    Args:
        terrain_mesh: Mesh-Objekt mit .faces und .face_props
        terrain_vertex_manager: VertexManager für Terrain (UNVERÄNDERT!)
        horizon_mesh: Horizon-Mesh-Objekt (wird erweitert)
        horizon_vertex_manager: Separater VertexManager für Horizon
        grid_bounds: (x_min, x_max, y_min, y_max) - WIRD IGNORIERT! (siehe oben)
        grid_spacing: Spacing des Horizon-Grids (default 200m)

    Returns:
        Tuple (stitching_faces, terrain_ring_vertices_global_coords)
        - stitching_faces: Liste von Faces mit Indizes im Horizon-VM
        - terrain_ring_vertices_global_coords: (N, 3) Koordinaten der Terrain-Ring-Vertices (für Horizon-VM)
    """
    print(f"  [Boundary-Stitching] Extrahiere Terrain-Boundary-Kontur...")
    terrain_boundary_vertices, actual_bounds = extract_terrain_boundary_edges(terrain_mesh, terrain_vertex_manager)
    print(f"  [Boundary-Stitching] {len(terrain_boundary_vertices)} Terrain-Boundary-Vertices")

    if len(terrain_boundary_vertices) == 0:
        print(f"  [!] Keine Terrain-Boundaries gefunden")
        return [], np.array([])

    print(f"  [Boundary-Stitching] Extrahiere Horizon-Ring (innere Reihe)...")
    # WICHTIG: Nutze TATSÄCHLICHE Terrain-Bounds, nicht übergebene grid_bounds!
    horizon_ring_vertices_local = extract_horizon_boundary_ring(horizon_vertex_manager, horizon_mesh, actual_bounds)
    print(f"  [Boundary-Stitching] {len(horizon_ring_vertices_local)} Horizon-Ring-Vertices (im Horizon-VM)")

    if len(horizon_ring_vertices_local) == 0:
        print(f"  [!] Keine Horizon-Ring-Vertices gefunden")
        return [], np.array([])

    # === SAUBERE LÖSUNG: Kopiere Terrain-Ring-Vertices in den HORIZON-VM ===
    # (nicht in den Terrain-VM - der bleibt unverändert!)
    print(f"  [Boundary-Stitching] Kopiere Terrain-Ring-Vertices in HORIZON-VertexManager...")
    
    # Hole Terrain-Ring-Vertex-Koordinaten
    terrain_ring_vertices_coords = terrain_vertex_manager.vertices[terrain_boundary_vertices]
    
    # Füge sie in den Horizon-VM ein
    terrain_ring_vertices_horizon_idx = np.array(
        horizon_vertex_manager.add_vertices_direct_nohash(terrain_ring_vertices_coords), 
        dtype=np.int32
    )
    
    print(f"  [Boundary-Stitching] Terrain-Vertices jetzt auch im Horizon-VM (Indizes {terrain_ring_vertices_horizon_idx[0]}..{terrain_ring_vertices_horizon_idx[-1]})")

    print(f"  [Boundary-Stitching] Generiere Ring-Strip-Stitching (Triangle-Strips zwischen zwei Polygonen)...")
    # JETZT BEIDE RINGE IM GLEICHEN HORIZON-VM!
    faces = stitch_ring_strip(
        terrain_ring_vertices_horizon_idx,  # Terrain-Ring im Horizon-VM
        horizon_ring_vertices_local,          # Horizon-Ring im Horizon-VM
        horizon_vertex_manager,
        max_distance=grid_spacing * 1.5
    )

    print(f"  [Boundary-Stitching] FERTIG: {len(faces)} Stitching-Faces")

    return faces, terrain_ring_vertices_coords
