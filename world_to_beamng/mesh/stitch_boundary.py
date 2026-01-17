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
    # WICHTIG: Der Horizon hat ein LOCH in der Mitte (wo das Terrain ist).
    # - Innere Boundary: Rand des Lochs (direkt außerhalb des Terrain-Bereichs)
    # - Äußere Boundary: Äußerer Rand des Horizon-Meshes (weit draußen)

    vertices = horizon_vertex_manager.vertices
    x_min, x_max, y_min, y_max = grid_bounds

    # Importiere HORIZON_GRID_SPACING
    from .. import config

    # EINFACHE GEOMETRISCHE FILTERUNG:
    # Innere Boundary = Vertices, die nahe am Terrain-Bereich liegen
    # Erweitere die Terrain-Bounds um ein paar Grid-Spacings
    buffer = config.HORIZON_GRID_SPACING * 2.0  # 400m Buffer

    inner_x_min = x_min - buffer
    inner_x_max = x_max + buffer
    inner_y_min = y_min - buffer
    inner_y_max = y_max + buffer

    inner_boundary = []
    outer_boundary = []

    for v_idx in boundary_vertices:
        v = vertices[v_idx]
        x, y = v[0], v[1]

        # Prüfe ob Vertex innerhalb des erweiterten Bereichs liegt
        if inner_x_min <= x <= inner_x_max and inner_y_min <= y <= inner_y_max:
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
    Verbindet zwei geschlossene Polygone (Ring) mit Triangle-Strips (seiten-bewusst).

    LOGIC:
    - Terrain-Boundary: äußeres geschlossenes Polygon (8659 Vertices)
    - Horizon-Ring: inneres geschlossenes Polygon (69 Vertices)
    - Erzeuge Triangle-Strips zwischen den beiden Ringen

    WICHTIG: Seiten-basierte Zuordnung!
    - Terrain WEST wird nur mit Horizon WEST verbunden
    - Terrain NORD wird nur mit Horizon NORD verbunden
    - etc. (keine Sprünge über Ecken)

    ALGORITHM:
    1. Zerlege Terrain-Ring in 4 Seiten (Nord, Süd, Ost, West)
    2. Zerlege Horizon-Ring in 4 Seiten (gleiche Kriterien)
    3. Verbinde Seite-zu-Seite mit Triangle-Strips
    4. Prüfe Max-Distance zwischen benachbarten Vertices

    Args:
        terrain_vertices_idx: Äußeres Polygon (Terrain-Boundary)
        horizon_vertices_idx: Inneres Polygon (Horizon-Ring)
        vertex_manager: VertexManager
        max_distance: Maximale Edge-Länge (300m)

    Returns:
        Liste von (v0, v1, v2) Triangles
    """
    if len(terrain_vertices_idx) == 0 or len(horizon_vertices_idx) == 0:
        return []

    vertices = vertex_manager.vertices

    # === STEP 1: Berechne Terrain- und Horizon-Bounds ===
    terrain_coords = vertices[terrain_vertices_idx][:, :2]
    terrain_x_min = terrain_coords[:, 0].min()
    terrain_x_max = terrain_coords[:, 0].max()
    terrain_y_min = terrain_coords[:, 1].min()
    terrain_y_max = terrain_coords[:, 1].max()

    # WICHTIG: Verwende TERRAIN Bounds für die Seiten-Zuordnung!
    # Die Horizon-Ring-Vertices liegen auf einem regulären Grid - EXAKT auf horizontalen/vertikalen Linien
    # KEINE Toleranz nötig, da die Quads perfekt ausgerichtet sind!
    horizon_coords = vertices[horizon_vertices_idx][:, :2]

    tolerance = 1.0  # 1m: Nur für Rundungsfehler, Vertices liegen exakt auf Grid-Linien

    # Stitching-Toleranz und Bounds für Filterung

    # === STEP 2: Zerlege beide Ringe in 4 Seiten (mit Linien-Filter) ===
    def assign_sides_with_line_filter(
        vertices_idx, coords, x_min, x_max, y_min, y_max, coarse_tolerance=300.0, line_tolerance=1.0
    ):
        """
        Zweistufige Filterung:
        1. Grobe Selektion: Vertices nahe am Terrain-Rand (< coarse_tolerance)
        2. Feine Selektion: Nur Vertices auf einer gemeinsamen Linie (Abweichung < line_tolerance)

        Returns: Dictionary mit Listen von Vertex-Indizes pro Seite
        """
        sides = {"NORTH": [], "SOUTH": [], "EAST": [], "WEST": []}

        # NORTH: Vertices mit Y nahe y_max
        candidates = []
        for i, v_idx in enumerate(vertices_idx):
            v = coords[i]
            if abs(v[1] - y_max) < coarse_tolerance:
                candidates.append((v_idx, v[1]))  # (index, y-coord)

        if len(candidates) > 0:
            # Finde die gemeinsame Y-Linie (Median oder Mehrheit)
            y_values = [y for _, y in candidates]
            y_median = np.median(y_values)
            # Filtere: Nur Vertices auf der Linie (< 1m Abweichung)
            for v_idx, y in candidates:
                if abs(y - y_median) < line_tolerance:
                    sides["NORTH"].append(v_idx)

        # SOUTH: Vertices mit Y nahe y_min
        candidates = []
        for i, v_idx in enumerate(vertices_idx):
            v = coords[i]
            if abs(v[1] - y_min) < coarse_tolerance:
                candidates.append((v_idx, v[1]))

        if len(candidates) > 0:
            y_values = [y for _, y in candidates]
            y_median = np.median(y_values)
            for v_idx, y in candidates:
                if abs(y - y_median) < line_tolerance:
                    sides["SOUTH"].append(v_idx)

        # EAST: Vertices mit X nahe x_max
        candidates = []
        for i, v_idx in enumerate(vertices_idx):
            v = coords[i]
            if abs(v[0] - x_max) < coarse_tolerance:
                candidates.append((v_idx, v[0]))  # (index, x-coord)

        if len(candidates) > 0:
            x_values = [x for _, x in candidates]
            x_median = np.median(x_values)
            for v_idx, x in candidates:
                if abs(x - x_median) < line_tolerance:
                    sides["EAST"].append(v_idx)

        # WEST: Vertices mit X nahe x_min
        candidates = []
        for i, v_idx in enumerate(vertices_idx):
            v = coords[i]
            if abs(v[0] - x_min) < coarse_tolerance:
                candidates.append((v_idx, v[0]))

        if len(candidates) > 0:
            x_values = [x for _, x in candidates]
            x_median = np.median(x_values)
            for v_idx, x in candidates:
                if abs(x - x_median) < line_tolerance:
                    sides["WEST"].append(v_idx)

        return sides

    # Terrain-Seiten
    # Terrain: Präzisere Filterung (50 cm Toleranz)
    terrain_sides = assign_sides_with_line_filter(
        terrain_vertices_idx,
        terrain_coords,
        terrain_x_min,
        terrain_x_max,
        terrain_y_min,
        terrain_y_max,
        coarse_tolerance=300.0,
        line_tolerance=0.5,
    )

    # Horizon: 1 m Toleranz (Quad-Grid)
    horizon_sides = assign_sides_with_line_filter(
        horizon_vertices_idx,
        horizon_coords,
        terrain_x_min,
        terrain_x_max,
        terrain_y_min,
        terrain_y_max,
        coarse_tolerance=300.0,
        line_tolerance=1.0,
    )

    # Validierung: Prüfe auf nicht zugewiesene Horizon-Vertices
    all_horizon_assigned = set()
    for side in ["NORTH", "SOUTH", "EAST", "WEST"]:
        all_horizon_assigned.update(horizon_sides[side])

    horizon_no_side = len(horizon_vertices_idx) - len(all_horizon_assigned)
    if horizon_no_side > 0:
        print(f"    [!] WARNUNG: {horizon_no_side} Horizon-Vertices OHNE Seite (von {len(horizon_vertices_idx)} total)")

    # === STEP 3: Verbinde Seite-zu-Seite mit sequenziellem Strip ===
    faces = []

    for side in ["NORTH", "SOUTH", "EAST", "WEST"]:
        terrain_side_vertices = terrain_sides[side]
        horizon_side_vertices = horizon_sides[side]

        if len(terrain_side_vertices) == 0 or len(horizon_side_vertices) == 0:
            print(f"    [!] FEHLER: Seite {side} hat keine Vertices (Terrain: {len(terrain_side_vertices)}, Horizon: {len(horizon_side_vertices)})")
            continue

        # Sortiere beide nach Position entlang der Seite
        if side == "NORTH":
            # Sortiere von West zu Ost (nach X)
            terrain_side_vertices = sorted(terrain_side_vertices, key=lambda v: vertices[v, 0])
            horizon_side_vertices = sorted(horizon_side_vertices, key=lambda v: vertices[v, 0])
        elif side == "SOUTH":
            # Sortiere von Ost zu West (nach -X)
            terrain_side_vertices = sorted(terrain_side_vertices, key=lambda v: -vertices[v, 0])
            horizon_side_vertices = sorted(horizon_side_vertices, key=lambda v: -vertices[v, 0])
        elif side == "EAST":
            # Sortiere von Nord zu Süd (nach -Y)
            terrain_side_vertices = sorted(terrain_side_vertices, key=lambda v: -vertices[v, 1])
            horizon_side_vertices = sorted(horizon_side_vertices, key=lambda v: -vertices[v, 1])
        elif side == "WEST":
            # Sortiere von Süd zu Nord (nach Y)
            terrain_side_vertices = sorted(terrain_side_vertices, key=lambda v: vertices[v, 1])
            horizon_side_vertices = sorted(horizon_side_vertices, key=lambda v: vertices[v, 1])

        # === Sequenzieller Strip: Gehe Terrain-Kante ab und verbinde mit Horizon-Ring ===
        h_curr_idx = 0  # Aktueller Horizon-Vertex Index
        h_prev_idx = -1  # Vorheriger Horizon-Vertex (für Detektion von Wechsel)

        for t_idx in range(len(terrain_side_vertices)):
            t_curr = terrain_side_vertices[t_idx]
            t_next = terrain_side_vertices[(t_idx + 1) % len(terrain_side_vertices)]

            t_curr_pos = vertices[t_curr][:2]

            # Finde NÄCHSTEN Horizon-Vertex zu t_curr (nur auf dieser Seite!)
            distances_to_horizon = np.linalg.norm(vertices[np.array(horizon_side_vertices)][:, :2] - t_curr_pos, axis=1)
            h_next_idx = np.argmin(distances_to_horizon)
            h_curr = horizon_side_vertices[h_curr_idx]
            h_next = horizon_side_vertices[h_next_idx]

            # Prüfe Distanzen
            dist_t_curr_h_curr = np.linalg.norm(vertices[t_curr][:2] - vertices[h_curr][:2])
            dist_t_next_h_curr = np.linalg.norm(vertices[t_next][:2] - vertices[h_curr][:2])

            # === Erzeuge Triangle zwischen Terrain und Horizon ===
            if dist_t_curr_h_curr <= max_distance and dist_t_next_h_curr <= max_distance:
                faces.append((t_curr, t_next, h_curr))

            # === Wechsle zu nächstem Horizon-Vertex wenn dieser näher ist ===
            if h_next_idx != h_curr_idx and h_next_idx != h_prev_idx:
                # Es gibt einen näheren Horizon-Vertex
                dist_t_next_h_next = np.linalg.norm(vertices[t_next][:2] - vertices[h_next][:2])
                if dist_t_next_h_next < dist_t_next_h_curr:
                    # === FÜLL-DREIECKE: Wenn Horizon-Index springt ===
                    index_jump = h_next_idx - h_curr_idx

                    # Behandle auch negative Sprünge (Wrap-around am Ende der Liste)
                    if index_jump < 0:
                        index_jump += len(horizon_side_vertices)

                    if index_jump >= 1:
                        # Fülle Lücke mit Dreiecken für jeden Sprung
                        for fill_offset in range(1, index_jump + 1):
                            fill_idx = (h_curr_idx + fill_offset) % len(horizon_side_vertices)
                            h_fill = horizon_side_vertices[fill_idx]
                            h_fill_prev = horizon_side_vertices[
                                (h_curr_idx + fill_offset - 1) % len(horizon_side_vertices)
                            ]

                            # Füll-Dreieck: t_next -> h_fill_prev -> h_fill (nutze nächsten Terrain-Vertex für konsistente Reihenfolge)
                            faces.append((t_next, h_fill_prev, h_fill))

                    # Nächster Horizon-Vertex ist näher - wechsle
                    h_prev_idx = h_curr_idx
                    h_curr_idx = h_next_idx

        # === Abschluss: Falls letzter Horizon-Eckpunkt noch nicht genutzt wurde, fülle bis zum Ende ===
        last_t_vertex = terrain_side_vertices[-1]
        last_horizon_idx = len(horizon_side_vertices) - 1
        if h_curr_idx < last_horizon_idx:
            for fill_idx in range(h_curr_idx + 1, last_horizon_idx + 1):
                h_fill = horizon_side_vertices[fill_idx]
                h_fill_prev = horizon_side_vertices[fill_idx - 1]

                # Füll-Dreieck mit letztem Terrain-Vertex zur Ecke hin
                faces.append((last_t_vertex, h_fill_prev, h_fill))

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
    terrain_boundary_vertices, actual_bounds = extract_terrain_boundary_edges(terrain_mesh, terrain_vertex_manager)

    if len(terrain_boundary_vertices) == 0:
        print(f"  [!] Keine Terrain-Boundaries gefunden")
        return [], np.array([])

    # WICHTIG: Nutze TATSÄCHLICHE Terrain-Bounds, nicht übergebene grid_bounds!
    horizon_ring_vertices_local = extract_horizon_boundary_ring(horizon_vertex_manager, horizon_mesh, actual_bounds)

    if len(horizon_ring_vertices_local) == 0:
        print(f"  [!] Keine Horizon-Ring-Vertices gefunden")
        return [], np.array([])

    # === SAUBERE LÖSUNG: Kopiere Terrain-Ring-Vertices in den HORIZON-VM ===
    # (nicht in den Terrain-VM - der bleibt unverändert!)
    terrain_ring_vertices_coords = terrain_vertex_manager.vertices[terrain_boundary_vertices]
    terrain_ring_vertices_horizon_idx = np.array(
        horizon_vertex_manager.add_vertices_direct_nohash(terrain_ring_vertices_coords), dtype=np.int32
    )
    # JETZT BEIDE RINGE IM GLEICHEN HORIZON-VM!
    faces = stitch_ring_strip(
        terrain_ring_vertices_horizon_idx,  # Terrain-Ring im Horizon-VM
        horizon_ring_vertices_local,  # Horizon-Ring im Horizon-VM
        horizon_vertex_manager,
        max_distance=grid_spacing * 1.5,
    )

    print(f"  [Boundary-Stitching] FERTIG: {len(faces)} Stitching-Faces")

    return faces
