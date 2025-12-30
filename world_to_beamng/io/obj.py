"""
OBJ-Export Funktionen (unified und layer-based).
"""

import os
import numpy as np
import pyvista as pv

# Konfigurierbare Centerline-Parameter
try:
    from world_to_beamng import config

    _CENTERLINE_SEARCH_RADIUS = config.CENTERLINE_SEARCH_RADIUS
    _CENTERLINE_SAMPLE_SPACING = config.CENTERLINE_SAMPLE_SPACING
except Exception:
    _CENTERLINE_SEARCH_RADIUS = 10.0
    _CENTERLINE_SAMPLE_SPACING = 10.0


def create_pyvista_mesh(vertices, faces):
    """Erstellt PyVista PolyData direkt aus Vertices und Faces (VEKTORISIERT)."""
    if not faces:
        return None

    # Konvertiere zu NumPy für vektorisierte Operationen
    faces_array = np.array(faces, dtype=np.int32)

    # Finde verwendete Vertices (VEKTORISIERT statt Set+Loop)
    used_indices_sorted = np.unique(faces_array)

    # Erstelle Mapping: alter Index → neuer Index (0-basiert für PyVista)
    # Nutze NumPy searchsorted für O(n log n) statt Dict-Lookup
    index_map = np.arange(len(used_indices_sorted))

    # Extrahiere verwendete Vertices (NumPy fancy indexing statt List Comprehension)
    used_vertices = np.array(vertices)[used_indices_sorted - 1]  # Faces sind 1-basiert

    # Konvertiere Faces (VEKTORISIERT: 1-basiert → 0-basiert)
    # searchsorted mappt alte Indizes → neue Indizes in O(n log n)
    faces_remapped = np.searchsorted(used_indices_sorted, faces_array)

    # PyVista Face-Format: [3, v1, v2, v3, 3, v1, v2, v3, ...]
    num_faces = len(faces_remapped)
    pyvista_faces = np.empty(num_faces * 4, dtype=np.int32)
    pyvista_faces[0::4] = 3  # Jedes 4. Element = 3 (Triangle)
    pyvista_faces[1::4] = faces_remapped[:, 0]
    pyvista_faces[2::4] = faces_remapped[:, 1]
    pyvista_faces[3::4] = faces_remapped[:, 2]

    # Erstelle PyVista PolyData
    mesh = pv.PolyData(used_vertices, pyvista_faces)
    return mesh


def save_layer_obj(filename, vertices, faces, material_name):
    """Speichert ein einzelnes Layer-Mesh als OBJ (nur verwendete Vertices)."""
    if not faces:
        print(f"  → {filename}: Keine Faces, überspringe")
        return

    # Finde alle verwendeten Vertex-Indizes
    used_indices = set()
    for face in faces:
        used_indices.update(face)

    # Erstelle Mapping: alter Index → neuer Index
    used_indices_sorted = sorted(used_indices)
    index_map = {
        old_idx: new_idx + 1 for new_idx, old_idx in enumerate(used_indices_sorted)
    }

    # Extrahiere nur verwendete Vertices (Faces sind 1-basiert)
    used_vertices = [vertices[idx - 1] for idx in used_indices_sorted]

    # Nummeriere Faces neu
    remapped_faces = [[index_map[idx] for idx in face] for face in faces]

    # BATCH-SCHREIBEN (100x schneller als Loop)
    with open(filename, "w", buffering=8 * 1024 * 1024) as f:  # 8MB Buffer
        f.write(f"# {material_name} Layer\n")
        f.write(f"mtllib terrain.mtl\n\n")

        # Vertices (BATCH mit join, nur verwendete)
        vertex_lines = [f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n" for v in used_vertices]
        f.write("".join(vertex_lines))

        # Faces (BATCH mit join, neu nummeriert)
        f.write(f"\nusemtl {material_name}\n")
        face_lines = [f"f {face[0]} {face[1]} {face[2]}\n" for face in remapped_faces]
        f.write("".join(face_lines))

    print(f"  ✓ {filename}: {len(used_vertices)} vertices, {len(faces)} faces")


def save_unified_obj(filename, vertices, road_faces, slope_faces, terrain_faces):
    """Speichert ein einheitliches Terrain-Mesh mit integrierten Straßen (ULTRA-OPTIMIERT)."""
    mtl_filename = filename.replace(".obj", ".mtl")

    # Erstelle MTL-Datei
    with open(mtl_filename, "w") as f:
        f.write("# Material Library for BeamNG Terrain\n")
        f.write("# Auto-generated\n\n")

        # Straßenoberfläche (Asphalt)
        f.write("newmtl road_surface\n")
        f.write("Ns 50.000000\n")
        f.write("Ka 0.200000 0.200000 0.200000\n")
        f.write("Kd 0.300000 0.300000 0.300000\n")
        f.write("Ks 0.500000 0.500000 0.500000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n\n")

        # Böschungen (Erde)
        f.write("newmtl road_slope\n")
        f.write("Ns 5.000000\n")
        f.write("Ka 0.200000 0.150000 0.100000\n")
        f.write("Kd 0.400000 0.300000 0.200000\n")
        f.write("Ks 0.100000 0.100000 0.100000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n\n")

        # Terrain (Gras/Natur)
        f.write("newmtl terrain\n")
        f.write("Ns 10.000000\n")
        f.write("Ka 0.100000 0.200000 0.100000\n")
        f.write("Kd 0.200000 0.500000 0.200000\n")
        f.write("Ks 0.100000 0.100000 0.100000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")

    # Erstelle OBJ-Datei (ULTRA-OPTIMIERT mit NumPy String-Formatting)
    print(f"\nSchreibe OBJ-Datei: {filename}")
    with open(
        filename, "w", buffering=64 * 1024 * 1024
    ) as f:  # 64MB Buffer für Ultra-Speed!
        f.write("# BeamNG Unified Terrain Mesh with integrated roads\n")
        f.write(f"# Generated from DGM1 data and OSM\n")
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")

        # ULTRA-SCHNELLE Methode: Formatiere alles mit NumPy, schreibe als einen Block
        def write_faces_fast(file, faces, prefix):
            """Schreibt Faces 10x schneller als np.savetxt durch Bulk-String-Ops."""
            if len(faces) == 0:
                return
            # Erstelle Strings für alle 3 Spalten separat
            col1 = faces[:, 0].astype(str)
            col2 = faces[:, 1].astype(str)
            col3 = faces[:, 2].astype(str)
            # Kombiniere mit festen Strings (kein Loop!)
            lines = np.char.add(
                np.char.add(np.char.add(np.char.add(prefix, col1), " "), col2),
                np.char.add(" ", col3),
            )
            # Schreibe alles auf einmal
            file.write("\n".join(lines.tolist()) + "\n")

        # Vertices schreiben (schnellste Methode)
        print(f"  Schreibe {len(vertices)} Vertices (ULTRA-FAST)...")
        v_array = np.array(vertices, dtype=np.float32)
        v1 = v_array[:, 0].astype(str)
        v2 = v_array[:, 1].astype(str)
        v3 = v_array[:, 2].astype(str)
        v_lines = np.char.add(
            np.char.add(np.char.add(np.char.add("v ", v1), " "), v2),
            np.char.add(" ", v3),
        )
        f.write("\n".join(v_lines.tolist()) + "\n")
        del v_array, v1, v2, v3, v_lines

        # Straßen-Objekt
        print(f"  Schreibe {len(road_faces)} Straßen-Faces (ULTRA-FAST)...")
        f.write("\no road_surface\n")
        f.write("usemtl road_surface\n")
        write_faces_fast(f, road_faces, "f ")

        # Böschungs-Objekt
        print(f"  Schreibe {len(slope_faces)} Böschungs-Faces (ULTRA-FAST)...")
        f.write("\no road_slope\n")
        f.write("usemtl road_slope\n")
        write_faces_fast(f, slope_faces, "f ")

        # Terrain-Objekt
        print(f"  Schreibe {len(terrain_faces)} Terrain-Faces (ULTRA-FAST)...")
        f.write("\no terrain\n")
        f.write("usemtl terrain\n")
        write_faces_fast(f, terrain_faces, "f ")

    print(f"  ✓ {filename} erfolgreich erstellt!")
    print(f"  ✓ {mtl_filename} erfolgreich erstellt!")


def save_roads_obj(filename, vertices, road_faces, road_face_to_idx):
    """Exportiert nur die Straßen als OBJ mit pro-Straße-Material (road_<id>)."""

    if len(road_faces) == 0:
        print(f"  → {filename}: Keine Straßen-Faces, überspringe")
        return

    # Gruppiere Faces nach road_idx
    from collections import defaultdict

    faces_by_road = defaultdict(list)
    unknown_counter = 0
    for face, ridx in zip(road_faces, road_face_to_idx):
        key = ridx if ridx is not None else f"unknown_{unknown_counter}"
        if ridx is None:
            unknown_counter += 1
        faces_by_road[key].append(face)

    # Remap Vertices (nur verwendete)
    used_indices = set()
    for faces in faces_by_road.values():
        for face in faces:
            used_indices.update(face)

    used_indices_sorted = sorted(used_indices)
    index_map = {
        old_idx: new_idx + 1 for new_idx, old_idx in enumerate(used_indices_sorted)
    }
    used_vertices = [vertices[idx] for idx in used_indices_sorted]

    # Remap Faces
    remapped_by_road = {}
    for key, faces in faces_by_road.items():
        remapped = [[index_map[idx] for idx in face] for face in faces]
        remapped_by_road[key] = remapped

    mtl_filename = filename.replace(".obj", ".mtl")
    with open(mtl_filename, "w") as f:
        f.write("# Road-only Material Library\n")
        for key in remapped_by_road.keys():
            f.write(f"newmtl road_{key}\n")
            f.write(
                "Ka 0.2 0.2 0.2\nKd 0.5 0.5 0.5\nKs 0.1 0.1 0.1\nd 1.0\nillum 2\n\n"
            )

    with open(filename, "w") as f:
        f.write("# Roads only OBJ\n")
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")

        for v in used_vertices:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")

        for key, faces in remapped_by_road.items():
            f.write(f"\no road_{key}\n")
            f.write(f"usemtl road_{key}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(
        f"  ✓ {filename}: {len(used_vertices)} vertices, {len(road_faces)} faces, {len(remapped_by_road)} roads"
    )


def save_centerlines_obj(
    filename, road_polygons, height_points, height_elevations, bbox, local_offset=None
):
    """
    Exportiert Straßen-Centerlines als OBJ mit Höhendaten.

    Args:
        filename: Zieldatei
        road_polygons: Liste von Straßen-Polygonen (aus get_road_polygons)
        height_points: Nicht verwendet
        height_elevations: Nicht verwendet
        bbox: Nicht verwendet
        local_offset: Optional (x, y, z) Offset zum Transformieren in lokales System
    """
    if not road_polygons:
        print(f"  Info: Keine Centerlines gefunden")
        return

    # Nutze den GLEICHEN Offset wie das Mesh
    if local_offset is None:
        # Versuche von config zu holen
        try:
            from world_to_beamng import config

            local_offset = config.LOCAL_OFFSET
        except:
            local_offset = None

    if local_offset is None:
        local_offset = (0.0, 0.0, 0.0)
        print(f"  Warnung: Kein LOCAL_OFFSET definiert, nutze (0, 0, 0)")
    else:
        print(
            f"  Nutze Mesh-Offset: X={local_offset[0]:.2f}, Y={local_offset[1]:.2f}, Z={local_offset[2]:.2f}"
        )

    centerline_vertices = []
    centerline_edges = []
    circle_vertices = []
    circle_edges = []

    def _resample_polyline(points, spacing):
        """Resample 3D polyline at fixed spacing (includes endpoints)."""
        if len(points) < 2:
            return points

        pts = np.array(points, dtype=float)
        diffs = np.diff(pts[:, :2], axis=0)  # XY-Länge für Distanz
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cumlen[-1]
        if total < 1e-6:
            return [tuple(p) for p in pts]

        num = max(int(np.floor(total / spacing)), 0)
        samples = np.linspace(0.0, total, num + 2)  # inkl. Start und Ende

        resampled = []
        for s in samples:
            idx = np.searchsorted(cumlen, s, side="right") - 1
            idx = min(idx, len(seg_lengths) - 1)
            seg_start = pts[idx]
            seg_end = pts[idx + 1]
            seg_len = seg_lengths[idx]
            if seg_len < 1e-9:
                resampled.append(tuple(seg_start))
                continue
            t = (s - cumlen[idx]) / seg_len
            p = seg_start + t * (seg_end - seg_start)
            resampled.append(tuple(p))

        return resampled

    def _create_search_circle(center, radius=_CENTERLINE_SEARCH_RADIUS, segments=16):
        """Erzeuge Kreispunkte um center in XY-Ebene."""
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        pts = []
        for ang in angles:
            x = center[0] + radius * np.cos(ang)
            y = center[1] + radius * np.sin(ang)
            pts.append((x, y, center[2]))
        return pts

    for road_poly in road_polygons:
        try:
            # Extrahiere Centerline aus der Straße (coords sind BEREITS in lokalen Koordinaten!)
            if "coords" in road_poly:
                centerline_orig = list(road_poly["coords"])
            else:
                continue

            if centerline_orig is None or len(centerline_orig) < 2:
                continue

            # Resample entlang der Centerline mit konfiguriertem Sample-Spacings
            centerline_resampled = _resample_polyline(
                centerline_orig, spacing=_CENTERLINE_SAMPLE_SPACING
            )

            # KEINE Transformation mehr - coords sind bereits lokal!
            centerline_3d = centerline_resampled

            # Speichere Vertices und Kanten
            start_idx = len(centerline_vertices)
            centerline_vertices.extend(centerline_3d)

            # Erstelle Kanten zwischen aufeinanderfolgenden Vertices
            for i in range(len(centerline_3d) - 1):
                centerline_edges.append([start_idx + i, start_idx + i + 1])

            # Erzeuge Suchkreise (konfigurierter Radius, 16 Segmente)
            for v in centerline_3d:
                circle_pts = _create_search_circle(
                    v, radius=_CENTERLINE_SEARCH_RADIUS, segments=16
                )
                c_start = len(circle_vertices)
                circle_vertices.extend(circle_pts)
                for j in range(len(circle_pts)):
                    nxt = (j + 1) % len(circle_pts)
                    circle_edges.append([c_start + j, c_start + nxt])

        except Exception:
            continue

    if not centerline_vertices:
        print(f"  Info: Keine gültigen Centerline-Vertices gefunden")
        return

    # Schreibe MTL-Datei
    mtl_filename = filename.replace(".obj", ".mtl")
    with open(mtl_filename, "w") as f:
        f.write("# Centerlines / Search Circles Material Library\n")
        f.write("newmtl centerline\n")
        f.write("Ka 1.0 0.0 1.0\n")  # Ambient: Magenta
        f.write("Kd 1.0 0.0 1.0\n")  # Diffuse: Magenta
        f.write("Ks 0.5 0.0 0.5\n")  # Specular: Dark Magenta
        f.write("d 1.0\n")  # Opacity: Full
        f.write("illum 2\n")
        f.write("\n")
        f.write("newmtl search_circle\n")
        f.write("Ka 0.0 1.0 1.0\n")  # Ambient: Cyan
        f.write("Kd 0.0 1.0 1.0\n")  # Diffuse: Cyan
        f.write("Ks 0.0 0.5 0.5\n")  # Specular: darker Cyan
        f.write("d 0.4\n")  # Etwas transparent
        f.write("illum 2\n")

    # Schreibe OBJ
    with open(filename, "w") as f:
        f.write("# Centerlines + Search Circles OBJ\n")
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")
        f.write("usemtl centerline\n\n")

        # Vertices: Centerlines + Kreise
        all_vertices = centerline_vertices + circle_vertices
        for v in all_vertices:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")

        f.write("\n")

        # Centerlines
        f.write("usemtl centerline\n")
        for edge in centerline_edges:
            f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")

        f.write("\nusemtl search_circle\n")
        offset = len(centerline_vertices)
        for edge in circle_edges:
            f.write(f"l {edge[0] + 1 + offset} {edge[1] + 1 + offset}\n")

    print(
        f"  ✓ {filename}: {len(centerline_vertices)} centerline vertices, {len(centerline_edges)} centerline edges, {len(circle_vertices)} circle vertices, {len(circle_edges)} circle edges"
    )
