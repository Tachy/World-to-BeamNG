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
def save_unified_obj(
    filename, vertices, road_faces, slope_faces, terrain_faces, junction_faces=None
):
    """Speichert ein einheitliches Terrain-Mesh mit integrierten Strassen und Junctions."""

    if junction_faces is None:
        junction_faces = np.array([], dtype=np.int32)

    mtl_filename = filename.replace(".obj", ".mtl")

    # Erstelle MTL-Datei
    with open(mtl_filename, "w") as f:
        f.write("# Material Library for BeamNG Terrain\n")
        f.write("# Auto-generated\n\n")

        # Strassenoberfläche (Asphalt)
        f.write("newmtl road_surface\n")
        f.write("Ns 50.000000\n")
        f.write("Ka 0.200000 0.200000 0.200000\n")
        f.write("Kd 0.300000 0.300000 0.300000\n")
        f.write("Ks 0.500000 0.500000 0.500000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n\n")

        # Boeschungen (Erde)
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
        f.write("illum 2\n\n")

        # Junction-Quads (Gruen fuer Sichtbarkeit)
        f.write("newmtl junction_quad_surface\n")
        f.write("Ns 50.000000\n")
        f.write("Ka 0.200000 0.300000 0.200000\n")
        f.write("Kd 0.400000 0.700000 0.400000\n")
        f.write("Ks 0.300000 0.300000 0.300000\n")
        f.write("Ni 1.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")

    # Erstelle OBJ-Datei
    print(f"\nSchreibe OBJ-Datei: {filename}")
    with open(
        filename, "w", buffering=64 * 1024 * 1024
    ) as f:  # 64MB Buffer fuer Ultra-Speed!
        f.write("# BeamNG Unified Terrain Mesh with integrated roads and junctions\n")
        f.write(f"# Generated from DGM1 data and OSM\n")
        f.write(f"mtllib {os.path.basename(mtl_filename)}\n\n")

        # ULTRA-SCHNELLE Methode: Formatiere alles mit NumPy, schreibe als einen Block
        def write_faces_fast(file, faces, prefix):
            """Schreibt Faces 10x schneller als np.savetxt durch Bulk-String-Ops."""
            if len(faces) == 0:
                return
            # Erstelle Strings fuer alle 3 Spalten separat
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

        # Strassen-Objekt
        print(f"  Schreibe {len(road_faces)} Strassen-Faces (ULTRA-FAST)...")
        f.write("\no road_surface\n")
        f.write("usemtl road_surface\n")
        write_faces_fast(f, road_faces, "f ")

        # Junction-Quads mit eigenem Material (fuer sichtbare Visualisierung)
        if junction_faces is not None and len(junction_faces) > 0:
            print(
                f"  Schreibe {len(junction_faces)} Junction-Faces mit eigenem Material..."
            )
            f.write("\no junction_quad\n")
            f.write("usemtl junction_quad_surface\n")
            write_faces_fast(f, junction_faces, "f ")
            print(f"    [OK] Junction-Faces geschrieben!")

        # Boeschungs-Objekt
        print(f"  Schreibe {len(slope_faces)} Boeschungs-Faces (ULTRA-FAST)...")
        f.write("\no road_slope\n")
        f.write("usemtl road_slope\n")
        write_faces_fast(f, slope_faces, "f ")

        # Terrain-Objekt
        print(f"  Schreibe {len(terrain_faces)} Terrain-Faces (ULTRA-FAST)...")
        f.write("\no terrain\n")
        f.write("usemtl terrain\n")
        write_faces_fast(f, terrain_faces, "f ")

    print(f"  [OK] {filename} erfolgreich erstellt!")
    print(f"  [OK] {mtl_filename} erfolgreich erstellt!")


def save_roads_obj(filename, vertices, road_faces, road_face_to_idx):
    """Exportiert nur die Strassen als OBJ mit pro-Strasse-Material (road_<id>)."""

    if len(road_faces) == 0:
        print(f"  -> {filename}: Keine Strassen-Faces, ueberspringe")
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
        f"  [OK] {filename}: {len(used_vertices)} vertices, {len(road_faces)} faces, {len(remapped_by_road)} roads"
    )

def save_ebene1_roads(vertices, road_faces, road_face_to_idx):
    """Exportiert roads.obj als ebene1.obj"""
    save_roads_obj("ebene1.obj", vertices, road_faces, road_face_to_idx)
    print(f"  [OK] ebene1.obj: Roads exportiert")


def save_ebene2_centerlines_junctions(
    road_polygons,
    junctions,
    all_vertices=None,
    remesh_boundaries=None,
    radius_circles=None,
):
    """
    Exportiert Centerlines + Junction-Points als ebene2.obj.

    Args:
        road_polygons: Liste von Road-Dicts mit 'coords'
        junctions: Liste von Junction-Dicts mit 'position'
        all_vertices: Optional - Vertices aus VertexManager (für Konsistenz)
        remesh_boundaries: Optional Liste von Boundary-Pfaden (Nx3) für Debug
    """
    filename = "ebene2.obj"
    mtl_filename = "ebene2.mtl"

    # Sammle alle Centerline-Vertices und Edges
    centerline_vertices = []
    centerline_edges = []

    for road in road_polygons:
        coords = road.get("coords", [])
        if len(coords) < 2:
            continue

        start_idx = len(centerline_vertices)

        # Koordinaten sind bereits in polygon.py transformiert (UTM -> Local, einmalig!)
        for coord in coords:
            centerline_vertices.append([coord[0], coord[1], coord[2]])

        # Erstelle Edges (verbinde aufeinanderfolgende Punkte)
        for i in range(len(coords) - 1):
            centerline_edges.append([start_idx + i, start_idx + i + 1])

    # Junction-Points als separate Vertices (nach Centerlines)
    junction_offset = len(centerline_vertices)
    junction_vertices = []
    junction_labels = []

    # Remesh-Boundaries (optional) als Linienzug
    remesh_boundaries = remesh_boundaries or []
    boundary_vertices = []
    boundary_edges = []

    # Mehrere Search-Radius-Kreise (optional)
    # radius_circles: Liste von Dicts {"junction_idx": int|None, "circle": [[x,y,z], ...]}
    radius_circles = radius_circles or []
    radius_circle_vertices = []  # flache Liste aller Kreis-Punkte
    radius_circle_edges = []

    # Fallback: wenn keine Kreise übergeben, versuche letzten Debug-Kreis aus remesh_debug_data.json zu laden
    if not radius_circles:
        try:
            import json
            import os

            if os.path.exists("remesh_debug_data.json"):
                with open("remesh_debug_data.json", "r") as debug_f:
                    debug_data = json.load(debug_f)
                    circle = debug_data.get("search_radius_circle")
                    if circle:
                        radius_circles.append({"junction_idx": None, "circle": circle})
        except Exception:
            pass  # Ignoriere Fehler beim Laden

    for idx, junction in enumerate(junctions):
        pos = junction.get("position", junction.get("pos", None))
        if pos is None:
            continue
        # Konvertiere zu reinen Python floats (nicht NumPy scalars)
        x_val = float(pos[0])
        y_val = float(pos[1])
        z_val = float(pos[2])
        junction_vertices.append([x_val, y_val, z_val])
        junction_labels.append(idx)  # Speichere die ursprüngliche Junction-ID

    # Schreibe MTL-Datei
    with open(mtl_filename, "w") as f:
        f.write("# ebene2.mtl - Centerlines + Junctions\n\n")

        # Material für Centerlines (grün)
        f.write("newmtl centerline\n")
        f.write("Ka 0.0 0.5 0.0\n")
        f.write("Kd 0.0 0.8 0.0\n")
        f.write("Ks 0.1 0.1 0.1\n")
        f.write("d 1.0\n")
        f.write("illum 2\n\n")

        # Material für Junction-Points (rot)
        f.write("newmtl junction_point\n")
        f.write("Ka 0.5 0.0 0.0\n")
        f.write("Kd 1.0 0.0 0.0\n")
        f.write("Ks 0.3 0.3 0.3\n")
        f.write("d 1.0\n")
        f.write("illum 2\n\n")

        # Material für Remesh-Boundaries (blau)
        f.write("newmtl remesh_boundary\n")
        f.write("Ka 0.0 0.0 0.6\n")
        f.write("Kd 0.0 0.0 1.0\n")
        f.write("Ks 0.2 0.2 0.2\n")
        f.write("d 1.0\n")
        f.write("illum 2\n\n")

        # Material für Search-Radius-Kreis (gelb)
        f.write("newmtl search_radius\n")
        f.write("Ka 0.6 0.6 0.0\n")
        f.write("Kd 1.0 1.0 0.0\n")
        f.write("Ks 0.2 0.2 0.2\n")
        f.write("d 1.0\n")
        f.write("illum 2\n\n")

    # Schreibe OBJ-Datei
    with open(filename, "w") as f:
        f.write("# ebene2.obj - Centerlines + Junction Points\n")
        f.write(f"mtllib {mtl_filename}\n\n")

        # Centerline-Vertices
        for v in centerline_vertices:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")

        # Junction-Vertices
        for i, v in enumerate(junction_vertices):
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")

        # Boundary-Vertices (optional)
        if remesh_boundaries:
            boundary_vertex_offset = len(centerline_vertices) + len(junction_vertices)
            for path in remesh_boundaries:
                start_idx = boundary_vertex_offset + len(boundary_vertices)
                path_list = np.asarray(path, dtype=float)
                boundary_vertices.extend(path_list.tolist())
                n = len(path_list)
                if n >= 2:
                    for i in range(n):
                        v1 = start_idx + i
                        v2 = start_idx + (i + 1) % n
                        boundary_edges.append([v1, v2])
            for v in boundary_vertices:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")

        # Radius-Kreis-Vertices (optional, mehrere)
        radius_circle_offset = (
            len(centerline_vertices) + len(junction_vertices) + len(boundary_vertices)
        )
        if radius_circles:
            for circle_entry in radius_circles:
                circle = circle_entry.get("circle", [])
                start_idx = radius_circle_offset + len(radius_circle_vertices)
                for v in circle:
                    f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
                n_circle = len(circle)
                if n_circle >= 2:
                    for i in range(n_circle):
                        v1 = start_idx + i
                        v2 = start_idx + (i + 1) % n_circle
                        radius_circle_edges.append([v1, v2])
                radius_circle_vertices.extend(circle)

        f.write("\n")

        # Centerline-Edges (grün)
        f.write("usemtl centerline\n")
        for edge in centerline_edges:
            f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")

        f.write("\n")

        # Junction-Points als einzelne Vertices (rot)
        # Schreibe als "Point" (p) mit Junction-Nummer als Label-Kommentar
        # Format für mesh_viewer: "p <vertex_idx>  # Junction <junction_number>"
        f.write("usemtl junction_point\n")
        for i, label in enumerate(junction_labels):
            # vertex_idx = Index im OBJ-File (1-basiert)
            # i ist die Position in der junction_vertices Liste
            # junction_offset ist wo die Junction-Vertices anfangen
            vertex_idx = junction_offset + i + 1  # 1-basiert
            # Label-Kommentar wird vom mesh_viewer geparst und als Text-Label angezeigt
            f.write(f"p {vertex_idx}  # Junction {label}\n")

        # Boundary-Linien (blau)
        if boundary_edges:
            f.write("\nusemtl remesh_boundary\n")
            for edge in boundary_edges:
                f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")

        # Radius-Kreis (gelb)
        if radius_circle_edges:
            f.write("\nusemtl search_radius\n")
            for edge in radius_circle_edges:
                f.write(f"l {edge[0] + 1} {edge[1] + 1}\n")

    print(
        f"  [OK] {filename}: {len(centerline_vertices)} centerline vertices, "
        f"{len(centerline_edges)} edges, {len(junction_vertices)} junction points"
    )
