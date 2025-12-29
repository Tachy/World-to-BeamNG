"""
OBJ-Export Funktionen (unified und layer-based).
"""

import os
import numpy as np
import pyvista as pv


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
