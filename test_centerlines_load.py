#!/usr/bin/env python3
"""Test script to verify centerlines loading"""

import numpy as np

# Simuliere _load_centerlines Methode
obj_file = "debug_centerlines.obj"
vertices = []
edges = []
search_circle_vertices = []
search_circle_faces = []
search_radius = 7.0

try:
    for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            with open(obj_file, "r", encoding=encoding) as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        vertices.append(
                            [float(parts[1]), float(parts[2]), float(parts[3])]
                        )
                    elif line.startswith("l "):
                        parts = line.strip().split()[1:]
                        indices = [int(p.split("/")[0]) - 1 for p in parts]
                        for i in range(len(indices) - 1):
                            edges.append([indices[i], indices[i + 1]])

                print(f"[OK] Encoding {encoding} erfolgreich")
                print(f"  Vertices geladen: {len(vertices)}")
                print(f"  Edges geladen: {len(edges)}")

                # Konvertiere zu NumPy Arrays
                vertices_arr = np.array(vertices)
                print(f"  vertices_arr shape: {vertices_arr.shape}")
                print(f"  vertices_arr dtype: {vertices_arr.dtype}")
                print(f"  vertices_arr bounds:")
                print(
                    f"    X: [{vertices_arr[:, 0].min()}, {vertices_arr[:, 0].max()}]"
                )
                print(
                    f"    Y: [{vertices_arr[:, 1].min()}, {vertices_arr[:, 1].max()}]"
                )
                print(
                    f"    Z: [{vertices_arr[:, 2].min()}, {vertices_arr[:, 2].max()}]"
                )

                # Test Rendering-Code Simulation
                print(f"\n  Test rendering code:")
                centerline_cells = np.array(
                    [[2, edge[0], edge[1]] for edge in edges]
                ).flatten()
                print(f"  centerline_cells shape: {centerline_cells.shape}")
                print(f"  centerline_cells dtype: {centerline_cells.dtype}")
                print(f"  First 30 elements: {centerline_cells[:30]}")

                break
        except (UnicodeDecodeError, ValueError) as e:
            print(f"[x] Encoding {encoding} fehlgeschlagen: {e}")
            continue

except FileNotFoundError:
    print(f"[x] Datei {obj_file} nicht gefunden!")
