"""
Analysiert die Faces in beamng.obj und sieht nach,
welche Vertices in den neuen Faces verwendet werden.
"""

import numpy as np
import re

# Lese die OBJ-Datei
print("[*] Lese beamng.obj...")
vertices = []
faces = []

with open("beamng.obj", "r") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "v":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            # Face: kann "v", "v/vt" oder "v/vt/vn" sein
            face_indices = []
            for part in parts[1:]:
                # Nimm nur die Vertex-Nummer (vor dem ersten /)
                vertex_idx = int(part.split("/")[0]) - 1  # OBJ ist 1-indexed
                face_indices.append(vertex_idx)
            faces.append(face_indices)

vertices = np.array(vertices)
print(f"[+] {len(vertices)} Vertices, {len(faces)} Faces geladen")

# Die gesnappten Vertices sind bei Index 87027, 86751, 86752, ...
gesnapped_indices = [
    87027,
    86751,
    86752,
    86753,
    86754,
    86755,
    87031,
    86756,
    87032,
    87033,
    86757,
    78546,
    78547,
]
print(f"\n[*] Gesnapped vertex indices: {gesnapped_indices}")

# Finde alle Faces, die gesnapped vertices enthalten
faces_with_gesnapped = []
for face_idx, face in enumerate(faces):
    has_gesnapped = any(v_idx in gesnapped_indices for v_idx in face)
    if has_gesnapped:
        faces_with_gesnapped.append(face_idx)

print(f"[+] {len(faces_with_gesnapped)} Faces mit gesnapped vertices")

# Zeige die ersten 5 Faces mit ihrer Z-Struktur
print(f"\n[*] Erste 5 Faces mit gesnapped vertices:")
for face_idx in faces_with_gesnapped[:5]:
    face = faces[face_idx]
    z_values = [vertices[v_idx, 2] for v_idx in face]
    gesnapped_mask = [v_idx in gesnapped_indices for v_idx in face]

    print(f"\n  Face {face_idx + 1}:")
    for v_idx, z_val, is_gesnapped in zip(face, z_values, gesnapped_mask):
        marker = " [GESNAPPED]" if is_gesnapped else " (interpoliert)"
        print(f"    v{v_idx+1} Z={z_val:.4f}{marker}")

# Prüfe ob die neuen Vertices (ab v86750) existieren und welche Z-Werte sie haben
print(f"\n[*] Prüfe neue/große Vertex-Indizes:")
# v86750 wäre Index 86749, aber das ist im OBJ vielleicht anders
# Schaue auf die hohen Indizes
max_vertex_idx = len(vertices)
print(f"[+] Max Vertex Index: {max_vertex_idx}")

# Zeige die letzten 30 Vertices
print(f"\n[*] Letzte 30 Vertices (höchste Indizes):")
for i in range(max(0, len(vertices) - 30), len(vertices)):
    print(f"  v{i+1}: Z = {vertices[i, 2]:.6f}")
