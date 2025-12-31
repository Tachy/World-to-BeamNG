"""Debug-Script für Böschungs-Mesh-Analyse"""

# Lese OBJ-Datei
obj_file = "beamng.obj"

vertices = []
slope_faces = []
road_faces = []
terrain_faces = []

current_material = None

with open(obj_file, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("usemtl"):
            current_material = line.split()[1]
        elif line.startswith("f "):
            parts = line.split()[1:]
            face = [int(p.split("/")[0]) for p in parts]
            if current_material == "road_slope":
                slope_faces.append(face)
            elif current_material == "road_surface":
                road_faces.append(face)
            elif current_material == "terrain":
                terrain_faces.append(face)

print(f"Total Vertices: {len(vertices)}")
print(f"Road Faces: {len(road_faces)}")
print(f"Slope Faces: {len(slope_faces)}")
print(f"Terrain Faces: {len(terrain_faces)}")

# Analysiere die Vertex-Verteilung
if slope_faces:
    # Finde min/max Indices in Slope-Faces
    all_slope_indices = set()
    for face in slope_faces:
        all_slope_indices.update(face)

    min_slope_idx = min(all_slope_indices)
    max_slope_idx = max(all_slope_indices)

    print(f"\n=== Slope-Face Index-Bereich ===")
    print(f"Min Index: {min_slope_idx}")
    print(f"Max Index: {max_slope_idx}")
    print(f"Anzahl unique Slope-Vertices: {len(all_slope_indices)}")

    # Analysiere die ersten Slope-Vertices
    print(f"\n=== Erste 10 Slope-Vertices (ab Index {min_slope_idx}) ===")
    for i in range(min_slope_idx, min(min_slope_idx + 10, len(vertices) + 1)):
        v = vertices[i - 1]
        print(f"v{i}: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")

    print(f"\n=== Erste 10 Boeschungs-Faces ===")
    for i, face in enumerate(slope_faces[:10]):
        print(f"Face {i}: {face}")
        # Prüfe ob Indices gültig sind
        for idx in face:
            if idx < 1 or idx > len(vertices):
                print(f"  [!]️ Ungueltiger Index: {idx} (max: {len(vertices)})")
            else:
                v = vertices[idx - 1]
                print(f"  v{idx}: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")
        print()
else:
    print("\n[!]️ KEINE Boeschungs-Faces gefunden!")
