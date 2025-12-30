import numpy as np


def get_obj_bounds(filename):
    vertices = []
    try:
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(filename, "r", encoding=encoding) as f:
                    for line in f:
                        if line.startswith("v "):
                            parts = line.strip().split()
                            vertices.append(
                                [float(parts[1]), float(parts[2]), float(parts[3])]
                            )
                break
            except (UnicodeDecodeError, ValueError):
                continue
    except FileNotFoundError:
        return None

    if not vertices:
        return None

    v = np.array(vertices)
    return {"min": v.min(axis=0), "max": v.max(axis=0), "count": len(vertices)}


mesh_bounds = get_obj_bounds("beamng.obj")
center_bounds = get_obj_bounds("debug_centerlines.obj")

print("=== MESH (beamng.obj) ===")
if mesh_bounds:
    print(f"Vertices: {mesh_bounds['count']}")
    print(f"Min: {mesh_bounds['min']}")
    print(f"Max: {mesh_bounds['max']}")
    print(f"Range X: {mesh_bounds['max'][0] - mesh_bounds['min'][0]:.1f}")
    print(f"Range Y: {mesh_bounds['max'][1] - mesh_bounds['min'][1]:.1f}")
    print(f"Range Z: {mesh_bounds['max'][2] - mesh_bounds['min'][2]:.1f}")
else:
    print("Nicht gefunden")

print("\n=== CENTERLINES (debug_centerlines.obj) ===")
if center_bounds:
    print(f"Vertices: {center_bounds['count']}")
    print(f"Min: {center_bounds['min']}")
    print(f"Max: {center_bounds['max']}")
    print(f"Range X: {center_bounds['max'][0] - center_bounds['min'][0]:.1f}")
    print(f"Range Y: {center_bounds['max'][1] - center_bounds['min'][1]:.1f}")
    print(f"Range Z: {center_bounds['max'][2] - center_bounds['min'][2]:.1f}")

    if mesh_bounds:
        scale_x = (mesh_bounds["max"][0] - mesh_bounds["min"][0]) / (
            center_bounds["max"][0] - center_bounds["min"][0]
        )
        scale_y = (mesh_bounds["max"][1] - mesh_bounds["min"][1]) / (
            center_bounds["max"][1] - center_bounds["min"][1]
        )
        print(f"\nMöglicher Scale Factor X: {scale_x:.4f}")
        print(f"Möglicher Scale Factor Y: {scale_y:.4f}")
else:
    print("Nicht gefunden")
