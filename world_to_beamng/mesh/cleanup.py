"""
Face Cleanup Utilities.
"""

import numpy as np


def cleanup_duplicate_faces(all_faces):
    """Entfernt doppelte Faces (gleiche Vertex-Tripel, unabhängig von Reihenfolge)."""
    if not all_faces:
        return []

    faces_np = np.array(all_faces, dtype=np.int64)

    # Sortiere Vertices in jedem Face für Vergleich
    faces_sorted = np.sort(faces_np, axis=1)

    # Finde eindeutige Faces
    _, unique_idx = np.unique(faces_sorted, axis=0, return_index=True)

    # Behalte Originale Reihenfolge (für korrekte Normalen)
    unique_faces = faces_np[np.sort(unique_idx)]

    num_removed = len(all_faces) - len(unique_faces)
    if num_removed > 0:
        print(f"  ✓ {num_removed} doppelte Faces entfernt")

    return unique_faces.tolist()
