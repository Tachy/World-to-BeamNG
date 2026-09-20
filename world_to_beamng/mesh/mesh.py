"""
Mesh-Klasse: Verwaltung von Faces mit Properties.

Hält Referenz zum VertexManager und verwaltet:
- Faces (als [v0, v1, v2] Indices)
- Face-Properties (Material, Surface, Friction, etc.)
"""

from collections import defaultdict


class Mesh:
    """Verwaltet ein Mesh mit Vertices (via VertexManager) und Faces mit Properties."""

    def __init__(self, vertex_manager):
        """
        Initialisiere Mesh.

        Args:
            vertex_manager: VertexManager Instanz für Vertex-Verwaltung
        """
        self.vertex_manager = vertex_manager
        self.faces = []  # Liste von [v0, v1, v2]

        # Indexed UV-System (wie Vertices)
        self.uvs = []  # Globale UV-Liste: [(u0, v0), (u1, v1), ...]
        self.uv_indices = {}  # face_idx -> [uv_idx0, uv_idx1, uv_idx2]

        self.vertex_normals = None  # Wird von compute_smooth_normals gesetzt

        # Statistiken
        self.material_counts = defaultdict(int)

    def get_statistics(self):
        """Gebe Statistiken zurück."""
        return {
            "total_faces": len(self.faces),
            "total_vertices": self.vertex_manager.get_count(),
            "material_counts": dict(self.material_counts),
        }
