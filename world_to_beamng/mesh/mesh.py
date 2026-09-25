"""
Mesh class: management of faces with properties.

Holds a reference to the VertexManager and manages:
- Faces (as [v0, v1, v2] indices)
- Face properties (material, surface, friction, etc.)
"""

from collections import defaultdict


class Mesh:
    """Manages a mesh with vertices (via VertexManager) and faces with properties."""

    def __init__(self, vertex_manager):
        """
        Initialize the mesh.

        Args:
            vertex_manager: VertexManager instance for vertex management
        """
        self.vertex_manager = vertex_manager
        self.faces = []  # List of [v0, v1, v2]

        # Indexed UV system (like vertices)
        self.uvs = []  # Global UV list: [(u0, v0), (u1, v1), ...]
        self.uv_indices = {}  # face_idx -> [uv_idx0, uv_idx1, uv_idx2]

        self.vertex_normals = None  # Set by compute_smooth_normals

        # Statistics
        self.material_counts = defaultdict(int)

    def get_statistics(self):
        """Return statistics."""
        return {
            "total_faces": len(self.faces),
            "total_vertices": self.vertex_manager.get_count(),
            "material_counts": dict(self.material_counts),
        }
