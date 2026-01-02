"""
Mesh-Klasse: Verwaltung von Faces mit Properties.

Hält Referenz zum VertexManager und verwaltet:
- Faces (als [v0, v1, v2] Indices)
- Face-Properties (Material, Surface, Friction, etc.)
"""

import numpy as np
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
        self.face_props = {}  # face_idx -> {material, surface, friction, ...}

        # Statistiken
        self.material_counts = defaultdict(int)

    def add_face(self, v0, v1, v2, material="terrain", **props):
        """
        Füge ein Face zum Mesh hinzu.

        Args:
            v0, v1, v2: Vertex-Indices
            material: Material-Typ (terrain, road, slope, etc.)
            **props: Beliebige zusätzliche Properties

        Returns:
            face_idx: Index des neuen Faces
        """
        face_idx = len(self.faces)
        self.faces.append([v0, v1, v2])

        # Speichere Properties
        self.face_props[face_idx] = {"material": material, **props}

        # Update Stats
        self.material_counts[material] += 1

        return face_idx

    def add_faces(self, faces_list, material="terrain", **props):
        """
        Füge mehrere Faces auf einmal hinzu.

        Args:
            faces_list: Liste von [v0, v1, v2]
            material: Material-Typ für alle Faces
            **props: Properties für alle Faces

        Returns:
            List von Face-Indices
        """
        face_indices = []
        for v0, v1, v2 in faces_list:
            idx = self.add_face(v0, v1, v2, material=material, **props)
            face_indices.append(idx)
        return face_indices

    def get_faces_by_property(self, prop_name, value):
        """
        Finde alle Faces mit einer bestimmten Property.

        Args:
            prop_name: Property-Name (z.B. "material")
            value: Gesuchter Wert

        Returns:
            List von Face-Indices
        """
        return [
            idx
            for idx, props in self.face_props.items()
            if props.get(prop_name) == value
        ]

    def get_faces_array(self, prop_name=None, value=None, dtype=np.int32):
        """
        Gebe Faces als NumPy Array zurück.

        Args:
            prop_name: Filter nach Property (optional)
            value: Filter-Wert (optional)
            dtype: NumPy dtype (default: int32)

        Returns:
            numpy array (n, 3) oder empty array
        """
        if prop_name is not None:
            face_indices = self.get_faces_by_property(prop_name, value)
            faces = [self.faces[i] for i in face_indices]
        else:
            faces = self.faces

        if len(faces) == 0:
            return np.empty((0, 3), dtype=dtype)

        return np.array(faces, dtype=dtype)

    def get_faces_with_materials(self, materials=None):
        """
        Gebe Faces mit ihren Materialien zurück.

        Args:
            materials: Filter auf bestimmte Materialien (optional)

        Returns:
            (faces_array, materials_list): Tuple
        """
        if materials is None:
            # Alle Faces
            face_indices = range(len(self.faces))
        else:
            # Nur bestimmte Materialien
            face_indices = []
            for material in materials:
                face_indices.extend(self.get_faces_by_property("material", material))

        faces = [self.faces[i] for i in sorted(face_indices)]
        materials_per_face = [
            self.face_props[i]["material"] for i in sorted(face_indices)
        ]

        faces_array = (
            np.array(faces, dtype=np.int32)
            if faces
            else np.empty((0, 3), dtype=np.int32)
        )

        return faces_array, materials_per_face

    def get_face_count(self):
        """Gebe Anzahl der Faces zurück."""
        return len(self.faces)

    def get_vertex_count(self):
        """Gebe Anzahl der Vertices zurück."""
        return self.vertex_manager.get_count()

    def get_vertices(self):
        """Gebe Vertex-Array zurück."""
        return np.asarray(self.vertex_manager.get_array())

    def get_statistics(self):
        """Gebe Statistiken zurück."""
        return {
            "total_faces": len(self.faces),
            "total_vertices": self.vertex_manager.get_count(),
            "material_counts": dict(self.material_counts),
        }

    def cleanup_duplicate_faces(self):
        """Entferne doppelte Faces (gleiches Set von Vertices)."""
        seen = set()
        unique_indices = []

        for idx, (v0, v1, v2) in enumerate(self.faces):
            # Normalisiere zu sortiertes Tuple (ignoriere Reihenfolge)
            key = tuple(sorted([v0, v1, v2]))

            if key not in seen:
                seen.add(key)
                unique_indices.append(idx)

        # Filtere Faces und Properties
        old_faces = self.faces
        old_props = self.face_props

        self.faces = [old_faces[i] for i in unique_indices]
        self.face_props = {
            new_idx: old_props[old_idx]
            for new_idx, old_idx in enumerate(unique_indices)
        }

        removed_count = len(old_faces) - len(self.faces)
        return removed_count
