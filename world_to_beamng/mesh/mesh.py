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
        self.face_uvs = {}  # face_idx -> {v0: (u0, v0), v1: (u1, v1), v2: (u2, v2)} or None
        self.vertex_normals = None  # Wird von compute_smooth_normals gesetzt

        # Statistiken
        self.material_counts = defaultdict(int)

    def add_face(self, v0, v1, v2, material="terrain", uv_coords=None, **props):
        """
        Füge ein Face zum Mesh hinzu mit automatischer Winding-Order Korrektur.

        Args:
            v0, v1, v2: Vertex-Indices
            material: Material-Typ (terrain, road, slope, etc.)
            uv_coords: Optional dict mit UV-Koordinaten {v0: (u,v), v1: (u,v), v2: (u,v)}
            **props: Beliebige zusätzliche Properties

        Returns:
            face_idx: Index des neuen Faces

        HINWEIS: Die Winding-Order wird automatisch korrigiert, damit die Face-Normale
        nach oben zeigt (Z > 0). Dies ist zentral an einer Stelle implementiert und
        garantiert die Sichtbarkeit in BeamNG für alle Geometrie-Typen.

        OPTIMIERUNG: Verwendet nur Cross-Product Z-Komponente (keine Normalisierung nötig).
        Dadurch ~2-3x schneller als vollständige Normal-Berechnung.

        UV-MAPPING: Optional können UV-Koordinaten pro Vertex angegeben werden.
        Format: {v0: (u,v), v1: (u,v), v2: (u,v)}
        """
        # Hole Vertex-Positionen direkt (schnell, da bereits numpy-Arrays)
        vertices = self.vertex_manager.vertices

        # Berechne Cross Product Z-Komponente direkt (ohne volle Normal-Berechnung)
        # cross(edge1, edge2)[2] = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x)
        v0_pos = vertices[v0]
        v1_pos = vertices[v1]
        v2_pos = vertices[v2]

        # Cross Product Z-Komponente (2D determinante)
        cross_z = (v1_pos[0] - v0_pos[0]) * (v2_pos[1] - v0_pos[1]) - (v1_pos[1] - v0_pos[1]) * (v2_pos[0] - v0_pos[0])

        # Wenn Z <= 0, zeigt Normal nach unten → Vertausche v1 und v2
        # HINWEIS: Dies funktioniert auch wenn nur Z-Komponente negativ ist (schräge Flächen)
        if cross_z <= 0:
            v1, v2 = v2, v1  # Tausche für CCW Ordnung
            # Wenn UVs vorhanden, auch vertauschen
            if uv_coords is not None:
                uv_coords = {v0: uv_coords.get(v0), v1: uv_coords.get(v2), v2: uv_coords.get(v1)}

        # Speichere Face mit korrigierter Ordnung
        face_idx = len(self.faces)
        self.faces.append([v0, v1, v2])

        # Speichere Properties
        self.face_props[face_idx] = {"material": material, **props}

        # Speichere UV-Koordinaten (oder None, falls nicht vorhanden)
        self.face_uvs[face_idx] = uv_coords

        # Update Stats
        self.material_counts[material] += 1

        return face_idx

    def add_faces(self, faces_list, material="terrain", uv_list=None, **props):
        """
        Füge mehrere Faces auf einmal hinzu.

        Args:
            faces_list: Liste von [v0, v1, v2]
            material: Material-Typ für alle Faces
            uv_list: Optional Liste von UV-Dicts pro Face (parallel zu faces_list)
            **props: Properties für alle Faces

        Returns:
            List von Face-Indices
        """
        face_indices = []
        for idx, (v0, v1, v2) in enumerate(faces_list):
            uv = uv_list[idx] if uv_list and idx < len(uv_list) else None
            face_idx = self.add_face(v0, v1, v2, material=material, uv_coords=uv, **props)
            face_indices.append(face_idx)
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
        return [idx for idx, props in self.face_props.items() if props.get(prop_name) == value]

    def remove_faces(self, face_indices):
        """
        Entferne Faces aus dem Mesh.

        Args:
            face_indices: Set oder Liste von Face-Indices zum Entfernen

        Returns:
            Anzahl entfernter Faces
        """
        if not face_indices:
            return 0

        face_indices_set = set(face_indices)
        removed_count = len(face_indices_set)

        # Neue Listen ohne die zu löschenden Faces
        new_faces = []
        new_face_props = {}
        new_face_uvs = {}
        old_to_new_idx = {}  # Mapping: alter Index -> neuer Index

        new_idx = 0
        for old_idx, face in enumerate(self.faces):
            if old_idx not in face_indices_set:
                new_faces.append(face)
                new_face_props[new_idx] = self.face_props.get(old_idx, {})
                new_face_uvs[new_idx] = self.face_uvs.get(old_idx, None)
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1

        # Ersetze alte Daten
        self.faces = new_faces
        self.face_props = new_face_props
        self.face_uvs = new_face_uvs

        # Update Material-Counts
        self.material_counts.clear()
        for props in self.face_props.values():
            mat = props.get("material", "terrain")
            self.material_counts[mat] += 1

        return removed_count

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
        materials_per_face = [self.face_props[i]["material"] for i in sorted(face_indices)]

        faces_array = np.array(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32)

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

    def compute_missing_uvs(self, tile_bounds=None, material_whitelist=None):
        """
        Berechne UV-Koordinaten für alle Faces die keine haben.

        Nutzt planar XY-Projektion normalisiert auf Tile-Bounds oder Mesh-Bounds.

        Args:
            tile_bounds: Optional (x_min, x_max, y_min, y_max) - Wenn gesetzt, normalisiere auf diese Bounds
                        (wichtig für Multi-Tile UV-Konsistenz)
            material_whitelist: Optional Set von Material-Namen - nur diese Faces UVs berechnen

        Returns:
            Anzahl der berechneten UVs
        """
        vertices = self.vertex_manager.get_array()
        if vertices is None or len(vertices) == 0:
            return 0

        vertices_array = np.asarray(vertices)

        # Bestimme Bounds
        if tile_bounds is not None:
            x_min, x_max, y_min, y_max = tile_bounds
        else:
            x_min = vertices_array[:, 0].min()
            x_max = vertices_array[:, 0].max()
            y_min = vertices_array[:, 1].min()
            y_max = vertices_array[:, 1].max()

        # Normalisierungsfaktoren
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        # Sammle zu berechnende Faces vorab (Filter nach fehlenden UVs & Material)
        target_indices = []
        if material_whitelist is None:
            for fi in range(len(self.faces)):
                if self.face_uvs.get(fi) is None:
                    target_indices.append(fi)
        else:
            allowed = material_whitelist
            for fi in range(len(self.faces)):
                if self.face_uvs.get(fi) is None:
                    mat = self.face_props.get(fi, {}).get("material")
                    if mat in allowed:
                        target_indices.append(fi)

        if not target_indices:
            return 0

        faces_arr = np.asarray(self.faces, dtype=np.int32)
        idx_arr = np.array(target_indices, dtype=np.int32)
        face_vertices = faces_arr[idx_arr]

        # Hole alle Punkte der Ziel-Faces in einem Rutsch
        v0_idx = face_vertices[:, 0]
        v1_idx = face_vertices[:, 1]
        v2_idx = face_vertices[:, 2]

        p0 = vertices_array[v0_idx]
        p1 = vertices_array[v1_idx]
        p2 = vertices_array[v2_idx]

        # Normalisierte UVs (0..1)
        u0 = (p0[:, 0] - x_min) / x_range
        v0_uv = (p0[:, 1] - y_min) / y_range
        u1 = (p1[:, 0] - x_min) / x_range
        v1_uv = (p1[:, 1] - y_min) / y_range
        u2 = (p2[:, 0] - x_min) / x_range
        v2_uv = (p2[:, 1] - y_min) / y_range

        # Zurückschreiben in face_uvs
        for k, fi in enumerate(idx_arr.tolist()):
            self.face_uvs[fi] = {
                int(v0_idx[k]): (float(u0[k]), float(v0_uv[k])),
                int(v1_idx[k]): (float(u1[k]), float(v1_uv[k])),
                int(v2_idx[k]): (float(u2[k]), float(v2_uv[k])),
            }

        return len(idx_arr)

    def compute_smooth_normals(self):
        """
        Berechne geglättete Vertex-Normalen aus allen Faces (durch Mittelung der Face-Normalen).

        OPTIMIERT: Vollständig vektorisiert mit numpy.add.at für schnelle Akkumulation.
        ~100x schneller als Loop-Version für große Meshes.

        Returns:
            numpy array (n_vertices, 3) mit normalisierten Normalen
        """
        vertices = self.vertex_manager.get_array()
        if vertices is None or len(vertices) == 0 or len(self.faces) == 0:
            self.vertex_normals = np.empty((0, 3), dtype=np.float32)
            return self.vertex_normals

        vertices_array = np.asarray(vertices, dtype=np.float32)
        n_vertices = len(vertices_array)
        normals = np.zeros((n_vertices, 3), dtype=np.float32)

        # Konvertiere Faces zu numpy array
        faces_arr = np.array(self.faces, dtype=np.int32)

        # Hole alle Vertex-Indizes
        v0_idx = faces_arr[:, 0]
        v1_idx = faces_arr[:, 1]
        v2_idx = faces_arr[:, 2]

        # Hole alle Vertices auf einmal (vektorisiert)
        p0 = vertices_array[v0_idx]
        p1 = vertices_array[v1_idx]
        p2 = vertices_array[v2_idx]

        # Berechne alle Face-Normalen auf einmal (vektorisiert)
        edge1 = p1 - p0
        edge2 = p2 - p0
        face_normals = np.cross(edge1, edge2)

        # Normalisiere Face-Normalen (vektorisiert)
        face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)

        # Filter degenerierte Faces (Norm ~ 0)
        valid_mask = face_norms.squeeze() > 1e-12
        face_normals[valid_mask] /= face_norms[valid_mask]

        # Akkumuliere Face-Normalen zu Vertices (vektorisiert mit add.at)
        # Dies ist ~100x schneller als Python-Loop
        np.add.at(normals, v0_idx, face_normals)
        np.add.at(normals, v1_idx, face_normals)
        np.add.at(normals, v2_idx, face_normals)

        # Normalisiere alle Vertex-Normalen (vektorisiert)
        vertex_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        nonzero_mask = vertex_norms.squeeze() > 1e-12
        normals[nonzero_mask] /= vertex_norms[nonzero_mask]

        self.vertex_normals = normals
        return self.vertex_normals

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
        self.face_props = {new_idx: old_props[old_idx] for new_idx, old_idx in enumerate(unique_indices)}

        removed_count = len(old_faces) - len(self.faces)
        return removed_count
