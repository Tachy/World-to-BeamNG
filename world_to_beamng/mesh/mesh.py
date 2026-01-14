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

        # Indexed UV-System (wie Vertices)
        self.uvs = []  # Globale UV-Liste: [(u0, v0), (u1, v1), ...]
        self.uv_indices = {}  # face_idx -> [uv_idx0, uv_idx1, uv_idx2]
        self._uv_lookup = {}  # Deduplication: (u, v) -> uv_idx

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

        OPTIMIERUNG: Cross-Product Z-Komponente nur, keine volle Normal-Berechnung.
        UV-System: Indexed UVs mit Deduplication für minimalen Memory-Footprint.
        """
        # Hole Vertex-Positionen (Cache-freundlich: 3 aufeinanderfolgende Array-Zugriffe)
        verts = self.vertex_manager.vertices
        v0_pos, v1_pos, v2_pos = verts[v0], verts[v1], verts[v2]

        # Cross Product Z-Komponente (2D Determinante) - inline für Speed
        cross_z = (v1_pos[0] - v0_pos[0]) * (v2_pos[1] - v0_pos[1]) - (v1_pos[1] - v0_pos[1]) * (v2_pos[0] - v0_pos[0])

        # Winding-Order Korrektur (wenn Z <= 0 → Normal zeigt nach unten)
        if cross_z <= 0:
            v1, v2 = v2, v1
            # UV-Swap wenn vorhanden (optimiert: nur wenn nötig)
            if uv_coords:
                # Direkt swap statt neue Dict-Creation
                temp = uv_coords.get(v1)
                uv_coords[v1] = uv_coords.get(v2)
                uv_coords[v2] = temp

        # Speichere Face
        face_idx = len(self.faces)
        self.faces.append([v0, v1, v2])
        self.face_props[face_idx] = {"material": material, **props}

        # UV-System: Indexed UVs (optimiert)
        if uv_coords:
            try:
                uv_idx0 = self.add_uv(*uv_coords[v0])
                uv_idx1 = self.add_uv(*uv_coords[v1])
                uv_idx2 = self.add_uv(*uv_coords[v2])
                self.uv_indices[face_idx] = [uv_idx0, uv_idx1, uv_idx2]
            except (KeyError, TypeError):
                pass  # Kein UV wenn unvollständig

        # Update Stats (inline increment)
        self.material_counts[material] += 1

        return face_idx

    def add_uv(self, u, v, deduplicate=True):
        """
        Füge UV-Koordinaten zum globalen UV-Pool hinzu.

        Args:
            u, v: UV-Koordinaten
            deduplicate: Bei True werden identische UVs dedupliziert (Standard)

        Returns:
            uv_idx: Index der UV-Koordinate
        """
        if deduplicate:
            # Runde EINMAL (nicht zweimal wie vorher)
            uv_rounded = (round(u, 6), round(v, 6))
            existing_idx = self._uv_lookup.get(uv_rounded)
            if existing_idx is not None:
                return existing_idx
            # Neu: Speichere gerundete Werte direkt
            uv_idx = len(self.uvs)
            self.uvs.append(uv_rounded)
            self._uv_lookup[uv_rounded] = uv_idx
            return uv_idx
        else:
            # Ohne Dedup: direkt append
            uv_idx = len(self.uvs)
            self.uvs.append((u, v))
            return uv_idx

    def set_face_uv_indices(self, face_idx, uv_idx0, uv_idx1, uv_idx2):
        """
        Setze UV-Indizes für ein Face.

        Args:
            face_idx: Face-Index
            uv_idx0, uv_idx1, uv_idx2: Indizes in self.uvs
        """
        self.uv_indices[face_idx] = [uv_idx0, uv_idx1, uv_idx2]

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
        new_uv_indices = {}
        old_to_new_idx = {}  # Mapping: alter Index -> neuer Index

        new_idx = 0
        for old_idx, face in enumerate(self.faces):
            if old_idx not in face_indices_set:
                new_faces.append(face)
                new_face_props[new_idx] = self.face_props.get(old_idx, {})
                new_uv_indices[new_idx] = self.uv_indices.get(old_idx, None)
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1

        # Ersetze alte Daten
        self.faces = new_faces
        self.face_props = new_face_props
        self.uv_indices = new_uv_indices

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

    def preserve_road_uvs(self):
        """
        Speichere Road-UV-Daten BEVOR Terrain-UVs berechnet werden.
        
        Roads haben bereits UV-Koordinaten aus road_mesh.py, die müssen bewahrt werden.
        Dieses System speichert die tatsächlichen UV-Koordinaten pro Road-Face,
        damit sie später in tile_slicer korrekt remapped werden können.
        
        Returns:
            dict: {face_idx: [(u0,v0), (u1,v1), (u2,v2)]} für alle Road-Faces
        """
        road_uv_data = {}
        road_faces_with_mat = 0
        road_faces_in_uv_indices = 0
        
        for face_idx, props in self.face_props.items():
            mat = props.get("material", "")
            # Prüfe ob Material mit "road" anfängt oder enthält
            if mat and ("road" in mat.lower() or "asphalt" in mat.lower() or "gravel" in mat.lower() or "dirt" in mat.lower()):
                road_faces_with_mat += 1
                # Hole die tatsächlichen UV-Koordinaten für dieses Face
                if face_idx in self.uv_indices:
                    road_faces_in_uv_indices += 1
                    uv_indices = self.uv_indices[face_idx]
                    uv_coords = []
                    for uv_idx in uv_indices:
                        if uv_idx < len(self.uvs):
                            uv_coords.append(self.uvs[uv_idx])
                        else:
                            uv_coords.append((0.0, 0.0))  # Fallback
                    
                    if len(uv_coords) == 3:
                        road_uv_data[face_idx] = uv_coords
        
        print(f"  [DEBUG] Road-Material-Faces: {road_faces_with_mat}")
        print(f"  [DEBUG] Road-Faces IN uv_indices: {road_faces_in_uv_indices}")
        print(f"  [INFO] {len(road_uv_data)} Road-Faces mit existierenden UVs identifiziert")
        return road_uv_data

    def compute_terrain_uvs_batch(self, material_filter=None, preserve_road_uv_data=None):
        """
        Berechnet UVs nur für Faces ohne existierende UVs mit bestimmtem Material (vektorisiert).

        Verwendet Vertex-XY-Positionen normalisiert auf Terrain-Bounds für 1:1 UV-Mapping.
        Respektiert bereits vorhandene UVs (z.B. von Roads) - überschreibt sie NICHT!

        Args:
            material_filter: Liste von Materials (z.B. ["terrain"]) oder None für alle
            preserve_road_uv_data: dict von preserve_road_uvs() - speichert Road-UV-Daten

        Returns:
            Anzahl Faces mit neuen UVs versehen
        """
        if len(self.faces) == 0:
            return 0

        # Filtere Faces nach Material
        # WICHTIG: Überspringe Faces die bereits UV-Indizes haben!
        face_indices = []
        for face_idx, props in self.face_props.items():
            mat = props.get("material")
            if material_filter is None or mat in material_filter:
                # Nur Faces ohne existierende UVs bearbeiten!
                # (oder auch wenn sie spezielle Road-UVs haben - diese müssen bewahrt werden)
                if preserve_road_uv_data and face_idx in preserve_road_uv_data:
                    # Road-Face mit speziellen UVs - überspringen
                    continue
                if face_idx not in self.uv_indices:
                    face_indices.append(face_idx)

                if face_idx not in self.uv_indices:
                    face_indices.append(face_idx)

        if not face_indices:
            return 0

        print(f"    Berechne UVs für {len(face_indices)} Faces ohne existierende UVs (batch-optimiert)...")

        # Hole alle Vertex-Positionen
        verts = np.asarray(self.vertex_manager.get_array(), dtype=np.float32)

        # Sammle alle relevanten Vertex-Indizes
        relevant_vert_indices = set()
        for face_idx in face_indices:
            relevant_vert_indices.update(self.faces[face_idx])

        # Berechne Bounds aus Vertex-XY-Positionen (vektorisiert)
        relevant_verts_xy = verts[list(relevant_vert_indices)][:, :2]
        x_min, y_min = relevant_verts_xy.min(axis=0)
        x_max, y_max = relevant_verts_xy.max(axis=0)

        x_range = float(x_max - x_min) if x_max > x_min else 1.0
        y_range = float(y_max - y_min) if y_max > y_min else 1.0

        # Berechne UVs für ALLE Vertices auf einmal (vektorisiert)
        all_uvs_x = (verts[:, 0] - x_min) / x_range
        all_uvs_y = (verts[:, 1] - y_min) / y_range

        # === BATCH-OPTIMIERUNG: Alle UVs pre-compute + deduplicate ===
        # Runde alle UVs zu float16 für Deduplication
        all_uvs_x_f16 = np.float16(all_uvs_x)
        all_uvs_y_f16 = np.float16(all_uvs_y)

        # Erstelle Lookup-Table: (u,v) → uv_idx (nur für relevante Vertices)
        uv_lookup_batch = {}  # (u_f16, v_f16) → uv_idx
        uv_mapping = {}  # vert_idx → uv_idx

        for vert_idx in relevant_vert_indices:
            u = all_uvs_x_f16[vert_idx]
            v = all_uvs_y_f16[vert_idx]
            uv_key = (u, v)

            # Nutze bestehende UV wenn vorhanden, sonst erstelle neue
            if uv_key not in uv_lookup_batch:
                uv_lookup_batch[uv_key] = len(self.uvs)
                self.uvs.append((float(u), float(v)))

            uv_mapping[vert_idx] = uv_lookup_batch[uv_key]

        # Setze uv_indices für alle Faces (schnell - nur Array-Lookup!)
        for face_idx in face_indices:
            v0, v1, v2 = self.faces[face_idx]
            self.uv_indices[face_idx] = [uv_mapping[v0], uv_mapping[v1], uv_mapping[v2]]

        print(f"    [OK] {len(face_indices)} Faces mit UVs versehen ({len(self.uvs)} unique UVs)")
        return len(face_indices)
