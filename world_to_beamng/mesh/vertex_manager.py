"""
Zentrale Verwaltung aller Mesh-Vertices mit automatischer Deduplication.
"""

import numpy as np
from scipy.spatial import cKDTree


class VertexManager:
    """
    Verwaltet alle Mesh-Vertices zentral mit automatischer Deduplizierung.

    Verhindert doppelte Vertices innerhalb einer definierten Toleranz und
    gibt konsistente globale Indices zurueck.

    OPTIMIZATION: Vertices werden intern als NumPy-Array gehalten für Performance!
    """

    def __init__(self, tolerance=0.001):
        """
        Initialisiert den VertexManager.

        Args:
            tolerance: Minimaler Abstand zwischen Vertices (in Metern).
                      Vertices näher als dieser Wert werden als identisch behandelt.
        """
        # OPTIMIZATION: Vertices als NumPy-Array statt Liste
        # Startet mit kapazität für ~1000 Vertices, wird bei Bedarf erweitert
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.tolerance = tolerance
        self.tolerance_sq = tolerance * tolerance

        # Schnelles räumliches Hash: cell -> [vertex_indices]
        self.cell_size = tolerance
        self.spatial_hash = {}

        # KDTree bleibt optional als Fallback (derzeit nicht genutzt)
        self.kdtree = None
        self.rebuild_threshold = 1000

    def add_vertex(self, x, y, z):
        """
        Fuegt einen Vertex hinzu oder gibt Index eines existierenden zurueck.

        Args:
            x, y, z: Koordinaten des Vertex

        Returns:
            int: Globaler Index des Vertex (0-basiert)
        """
        new_point = np.array([x, y, z], dtype=np.float32)

        existing_idx = self._find_existing(new_point)
        if existing_idx is not None:
            return existing_idx

        new_idx = len(self.vertices)
        # OPTIMIZATION: Append zu NumPy-Array durch vstack statt append zu Liste
        self.vertices = np.vstack([self.vertices, new_point.reshape(1, 3)])
        self._add_to_hash(new_idx, new_point)
        return new_idx

    def add_vertices_direct(self, xs, ys, zs):
        """
        Fuegt viele Vertices OHNE Deduplication hinzu (schnell).

        Hinweis: Nur nutzen, wenn sicher keine Überschneidung mit existierenden
        Vertices besteht (z. B. initialer Grid-Aufbau). Baut den KDTree am Ende
        neu auf, damit spätere Queries korrekt funktionieren.

        Returns: Liste globaler Indices (0-basiert)
        """
        start_idx = len(self.vertices)
        coords = np.column_stack([xs, ys, zs]).astype(np.float32)
        # OPTIMIZATION: vstack statt extend
        self.vertices = np.vstack([self.vertices, coords])
        for i, pt in enumerate(coords):
            self._add_to_hash(start_idx + i, pt)
        self._rebuild_kdtree()
        return list(range(start_idx, start_idx + len(coords)))

    def add_vertices_direct_nohash(self, coords):
        """Fuegt viele Vertices ohne Dedup und ohne Hash/KDTree-Update hinzu (maximale Speed).

        Nur nutzen, wenn danach keine Dedup-Queries mehr noetig sind und keine Überschneidungen
        zu bestehenden Vertices zu erwarten sind.
        """
        coords_arr = np.asarray(coords, dtype=np.float32)
        if coords_arr.size == 0:
            return []

        start_idx = len(self.vertices)
        # OPTIMIZATION: Single vstack statt Loop
        if len(coords_arr.shape) == 1:
            coords_arr = coords_arr.reshape(1, -1)
        self.vertices = np.vstack([self.vertices, coords_arr])
        end_idx = start_idx + len(coords_arr)
        return list(range(start_idx, end_idx))

    def add_vertices_bulk(self, coords):
        """Batch-Insert vieler Vertices ohne Dedup/Hash (ultra-schnell für >1000 Vertices).

        Dies ist die schnellste Methode für Road-Vertices ohne Überschneidungen.
        Nutze diese für das globale Vertex-Insert in road_mesh.py.
        """
        coords_arr = np.asarray(coords, dtype=np.float32)
        if coords_arr.size == 0:
            return []

        start_idx = len(self.vertices)
        if len(coords_arr.shape) == 1:
            coords_arr = coords_arr.reshape(1, -1)
        # MEGA-OPTIMIZATION: Einziger vstack für alle Vertices
        self.vertices = np.vstack([self.vertices, coords_arr])
        return np.arange(start_idx, start_idx + len(coords_arr), dtype=int).tolist()

    def add_vertices_batch_dedup(self, coords):
        """
        Fuegt viele Vertices mit Deduplication auf einmal hinzu (vektorisiert).

        Args:
            coords: Iterable von (x, y, z)

        Returns:
            list[int]: Globale Indices der eingefuegten/gefundenen Vertices
        """
        coords_arr = np.asarray(coords, dtype=np.float32)
        if coords_arr.size == 0:
            return []

        return self.add_vertices_batch_dedup_fast(coords_arr)

    def add_vertices_batch_dedup_fast(self, coords_arr):
        """Schneller deduplizierender Batch-Insert (ohne per-Vertex np.array-Kopien)."""
        coords_arr = np.asarray(coords_arr, dtype=np.float32)
        if coords_arr.size == 0:
            return []

        # Deduplizierung aktiv
        indices = []
        for coord in coords_arr:
            existing_idx = self._find_existing(coord)
            if existing_idx is not None:
                indices.append(existing_idx)
            else:
                new_idx = len(self.vertices)
                self.vertices = np.vstack([self.vertices, coord.reshape(1, 3)])
                self._add_to_hash(new_idx, coord)
                indices.append(new_idx)
        return indices

    # --- Interne Helfer fuer Spatial Hash ---
    def _cell_key(self, point):
        return (
            int(np.floor(point[0] / self.cell_size)),
            int(np.floor(point[1] / self.cell_size)),
            int(np.floor(point[2] / self.cell_size)),
        )

    def _add_to_hash(self, idx, point):
        key = self._cell_key(point)
        bucket = self.spatial_hash.get(key)
        if bucket is None:
            self.spatial_hash[key] = [idx]
        else:
            bucket.append(idx)

    def _find_existing(self, point):
        key = self._cell_key(point)
        px, py, pz = point
        # Pruefe eigene und Nachbarzellen (3x3x3)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                    bucket = self.spatial_hash.get(neighbor_key)
                    if not bucket:
                        continue
                    for vidx in bucket:
                        vx, vy, vz = self.vertices[vidx]
                        dist_sq = (vx - px) ** 2 + (vy - py) ** 2 + (vz - pz) ** 2
                        if dist_sq < self.tolerance_sq:
                            return int(vidx)
        return None

    def add_vertex_tuple(self, vertex_tuple):
        """
        Fuegt einen Vertex als Tuple hinzu.

        Args:
            vertex_tuple: (x, y, z) Tuple

        Returns:
            int: Globaler Index des Vertex
        """
        return self.add_vertex(vertex_tuple[0], vertex_tuple[1], vertex_tuple[2])

    def get_vertex(self, index):
        """
        Gibt Vertex an gegebenem Index zurueck.

        Args:
            index: Globaler Vertex-Index

        Returns:
            np.array: [x, y, z] Koordinaten
        """
        return self.vertices[index]

    def get_array(self):
        """
        Gibt alle Vertices als NumPy-Array zurueck.

        Returns:
            np.ndarray: (N, 3) Array mit allen Vertices
        """
        return self.vertices

    def get_count(self):
        """
        Gibt Anzahl der Vertices zurueck.

        Returns:
            int: Anzahl Vertices
        """
        return len(self.vertices)

    def _rebuild_kdtree(self):
        """Baut KDTree neu auf (derzeit optional)."""
        if len(self.vertices) > 0:
            vertex_array = np.array(self.vertices, dtype=np.float32)
            self.kdtree = cKDTree(vertex_array)

    def __len__(self):
        """Gibt Anzahl der Vertices zurueck."""
        return len(self.vertices)

    def __repr__(self):
        return f"VertexManager({len(self.vertices)} vertices, tolerance={self.tolerance}m)"
