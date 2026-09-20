"""
Zentrale Verwaltung aller Mesh-Vertices mit automatischer Deduplication.
"""

import numpy as np


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

    def get_count(self):
        """
        Gibt Anzahl der Vertices zurueck.

        Returns:
            int: Anzahl Vertices
        """
        return len(self.vertices)

    def __len__(self):
        """Gibt Anzahl der Vertices zurueck."""
        return len(self.vertices)

    def __repr__(self):
        return f"VertexManager({len(self.vertices)} vertices, tolerance={self.tolerance}m)"
