"""
Central management of all mesh vertices with automatic deduplication.
"""

import numpy as np


class VertexManager:
    """
    Manages all mesh vertices centrally with automatic deduplication.

    Prevents duplicate vertices within a defined tolerance and
    returns consistent global indices.

    OPTIMIZATION: Vertices are kept internally as a NumPy array for performance!
    """

    def __init__(self, tolerance=0.001):
        """
        Initializes the VertexManager.

        Args:
            tolerance: Minimum distance between vertices (in meters).
                      Vertices closer than this value are treated as identical.
        """
        # OPTIMIZATION: Vertices as a NumPy array instead of a list
        # Starts with capacity for ~1000 vertices, grows as needed
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.tolerance = tolerance
        self.tolerance_sq = tolerance * tolerance

        # Fast spatial hash: cell -> [vertex_indices]
        self.cell_size = tolerance
        self.spatial_hash = {}

    def add_vertex(self, x, y, z):
        """
        Adds a vertex or returns the index of an existing one.

        Args:
            x, y, z: Coordinates of the vertex

        Returns:
            int: Global index of the vertex (0-based)
        """
        new_point = np.array([x, y, z], dtype=np.float32)

        existing_idx = self._find_existing(new_point)
        if existing_idx is not None:
            return existing_idx

        new_idx = len(self.vertices)
        # OPTIMIZATION: Append to NumPy array via vstack instead of appending to a list
        self.vertices = np.vstack([self.vertices, new_point.reshape(1, 3)])
        self._add_to_hash(new_idx, new_point)
        return new_idx

    def add_vertices_direct_nohash(self, coords):
        """Adds many vertices without dedup and without hash/KDTree update (maximum speed).

        Only use when no dedup queries are needed afterwards and no overlaps
        with existing vertices are expected.
        """
        coords_arr = np.asarray(coords, dtype=np.float32)
        if coords_arr.size == 0:
            return []

        start_idx = len(self.vertices)
        # OPTIMIZATION: Single vstack instead of a loop
        if len(coords_arr.shape) == 1:
            coords_arr = coords_arr.reshape(1, -1)
        self.vertices = np.vstack([self.vertices, coords_arr])
        end_idx = start_idx + len(coords_arr)
        return list(range(start_idx, end_idx))

    # --- Internal helpers for the spatial hash ---
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
        # Check own and neighboring cells (3x3x3)
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
        Returns the number of vertices.

        Returns:
            int: Number of vertices
        """
        return len(self.vertices)

    def __len__(self):
        """Returns the number of vertices."""
        return len(self.vertices)

    def __repr__(self):
        return f"VertexManager({len(self.vertices)} vertices, tolerance={self.tolerance}m)"
