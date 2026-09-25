"""
Central cache manager for all cache operations.

Simplifies and centralizes cache access.
"""

import json
from pathlib import Path
from typing import Optional, Dict
import hashlib


class CacheManager:
    """
    Central cache manager.

    Features:
    - Get-or-compute pattern
    - Multiple cache backends (JSON, NPZ, Pickle)
    - Cache invalidation
    - Hash-based keys
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize the CacheManager.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str, extension: str = ".json") -> Path:
        """
        Get the cache path for a key.

        Args:
            key: Cache key
            extension: File extension

        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{key}{extension}"

    def exists(self, key: str, extension: str = ".json") -> bool:
        """
        Check whether a cache entry exists.

        Args:
            key: Cache key
            extension: File extension

        Returns:
            True if the cache entry exists
        """
        return self.get_path(key, extension).exists()

    def get_npz(self, key: str) -> Optional[Dict]:
        """
        Load an NPZ from the cache.

        Args:
            key: Cache key

        Returns:
            Dict of numpy arrays or None
        """
        import numpy as np

        path = self.get_path(key, ".npz")
        if not path.exists():
            return None

        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def set_npz(self, key: str, **arrays):
        """
        Store an NPZ in the cache.

        Args:
            key: Cache key
            **arrays: Named numpy arrays
        """
        import numpy as np

        path = self.get_path(key, ".npz")
        np.savez_compressed(path, **arrays)

    @staticmethod
    def hash_file(filepath: Path) -> str:
        """
        Create a hash of a file.

        Args:
            filepath: Path to the file

        Returns:
            MD5 hash
        """
        if not filepath.exists():
            return ""

        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)

        return md5.hexdigest()[:12]
