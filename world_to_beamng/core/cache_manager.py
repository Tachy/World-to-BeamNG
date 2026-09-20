"""
Zentraler Cache-Manager für alle Cache-Operationen.

Vereinfacht und zentralisiert Cache-Zugriffe.
"""

import json
from pathlib import Path
from typing import Optional, Dict
import hashlib


class CacheManager:
    """
    Zentraler Cache-Manager.

    Features:
    - Get-or-compute Pattern
    - Multiple Cache-Backends (JSON, NPZ, Pickle)
    - Cache-Invalidierung
    - Hash-basierte Keys
    """

    def __init__(self, cache_dir: Path):
        """
        Initialisiere CacheManager.

        Args:
            cache_dir: Verzeichnis für Cache-Dateien
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str, extension: str = ".json") -> Path:
        """
        Hole Cache-Pfad für Key.

        Args:
            key: Cache-Key
            extension: Dateiendung

        Returns:
            Pfad zur Cache-Datei
        """
        return self.cache_dir / f"{key}{extension}"

    def exists(self, key: str, extension: str = ".json") -> bool:
        """
        Prüfe ob Cache-Eintrag existiert.

        Args:
            key: Cache-Key
            extension: Dateiendung

        Returns:
            True wenn Cache existiert
        """
        return self.get_path(key, extension).exists()

    def get_npz(self, key: str) -> Optional[Dict]:
        """
        Lade NPZ aus Cache.

        Args:
            key: Cache-Key

        Returns:
            Dict mit numpy arrays oder None
        """
        import numpy as np

        path = self.get_path(key, ".npz")
        if not path.exists():
            return None

        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def set_npz(self, key: str, **arrays):
        """
        Speichere NPZ in Cache.

        Args:
            key: Cache-Key
            **arrays: Benannte numpy arrays
        """
        import numpy as np

        path = self.get_path(key, ".npz")
        np.savez_compressed(path, **arrays)

    @staticmethod
    def hash_file(filepath: Path) -> str:
        """
        Erstelle Hash von Datei.

        Args:
            filepath: Pfad zur Datei

        Returns:
            MD5-Hash
        """
        if not filepath.exists():
            return ""

        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)

        return md5.hexdigest()[:12]
