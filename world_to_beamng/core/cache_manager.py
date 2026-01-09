"""
Zentraler Cache-Manager für alle Cache-Operationen.

Vereinfacht und zentralisiert Cache-Zugriffe.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Callable, Dict
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

    def get_json(self, key: str) -> Optional[Dict]:
        """
        Lade JSON aus Cache.

        Args:
            key: Cache-Key

        Returns:
            Dict oder None
        """
        path = self.get_path(key, ".json")
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_json(self, key: str, data: Dict):
        """
        Speichere JSON in Cache.

        Args:
            key: Cache-Key
            data: Zu speichernde Daten
        """
        path = self.get_path(key, ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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

    def get_pickle(self, key: str) -> Optional[Any]:
        """
        Lade Pickle aus Cache.

        Args:
            key: Cache-Key

        Returns:
            Beliebiges Objekt oder None
        """
        path = self.get_path(key, ".pkl")
        if not path.exists():
            return None

        with open(path, "rb") as f:
            return pickle.load(f)

    def set_pickle(self, key: str, obj: Any):
        """
        Speichere Objekt als Pickle.

        Args:
            key: Cache-Key
            obj: Zu speicherndes Objekt
        """
        path = self.get_path(key, ".pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any], cache_type: str = "json") -> Any:
        """
        Hole aus Cache oder berechne und speichere.

        Args:
            key: Cache-Key
            compute_fn: Funktion zur Berechnung wenn nicht im Cache
            cache_type: "json", "npz" oder "pickle"

        Returns:
            Gecachte oder neu berechnete Daten
        """
        # Prüfe Cache
        if cache_type == "json":
            cached = self.get_json(key)
            if cached is not None:
                return cached
        elif cache_type == "npz":
            cached = self.get_npz(key)
            if cached is not None:
                return cached
        elif cache_type == "pickle":
            cached = self.get_pickle(key)
            if cached is not None:
                return cached
        else:
            raise ValueError(f"Unknown cache_type: {cache_type}")

        # Berechne
        result = compute_fn()

        # Speichere
        if cache_type == "json":
            self.set_json(key, result)
        elif cache_type == "npz":
            self.set_npz(key, **result)
        elif cache_type == "pickle":
            self.set_pickle(key, result)

        return result

    def invalidate(self, pattern: str = "*"):
        """
        Lösche Cache-Einträge.

        Args:
            pattern: Glob-Pattern für zu löschende Dateien
        """
        for path in self.cache_dir.glob(pattern):
            if path.is_file():
                path.unlink()

    def clear_all(self):
        """Lösche gesamten Cache."""
        self.invalidate("*")

    @staticmethod
    def hash_file(filepath: str) -> str:
        """
        Erstelle Hash von Datei.

        Args:
            filepath: Pfad zur Datei

        Returns:
            MD5-Hash
        """
        if not os.path.exists(filepath):
            return ""

        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)

        return md5.hexdigest()[:12]

    @staticmethod
    def hash_string(text: str) -> str:
        """
        Erstelle Hash von String.

        Args:
            text: Zu hashender Text

        Returns:
            MD5-Hash
        """
        return hashlib.md5(text.encode()).hexdigest()[:12]
