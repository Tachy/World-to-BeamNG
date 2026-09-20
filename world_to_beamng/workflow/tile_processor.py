"""
Tile-Processing Logik.

Extrahiert die Tile-Lade und Verarbeitungslogik aus multitile.py.
"""

import zipfile
from world_to_beamng.logging_config import LoggerConfig
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from ..core.cache_manager import CacheManager

logger = LoggerConfig.get_logger()


class TileProcessor:
    """
    Verarbeitet einzelne DGM-Tiles.

    Verantwortlich für:
    - Laden von Höhendaten
    - Caching
    - Koordinaten-Transformation
    """

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def load_height_data(
        self, tile: Dict, tile_hash: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Lade Höhendaten einer DGM1-Kachel (mit Cache).

        Args:
            tile: Tile-Metadaten Dict
            tile_hash: Optional - Hash für Cache

        Returns:
            Tuple (height_points, height_elevations) oder (None, None)
        """
        filepath = tile.get("filepath")
        if not filepath or not Path(filepath).exists():
            logger.error(f"  [!] DGM1-Datei fehlt: {filepath}")
            return None, None

        # Hash berechnen
        if tile_hash is None:
            tile_hash = self.cache.hash_file(filepath)

        # Prüfe Cache
        cache_key = f"height_raw_{tile_hash}"
        cached = self.cache.get_npz(cache_key)
        if cached:
            return cached["points"], cached["elevations"]

        # Lade Datei
        logger.info(f"  [→] Lade DGM1: {Path(filepath).name}")
        points, elevations = self._load_from_zip(filepath)

        if points is None or elevations is None:
            return None, None

        # Speichere in Cache
        self.cache.set_npz(cache_key, points=points, elevations=elevations)

        return points, elevations

    def load_height_data_multi(self, tiles: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Lädt und kombiniert die Höhendaten mehrerer DGM1-Kacheln zu einer
        einzigen Punktwolke (vstack/hstack) - dieselbe Kombinationslogik wie
        beim Laden der 4 Sub-Kacheln innerhalb eines einzelnen DGM1-ZIPs
        (_load_from_zip), nur eine Ebene höher für mehrere ZIP-Dateien.

        Setzt voraus, dass die Kacheln einen lückenlosen, rechteckigen
        Bereich bilden (Nutzer-Verantwortung, siehe utils.tile_scanner).

        Args:
            tiles: Liste von Tile-Metadaten-Dicts (wie scan_lgl_tiles() sie liefert)

        Returns:
            Tuple (points, elevations) oder (None, None), falls eine Kachel fehlschlägt
        """
        all_points = []
        all_elevations = []

        for tile in tiles:
            points, elevations = self.load_height_data(tile)
            if points is None:
                logger.error(f"  [!] Höhendaten für {tile.get('filename')} fehlen - Gesamtfläche unvollständig")
                return None, None
            all_points.append(points)
            all_elevations.append(elevations)

        if not all_points:
            return None, None

        return np.vstack(all_points), np.hstack(all_elevations)

    def _load_from_zip(self, filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Lade Höhendaten aus ZIP.

        Ein LGL DGM1-ZIP enthält 4 XYZ-Dateien (2×2 Kacheln à 1000×1000m).

        Args:
            filepath: Pfad zum ZIP

        Returns:
            Tuple (points, elevations)
        """
        all_points = []
        all_elevations = []

        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                for fname in zf.namelist():
                    if fname.endswith(".xyz"):
                        with zf.open(fname) as f:
                            data = np.loadtxt(f, delimiter=" ", dtype=float)
                            if data.size == 0:
                                continue

                            points = data[:, :2]  # X, Y
                            elevations = data[:, 2]  # Z

                            all_points.append(points)
                            all_elevations.append(elevations)

        except Exception as e:
            logger.error(f"  [!] Fehler beim Laden von {filepath}: {e}")
            return None, None

        if not all_points:
            return None, None

        # Kombiniere alle Kacheln
        combined_points = np.vstack(all_points)
        combined_elevations = np.hstack(all_elevations)

        return combined_points, combined_elevations

    def ensure_local_offset(
        self, global_offset: Tuple[float, float], height_points: np.ndarray, height_elevations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transformiere globale Koordinaten zu lokalen (relativ zu global_offset).

        Args:
            global_offset: (origin_x, origin_y) globaler Offset
            height_points: N×2 Array mit Punkten
            height_elevations: N Array mit Höhen

        Returns:
            Tuple (lokale_points, elevations)
        """
        origin_x, origin_y = global_offset

        local_points = height_points.copy()
        local_points[:, 0] -= origin_x
        local_points[:, 1] -= origin_y

        return local_points, height_elevations
