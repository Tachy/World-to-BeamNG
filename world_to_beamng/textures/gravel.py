"""
Prozedurale, kachelbare Kies-Textur für Flachdächer - wird einmalig erzeugt und in data/textures abgelegt (nicht bei
jedem Export): automatisch, falls sie fehlt (textures/registry.py), oder gezielt per tools/generate_gravel_texture.py.

Periodisches Voronoi-Muster: jede Zelle ist ein Kieselstein (Kuppel), die Fugen dazwischen sind dunkel.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

from .. import config
from ..facade.texture_utils import fbm, gray_to_rgb, normal_from_height, to_uint8
from . import library

_PEBBLE_SIZE_M = 0.014  # mittlerer Korndurchmesser (Kies 8-16 mm)
_PALETTE = ((150, 146, 138), (128, 124, 118), (170, 164, 152), (140, 134, 122), (112, 110, 106), (160, 150, 132))
_RELIEF_PX = 3.0


class GravelTextureGenerator:
    """Erzeugt Albedo, Normalmap und Roughness einer kachelbaren Kiesfläche."""

    def __init__(self, size_px: int = config.FLAT_ROOF_GRAVEL_TEXTURE_PX, repeat_m: float = config.FLAT_ROOF_GRAVEL_REPEAT_M):
        self._size = size_px
        self._pebbles_per_side = max(8, int(round(repeat_m / _PEBBLE_SIZE_M)))

    def generate(self, seed: Optional[int] = 4242) -> Dict[str, np.ndarray]:
        """
        Returns:
            {"albedo", "normal", "roughness"}: uint8-RGB-Bilder (size_px x size_px x 3)
        """
        rng = np.random.default_rng(seed)
        n, size = self._pebbles_per_side, self._size

        grid = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), axis=-1).reshape(-1, 2)
        centres = ((grid + rng.random(grid.shape)) / n) % 1.0
        tree = cKDTree(centres, boxsize=1.0)  # periodische Nachbarschaft = nahtlose Kachel

        coords = (np.arange(size) + 0.5) / size
        pixels = np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1).reshape(-1, 2)
        distance, nearest = tree.query(pixels, k=2, workers=-1)
        first, second, pebble = distance[:, 0], distance[:, 1], nearest[:, 0]

        spacing = 1.0 / n
        gap = np.clip((second - first) / (0.30 * spacing), 0.0, 1.0).reshape(size, size)
        dome = np.clip(1.0 - (first / (0.8 * spacing)) ** 2, 0.0, 1.0).reshape(size, size)

        palette = np.array(_PALETTE, dtype=np.float64) / 255.0
        color = palette[rng.integers(0, len(palette), len(centres))] * (0.75 + 0.5 * rng.random(len(centres)))[:, None]
        albedo = color[pebble].reshape(size, size, 3)
        albedo = albedo * ((0.30 + 0.70 * gap**0.7) * (0.85 + 0.25 * dome))[..., None] * (1.0 + 0.07 * fbm(size, size, rng))[..., None]

        height = dome * gap * _RELIEF_PX
        roughness = 0.82 + 0.10 * (1.0 - dome)
        return {"albedo": to_uint8(albedo), "normal": normal_from_height(height, 1.0), "roughness": gray_to_rgb(roughness)}


def generate_gravel_texture(library_dir: Optional[Path] = None, seed: int = 4242) -> Path:
    """
    Erzeugt die Kies-Textur und legt sie in der Textur-Bibliothek ab (data/textures/roof_gravel).

    Returns:
        Ordner der Textur
    """
    generated = GravelTextureGenerator().generate(seed=seed)
    maps = {"color": generated["albedo"], "normal": generated["normal"], "roughness": generated["roughness"]}
    return library.store_texture(
        config.FLAT_ROOF_GRAVEL_TEXTURE,
        maps,
        tile_m=config.FLAT_ROOF_GRAVEL_REPEAT_M,
        source=f"prozedural (textures/gravel.py), Seed {seed}, {config.FLAT_ROOF_GRAVEL_TEXTURE_PX} px",
        library_dir=library_dir,
    )
