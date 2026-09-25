"""
Procedural, tileable steel texture for bridge railings (posts + handrail) - generated once and stored in
data/textures (not on every export), automatically if it is missing (textures/registry.py).

Galvanized/painted steel: medium gray with a fine, coarsely brushed finish and lower roughness than
concrete (glossier).
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import config
from ..facade.texture_utils import fbm, gray_to_rgb, normal_from_height, periodic_noise, to_uint8
from . import library


class RailingTextureGenerator:
    """Generates albedo, normal map and roughness of a galvanized steel surface (bridge railing)."""

    def __init__(self, size_px: int = config.RAILING_TEXTURE_PX, repeat_m: float = config.RAILING_TEXTURE_TILE_M):
        self._size = size_px
        self._repeat_m = repeat_m

    def generate(self, seed: Optional[int] = 3131) -> Dict[str, np.ndarray]:
        """
        Returns:
            {"albedo", "normal", "roughness"}: uint8 RGB images (size_px x size_px x 3)
        """
        rng = np.random.default_rng(seed)
        size = self._size

        base_gray = 0.5 + 0.04 * fbm(size, size, rng, betas=(1.5, 2.5, 4.0), weights=(0.4, 0.35, 0.25))
        streaks = 0.03 * periodic_noise(size, size, rng, beta=1.0)  # coarse brushed finish
        gray = np.clip(base_gray + streaks, 0.0, 1.0)

        albedo = gray_to_rgb(gray).astype(np.float64) / 255.0
        albedo = albedo * (0.95 + 0.05 * fbm(size, size, rng, betas=(5.0,), weights=(1.0,)))[..., None]  # fine grain

        height = 0.15 * periodic_noise(size, size, rng, beta=4.0)  # fine finish in the normal map
        roughness = np.full((size, size), 0.45)  # glossier than concrete

        return {"albedo": to_uint8(albedo), "normal": normal_from_height(height, 0.3), "roughness": gray_to_rgb(roughness)}


def generate_railing_texture(library_dir: Optional[Path] = None, seed: int = 3131) -> Path:
    """
    Generates the steel texture and stores it in the texture library (data/textures/bridge_railing).

    Returns:
        Folder of the texture
    """
    generated = RailingTextureGenerator().generate(seed=seed)
    maps = {"color": generated["albedo"], "normal": generated["normal"], "roughness": generated["roughness"]}
    return library.store_texture(
        config.RAILING_TEXTURE_NAME,
        maps,
        tile_m=config.RAILING_TEXTURE_TILE_M,
        source=f"procedural (textures/steel.py), seed {seed}, {config.RAILING_TEXTURE_PX} px",
        library_dir=library_dir,
    )
