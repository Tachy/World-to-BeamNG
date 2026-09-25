"""
Procedural, tileable concrete texture for bridge piers, tunnel walls/ceiling/portals and gallery roof/supports -
generated once and stored in data/textures (not on every export), automatically if it is missing
(textures/registry.py).

Smooth-formed surface: light gray noise (large blotches + fine grain) plus a fine formwork structure in
the normal map.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import config
from ..facade.texture_utils import fbm, gray_to_rgb, normal_from_height, periodic_noise, to_uint8
from . import library


class ConcreteTextureGenerator:
    """Generates albedo, normal map and roughness of a smooth-formed concrete surface."""

    def __init__(self, size_px: int = config.CONCRETE_TEXTURE_PX, repeat_m: float = config.CONCRETE_TEXTURE_TILE_M):
        self._size = size_px
        self._repeat_m = repeat_m

    def generate(self, seed: Optional[int] = 8181) -> Dict[str, np.ndarray]:
        """
        Returns:
            {"albedo", "normal", "roughness"}: uint8 RGB images (size_px x size_px x 3)
        """
        rng = np.random.default_rng(seed)
        size = self._size

        base_gray = 0.62 + 0.06 * fbm(size, size, rng, betas=(1.3, 2.2, 3.2), weights=(0.5, 0.3, 0.2))
        stain = 0.05 * periodic_noise(size, size, rng, beta=0.6)  # larger, soft water stains
        gray = np.clip(base_gray + stain, 0.0, 1.0)

        albedo = gray_to_rgb(gray).astype(np.float64) / 255.0
        albedo = albedo * (0.92 + 0.08 * fbm(size, size, rng, betas=(4.0,), weights=(1.0,)))[..., None]  # fine grain

        height = 0.4 * periodic_noise(size, size, rng, beta=3.0)  # fine formwork structure
        roughness = np.full((size, size), 0.85)

        return {"albedo": to_uint8(albedo), "normal": normal_from_height(height, 0.6), "roughness": gray_to_rgb(roughness)}


def generate_concrete_texture(library_dir: Optional[Path] = None, seed: int = 8181) -> Path:
    """
    Generates the concrete texture and stores it in the texture library (data/textures/tunnel_concrete).

    Returns:
        Folder of the texture
    """
    generated = ConcreteTextureGenerator().generate(seed=seed)
    maps = {"color": generated["albedo"], "normal": generated["normal"], "roughness": generated["roughness"]}
    return library.store_texture(
        config.CONCRETE_TEXTURE_NAME,
        maps,
        tile_m=config.CONCRETE_TEXTURE_TILE_M,
        source=f"procedural (textures/concrete.py), seed {seed}, {config.CONCRETE_TEXTURE_PX} px",
        library_dir=library_dir,
    )
