"""
Prozedurale, kachelbare Stahl-Textur für Brücken-Geländer (Pfosten + Handlauf) - wird einmalig erzeugt und in
data/textures abgelegt (nicht bei jedem Export), automatisch falls sie fehlt (textures/registry.py).

Verzinkter/lackierter Stahl: mittleres Grau mit feinem, grob gebürstetem Schliff und geringerer Rauheit als
Beton (glänzender).
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import config
from ..facade.texture_utils import fbm, gray_to_rgb, normal_from_height, periodic_noise, to_uint8
from . import library


class RailingTextureGenerator:
    """Erzeugt Albedo, Normalmap und Roughness einer verzinkten Stahlfläche (Brücken-Geländer)."""

    def __init__(self, size_px: int = config.RAILING_TEXTURE_PX, repeat_m: float = config.RAILING_TEXTURE_TILE_M):
        self._size = size_px
        self._repeat_m = repeat_m

    def generate(self, seed: Optional[int] = 3131) -> Dict[str, np.ndarray]:
        """
        Returns:
            {"albedo", "normal", "roughness"}: uint8-RGB-Bilder (size_px x size_px x 3)
        """
        rng = np.random.default_rng(seed)
        size = self._size

        base_gray = 0.5 + 0.04 * fbm(size, size, rng, betas=(1.5, 2.5, 4.0), weights=(0.4, 0.35, 0.25))
        streaks = 0.03 * periodic_noise(size, size, rng, beta=1.0)  # grober gebürsteter Schliff
        gray = np.clip(base_gray + streaks, 0.0, 1.0)

        albedo = gray_to_rgb(gray).astype(np.float64) / 255.0
        albedo = albedo * (0.95 + 0.05 * fbm(size, size, rng, betas=(5.0,), weights=(1.0,)))[..., None]  # feine Körnung

        height = 0.15 * periodic_noise(size, size, rng, beta=4.0)  # feiner Schliff in der Normalmap
        roughness = np.full((size, size), 0.45)  # glänzender als Beton

        return {"albedo": to_uint8(albedo), "normal": normal_from_height(height, 0.3), "roughness": gray_to_rgb(roughness)}


def generate_railing_texture(library_dir: Optional[Path] = None, seed: int = 3131) -> Path:
    """
    Erzeugt die Stahl-Textur und legt sie in der Textur-Bibliothek ab (data/textures/bridge_railing).

    Returns:
        Ordner der Textur
    """
    generated = RailingTextureGenerator().generate(seed=seed)
    maps = {"color": generated["albedo"], "normal": generated["normal"], "roughness": generated["roughness"]}
    return library.store_texture(
        config.RAILING_TEXTURE_NAME,
        maps,
        tile_m=config.RAILING_TEXTURE_TILE_M,
        source=f"prozedural (textures/steel.py), Seed {seed}, {config.RAILING_TEXTURE_PX} px",
        library_dir=library_dir,
    )
