"""
Prozedurale, kachelbare Beton-Textur für Brücken-Pfeiler, Tunnel-Wände/-Decke/-Portale und Galerie-Dach/-Stützen -
wird einmalig erzeugt und in data/textures abgelegt (nicht bei jedem Export), automatisch falls sie fehlt
(textures/registry.py).

Schalglatte Fläche: leichtes Grau-Rauschen (große Flecken + feine Körnung) plus eine feine Schalungsstruktur in
der Normalmap.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .. import config
from ..facade.texture_utils import fbm, gray_to_rgb, normal_from_height, periodic_noise, to_uint8
from . import library


class ConcreteTextureGenerator:
    """Erzeugt Albedo, Normalmap und Roughness einer schalglatten Betonfläche."""

    def __init__(self, size_px: int = config.CONCRETE_TEXTURE_PX, repeat_m: float = config.CONCRETE_TEXTURE_TILE_M):
        self._size = size_px
        self._repeat_m = repeat_m

    def generate(self, seed: Optional[int] = 8181) -> Dict[str, np.ndarray]:
        """
        Returns:
            {"albedo", "normal", "roughness"}: uint8-RGB-Bilder (size_px x size_px x 3)
        """
        rng = np.random.default_rng(seed)
        size = self._size

        base_gray = 0.62 + 0.06 * fbm(size, size, rng, betas=(1.3, 2.2, 3.2), weights=(0.5, 0.3, 0.2))
        stain = 0.05 * periodic_noise(size, size, rng, beta=0.6)  # größere, weiche Wasserflecken
        gray = np.clip(base_gray + stain, 0.0, 1.0)

        albedo = gray_to_rgb(gray).astype(np.float64) / 255.0
        albedo = albedo * (0.92 + 0.08 * fbm(size, size, rng, betas=(4.0,), weights=(1.0,)))[..., None]  # feine Körnung

        height = 0.4 * periodic_noise(size, size, rng, beta=3.0)  # feine Schalungsstruktur
        roughness = np.full((size, size), 0.85)

        return {"albedo": to_uint8(albedo), "normal": normal_from_height(height, 0.6), "roughness": gray_to_rgb(roughness)}


def generate_concrete_texture(library_dir: Optional[Path] = None, seed: int = 8181) -> Path:
    """
    Erzeugt die Beton-Textur und legt sie in der Textur-Bibliothek ab (data/textures/tunnel_concrete).

    Returns:
        Ordner der Textur
    """
    generated = ConcreteTextureGenerator().generate(seed=seed)
    maps = {"color": generated["albedo"], "normal": generated["normal"], "roughness": generated["roughness"]}
    return library.store_texture(
        config.CONCRETE_TEXTURE_NAME,
        maps,
        tile_m=config.CONCRETE_TEXTURE_TILE_M,
        source=f"prozedural (textures/concrete.py), Seed {seed}, {config.CONCRETE_TEXTURE_PX} px",
        library_dir=library_dir,
    )
