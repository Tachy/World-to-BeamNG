"""
Procedural, seamlessly tileable plaster texture.

One structure (normal map, roughness, brightness variation) for all plaster colors; each color gets its own
albedo texture from the same structure. All layers are periodic: no edges, no cells.
"""

from typing import Dict

import numpy as np

from .. import config
from .facade_styles import PLASTER_COLORS
from .texture_utils import gray_to_rgb, normal_from_height, periodic_noise, to_uint8

PLASTER_VERSION = 1  # increase on pattern changes: forces regeneration of the files


class PlasterTextureGenerator:
    """Generates the plaster textures (rough plaster: fine grain, very subtle large-scale variation)."""

    def __init__(self, size_px: int = config.FACADE_PLASTER_TEXTURE_PX):
        self._size = size_px

    def generate(self, seed: int = 7) -> Dict[str, object]:
        """
        Returns:
            {"albedo": {color name: uint8-RGB}, "normal": uint8-RGB, "roughness": uint8-RGB}
        """
        rng = np.random.default_rng(seed)
        n = self._size
        grain = periodic_noise(n, n, rng, 0.3)  # fine grain
        mid = periodic_noise(n, n, rng, 1.6)
        low = periodic_noise(n, n, rng, 2.6)  # large-scale, deliberately weak: the repetition should not be noticeable

        tone = 1.0 + 0.045 * grain + 0.028 * mid + 0.015 * low
        height = 0.9 * grain + 0.6 * mid
        albedo = {
            color.name: to_uint8(np.clip(np.array(color.rgb) / 255.0 * tone[..., None], 0.0, 1.0)) for color in PLASTER_COLORS
        }
        roughness = 0.90 + 0.05 * grain
        return {"albedo": albedo, "normal": normal_from_height(height, 1.1), "roughness": gray_to_rgb(roughness)}
