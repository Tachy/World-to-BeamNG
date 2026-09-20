"""
Prozedurale, fugenlos kachelbare Putztextur.

Eine Struktur (Normalmap, Roughness, Helligkeitsverlauf) für alle Putzfarben; jede Farbe bekommt ihre eigene
Albedo-Textur aus derselben Struktur. Alle Ebenen sind periodisch: keine Kanten, keine Zellen.
"""

from typing import Dict

import numpy as np

from .. import config
from .facade_styles import PLASTER_COLORS
from .texture_utils import gray_to_rgb, normal_from_height, periodic_noise, to_uint8

PLASTER_VERSION = 1  # bei Änderungen am Muster erhöhen: erzwingt Neuerzeugen der Dateien


class PlasterTextureGenerator:
    """Erzeugt die Putztexturen (Rauputz: feines Korn, sehr zurückhaltende großflächige Schwankung)."""

    def __init__(self, size_px: int = config.FACADE_PLASTER_TEXTURE_PX):
        self._size = size_px

    def generate(self, seed: int = 7) -> Dict[str, object]:
        """
        Returns:
            {"albedo": {Farbname: uint8-RGB}, "normal": uint8-RGB, "roughness": uint8-RGB}
        """
        rng = np.random.default_rng(seed)
        n = self._size
        grain = periodic_noise(n, n, rng, 0.3)  # feines Korn
        mid = periodic_noise(n, n, rng, 1.6)
        low = periodic_noise(n, n, rng, 2.6)  # großflächig, bewusst schwach: die Wiederholung soll nicht auffallen

        tone = 1.0 + 0.045 * grain + 0.028 * mid + 0.015 * low
        height = 0.9 * grain + 0.6 * mid
        albedo = {
            color.name: to_uint8(np.clip(np.array(color.rgb) / 255.0 * tone[..., None], 0.0, 1.0)) for color in PLASTER_COLORS
        }
        roughness = 0.90 + 0.05 * grain
        return {"albedo": albedo, "normal": normal_from_height(height, 1.1), "roughness": gray_to_rgb(roughness)}
