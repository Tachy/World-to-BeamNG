"""
Foto -> nahtlos kachelnde Textur samt Normal- und Roughness-Map (einmalig per tools/make_seamless_texture.py).

Ablauf: quadratisch zuschneiden, auf die Zielgröße skalieren, großflächige Beleuchtung ausgleichen, die Ränder über
eine um die halbe Kachel versetzte Kopie überblenden (Kachelnaht) und aus der Helligkeit Höhe und Rauheit ableiten
(dunkle Fugen = tief und rau).
"""

from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .. import config
from ..facade.texture_utils import gaussian_blur_wrap, gray_to_rgb, normal_from_height, to_uint8

LIGHTING_SIGMA_FRAC = 0.12  # Radius des Beleuchtungsausgleichs relativ zur Kachelgröße
DEFAULT_BLEND = 0.4  # Breite der Überblendung an den Rändern relativ zur Kachelgröße (höchstens 0.5)
NORMAL_STRENGTH_PER_1024 = 4.0  # Normalmap-Stärke bei 1024 px; wächst mit der Auflösung, damit der Eindruck gleich bleibt
_LUMA = np.array([0.299, 0.587, 0.114])


def crop_square(photo: np.ndarray, x: Optional[int] = None, y: Optional[int] = None, side: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Quadratischer Ausschnitt; ohne Angaben das größte zentrierte Quadrat.

    Returns:
        (Ausschnitt, Kantenlänge in Bildpunkten)
    """
    height, width = photo.shape[:2]
    side = side or min(height, width)
    x = (width - side) // 2 if x is None else x
    y = (height - side) // 2 if y is None else y
    if side <= 0 or x < 0 or y < 0 or x + side > width or y + side > height:
        raise ValueError(f"Ausschnitt x={x}, y={y}, Kante={side} liegt nicht im Foto ({width} x {height})")
    return photo[y : y + side, x : x + side], side


def flatten_lighting(image: np.ndarray, sigma_frac: float = LIGHTING_SIGMA_FRAC) -> np.ndarray:
    """Gleicht großflächige Helligkeitsverläufe (Schattenwurf, Vignette) aus; die mittlere Helligkeit bleibt."""
    sigma = sigma_frac * min(image.shape[:2])
    luma = gaussian_filter(image @ _LUMA, sigma=sigma, mode="reflect")
    flat = image * (luma.mean() / np.maximum(luma, 1e-3))[..., None]
    return np.clip(flat * (image.mean() / max(float(flat.mean()), 1e-6)), 0.0, 1.0)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_seamless(image: np.ndarray, blend: float = DEFAULT_BLEND) -> np.ndarray:
    """
    Macht das Bild an allen vier Rändern kachelbar.

    Die um die halbe Kantenlänge versetzte Kopie ist an den Bildrändern stetig (dort lagen im Original Nachbarpixel),
    das Original in der Bildmitte. Gewichtet wird zum Rand hin zur Kopie; die Varianz wird in der Überblendung
    erhalten, damit sie nicht flauer aussieht.
    """
    blend = min(blend, 0.5)
    height, width = image.shape[:2]

    def edge_weight(count: int) -> np.ndarray:
        position = (np.arange(count) + 0.5) / count
        return _smoothstep(np.minimum(position, 1.0 - position) / blend)

    weight = (edge_weight(height)[:, None] * edge_weight(width)[None, :])[..., None]
    shifted = np.roll(image, (height // 2, width // 2), axis=(0, 1))
    mean = image.mean(axis=(0, 1))
    mixed = image * weight + shifted * (1.0 - weight)
    tile = mean + (mixed - mean) / np.sqrt(weight**2 + (1.0 - weight) ** 2)
    return np.clip(tile, 0.0, 1.0)


def derive_maps(tile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normal- und Roughness-Map (uint8 RGB) aus der Helligkeit der Kachel; dunkel = tief und rauer.
    """
    size = min(tile.shape[:2])
    luma = tile @ _LUMA
    height = gaussian_blur_wrap(luma, max(0.7, size / 1024.0))
    normal = normal_from_height(height, NORMAL_STRENGTH_PER_1024 * size / 1024.0)

    reference = max(float(np.percentile(luma, 90)), 1e-3)
    roughness = 0.78 + 0.20 * (1.0 - np.clip(luma / reference, 0.0, 1.0))
    return normal, gray_to_rgb(roughness)


def build_from_photo(
    photo: np.ndarray,
    photo_width_m: float,
    size_px: int = 2048,
    crop: Optional[Tuple[int, int, int]] = None,
    blend: float = DEFAULT_BLEND,
) -> Dict:
    """
    Args:
        photo: uint8-RGB-Foto
        photo_width_m: reale Breite des gesamten Fotos in Metern (bestimmt die Kachelgröße)
        size_px: Kantenlänge der Kachel
        crop: (x, y, Kantenlänge) des Quadrats in Foto-Bildpunkten; Standard: größtes zentriertes Quadrat
        blend: siehe make_seamless

    Returns:
        {"maps": {"color", "normal", "roughness"} als uint8-RGB, "tile_m": reale Kantenlänge der Kachel in Metern}
    """
    square, side = crop_square(photo, *crop) if crop else crop_square(photo)
    resized = Image.fromarray(np.ascontiguousarray(square), "RGB").resize((size_px, size_px), Image.LANCZOS)
    tile = make_seamless(flatten_lighting(np.asarray(resized, dtype=np.float64) / 255.0), blend)
    normal, roughness = derive_maps(tile)
    return {
        "maps": {"color": to_uint8(tile), "normal": normal, "roughness": roughness},
        "tile_m": photo_width_m * side / photo.shape[1],
    }
