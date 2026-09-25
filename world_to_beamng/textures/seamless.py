"""
Photo -> seamlessly tiling texture including normal and roughness map (once via tools/make_seamless_texture.py).

Procedure: crop to a square, scale to the target size, even out large-scale lighting, blend the edges via a copy
offset by half a tile (tile seam) and derive height and roughness from the brightness
(dark joints = deep and rough).
"""

from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .. import config
from ..facade.texture_utils import gaussian_blur_wrap, gray_to_rgb, normal_from_height, to_uint8

LIGHTING_SIGMA_FRAC = 0.12  # Radius of the lighting compensation relative to the tile size
DEFAULT_BLEND = 0.4  # Width of the blend at the edges relative to the tile size (at most 0.5)
NORMAL_STRENGTH_PER_1024 = 4.0  # Normal map strength at 1024 px; grows with the resolution so the impression stays the same
_LUMA = np.array([0.299, 0.587, 0.114])


def crop_square(photo: np.ndarray, x: Optional[int] = None, y: Optional[int] = None, side: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Square crop; without arguments the largest centered square.

    Returns:
        (crop, edge length in pixels)
    """
    height, width = photo.shape[:2]
    side = side or min(height, width)
    x = (width - side) // 2 if x is None else x
    y = (height - side) // 2 if y is None else y
    if side <= 0 or x < 0 or y < 0 or x + side > width or y + side > height:
        raise ValueError(f"Crop x={x}, y={y}, edge={side} is not inside the photo ({width} x {height})")
    return photo[y : y + side, x : x + side], side


def flatten_lighting(image: np.ndarray, sigma_frac: float = LIGHTING_SIGMA_FRAC) -> np.ndarray:
    """Evens out large-scale brightness gradients (cast shadows, vignette); the mean brightness is preserved."""
    sigma = sigma_frac * min(image.shape[:2])
    luma = gaussian_filter(image @ _LUMA, sigma=sigma, mode="reflect")
    flat = image * (luma.mean() / np.maximum(luma, 1e-3))[..., None]
    return np.clip(flat * (image.mean() / max(float(flat.mean()), 1e-6)), 0.0, 1.0)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_seamless(image: np.ndarray, blend: float = DEFAULT_BLEND) -> np.ndarray:
    """
    Makes the image tileable at all four edges.

    The copy offset by half the edge length is continuous at the image edges (in the original, neighboring pixels lay
    there), the original in the image center. Weighting shifts toward the copy at the edge; the variance is
    preserved in the blend so that it does not look flatter.
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
    Normal and roughness map (uint8 RGB) from the brightness of the tile; dark = deep and rougher.
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
        photo: uint8 RGB photo
        photo_width_m: real width of the entire photo in meters (determines the tile size)
        size_px: Edge length of the tile
        crop: (x, y, edge length) of the square in photo pixels; default: largest centered square
        blend: see make_seamless

    Returns:
        {"maps": {"color", "normal", "roughness"} as uint8 RGB, "tile_m": real edge length of the tile in meters}
    """
    square, side = crop_square(photo, *crop) if crop else crop_square(photo)
    resized = Image.fromarray(np.ascontiguousarray(square), "RGB").resize((size_px, size_px), Image.LANCZOS)
    tile = make_seamless(flatten_lighting(np.asarray(resized, dtype=np.float64) / 255.0), blend)
    normal, roughness = derive_maps(tile)
    return {
        "maps": {"color": to_uint8(tile), "normal": normal, "roughness": roughness},
        "tile_m": photo_width_m * side / photo.shape[1],
    }
