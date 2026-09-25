"""
NumPy building blocks for procedural, tileable textures (periodic = seamless at all four edges).
"""

import numpy as np

from .. import config


def periodic_noise(height: int, width: int, rng: np.random.Generator, beta: float = 2.0) -> np.ndarray:
    """
    Periodic noise in [-1, 1] with a 1/f^beta spectrum (larger beta = softer/larger-scale).
    """
    spectrum = np.fft.fft2(rng.standard_normal((height, width)))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    filtered = np.real(np.fft.ifft2(spectrum / radius**beta))
    filtered -= filtered.mean()
    return filtered / (np.abs(filtered).max() + 1e-12)


def fbm(height: int, width: int, rng: np.random.Generator, betas=(1.2, 2.0, 3.0), weights=(0.25, 0.4, 0.35)) -> np.ndarray:
    """Several noise layers, summed with weights, in [-1, 1]."""
    total = sum(w * periodic_noise(height, width, rng, b) for b, w in zip(betas, weights))
    return total / (np.abs(total).max() + 1e-12)


def gaussian_blur_wrap(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur with wrap-around edges (FFT)."""
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    kernel = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (fx * fx + fy * fy))
    return np.real(np.fft.ifft2(np.fft.fft2(image) * kernel))


def normal_from_height(height: np.ndarray, strength: float) -> np.ndarray:
    """
    Normal map (uint8 RGB) from a height field; the edges wrap around, the green channel follows
    config.TEXTURE_NORMAL_GREEN_UP.
    """
    d_dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    d_dy_img = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5  # image rows downward
    nx = -d_dx * strength
    ny = d_dy_img * strength if config.TEXTURE_NORMAL_GREEN_UP else -d_dy_img * strength
    nz = np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx / length, ny / length, nz / length], axis=-1)
    return np.clip((normal * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Float image in [0, 1] -> uint8."""
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Float grayscale image in [0, 1] -> uint8 RGB (for BC4 data: only the red channel counts)."""
    return np.repeat(to_uint8(gray)[..., None], 3, axis=-1)
