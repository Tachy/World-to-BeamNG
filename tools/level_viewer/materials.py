"""
Display colors for materials and image loading for textures (terrain photo, minimap, horizon).

The viewer draws structures with flat colors: most material textures are vanilla BeamNG assets that only exist
inside the game's content zips, and a debug view reads better with distinct colors anyway.
"""

import zlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the aerial photos are 8192 px tiles

# Substring of the material name -> RGB (0..1); checked in order, first match wins
_PALETTE = (
    ("line_divider", (1.0, 0.85, 0.2)),
    ("line_", (1.0, 1.0, 1.0)),
    ("asphalt", (0.25, 0.25, 0.27)),
    ("cobblestone", (0.55, 0.45, 0.4)),
    ("gravel", (0.62, 0.6, 0.55)),
    ("dirt", (0.55, 0.42, 0.28)),
    ("concrete", (0.72, 0.72, 0.7)),
    ("railing", (0.35, 0.37, 0.4)),
    ("wall", (0.6, 0.55, 0.5)),
    ("roof", (0.6, 0.25, 0.18)),
    ("window", (0.3, 0.4, 0.5)),
    ("horizon", (0.45, 0.5, 0.4)),
)


def material_color(name: Optional[str], materials: Optional[dict] = None) -> Tuple[float, float, float]:
    """
    RGB (0..1) for a material: the palette entry matching its name, else a flat color from materials.json
    (Stages[0].diffuseColor / baseColorFactor), else a stable pseudo-random color derived from the name.
    """
    name = name or ""
    lower = name.lower()
    for key, rgb in _PALETTE:
        if key in lower:
            return rgb
    definition = (materials or {}).get(name) or (materials or {}).get(name.removesuffix("_structure")) or {}
    stages = definition.get("Stages") or [{}]
    stage = stages[0] if isinstance(stages[0], dict) else {}
    for key in ("diffuseColor", "baseColorFactor"):
        value = stage.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return tuple(float(np.clip(v, 0.0, 1.0)) for v in value[:3])
    seed = zlib.crc32(name.encode("utf-8"))
    return (0.35 + (seed & 0xFF) / 510.0, 0.35 + ((seed >> 8) & 0xFF) / 510.0, 0.35 + ((seed >> 16) & 0xFF) / 510.0)


def load_image(path: Path, max_size: int = 4096) -> Optional[np.ndarray]:
    """
    RGB uint8 image (row 0 = top), scaled down so that the longer side is at most `max_size`.

    DDS files go through imageio (Pillow reads only some DDS variants); returns None if the file cannot be read.
    """
    path = Path(path)
    try:
        if path.suffix.lower() == ".dds":
            import imageio.v2 as imageio

            array = np.asarray(imageio.imread(str(path)))
            image = Image.fromarray(array[..., :3] if array.ndim == 3 else array)
        else:
            image = Image.open(path)
            image.draft("RGB", (max_size, max_size))  # cheap downscale while decoding (JPEG only)
        image = image.convert("RGB")
        if max(image.size) > max_size:
            factor = int(np.ceil(max(image.size) / max_size))
            image = image.reduce(factor)
        return np.asarray(image)
    except Exception:
        return None
