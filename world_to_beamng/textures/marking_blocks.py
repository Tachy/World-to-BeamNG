"""
Texture of the block stripes (config.ROAD_MARKING_BLOCK_MATERIAL): blocks of BLOCK_LENGTH with GAP between them, in the
grey/opacity of the stock dash texture (config.ROAD_MARKING_BLOCK_GREY/OPACITY), uniform across the line. BeamNG has no stock block marking; it is generated at every export (a few KB). The
material entry lives in data/osm_to_beamng.json (road_markings.line_block_white) and its textureLength has to equal
the period (block + gap) - one texture repeat along the line.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

BLOCK_TEXTURE_NAME = "line_block_white"
ACROSS_PIXELS = 16  # the block is uniform across the line


def write_block_stripe_textures(
    directory: Path, block_length: float, gap_length: float, pixels_per_meter: int = 64, grey: int = 255, opacity: int = 255
) -> None:
    """Writes `<name>_b.color.png` (uniform `grey`) and `<name>_o.data.png` (opacity: blocks `opacity`, gaps 0) into `directory`;
    the image height is one period (block + gap) along the line, rounded up to a power of two."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # BeamNG only cooks textures with power-of-two sizes (otherwise the decal shows up as a thin red line): the period is
    # rounded up to the next power of two, block and gap keep their ratio
    height = 1 << max(0, math.ceil(math.log2(max(1.0, (block_length + gap_length) * pixels_per_meter))))
    block_px = max(1, round(height * block_length / (block_length + gap_length)))
    alpha = np.zeros((height, ACROSS_PIXELS), dtype=np.uint8)
    alpha[:block_px, :] = opacity
    Image.fromarray(alpha, mode="L").save(directory / f"{BLOCK_TEXTURE_NAME}_o.data.png")
    Image.fromarray(np.full((height, ACROSS_PIXELS, 3), grey, dtype=np.uint8), mode="RGB").save(
        directory / f"{BLOCK_TEXTURE_NAME}_b.color.png"
    )
