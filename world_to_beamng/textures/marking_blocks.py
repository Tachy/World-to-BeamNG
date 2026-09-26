"""
Texture of the block stripes (config.ROAD_MARKING_BLOCK_MATERIAL): blocks of BLOCK_LENGTH with GAP between them, white,
fully opaque across the line. BeamNG has no stock block marking; it is generated at every export (a few KB). The
material entry lives in data/osm_to_beamng.json (road_markings.line_block_white) and its textureLength has to equal
the period (block + gap) - one texture repeat along the line.
"""

from pathlib import Path

import numpy as np
from PIL import Image

BLOCK_TEXTURE_NAME = "line_block_white"
ACROSS_PIXELS = 16  # the block is uniform across the line


def write_block_stripe_textures(directory: Path, block_length: float, gap_length: float, pixels_per_meter: int = 64) -> None:
    """Writes `<name>_b.color.png` (white) and `<name>_o.data.png` (opacity: blocks 255, gaps 0) into `directory`;
    the image height is one period (block + gap) along the line."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    block_px = max(1, round(block_length * pixels_per_meter))
    height = block_px + max(0, round(gap_length * pixels_per_meter))
    opacity = np.zeros((height, ACROSS_PIXELS), dtype=np.uint8)
    opacity[:block_px, :] = 255
    Image.fromarray(opacity, mode="L").save(directory / f"{BLOCK_TEXTURE_NAME}_o.data.png")
    Image.fromarray(np.full((height, ACROSS_PIXELS, 3), 255, dtype=np.uint8), mode="RGB").save(
        directory / f"{BLOCK_TEXTURE_NAME}_b.color.png"
    )
