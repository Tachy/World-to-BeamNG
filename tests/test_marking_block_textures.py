"""Tests for world_to_beamng.textures.marking_blocks: generated texture of the block stripes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from PIL import Image

from world_to_beamng import config
from world_to_beamng.textures.marking_blocks import BLOCK_TEXTURE_NAME, write_block_stripe_textures


def test_texture_period_is_block_plus_gap_and_blocks_are_fully_opaque(tmp_path):
    write_block_stripe_textures(tmp_path, block_length=1.0, gap_length=1.0, pixels_per_meter=32)

    opacity = np.array(Image.open(tmp_path / f"{BLOCK_TEXTURE_NAME}_o.data.png").convert("L"))
    color = Image.open(tmp_path / f"{BLOCK_TEXTURE_NAME}_b.color.png").convert("RGB")
    column = opacity[:, opacity.shape[1] // 2]
    assert opacity.shape[0] == 64  # period 2 m at 32 px/m (test arguments, not the config)
    assert (column[:32] == 255).all() and (column[32:] == 0).all()  # 1 m block, 1 m gap
    assert (opacity[0] == 255).all()  # the block spans the whole width of the line
    assert np.array(color).min() == 255  # white


def test_config_texture_period_matches_the_marking_definition():
    period = config.ROAD_MARKING_BLOCK_LENGTH + config.ROAD_MARKING_BLOCK_GAP

    assert config.OSM_MAPPER.road_markings[config.ROAD_MARKING_BLOCK_MATERIAL]["textureLength"] == pytest.approx(period)


def test_material_paths_point_at_the_generated_files():
    textures = config.OSM_MAPPER.road_markings[config.ROAD_MARKING_BLOCK_MATERIAL]["textures"]

    assert textures["baseColorMap"].endswith(f"{BLOCK_TEXTURE_NAME}_b.color.png")
    assert textures["opacityMap"].endswith(f"{BLOCK_TEXTURE_NAME}_o.data.png")


def test_blocks_have_the_length_and_gap_of_the_dashes_of_the_dashed_divider():
    # the stock dashed texture holds 2 dashes and 2 gaps of equal length per repeat (measured: 6 m dash, 6 m gap at 24 m)
    dash = config.OSM_MAPPER.road_markings[config.ROAD_MARKING_DIVIDER_MATERIAL]["textureLength"] / 4.0

    assert config.ROAD_MARKING_BLOCK_LENGTH == pytest.approx(dash)
    assert config.ROAD_MARKING_BLOCK_GAP == pytest.approx(dash)
