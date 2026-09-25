"""Tests for world_to_beamng.textures.concrete: procedural concrete texture for bridges/tunnels/galleries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from world_to_beamng import config
from world_to_beamng.textures.concrete import ConcreteTextureGenerator, generate_concrete_texture


def test_generated_maps_have_the_configured_size_and_are_valid_rgb_images():
    maps = ConcreteTextureGenerator(size_px=64, repeat_m=2.0).generate(seed=1)

    for key in ("albedo", "normal", "roughness"):
        assert maps[key].shape == (64, 64, 3)
        assert maps[key].dtype == np.uint8


def test_generation_is_deterministic_for_the_same_seed():
    a = ConcreteTextureGenerator(size_px=32, repeat_m=2.0).generate(seed=7)
    b = ConcreteTextureGenerator(size_px=32, repeat_m=2.0).generate(seed=7)

    np.testing.assert_array_equal(a["albedo"], b["albedo"])


def test_different_seeds_produce_different_textures():
    a = ConcreteTextureGenerator(size_px=32, repeat_m=2.0).generate(seed=1)
    b = ConcreteTextureGenerator(size_px=32, repeat_m=2.0).generate(seed=2)

    assert not np.array_equal(a["albedo"], b["albedo"])


def test_generate_concrete_texture_stores_it_in_the_library(tmp_path):
    folder = generate_concrete_texture(library_dir=tmp_path, seed=3)

    assert folder == tmp_path / config.CONCRETE_TEXTURE_NAME
    for channel in ("color", "normal", "roughness"):
        assert (folder / f"{channel}.png").exists()
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert config.CONCRETE_TEXTURE_NAME in manifest
