"""Tests for world_to_beamng.textures.seamless: photo -> seamless tile with normal and roughness map."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng.textures import seamless


def _photo(height=300, width=400, seed=3):
    """Noisy 'photo' with a brightness gradient (lighting) - guaranteed not periodic at the edges."""
    rng = np.random.default_rng(seed)
    noise = rng.random((height, width, 3))
    ramp = np.linspace(0.3, 1.0, width)[None, :, None] * np.linspace(0.6, 1.0, height)[:, None, None]
    return np.clip(0.25 + 0.4 * noise * ramp + 0.3 * ramp, 0, 1)


def _seam(image):
    """Mean jump across the tile edges (wrap) relative to the mean jump between neighboring pixels."""
    wrap = np.abs(image[0] - image[-1]).mean() + np.abs(image[:, 0] - image[:, -1]).mean()
    inner = np.abs(np.diff(image, axis=0)).mean() + np.abs(np.diff(image, axis=1)).mean()
    return wrap / inner


def test_crop_is_a_centred_square_by_default():
    photo = _photo(300, 400)

    crop, side = seamless.crop_square(photo)

    assert crop.shape[:2] == (300, 300) and side == 300
    np.testing.assert_array_equal(crop, photo[:, 50:350])


def test_crop_can_be_placed_explicitly():
    photo = _photo(300, 400)

    crop, side = seamless.crop_square(photo, x=10, y=20, side=100)

    assert side == 100
    np.testing.assert_array_equal(crop, photo[20:120, 10:110])


def test_crop_outside_the_photo_is_rejected():
    with pytest.raises(ValueError):
        seamless.crop_square(_photo(300, 400), x=350, y=0, side=100)


def test_flattening_removes_the_large_scale_lighting_but_keeps_the_average():
    photo = _photo(256, 256)
    flat = seamless.flatten_lighting(photo)

    left, right = flat[:, :64].mean(), flat[:, -64:].mean()
    assert abs(left - right) < 0.3 * abs(photo[:, :64].mean() - photo[:, -64:].mean())
    assert flat.mean() == pytest.approx(photo.mean(), abs=0.03)


def test_the_seamless_tile_has_no_visible_seam():
    photo = _photo(256, 256)
    assert _seam(photo) > 2.0  # starting point: hard seam

    tile = seamless.make_seamless(photo)

    assert tile.shape == photo.shape
    assert _seam(tile) < 1.5


def test_seamless_keeps_the_contrast():
    photo = np.random.default_rng(5).random((256, 256, 3))  # without lighting gradient: comparable variance

    tile = seamless.make_seamless(photo)

    assert tile.std() > 0.8 * photo.std()
    assert tile.min() >= 0.0 and tile.max() <= 1.0


def test_derived_maps_are_valid_and_tile():
    tile = seamless.make_seamless(_photo(256, 256))

    normal, roughness = seamless.derive_maps(tile)

    assert normal.dtype == np.uint8 and normal.shape == (256, 256, 3)
    vectors = normal.astype(float) / 255.0 * 2.0 - 1.0
    assert np.abs(np.linalg.norm(vectors, axis=-1) - 1.0).max() < 0.03
    assert normal[..., 2].min() > 128  # points out of the surface
    assert roughness.shape == (256, 256, 3)
    assert 150 < roughness[..., 0].mean() < 240  # stone: rough


def test_dark_joints_are_rougher_than_bright_stone():
    tile = np.full((128, 128, 3), 0.7)
    tile[:, 60:68] = 0.1

    _, roughness = seamless.derive_maps(tile)

    assert roughness[:, 60:68, 0].mean() > roughness[:, :40, 0].mean()


def test_build_from_photo_returns_maps_and_the_real_tile_size():
    photo = (np.clip(_photo(300, 400), 0, 1) * 255).astype(np.uint8)

    result = seamless.build_from_photo(photo, photo_width_m=2.0, size_px=128)

    assert set(result["maps"]) == {"color", "normal", "roughness"}
    assert all(m.shape == (128, 128, 3) and m.dtype == np.uint8 for m in result["maps"].values())
    assert result["tile_m"] == pytest.approx(2.0 * 300 / 400)  # the square uses 300 of 400 pixels of width


def test_build_from_photo_respects_an_explicit_crop_for_the_scale():
    photo = (np.clip(_photo(300, 400), 0, 1) * 255).astype(np.uint8)

    result = seamless.build_from_photo(photo, photo_width_m=2.0, size_px=64, crop=(0, 0, 200))

    assert result["tile_m"] == pytest.approx(2.0 * 200 / 400)
