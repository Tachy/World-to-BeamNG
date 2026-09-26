"""
Tests for retouching bridges out of the aerial photo (io/aerial_bridge_fill.py): the photo shows the deck from above,
the terrain under the bridge gets the texture beside it mirrored into the bridge outline instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from world_to_beamng.io.aerial_bridge_fill import fill_bridge_areas, mirror_fill, rasterize_areas

RED, BLUE = (250, 20, 20), (20, 20, 250)


def _stripe_mask(shape, col_start, col_end):
    mask = np.zeros(shape, dtype=bool)
    mask[:, col_start:col_end] = True
    return mask


def test_pixels_outside_the_mask_stay_untouched():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    mask = _stripe_mask((30, 40), 15, 25)

    filled = mirror_fill(image, mask)

    assert np.array_equal(filled[~mask], image[~mask])


def test_a_pixel_at_the_edge_takes_the_mirrored_neighbour_pixel():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(20, 60, 3), dtype=np.uint8)
    mask = _stripe_mask((20, 60), 20, 40)  # 20 px wide: the far side weighs 1/21 at the edge

    filled = mirror_fill(image, mask).astype(float)

    # first masked column mirrors the last unmasked column on the left, the second one the column before it
    assert np.abs(filled[:, 20] - image[:, 19]).max() <= 256 / 21 + 1
    assert np.abs(filled[:, 21] - image[:, 18]).max() <= 256 * 2 / 21 + 1
    # symmetric on the right edge
    assert np.abs(filled[:, 39] - image[:, 40]).max() <= 256 / 21 + 1


def test_the_two_sides_blend_across_the_bridge():
    image = np.zeros((10, 40, 3), dtype=np.uint8)
    image[:, :20] = RED
    image[:, 20:] = BLUE
    mask = _stripe_mask((10, 40), 10, 30)

    filled = mirror_fill(image, mask).astype(float)
    red_share = filled[5, 10:30, 0]

    assert np.all(np.diff(red_share) <= 0)  # red fades out from the left edge to the right edge
    assert red_share[0] > 230 and red_share[-1] < 40


def test_a_feature_crossing_under_the_bridge_continues_through_it():
    # a river running across a bridge: the rows are the texture, the fill must keep them continuous
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[12:18] = BLUE
    image[:12] = RED
    image[18:] = RED
    mask = _stripe_mask((30, 40), 15, 25)

    filled = mirror_fill(image, mask)

    assert np.array_equal(filled, image)


def test_a_mask_at_the_image_border_is_filled_from_the_inner_side():
    image = np.zeros((10, 30, 3), dtype=np.uint8)
    image[:, 5:] = BLUE
    mask = _stripe_mask((10, 30), 0, 5)

    filled = mirror_fill(image, mask)

    assert np.array_equal(filled[:, :5], np.broadcast_to(np.array(BLUE, dtype=np.uint8), (10, 5, 3)))


def test_nothing_changes_without_unmasked_pixels_or_without_mask():
    image = np.full((8, 8, 3), 77, dtype=np.uint8)

    assert np.array_equal(mirror_fill(image, np.ones((8, 8), dtype=bool)), image)
    assert np.array_equal(mirror_fill(image, np.zeros((8, 8), dtype=bool)), image)


def test_areas_are_rasterized_with_row_zero_in_the_north():
    # photo covers x 0..20, y 0..10 at 1 m/px; the area is the north-east corner x 15..20, y 8..10
    area = np.array([[15.0, 8.0], [20.0, 8.0], [20.0, 10.0], [15.0, 10.0]])

    mask = rasterize_areas([area], bounds=(0.0, 20.0, 0.0, 10.0), size=(20, 10))

    assert mask.shape == (10, 20)
    assert mask[0:2, 15:20].all()
    assert mask.sum() == 10


def test_fill_bridge_areas_retouches_only_the_area_on_the_photo():
    image = Image.new("RGB", (40, 20), RED)
    image.paste(BLUE, (18, 0, 22, 20))  # the deck in the photo: 4 m wide, north-south
    deck = np.array([[18.0, 0.0], [22.0, 0.0], [22.0, 20.0], [18.0, 20.0]])

    filled = np.asarray(fill_bridge_areas(image, [deck], bounds=(0.0, 40.0, 0.0, 20.0)))

    assert (filled[..., 2] < 60).all()  # no blue deck left
    assert np.array_equal(filled[:, :18], np.asarray(image)[:, :18])


def test_fill_bridge_areas_without_areas_returns_the_photo_unchanged():
    image = Image.new("RGB", (10, 10), RED)

    assert fill_bridge_areas(image, [], bounds=(0.0, 10.0, 0.0, 10.0)) is image
