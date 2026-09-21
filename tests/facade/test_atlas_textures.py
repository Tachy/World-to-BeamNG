"""
Tests für Fenster-Atlas, Putztextur und Kies-Textur.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.facade.facade_styles import PLASTER_COLORS
from world_to_beamng.textures.gravel import GravelTextureGenerator
from world_to_beamng.facade.plaster_texture import PlasterTextureGenerator
from world_to_beamng.facade.window_atlas import SPRITE_SIZES_M, WindowAtlasGenerator, WindowAtlasLayout, WindowSprite

LAYOUT = WindowAtlasLayout()


@pytest.fixture(scope="module")
def windows():
    return WindowAtlasGenerator().generate()


@pytest.fixture(scope="module")
def plaster():
    return PlasterTextureGenerator(size_px=256).generate()


@pytest.fixture(scope="module")
def gravel():
    return GravelTextureGenerator(size_px=256, repeat_m=1.0).generate()


# ---------------------------------------------------------------- Fenster-Atlas


def test_sprites_have_their_real_size_in_metres():
    for sprite, (width_m, height_m) in SPRITE_SIZES_M.items():
        x, y, w, h = LAYOUT.rect_px(sprite)
        assert w == round(width_m * LAYOUT.px_per_m) and h == round(height_m * LAYOUT.px_per_m)


def test_sprites_do_not_overlap_including_their_gutter():
    g = LAYOUT.gutter_px
    blocks = [(x - g, y - g, x + w + g, y + h + g) for x, y, w, h in (LAYOUT.rect_px(s) for s in WindowSprite)]

    for x0, y0, x1, y1 in blocks:
        assert x0 >= 0 and y0 >= 0 and x1 <= LAYOUT.width_px and y1 <= LAYOUT.height_px
    for i, a in enumerate(blocks):
        for b in blocks[i + 1 :]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_uv_rect_covers_exactly_the_sprite():
    for sprite in WindowSprite:
        u0, v0, u1, v1 = LAYOUT.uv_rect(sprite)
        _, _, w, h = LAYOUT.rect_px(sprite)
        assert (u1 - u0) * LAYOUT.width_px == pytest.approx(w)
        assert (v1 - v0) * LAYOUT.height_px == pytest.approx(h)
        assert 0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0


def test_atlas_has_the_layout_size(windows):
    for image in windows.values():
        assert image.shape == (LAYOUT.height_px, LAYOUT.width_px, 3) and image.dtype == np.uint8


def test_gutter_repeats_the_edge_pixels(windows):
    g = config.FACADE_GUTTER_PX
    x, y, w, h = LAYOUT.rect_px(WindowSprite.WINDOW_PLAIN)

    top_gutter = windows["albedo"][y - g : y, x + w // 2].astype(int)
    top_edge = windows["albedo"][y, x + w // 2].astype(int)
    left_gutter = windows["albedo"][y + h // 2, x - g : x].astype(int)
    left_edge = windows["albedo"][y + h // 2, x].astype(int)

    assert np.abs(top_gutter - top_edge).max() == 0 and np.abs(left_gutter - left_edge).max() == 0


def test_normals_are_normalised_and_point_out_of_the_surface(windows):
    x, y, w, h = LAYOUT.rect_px(WindowSprite.WINDOW_TRANSOM)
    vectors = windows["normal"][y : y + h, x : x + w].astype(np.float64) / 255.0 * 2.0 - 1.0

    assert np.abs(np.linalg.norm(vectors, axis=-1) - 1.0).max() < 0.03
    assert vectors[..., 2].min() > 0.0


def test_the_reveal_shows_up_in_the_normal_map(windows):
    x, y, w, h = LAYOUT.rect_px(WindowSprite.WINDOW_PLAIN)
    normal = windows["normal"][y : y + h, x : x + w].astype(int)

    assert np.abs(normal[..., 0] - 128).max() > 20  # Kanten von Rahmen/Laibung kippen die Normale


def test_glass_is_smoother_than_the_frame(windows):
    x, y, w, h = LAYOUT.rect_px(WindowSprite.WINDOW_PLAIN)
    rough = windows["roughness"][y : y + h, x : x + w, 0]

    assert rough.min() < 40 and rough.max() > 100  # Glas ~0,08 -> 20; Rahmen/Laibung deutlich rauer


# ---------------------------------------------------------------- Putz


def test_one_albedo_per_plaster_colour_with_its_average_tone(plaster):
    assert set(plaster["albedo"]) == {color.name for color in PLASTER_COLORS}
    for color in PLASTER_COLORS:
        mean = plaster["albedo"][color.name].reshape(-1, 3).mean(axis=0)
        assert mean == pytest.approx(color.rgb, abs=4)


def test_white_dominates_and_red_tones_are_rare():
    weights = {color.name: color.weight for color in PLASTER_COLORS}

    assert weights["white"] > 500
    assert weights["salmon"] + weights["terracotta"] < 80


def test_plaster_tiles_without_seams(plaster):
    albedo = plaster["albedo"]["white"].astype(float)
    typical = np.abs(np.diff(albedo, axis=0)).mean()

    assert np.abs(albedo[0] - albedo[-1]).mean() < typical * 3  # Sprung über die Kante wie zwischen Nachbarpixeln
    assert np.abs(albedo[:, 0] - albedo[:, -1]).mean() < typical * 3


def test_plaster_normals_are_normalised(plaster):
    vectors = plaster["normal"].astype(np.float64) / 255.0 * 2.0 - 1.0

    assert np.abs(np.linalg.norm(vectors, axis=-1) - 1.0).max() < 0.03


def test_plaster_has_no_cell_structure(plaster):
    # Keine Wiederholung innerhalb der Textur: Zeilen-/Spaltenmittelwerte schwanken kaum (kein Raster, keine Fugen)
    gray = plaster["albedo"]["white"].astype(float).mean(axis=-1)

    assert gray.mean(axis=0).std() < 2.0 and gray.mean(axis=1).std() < 2.0


# ---------------------------------------------------------------- Kies


def test_gravel_is_tileable_and_rough(gravel):
    assert gravel["albedo"].shape == (256, 256, 3)
    albedo = gravel["albedo"].astype(float)
    typical = np.abs(np.diff(albedo, axis=0)).mean()

    assert np.abs(albedo[0] - albedo[-1]).mean() < typical * 3
    assert np.abs(albedo[:, 0] - albedo[:, -1]).mean() < typical * 3
    assert gravel["roughness"][..., 0].mean() > 190
