"""Prüft die eingecheckte Bibliothek data/textures: vollständig, quadratisch, Zweierpotenz, passend zur Config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from PIL import Image

from world_to_beamng import config
from world_to_beamng.textures import library

MANIFEST = library.load_manifest()


def test_the_library_has_the_gravel_texture_with_the_roof_scale():
    assert library.texture_tile_m(config.FLAT_ROOF_GRAVEL_TEXTURE, 0.0) == pytest.approx(config.FLAT_ROOF_GRAVEL_REPEAT_M)


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_every_texture_is_complete_square_and_a_power_of_two(name):
    sizes = set()
    for channel in ("color", "normal", "roughness"):
        path = config.TEXTURE_LIBRARY_DIR / name / f"{channel}.png"
        assert path.exists(), f"{path} fehlt"
        with Image.open(path) as image:
            assert image.width == image.height
            assert image.width & (image.width - 1) == 0
            sizes.add(image.size)
    assert len(sizes) == 1, "alle Kanäle einer Textur müssen gleich groß sein"
    assert MANIFEST[name]["tile_m"] > 0 and MANIFEST[name]["source"]
