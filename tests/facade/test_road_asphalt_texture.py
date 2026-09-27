"""
Tests for world_to_beamng.textures.road_asphalt: an ambientCG PBR ZIP becomes the asphalt road strip in the texture
library (several source tiles across the carriageway, whole tiles per DecalRoad texture length) and replaces BeamNG's
stock asphalt textures.
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from world_to_beamng import config
from world_to_beamng.textures import library
from world_to_beamng.textures.road_asphalt import (
    build_road_strip,
    import_road_asphalt,
    read_ambientcg_zip,
    road_asphalt_outdated,
    use_road_asphalt,
)

PX = 8


def _png(array, mode):
    buffer = io.BytesIO()
    Image.fromarray(array, mode).save(buffer, "PNG")
    return buffer.getvalue()


def _ambientcg_zip(path, asset="Road012A_2K-PNG", color=120, ao=255):
    """ambientCG layout: <asset>_Color.png (RGBA), _NormalGL/_NormalDX.png, _Roughness.png, _AmbientOcclusion.png."""
    gradient = np.tile(np.arange(PX, dtype=np.uint8)[None, :] * 20, (PX, 1))
    with zipfile.ZipFile(path, "w") as archive:
        rgba = np.dstack([np.full((PX, PX), color, np.uint8), gradient, gradient, np.full((PX, PX), 255, np.uint8)])
        archive.writestr(f"{asset}_Color.png", _png(rgba, "RGBA"))
        archive.writestr(f"{asset}_NormalGL.png", _png(np.dstack([gradient, np.full((PX, PX), 200, np.uint8), np.full((PX, PX), 255, np.uint8)]), "RGB"))
        archive.writestr(f"{asset}_NormalDX.png", _png(np.dstack([gradient, np.full((PX, PX), 55, np.uint8), np.full((PX, PX), 255, np.uint8)]), "RGB"))
        archive.writestr(f"{asset}_Roughness.png", _png(np.full((PX, PX), 180, np.uint8), "L"))
        archive.writestr(f"{asset}_AmbientOcclusion.png", _png(np.full((PX, PX), ao, np.uint8), "L"))
        archive.writestr(f"{asset}.mtlx", "<materialx/>")
    return path


def test_the_zip_is_read_with_the_green_up_normal_map(tmp_path):
    maps = read_ambientcg_zip(_ambientcg_zip(tmp_path / "Road012A_2K-PNG.zip"))

    assert set(maps) == {"color", "normal", "roughness", "ao"}
    assert maps["color"].shape == (PX, PX, 3)
    assert maps["normal"][0, 0, 1] == 200  # NormalGL (green up, like config.TEXTURE_NORMAL_GREEN_UP), not NormalDX


def test_ambient_occlusion_is_baked_into_the_colour(tmp_path):
    bright = read_ambientcg_zip(_ambientcg_zip(tmp_path / "a.zip", ao=255))
    dark = read_ambientcg_zip(_ambientcg_zip(tmp_path / "b.zip", ao=128))

    strip_bright = build_road_strip(bright, tiles_across=3, tiles_along=2, tile_px=PX)
    strip_dark = build_road_strip(dark, tiles_across=3, tiles_along=2, tile_px=PX)

    assert strip_dark["color"][..., 0].mean() == pytest.approx(strip_bright["color"][..., 0].mean() * 128 / 255, abs=1.5)


def test_the_strip_holds_whole_tiles_across_and_along(tmp_path):
    maps = read_ambientcg_zip(_ambientcg_zip(tmp_path / "a.zip"))

    strip = build_road_strip(maps, tiles_across=3, tiles_along=2, tile_px=PX)

    for channel in ("color", "normal", "roughness"):
        assert strip[channel].shape[:2] == (2 * PX, 3 * PX)  # rows = along the road, columns = across it
        assert strip[channel].dtype == np.uint8 and strip[channel].shape[2] == 3
    color = strip["color"]
    assert np.array_equal(color[:PX, :PX], color[PX:, 2 * PX:])  # the same tile everywhere: seamless along the road


def test_import_stores_the_strip_as_a_library_texture(tmp_path):
    zip_path = _ambientcg_zip(tmp_path / "Road012A_2K-PNG.zip")

    import_road_asphalt(tmp_path, zip_name=zip_path.name, name="road_asphalt", tiles_across=3, tiles_along=2, tile_px=PX)

    assert library.is_complete("road_asphalt", tmp_path)
    entry = library.load_manifest(tmp_path)["road_asphalt"]
    assert entry["tile_m"] == pytest.approx(config.ROAD_DECAL_TEXTURE_LENGTH)  # one strip = one texture repeat
    assert "Road012A_2K-PNG.zip" in entry["source"] and "ambientCG" in entry["source"]


def test_a_changed_zip_makes_the_imported_texture_outdated(tmp_path):
    zip_path = _ambientcg_zip(tmp_path / "Road012A_2K-PNG.zip")
    import_road_asphalt(tmp_path, zip_name=zip_path.name, name="road_asphalt", tiles_across=3, tiles_along=2, tile_px=PX)
    assert not road_asphalt_outdated(tmp_path, zip_name=zip_path.name, name="road_asphalt")

    _ambientcg_zip(zip_path, color=30)  # another download under the same name

    assert road_asphalt_outdated(tmp_path, zip_name=zip_path.name, name="road_asphalt")


def test_only_the_asphalt_surface_gets_the_new_textures():
    surface_types = {
        "asphalt_road_standard": {"textures": {"baseColorMap": "stock_b.dds", "ambientOcclusionMap": "stock_ao.dds"}},
        "dirt_road": {"textures": {"baseColorMap": "dirt_b.dds"}},
    }
    library_maps = {"baseColorMap": "road_asphalt_b.color.dds", "normalMap": "road_asphalt_nm.normal.dds",
                    "roughnessMap": "road_asphalt_r.data.dds"}

    use_road_asphalt(surface_types, "asphalt_road_standard", library_maps)

    assert surface_types["asphalt_road_standard"]["textures"] == library_maps  # stock AO/opacity dropped: baked in
    assert surface_types["dirt_road"]["textures"] == {"baseColorMap": "dirt_b.dds"}
