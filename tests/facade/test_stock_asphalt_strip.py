"""
Tests for the road strip built from BeamNG's installed stock asphalt (textures/road_asphalt.py): the stock tile is
repeated across and along the road so its grain keeps its real size on a DecalRoad, and the result only goes into the
level as DDS (BeamNG assets must not end up in the public texture library).
"""

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from PIL import Image

from world_to_beamng.textures import road_asphalt
from world_to_beamng.textures.road_asphalt import build_road_strip, write_stock_road_strip

PX = 8
SOURCE = "tileable/road/m_asphalt_new_01/t_asphalt_02"


def _png(array, mode):
    buffer = io.BytesIO()
    Image.fromarray(array, mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _install(tmp_path):
    rng = np.random.default_rng(3)
    materials = tmp_path / "install" / "content" / "assets" / "materials"
    materials.mkdir(parents=True)
    prefix = f"assets/materials/{SOURCE}"
    with zipfile.ZipFile(materials / "tileable.zip", "w") as archive:  # PNG bytes under the DDS names: PIL reads by content
        archive.writestr(f"{prefix}_b.color.dds", _png(rng.integers(0, 255, (PX, PX, 4), dtype=np.uint8), "RGBA"))
        archive.writestr(f"{prefix}_nm.normal.dds", _png(np.full((PX, PX, 3), (128, 128, 255), dtype=np.uint8), "RGB"))
        archive.writestr(f"{prefix}_ao.data.dds", _png(np.full((PX, PX), 128, dtype=np.uint8), "L"))
    return tmp_path / "install"


def test_strip_without_roughness_repeats_the_tile_and_bakes_the_ao():
    color = np.full((PX, PX, 3), 200, dtype=np.uint8)
    maps = {"color": color, "normal": np.full((PX, PX, 3), (128, 128, 255), dtype=np.uint8), "ao": np.full((PX, PX), 128, dtype=np.uint8)}
    strip = build_road_strip(maps, tiles_across=4, tiles_along=4, tile_px=PX)
    assert set(strip) == {"color", "normal"}
    assert strip["color"].shape == (4 * PX, 4 * PX, 3)
    assert abs(int(strip["color"][0, 0, 0]) - 100) <= 1  # 200 * 128/255


def test_stock_strip_is_written_as_dds_into_the_level_only(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(road_asphalt, "_write_dds", lambda pixels, out, name, fmt: written.append((name, pixels.shape)) or out / f"{name}.dds")
    maps = write_stock_road_strip(_install(tmp_path), tmp_path / "level_textures", SOURCE, tiles_across=4, tiles_along=4, tile_px=PX)
    assert set(maps) == {"baseColorMap", "normalMap"}
    assert maps["baseColorMap"].endswith("road_asphalt_stock_b.color.dds")
    assert sorted(written) == [("road_asphalt_stock_b.color", (4 * PX, 4 * PX, 3)), ("road_asphalt_stock_nm.normal", (4 * PX, 4 * PX, 3))]


def test_stock_strip_is_only_rebuilt_when_the_source_changes(tmp_path, monkeypatch):
    written = []

    def fake_write(pixels, out, name, fmt):
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.dds").write_bytes(b"dds")
        written.append(name)
        return out / f"{name}.dds"

    monkeypatch.setattr(road_asphalt, "_write_dds", fake_write)
    install = _install(tmp_path)
    write_stock_road_strip(install, tmp_path / "level_textures", SOURCE, tiles_across=4, tiles_along=4, tile_px=PX)
    write_stock_road_strip(install, tmp_path / "level_textures", SOURCE, tiles_across=4, tiles_along=4, tile_px=PX)
    assert len(written) == 2  # second call: unchanged source and tiling -> nothing rewritten
    write_stock_road_strip(install, tmp_path / "level_textures", SOURCE, tiles_across=5, tiles_along=4, tile_px=PX)
    assert len(written) == 4
