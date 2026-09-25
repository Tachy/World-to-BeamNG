"""Tests for world_to_beamng.textures.library: committed textures (data/textures) -> DDS in the level."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from PIL import Image

from world_to_beamng.facade import dds_export
from world_to_beamng.textures import library

CHANNELS = ("color", "normal", "roughness")


def _write_texture(library_dir, name, tile_m=1.5, value=100, channels=CHANNELS):
    folder = library_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    for channel in channels:
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8), "RGB").save(folder / f"{channel}.png")
    manifest_path = library_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"textures": {}}
    manifest["textures"][name] = {"tile_m": tile_m, "source": "test"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def written(monkeypatch):
    """Replaces texconv: writes an empty DDS and records (name, format, pixel mean, mips)."""
    calls = []

    def fake_write_dds(pixels, output_dir, name, dds_format, max_mip_levels):
        calls.append((name, dds_format, float(pixels.mean()), max_mip_levels))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{name}.dds"
        path.write_bytes(b"dds")
        return path

    monkeypatch.setattr(dds_export, "write_dds", fake_write_dds)
    return calls


def test_textures_become_dds_with_the_stock_formats_and_full_mips(tmp_path, written):
    lib, out = tmp_path / "lib", tmp_path / "out"
    _write_texture(lib, "roof_gravel", value=90)

    paths = library.ensure_library_textures(out, lib)

    assert {(name, fmt, mips) for name, fmt, _, mips in written} == {
        ("roof_gravel_b.color", dds_export.COLOR, 0),
        ("roof_gravel_nm.normal", dds_export.NORMAL, 0),
        ("roof_gravel_r.data", dds_export.DATA, 0),
    }
    assert set(paths["roof_gravel"]) == {"baseColorMap", "normalMap", "roughnessMap"}
    assert paths["roof_gravel"]["baseColorMap"].endswith("roof_gravel_b.color.dds")
    assert paths["roof_gravel"]["baseColorMap"].startswith("levels/")  # relative to the BeamNG user folder, with slashes
    assert "\\" not in paths["roof_gravel"]["baseColorMap"]


def test_unchanged_sources_are_not_converted_again(tmp_path, written):
    lib, out = tmp_path / "lib", tmp_path / "out"
    _write_texture(lib, "roof_gravel")
    library.ensure_library_textures(out, lib)
    first = len(written)

    library.ensure_library_textures(out, lib)

    assert len(written) == first


def test_a_changed_source_is_converted_again_but_only_that_texture(tmp_path, written):
    lib, out = tmp_path / "lib", tmp_path / "out"
    _write_texture(lib, "roof_gravel", value=90)
    _write_texture(lib, "rubble_stone_wall", value=120)
    library.ensure_library_textures(out, lib)
    written.clear()

    _write_texture(lib, "roof_gravel", value=200)
    library.ensure_library_textures(out, lib)

    assert {name for name, *_ in written} == {"roof_gravel_b.color", "roof_gravel_nm.normal", "roof_gravel_r.data"}


def test_a_deleted_dds_is_created_again(tmp_path, written):
    lib, out = tmp_path / "lib", tmp_path / "out"
    _write_texture(lib, "roof_gravel")
    library.ensure_library_textures(out, lib)
    (out / "roof_gravel_nm.normal.dds").unlink()
    written.clear()

    library.ensure_library_textures(out, lib)

    assert [name for name, *_ in written] == ["roof_gravel_nm.normal"]


def test_a_texture_with_missing_channels_is_skipped_with_a_warning(tmp_path, written, caplog):
    lib, out = tmp_path / "lib", tmp_path / "out"
    _write_texture(lib, "roof_gravel")
    _write_texture(lib, "rubble_stone_wall", channels=("color",))

    with caplog.at_level("WARNING", logger="world_to_beamng"):
        paths = library.ensure_library_textures(out, lib)

    assert "rubble_stone_wall" not in paths
    assert "roof_gravel" in paths
    assert "rubble_stone_wall" in caplog.text


def test_missing_library_gives_no_textures(tmp_path, written):
    assert library.ensure_library_textures(tmp_path / "out", tmp_path / "nothing") == {}


def test_tile_size_comes_from_the_manifest_with_a_fallback(tmp_path):
    lib = tmp_path / "lib"
    _write_texture(lib, "rubble_stone_wall", tile_m=1.37)

    assert library.texture_tile_m("rubble_stone_wall", 9.9, lib) == pytest.approx(1.37)
    assert library.texture_tile_m("unknown", 9.9, lib) == pytest.approx(9.9)
    assert library.texture_tile_m("rubble_stone_wall", 9.9, tmp_path / "nothing") == pytest.approx(9.9)


def test_stored_textures_are_found_again_and_keep_other_entries(tmp_path, written):
    lib, out = tmp_path / "lib", tmp_path / "out"
    maps = {channel: np.full((8, 8, 3), 50 + i, dtype=np.uint8) for i, channel in enumerate(CHANNELS)}

    library.store_texture("roof_gravel", maps, tile_m=2.0, source="generated", library_dir=lib)
    library.store_texture("rubble_stone_wall", maps, tile_m=1.1, source="photo", library_dir=lib)

    assert library.texture_tile_m("roof_gravel", 0.0, lib) == pytest.approx(2.0)
    assert library.load_manifest(lib)["rubble_stone_wall"]["source"] == "photo"
    assert set(library.ensure_library_textures(out, lib)) == {"roof_gravel", "rubble_stone_wall"}
    assert np.asarray(Image.open(lib / "roof_gravel" / "normal.png"))[0, 0, 0] == 51
