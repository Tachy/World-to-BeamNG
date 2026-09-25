"""Tests for world_to_beamng.textures.registry: pre-flight check of all textures the export needs.

Procedural textures (gravel) are generated once on demand; the pipeline cannot generate photo textures (rubble stone
wall): if one is missing, the export aborts.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from world_to_beamng import config
from world_to_beamng.facade import dds_export
from world_to_beamng.textures import library, registry
from world_to_beamng.textures.registry import MissingTexturesError, TextureSpec

CHANNELS = ("color", "normal", "roughness")


def _maps(value=100):
    return {channel: np.full((8, 8, 3), value, dtype=np.uint8) for channel in CHANNELS}


@pytest.fixture
def dds(monkeypatch):
    """Replaces texconv; records the written DDS names."""
    names = []

    def fake_write_dds(pixels, output_dir, name, dds_format, max_mip_levels):
        names.append(name)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{name}.dds").write_bytes(b"dds")
        return output_dir / f"{name}.dds"

    monkeypatch.setattr(dds_export, "write_dds", fake_write_dds)
    return names


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "lib", tmp_path / "out"


def _procedural(calls):
    def generate(library_dir):
        calls.append(library_dir)
        library.store_texture("procedural_tex", _maps(), 2.0, "procedural", library_dir)

    return TextureSpec("procedural_tex", "Test object A", lambda: True, generate=generate)


def _photo(required=lambda: True):
    return TextureSpec("photo_tex", "Test object B", required, generate=None, hint="tools/make_seamless_texture.py <photo> --name photo_tex --width-m <meters>")


def test_present_textures_are_converted_and_nothing_is_generated(dirs, dds):
    lib, out = dirs
    calls = []
    library.store_texture("procedural_tex", _maps(), 2.0, "vorhanden", lib)
    library.store_texture("photo_tex", _maps(), 1.5, "photo", lib)

    paths = registry.prepare_textures(out, lib, (_procedural(calls), _photo()))

    assert calls == []
    assert set(paths) == {"procedural_tex", "photo_tex"}
    assert paths["photo_tex"]["baseColorMap"].endswith("photo_tex_b.color.dds")


def test_a_missing_procedural_texture_is_generated_once_and_then_used(dirs, dds, caplog):
    lib, out = dirs
    calls = []
    library.store_texture("photo_tex", _maps(), 1.5, "photo", lib)

    with caplog.at_level("INFO", logger="world_to_beamng"):
        paths = registry.prepare_textures(out, lib, (_procedural(calls), _photo()))
        registry.prepare_textures(out, lib, (_procedural(calls), _photo()))  # second run: it exists now

    assert calls == [lib]  # generated exactly once
    assert "procedural_tex" in paths and (lib / "procedural_tex" / "color.png").exists()
    assert "procedural_tex" in caplog.text and "check it in" in caplog.text  # note: belongs in the repository


def test_a_missing_photo_texture_aborts_and_says_what_to_do(dirs, dds):
    lib, out = dirs
    calls = []

    with pytest.raises(MissingTexturesError) as error:
        registry.prepare_textures(out, lib, (_procedural(calls), _photo()))

    message = str(error.value)
    assert "photo_tex" in message and "Test object B" in message
    assert str(lib / "photo_tex") in message  # where the files have to be located
    assert "make_seamless_texture.py" in message and "--width-m" in message
    assert dds == []  # aborted before anything was written to the level


def test_all_missing_photo_textures_are_reported_together(dirs, dds):
    lib, out = dirs
    other = TextureSpec("second_photo", "Test object C", lambda: True, hint="process the photo")

    with pytest.raises(MissingTexturesError) as error:
        registry.prepare_textures(out, lib, (_photo(), other))

    assert "photo_tex" in str(error.value) and "second_photo" in str(error.value)


def test_a_photo_texture_with_a_missing_channel_counts_as_missing(dirs, dds):
    lib, out = dirs
    library.store_texture("photo_tex", _maps(), 1.5, "photo", lib)
    (lib / "photo_tex" / "normal.png").unlink()

    with pytest.raises(MissingTexturesError) as error:
        registry.prepare_textures(out, lib, (_photo(),))

    assert "normal.png" in str(error.value)


def test_a_texture_that_is_not_needed_is_neither_required_nor_converted(dirs, dds):
    lib, out = dirs

    paths = registry.prepare_textures(out, lib, (_photo(required=lambda: False),))

    assert paths == {} and dds == []


def test_prepared_textures_are_computed_once_and_refreshed_by_prepare(dirs, dds, monkeypatch):
    lib, out = dirs
    library.store_texture("photo_tex", _maps(), 1.5, "photo", lib)
    monkeypatch.setattr(config, "TEXTURE_LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "BEAMNG_DIR_TEXTURES", out)
    monkeypatch.setattr(registry, "REGISTRY", (_photo(),))
    registry.reset_cache()

    first = registry.prepared_textures()
    assert registry.prepared_textures() is first  # cached

    library.store_texture("photo_tex", _maps(200), 1.5, "neu", lib)
    registry.prepare_textures()
    assert registry.prepared_textures() is not first  # new export: freshly checked
    registry.reset_cache()


def test_prepared_textures_abort_too_when_nobody_ran_the_check_first(dirs, dds, monkeypatch):
    lib, out = dirs
    monkeypatch.setattr(config, "TEXTURE_LIBRARY_DIR", lib)
    monkeypatch.setattr(config, "BEAMNG_DIR_TEXTURES", out)
    monkeypatch.setattr(registry, "REGISTRY", (_photo(),))
    registry.reset_cache()

    with pytest.raises(MissingTexturesError):
        registry.prepared_textures()
    registry.reset_cache()


# --- the real registry -----------------------------------------------------------------------------------------------


def test_the_real_registry_defines_gravel_as_procedural_and_the_wall_as_a_photo_texture():
    specs = {spec.name: spec for spec in registry.REGISTRY}

    gravel, wall = specs[config.FLAT_ROOF_GRAVEL_TEXTURE], specs[config.WALL_TEXTURE_NAME]
    assert gravel.generate is not None and gravel.required()
    assert wall.generate is None and "make_seamless_texture.py" in wall.hint and "--name rubble_stone_wall" in wall.hint


def test_the_wall_texture_is_only_required_while_walls_are_enabled(monkeypatch):
    wall = next(spec for spec in registry.REGISTRY if spec.name == config.WALL_TEXTURE_NAME)

    monkeypatch.setattr(config, "WALLS_ENABLED", True)
    assert wall.required()
    monkeypatch.setattr(config, "WALLS_ENABLED", False)
    assert not wall.required()


def test_the_committed_gravel_is_complete_so_a_fresh_checkout_needs_no_generation():
    assert library.is_complete(config.FLAT_ROOF_GRAVEL_TEXTURE)


# --- Library: completeness ---------------------------------------------------------------------------------------


def test_is_complete_needs_a_manifest_entry_and_all_three_pngs(tmp_path):
    lib = tmp_path / "lib"
    assert not library.is_complete("x", lib)

    library.store_texture("x", _maps(), 1.0, "t", lib)
    assert library.is_complete("x", lib)

    (lib / "x" / "roughness.png").unlink()
    assert not library.is_complete("x", lib)
    assert library.missing_files("x", lib) == ["roughness.png"]

    library.store_texture("x", _maps(), 1.0, "t", lib)  # complete again ...
    manifest = json.loads((lib / "manifest.json").read_text(encoding="utf-8"))
    manifest["textures"].pop("x")  # ... but without a manifest entry
    (lib / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert not library.is_complete("x", lib)


def test_concrete_texture_is_registered_when_bridges_or_tunnels_are_enabled(monkeypatch):
    from world_to_beamng import config
    from world_to_beamng.textures import registry

    # raising=False: TUNNELS_ENABLED only exists from Task 11 on (Task 10 runs before) - monkeypatch then creates
    # the attribute locally for the test and reverts it at the end of the test instead of raising AttributeError.
    monkeypatch.setattr(config, "BRIDGES_ENABLED", True)
    monkeypatch.setattr(config, "TUNNELS_ENABLED", False, raising=False)
    assert any(spec.name == config.CONCRETE_TEXTURE_NAME for spec in registry.REGISTRY if spec.required())

    monkeypatch.setattr(config, "BRIDGES_ENABLED", False)
    monkeypatch.setattr(config, "TUNNELS_ENABLED", False, raising=False)
    assert not any(spec.name == config.CONCRETE_TEXTURE_NAME for spec in registry.REGISTRY if spec.required())
