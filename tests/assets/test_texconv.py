"""Tests for world_to_beamng/io/texconv.py: existing binary is used as is, a failed download leaves no file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from world_to_beamng.io.texconv import ensure_texconv


def test_existing_binary_is_returned_without_download(tmp_path):
    exe = tmp_path / "bin" / "texconv.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"MZ")

    assert ensure_texconv(exe, url="http://invalid.invalid/texconv.exe") == exe


def test_download_writes_the_file_atomically(tmp_path):
    source = tmp_path / "remote.exe"
    source.write_bytes(b"MZ" * 1000)
    exe = tmp_path / "bin" / "texconv.exe"

    assert ensure_texconv(exe, url=source.as_uri()).read_bytes() == b"MZ" * 1000
    assert not (tmp_path / "bin" / "texconv.exe.part").exists()


def test_failed_download_raises_with_the_manual_fallback_and_leaves_nothing(tmp_path):
    exe = tmp_path / "bin" / "texconv.exe"

    with pytest.raises(FileNotFoundError, match="download it manually"):
        ensure_texconv(exe, url=(tmp_path / "missing.exe").as_uri())
    assert not exe.exists() and not (tmp_path / "bin" / "texconv.exe.part").exists()
