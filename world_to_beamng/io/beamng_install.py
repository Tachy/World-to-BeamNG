"""Finds the BeamNG installation (for assets taken over from the content ZIPs)."""

import configparser
import os
from pathlib import Path


def get_beamng_install_dir() -> Path:
    """Reads the BeamNG install path from BeamNG.drive.ini (neighbor of userPathWithouVersion)."""
    ini_path = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "BeamNG" / "BeamNG.drive.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"BeamNG.drive.ini not found: {ini_path}")

    # The file is a simple "key = value" list without section headers -> ConfigParser needs a dummy section
    raw = ini_path.read_text(encoding="utf-8-sig")
    parser = configparser.ConfigParser()
    parser.read_string("[main]\n" + raw)
    install_path = parser["main"]["installpath"].strip().strip('"')
    return Path(install_path)
