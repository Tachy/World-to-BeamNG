"""Findet die BeamNG-Installation (für Assets, die aus den Content-ZIPs übernommen werden)."""

import configparser
import os
from pathlib import Path


def get_beamng_install_dir() -> Path:
    """Liest den BeamNG-Installationspfad aus BeamNG.drive.ini (userPathWithouVersion-Nachbar)."""
    ini_path = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "BeamNG" / "BeamNG.drive.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"BeamNG.drive.ini not found: {ini_path}")

    # Datei ist eine simple "key = value" Liste ohne Section-Header -> ConfigParser braucht Dummy-Section
    raw = ini_path.read_text(encoding="utf-8-sig")
    parser = configparser.ConfigParser()
    parser.read_string("[main]\n" + raw)
    install_path = parser["main"]["installpath"].strip().strip('"')
    return Path(install_path)
