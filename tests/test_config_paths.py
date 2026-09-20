"""
Tests: Konfiguration ohne persönliche Werte - Level-Ordner aus %LOCALAPPDATA%, API-Key nur aus der Umgebung.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG_FILE = ROOT / "world_to_beamng" / "config.py"


def _run(code: str, env_extra: dict) -> str:
    import os

    env = {**os.environ, **env_extra}
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_level_folder_follows_localappdata(tmp_path):
    out = _run("from world_to_beamng import config; print(config.BEAMNG_DIR)", {"LOCALAPPDATA": str(tmp_path)})

    assert Path(out) == tmp_path / "BeamNG" / "BeamNG.drive" / "current" / "levels" / "world_to_beamng"


def test_derived_folders_live_below_the_level_folder(tmp_path):
    out = _run(
        "from world_to_beamng import config; print(config.BEAMNG_DIR_TEXTURES); print(config.BEAMNG_DIR.parent.parent)",
        {"LOCALAPPDATA": str(tmp_path)},
    )
    textures, user_dir = out.splitlines()

    assert Path(textures) == tmp_path / "BeamNG" / "BeamNG.drive" / "current" / "levels" / "world_to_beamng" / "art" / "shapes" / "textures"
    assert Path(user_dir) == tmp_path / "BeamNG" / "BeamNG.drive" / "current"  # daraus liest ForestWorkflow die Baum-Assets


def test_no_secret_or_personal_path_is_committed_in_the_config():
    text = CONFIG_FILE.read_text(encoding="utf-8")

    assert not re.search(r"\b[0-9a-f]{32}\b", text), "sieht nach einem API-Key im Klartext aus"
    assert "C:/Users/" not in text and "C:\\Users\\" not in text
