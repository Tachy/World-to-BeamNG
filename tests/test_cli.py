"""Tests for world_to_beamng/cli.py: command line of world_to_beamng.py (--loglevel overrides LOG_LEVEL)."""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from world_to_beamng.cli import apply_cli

ROOT = Path(__file__).parent.parent


def test_loglevel_option_sets_the_log_level_case_insensitive(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    apply_cli(["--loglevel=info"])
    assert os.environ["LOG_LEVEL"] == "INFO"


def test_without_option_the_environment_stays(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    apply_cli([])
    assert os.environ["LOG_LEVEL"] == "ERROR"


def test_unknown_level_is_rejected():
    with pytest.raises(SystemExit):
        apply_cli(["--loglevel=verbose"])


def test_option_wins_over_the_environment_in_the_real_import_order():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from world_to_beamng.cli import apply_cli\n"
        "apply_cli(['--loglevel=debug'])\n"
        "from world_to_beamng import config\n"
        "import logging\n"
        "print(config.LOG_LEVEL, logging.getLevelName(logging.getLogger('world_to_beamng').getEffectiveLevel()))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "LOG_LEVEL": "ERROR"}, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["DEBUG", "DEBUG"]


def test_script_rejects_an_unknown_level_before_exporting():
    result = subprocess.run([sys.executable, str(ROOT / "world_to_beamng.py"), "--loglevel=verbose"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 2 and "--loglevel" in result.stderr
