"""Tests für world_to_beamng/logging_config.py: Root-Cause-Fix für stumme Modul-Logger.

Vorher konfigurierte LoggerConfig nur den Logger "w2b" - Module, die das
Standard-Idiom logging.getLogger(__name__) nutzen (z.B. world_to_beamng.
workflow.terrain_workflow), propagierten NICHT zu "w2b" und verwarfen
INFO/DEBUG-Meldungen lautlos (Root-Logger-Default-Level WARNING).
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from world_to_beamng.logging_config import LoggerConfig

ROOT = Path(__file__).parent.parent


def _run_fresh_process(code: str, env_extra: dict) -> str:
    """Führt `code` in einem NEUEN Python-Prozess aus - das LoggerConfig-Singleton ist
    prozessweit global, ein In-Process-Test würde die reale Import-Reihenfolge von
    world_to_beamng.config (siehe test_config_LOG_LEVEL_survives_the_real_import_order unten)
    nicht abbilden, da hier im Testlauf schon längst irgendein anderes Modul importiert wurde."""
    env = {**os.environ, **env_extra}
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture(autouse=True)
def reset_logger_singleton():
    LoggerConfig._instance = None
    LoggerConfig._logger = None
    yield
    LoggerConfig._instance = None
    LoggerConfig._logger = None


def test_module_style_logger_inherits_configured_level():
    LoggerConfig.get_instance(log_file=None, level=logging.INFO)

    module_logger = logging.getLogger("world_to_beamng.workflow.terrain_workflow")

    assert module_logger.getEffectiveLevel() == logging.INFO
    assert module_logger.isEnabledFor(logging.INFO)


def test_deeply_nested_module_logger_also_inherits_the_level():
    LoggerConfig.get_instance(log_file=None, level=logging.DEBUG)

    module_logger = logging.getLogger("world_to_beamng.forest.forest_instance_generator")

    assert module_logger.getEffectiveLevel() == logging.DEBUG


def test_level_can_be_given_as_a_name_string():
    LoggerConfig.get_instance(log_file=None, level="WARNING")

    module_logger = logging.getLogger("world_to_beamng.workflow.terrain_workflow")

    assert module_logger.getEffectiveLevel() == logging.WARNING


def test_unknown_level_name_raises_a_clear_error():
    with pytest.raises(ValueError, match="WARNNIG"):
        LoggerConfig.get_instance(log_file=None, level="WARNNIG")


def test_get_logger_returns_the_package_logger():
    logger = LoggerConfig.get_logger()

    assert logger.name == "world_to_beamng"


def test_noisy_third_party_loggers_are_dampened_to_warning():
    LoggerConfig.get_instance(log_file=None, level=logging.DEBUG)

    for name in ("urllib3", "PIL", "matplotlib"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING


def test_console_handler_does_not_choke_on_literal_square_brackets(capsys):
    LoggerConfig.get_instance(log_file=None, level=logging.INFO)
    logger = logging.getLogger("world_to_beamng.textures.registry")

    logger.info("  [OK] Texture foo (procedural)")  # darf NICHT als rich-Markup interpretiert werden

    captured = capsys.readouterr()
    assert "[OK] Texture foo" in captured.out


def test_config_LOG_LEVEL_env_var_survives_the_real_import_order():
    """Regression: world_to_beamng/config.py importierte früher `from .osm.osm_mapper import
    OSMMapper` VOR der eigenen LoggerConfig.get_instance()-Konfiguration - osm_mapper.py ruft
    beim eigenen Modul-Import bereits logger = LoggerConfig.get_logger() auf, was das Singleton
    (get_instance()-"nur einmal erzeugen"-Guard) mit den Default-Werten (INFO) fest einfror.
    config.py's eigener LOG_LEVEL-Aufruf wurde dadurch zum stillen No-Op. Muss in einem frischen
    Prozess laufen (siehe _run_fresh_process()-Docstring)."""
    out = _run_fresh_process(
        "from world_to_beamng import config\n"
        "import logging\n"
        "print(logging.getLogger('world_to_beamng.osm.osm_mapper').getEffectiveLevel())",
        {"LOG_LEVEL": "WARNING"},
    )

    assert int(out) == logging.WARNING


def test_config_LOG_LEVEL_defaults_to_warning_without_the_env_var():
    out = _run_fresh_process(
        "from world_to_beamng import config\n"
        "import logging\n"
        "print(logging.getLogger('world_to_beamng').getEffectiveLevel())",
        {"LOG_LEVEL": ""},  # sicherstellen, dass eine evtl. gesetzte Variable NICHT durchschlägt
    )

    assert int(out) == logging.WARNING
