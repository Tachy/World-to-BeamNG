"""Tests for world_to_beamng/logging_config.py: root-cause fix for silent module loggers.

Previously LoggerConfig only configured the logger "w2b" - modules using the
standard idiom logging.getLogger(__name__) (e.g. world_to_beamng.
workflow.terrain_workflow) did NOT propagate to "w2b" and silently discarded
INFO/DEBUG messages (root logger default level WARNING).
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
    """Runs `code` in a NEW Python process - the LoggerConfig singleton is
    process-wide global, an in-process test would not reproduce the real import order of
    world_to_beamng.config (see test_config_LOG_LEVEL_survives_the_real_import_order below),
    since some other module has long since been imported here in the test run."""
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

    logger.info("  [OK] Texture foo (procedural)")  # must NOT be interpreted as rich markup

    captured = capsys.readouterr()
    assert "[OK] Texture foo" in captured.out


def test_config_LOG_LEVEL_env_var_survives_the_real_import_order():
    """Regression: world_to_beamng/config.py used to import `from .osm.osm_mapper import
    OSMMapper` BEFORE its own LoggerConfig.get_instance() configuration - osm_mapper.py already calls
    logger = LoggerConfig.get_logger() on its own module import, which froze the singleton
    (get_instance() "create only once" guard) with the default values (INFO).
    config.py's own LOG_LEVEL call thereby became a silent no-op. Must run in a fresh
    process (see _run_fresh_process() docstring)."""
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
        {"LOG_LEVEL": ""},  # make sure a possibly set variable does NOT leak through
    )

    assert int(out) == logging.WARNING
