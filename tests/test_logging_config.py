"""Tests für world_to_beamng/logging_config.py: Root-Cause-Fix für stumme Modul-Logger.

Vorher konfigurierte LoggerConfig nur den Logger "w2b" - Module, die das
Standard-Idiom logging.getLogger(__name__) nutzen (z.B. world_to_beamng.
workflow.terrain_workflow), propagierten NICHT zu "w2b" und verwarfen
INFO/DEBUG-Meldungen lautlos (Root-Logger-Default-Level WARNING).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from world_to_beamng.logging_config import LoggerConfig


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
    LoggerConfig.get_instance(log_file=None, level=logging.DEBUG, verbose=True)

    module_logger = logging.getLogger("world_to_beamng.forest.forest_instance_generator")

    assert module_logger.getEffectiveLevel() == logging.DEBUG


def test_get_logger_returns_the_package_logger():
    logger = LoggerConfig.get_logger()

    assert logger.name == "world_to_beamng"


def test_noisy_third_party_loggers_are_dampened_to_warning():
    LoggerConfig.get_instance(log_file=None, level=logging.DEBUG, verbose=True)

    for name in ("urllib3", "PIL", "matplotlib"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING


def test_console_handler_does_not_choke_on_literal_square_brackets(capsys):
    LoggerConfig.get_instance(log_file=None, level=logging.INFO)
    logger = logging.getLogger("world_to_beamng.textures.registry")

    logger.info("  [OK] Textur foo (prozedural)")  # darf NICHT als rich-Markup interpretiert werden

    captured = capsys.readouterr()
    assert "[OK] Textur foo" in captured.out
