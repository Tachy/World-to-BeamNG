"""
Central logger configuration for World-to-BeamNG.

Supports:
- Output to the console (stdout)
- Optional output to a log file
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Uniform format for all modules
"""

import logging
from pathlib import Path
from typing import Optional, Union

from rich.logging import RichHandler

from .progress import console

# Allowed level names for LOG_LEVEL (config.py) - deliberately its own fixed mapping instead of the
# (partly deprecated/undocumented) string->level resolution of logging.getLevelName(), so that a
# typo in the environment variable raises a clear error instead of silently doing the wrong thing.
LEVEL_NAMES = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LoggerConfig:
    """
    Central logger configuration (singleton).

    Manages the package logger "world_to_beamng" and its:
    - Console output (always active, via RichHandler on the shared progress.py console)
    - File output (optional)
    - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - the only control: the `level` parameter
      (config.py sets it from the environment variable LOG_LEVEL, see there)
    - Uniform format
    """

    _instance: Optional["LoggerConfig"] = None
    _logger: Optional[logging.Logger] = None

    def __init__(self, log_file: Optional[Path] = None, level: Union[int, str] = logging.INFO):
        """
        Initialize the logger.

        Args:
            log_file: Path to the log file (None = console only)
            level: Logging level as int (logging.DEBUG/.../CRITICAL) or name ("DEBUG", "INFO", ...)
        """
        self.log_file = log_file
        self.level = self._resolve_level(level)
        self._setup_logger()

    @staticmethod
    def _resolve_level(level: Union[int, str]) -> int:
        if not isinstance(level, str):
            return level
        try:
            return LEVEL_NAMES[level.upper()]
        except KeyError:
            raise ValueError(f"Unknown log level {level!r} - allowed: {', '.join(LEVEL_NAMES)}") from None

    @classmethod
    def get_instance(cls, log_file: Optional[Path] = None, level: Union[int, str] = logging.INFO) -> "LoggerConfig":
        """
        Get the singleton instance (created on demand).

        Args:
            log_file: Path to the log file (None = console only)
            level: Logging level as int or name (see __init__)

        Returns:
            LoggerConfig singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(log_file, level)
        return cls._instance

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Get the central logger instance.

        Returns:
            logging.Logger for the world_to_beamng package
        """
        if cls._logger is None:
            cls.get_instance()
        return cls._logger

    def _setup_logger(self) -> None:
        """Configure the logger with a console handler and optionally a file handler."""
        logger_instance = logging.getLogger("world_to_beamng")
        logger_instance.setLevel(self.level)
        logger_instance.handlers.clear()  # Prevent duplicates on repeated calls

        # File formatter with extra info; the console takes over level color/badge via RichHandler
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s() | %(message)s",
            datefmt="%H:%M:%S",
        )

        # Console handler (always active, stdout) - shares the console with progress.py so that
        # log lines appear cleanly above active bars/spinners. markup=False is mandatory:
        # existing log messages contain literal square brackets ("[OK]", "[i]", "[!]", ...)
        # which must NOT be parsed as rich markup.
        console_handler = RichHandler(
            console=console,
            markup=False,
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(self.level)
        logger_instance.addHandler(console_handler)

        # File handler (optional)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(file_formatter)
            logger_instance.addHandler(file_handler)
            logger_instance.info(
                f"Logger initialized: file={self.log_file}, level={logging.getLevelName(self.level)}"
            )

        # Dampen chatty third-party loggers: they also use logging.getLogger(__name__)
        # and would otherwise be logged too, due to the now correctly propagating root-cause fix.
        for noisy in ("urllib3", "PIL", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Set the class variable so that get_logger() finds it
        LoggerConfig._logger = logger_instance
