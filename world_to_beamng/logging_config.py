"""
Zentrale Logger-Konfiguration für World-to-BeamNG.

Unterstützt:
- Ausgabe auf Console (stdout)
- Optional Ausgabe in Logfile
- Konfigurierbare Log-Level (DEBUG, INFO, WARNING, ERROR)
- Einheitliches Format für alle Module
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class LoggerConfig:
    """
    Zentrale Logger-Konfiguration (Singleton).

    Verwaltet:
    - Console-Output (immer aktiv)
    - File-Output (optional)
    - Log-Level (DEBUG, INFO, WARNING, ERROR)
    - Einheitliches Format
    """

    _instance: Optional["LoggerConfig"] = None
    _logger: Optional[logging.Logger] = None

    def __init__(self, log_file: Optional[Path] = None, level: int = logging.INFO, verbose: bool = False):
        """
        Initialisiere Logger.

        Args:
            log_file: Pfad zu Logfile (None = nur Console)
            level: Logging-Level (logging.DEBUG, INFO, WARNING, ERROR)
            verbose: True = Force DEBUG level (ignoriert level Parameter)
        """
        self.log_file = log_file
        self.level = logging.DEBUG if verbose else level
        self._setup_logger()

    @classmethod
    def get_instance(
        cls, log_file: Optional[Path] = None, level: int = logging.INFO, verbose: bool = False
    ) -> "LoggerConfig":
        """
        Hole Singleton-Instanz (erstelle bei Bedarf).

        Args:
            log_file: Pfad zu Logfile (None = nur Console)
            level: Logging-Level
            verbose: True = DEBUG level

        Returns:
            LoggerConfig Singleton-Instanz
        """
        if cls._instance is None:
            cls._instance = cls(log_file, level, verbose)
        return cls._instance

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Hole zentrale Logger-Instanz.

        Returns:
            logging.Logger für w2b-Modul
        """
        if cls._logger is None:
            cls.get_instance()
        return cls._logger

    @classmethod
    def reset(cls) -> None:
        """Setze Singleton-Instanz zurück (für Tests)."""
        cls._instance = None
        cls._logger = None

    def _setup_logger(self) -> None:
        """Konfiguriere Logger mit Console- und optional File-Handler."""
        logger_instance = logging.getLogger("w2b")
        logger_instance.setLevel(self.level)
        logger_instance.handlers.clear()  # Verhindere Duplikate bei mehrfachen Calls

        # Format: [TIME] [LEVEL] [Module:Function] Message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s() | %(message)s", datefmt="%H:%M:%S"
        )

        # Console-Handler (immer aktiv, stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)

        # File-Handler (optional)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)
            logger_instance.info(
                f"Logger initialisiert: Datei={self.log_file}, Level={logging.getLevelName(self.level)}"
            )

        # Setze Klassen-Variable damit get_logger() es findet
        LoggerConfig._logger = logger_instance
