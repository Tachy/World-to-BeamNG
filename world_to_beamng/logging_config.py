"""
Zentrale Logger-Konfiguration für World-to-BeamNG.

Unterstützt:
- Ausgabe auf Console (stdout)
- Optional Ausgabe in Logfile
- Konfigurierbare Log-Level (DEBUG, INFO, WARNING, ERROR)
- Einheitliches Format für alle Module
"""

import logging
from pathlib import Path
from typing import Optional, Union

from rich.logging import RichHandler

from .progress import console

# Erlaubte Level-Namen für LOG_LEVEL (config.py) - bewusst eine eigene, feste Zuordnung statt der
# (teils veralteten/undokumentierten) String->Level-Auflösung von logging.getLevelName(), damit ein
# Tippfehler in der Umgebungsvariable einen klaren Fehler wirft statt still das Falsche zu tun.
LEVEL_NAMES = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LoggerConfig:
    """
    Zentrale Logger-Konfiguration (Singleton).

    Verwaltet den Paket-Logger "world_to_beamng" und dessen:
    - Console-Output (immer aktiv, über RichHandler an shared progress.py Console)
    - File-Output (optional)
    - Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - einzige Steuerung: der `level`-Parameter
      (config.py setzt ihn aus der Umgebungsvariable LOG_LEVEL, siehe dort)
    - Einheitliches Format
    """

    _instance: Optional["LoggerConfig"] = None
    _logger: Optional[logging.Logger] = None

    def __init__(self, log_file: Optional[Path] = None, level: Union[int, str] = logging.INFO):
        """
        Initialisiere Logger.

        Args:
            log_file: Pfad zu Logfile (None = nur Console)
            level: Logging-Level als int (logging.DEBUG/.../CRITICAL) oder Name ("DEBUG", "INFO", ...)
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
        Hole Singleton-Instanz (erstelle bei Bedarf).

        Args:
            log_file: Pfad zu Logfile (None = nur Console)
            level: Logging-Level als int oder Name (siehe __init__)

        Returns:
            LoggerConfig Singleton-Instanz
        """
        if cls._instance is None:
            cls._instance = cls(log_file, level)
        return cls._instance

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Hole zentrale Logger-Instanz.

        Returns:
            logging.Logger für world_to_beamng-Paket
        """
        if cls._logger is None:
            cls.get_instance()
        return cls._logger

    def _setup_logger(self) -> None:
        """Konfiguriere Logger mit Console- und optional File-Handler."""
        logger_instance = logging.getLogger("world_to_beamng")
        logger_instance.setLevel(self.level)
        logger_instance.handlers.clear()  # Verhindere Duplikate bei mehrfachen Calls

        # File-Formatter mit Zusatzinfos; die Konsole übernimmt Level-Farbe/-Badge über RichHandler
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s() | %(message)s",
            datefmt="%H:%M:%S",
        )

        # Console-Handler (immer aktiv, stdout) - teilt sich die Konsole mit progress.py, damit
        # Log-Zeilen sauber oberhalb aktiver Balken/Spinner erscheinen. markup=False ist Pflicht:
        # bestehende Log-Nachrichten enthalten literale eckige Klammern ("[OK]", "[i]", "[!]", ...),
        # die NICHT als rich-Markup geparst werden dürfen.
        console_handler = RichHandler(
            console=console,
            markup=False,
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(self.level)
        logger_instance.addHandler(console_handler)

        # File-Handler (optional)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(self.level)
            file_handler.setFormatter(file_formatter)
            logger_instance.addHandler(file_handler)
            logger_instance.info(
                f"Logger initialized: file={self.log_file}, level={logging.getLevelName(self.level)}"
            )

        # Geschwätzige Third-Party-Logger dämpfen: sie nutzen ebenfalls logging.getLogger(__name__)
        # und würden sonst durch den jetzt korrekt propagierenden Root-Cause-Fix mitgeloggt.
        for noisy in ("urllib3", "PIL", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Setze Klassen-Variable damit get_logger() es findet
        LoggerConfig._logger = logger_instance
