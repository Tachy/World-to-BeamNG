import subprocess
import sys
import urllib.request
from pathlib import Path
import logging
from world_to_beamng.logging_config import LoggerConfig
logger = LoggerConfig.get_logger()


def install_requirements():
    logger.info("--- Installiere Python-Abhängigkeiten ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_texconv():
    bin_dir = "bin"
    texconv_path = Path(bin_dir) / "texconv.exe"

    # Offizieller Microsoft GitHub Release Link (64-bit)
    url = "https://github.com/microsoft/DirectXTex/releases/latest/download/texconv.exe"

    if not texconv_path.exists():
        logger.info("--- Lade texconv.exe herunter ---")
        Path(bin_dir).mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, texconv_path)
        logger.info(f"Gespeichert in: {texconv_path}")
    else:
        logger.info("--- texconv.exe bereits vorhanden ---")


if __name__ == "__main__":
    install_requirements()
    download_texconv()
    logger.info("\nSetup erfolgreich abgeschlossen! Du kannst den Generator jetzt starten.")
