"""
Microsoft's texconv.exe (DirectXTex) for PNG/TIFF -> DDS conversion, downloaded on first use into bin/.

setup_project.py calls ensure_texconv() as well; the export calls it lazily right before the first conversion, so
a missing binary no longer aborts the export as long as the machine is online once.
"""

import urllib.request
from pathlib import Path

from ..logging_config import LoggerConfig

logger = LoggerConfig.get_logger()

TEXCONV_PATH = Path("bin/texconv.exe")
TEXCONV_URL = "https://github.com/microsoft/DirectXTex/releases/latest/download/texconv.exe"


def ensure_texconv(path: Path = TEXCONV_PATH, url: str = TEXCONV_URL) -> Path:
    """
    Path of texconv.exe; downloads it (atomically via a .part file) if it is missing.

    Raises:
        FileNotFoundError: if it is missing and cannot be downloaded (e.g. offline) - with the manual fallback
    """
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    logger.info(f"  [i] texconv.exe missing - downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as f:
            while chunk := response.read(1 << 16):
                f.write(chunk)
        part.replace(path)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"texconv.exe not found at {path} and the download failed ({exc}); download it manually from {url}"
        ) from exc
    logger.info(f"  [OK] texconv.exe saved to {path}")
    return path
