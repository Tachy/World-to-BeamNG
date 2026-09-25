"""
One-time setup: installs the Python dependencies and downloads texconv.exe into bin/.

Only the standard library is imported before `pip install` has run (the package's logging needs `rich`).
"""

import subprocess
import sys


def install_requirements():
    print("--- Installing Python dependencies ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_texconv():
    # Imported only now: the package needs the requirements installed above
    from world_to_beamng.io.texconv import ensure_texconv

    path = ensure_texconv()
    print(f"--- texconv.exe ready: {path} ---")


if __name__ == "__main__":
    install_requirements()
    download_texconv()
    print()
    print("Setup finished successfully! You can start the generator now.")
