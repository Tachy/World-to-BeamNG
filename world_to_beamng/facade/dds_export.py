"""
PNG -> DDS über bin/texconv.exe, in denselben Formaten wie die BeamNG-Stock-Texturen.

Farbe BC7 sRGB, Normalmap BC5, Daten (Roughness) BC4.
"""

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

TEXCONV = Path("bin/texconv.exe")

COLOR = "BC7_UNORM_SRGB"
NORMAL = "BC5_UNORM"
DATA = "BC4_UNORM"


def write_dds(pixels: np.ndarray, output_dir: Path, name: str, dds_format: str, max_mip_levels: int) -> Path:
    """
    Schreibt `pixels` (H, W, 3 uint8) als `<name>.dds` in `output_dir`.

    Args:
        pixels: RGB-Bild
        output_dir: Zielordner (wird angelegt)
        name: Dateiname ohne .dds, z. B. "windows_b.color"
        dds_format: COLOR, NORMAL oder DATA
        max_mip_levels: Länge der Mip-Kette (0 = vollständig)

    Returns:
        Pfad der DDS-Datei
    """
    if not TEXCONV.exists():
        raise FileNotFoundError(f"texconv.exe not found: {TEXCONV}")

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{name}.png"
    Image.fromarray(pixels, "RGB").save(png, "PNG")
    try:
        command = [str(TEXCONV), "-f", dds_format, "-m", str(max_mip_levels), "-y", "-o", str(output_dir)]
        if dds_format == COLOR:
            command.append("-srgb")
        command.append(str(png))
        subprocess.run(command, capture_output=True, text=True, check=True)
    finally:
        png.unlink(missing_ok=True)

    dds = output_dir / f"{name}.dds"
    if not dds.exists():
        raise FileNotFoundError(f"texconv did not create {dds}")
    return dds
