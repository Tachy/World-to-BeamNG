"""
Erzeugt die prozeduralen Gebäude-Texturen (Putz je Farbe, Fenster-Atlas) als DDS im Level und liefert
ihre Pfade.

Die Dateien werden nur neu geschrieben, wenn sich Farben, Layout oder Generator geändert haben (Hash-Datei).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .. import config
from . import dds_export
from .facade_styles import PLASTER_COLORS
from .plaster_texture import PLASTER_VERSION, PlasterTextureGenerator
from .window_atlas import WINDOW_ATLAS_VERSION, WindowAtlasGenerator, WindowAtlasLayout

logger = logging.getLogger(__name__)

HASH_FILE = "building_textures.hash"

# Dateien früherer Stände (Zellen-Atlas): werden beim Neuerzeugen entfernt
_OBSOLETE_FILES = ("facade_atlas_b.color.dds", "facade_atlas_nm.normal.dds", "facade_atlas_r.data.dds")

# (Schlüssel, Dateiname ohne .dds, DDS-Format, Mip-Kette voll?, Bild aus den Generator-Ergebnissen)
_Entry = Tuple[str, str, str, bool, Callable[[Dict], object]]


def _entries() -> List[_Entry]:
    entries: List[_Entry] = []
    for color in PLASTER_COLORS:
        entries.append((f"plaster_color_{color.name}", f"plaster_{color.name}_b.color", dds_export.COLOR, True, lambda g, n=color.name: g["plaster"]["albedo"][n]))
    entries += [
        ("plaster_normal", "plaster_nm.normal", dds_export.NORMAL, True, lambda g: g["plaster"]["normal"]),
        ("plaster_roughness", "plaster_r.data", dds_export.DATA, True, lambda g: g["plaster"]["roughness"]),
        ("windows_color", "windows_b.color", dds_export.COLOR, False, lambda g: g["windows"]["albedo"]),
        ("windows_normal", "windows_nm.normal", dds_export.NORMAL, False, lambda g: g["windows"]["normal"]),
        ("windows_roughness", "windows_r.data", dds_export.DATA, False, lambda g: g["windows"]["roughness"]),
    ]
    return entries


def _settings_hash() -> str:
    layout = WindowAtlasLayout()
    payload = {
        "versions": [PLASTER_VERSION, WINDOW_ATLAS_VERSION],
        "colors": [repr(color) for color in PLASTER_COLORS],
        "window_layout": [layout.px_per_m, layout.gutter_px, layout.width_px, layout.height_px],
        "plaster_px": config.FACADE_PLASTER_TEXTURE_PX,
        "green_up": config.TEXTURE_NORMAL_GREEN_UP,
        "mips": config.FACADE_MAX_MIP_LEVELS,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def ensure_building_textures(output_dir: Path = None) -> Dict[str, str]:
    """
    Stellt die Gebäude-Texturen bereit (erzeugt sie bei Bedarf).

    Args:
        output_dir: Zielordner; Standard config.BEAMNG_DIR_TEXTURES

    Returns:
        {Schlüssel: Pfad relativ zum BeamNG-Userordner für materials.json}. Schlüssel: plaster_color_<Farbe>,
        plaster_normal, plaster_roughness, windows_color/normal/roughness.
    """
    output_dir = Path(output_dir or config.BEAMNG_DIR_TEXTURES)
    entries = _entries()
    hash_path = output_dir / HASH_FILE
    settings = _settings_hash()

    up_to_date = (
        hash_path.exists()
        and hash_path.read_text(encoding="utf-8").strip() == settings
        and all((output_dir / f"{name}.dds").exists() for _, name, _, _, _ in entries)
    )
    if not up_to_date:
        _generate(output_dir, entries)
        hash_path.write_text(settings, encoding="utf-8")

    return {key: str(config.RELATIVE_DIR_TEXTURES / f"{name}.dds") for key, name, _, _, _ in entries}


def _generate(output_dir: Path, entries: List[_Entry]) -> None:
    logger.info("  [i] Erzeuge Putz- und Fenstertexturen ...")
    generated = {
        "plaster": PlasterTextureGenerator().generate(),
        "windows": WindowAtlasGenerator().generate(),
    }
    for _, name, dds_format, full_mips, image in entries:
        # Kachelnde Texturen: volle Mip-Kette (gegen Flimmern). Fenster-Atlas: begrenzt (Sprites bluten sonst ineinander).
        mip_levels = 0 if full_mips else config.FACADE_MAX_MIP_LEVELS
        dds_export.write_dds(image(generated), output_dir, name, dds_format, mip_levels)
    for obsolete in _OBSOLETE_FILES:
        (output_dir / obsolete).unlink(missing_ok=True)
    logger.info(f"  [✓] Gebäude-Texturen in {output_dir}")
