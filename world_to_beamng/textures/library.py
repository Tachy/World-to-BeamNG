"""
Textur-Bibliothek: einmalig erzeugte oder fotobasierte Texturen aus `data/textures` (Teil des Repositories) werden als
DDS ins Level geschrieben.

Aufbau: `data/textures/manifest.json` ({"textures": {name: {"tile_m": Meter je Kachel, "source": Herkunft}}}) und je
Textur ein Ordner `data/textures/<name>/` mit `color.png`, `normal.png`, `roughness.png`. Eine neue Textur ist damit
nur ein neuer Ordner samt Manifest-Eintrag. Die DDS-Dateien werden nur neu geschrieben, wenn sich die PNGs ändern.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .. import config
from ..facade import dds_export

logger = logging.getLogger(__name__)

HASH_FILE = "library_textures.hash.json"

# (Kanal = PNG-Name, Material-Schlüssel, DDS-Suffix, DDS-Format)
_CHANNELS = (
    ("color", "baseColorMap", "_b.color", dds_export.COLOR),
    ("normal", "normalMap", "_nm.normal", dds_export.NORMAL),
    ("roughness", "roughnessMap", "_r.data", dds_export.DATA),
)


def load_manifest(library_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """{Textur-Name: Manifest-Eintrag}; leer, wenn es die Bibliothek (noch) nicht gibt."""
    manifest_path = Path(library_dir or config.TEXTURE_LIBRARY_DIR) / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8")).get("textures", {})


def texture_tile_m(name: str, default: float, library_dir: Optional[Path] = None) -> float:
    """Meter je Kachel der Textur laut Manifest; `default`, wenn sie dort nicht steht."""
    entry = load_manifest(library_dir).get(name)
    return float(entry["tile_m"]) if entry and "tile_m" in entry else default


def store_texture(name: str, maps: Dict[str, np.ndarray], tile_m: float, source: str, library_dir: Optional[Path] = None) -> Path:
    """
    Legt eine Textur in der Bibliothek ab (PNGs + Manifest-Eintrag); ersetzt eine gleichnamige.

    Args:
        maps: {"color", "normal", "roughness"} als uint8-RGB-Bilder
        tile_m: reale Kantenlänge einer Kachel in Metern
        source: Herkunft (z. B. "Foto vom 21.09.2026, Bruchsteinmauer Musterort" oder "prozedural, Seed 4242")
    """
    library_dir = Path(library_dir or config.TEXTURE_LIBRARY_DIR)
    folder = library_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    for channel, *_ in _CHANNELS:
        Image.fromarray(maps[channel], "RGB").save(folder / f"{channel}.png", optimize=True)

    manifest_path = library_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"textures": {}}
    manifest["textures"][name] = {"tile_m": round(float(tile_m), 4), "source": source}
    manifest["textures"] = dict(sorted(manifest["textures"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return folder


def missing_files(name: str, library_dir: Optional[Path] = None) -> List[str]:
    """Namen der PNGs (color.png, normal.png, roughness.png), die im Ordner der Textur fehlen."""
    folder = Path(library_dir or config.TEXTURE_LIBRARY_DIR) / name
    return [f"{channel}.png" for channel, *_ in _CHANNELS if not (folder / f"{channel}.png").exists()]


def is_complete(name: str, library_dir: Optional[Path] = None) -> bool:
    """Manifest-Eintrag und alle drei PNGs sind vorhanden."""
    return name in load_manifest(library_dir) and not missing_files(name, library_dir)


def _source_hash(folder: Path) -> str:
    digest = hashlib.sha1()
    for channel, *_ in _CHANNELS:
        digest.update((folder / f"{channel}.png").read_bytes())
    return digest.hexdigest()


def ensure_library_textures(output_dir: Optional[Path] = None, library_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Schreibt die Bibliothekstexturen als DDS in den Level-Texturordner (nur bei Änderung).

    Args:
        output_dir: Zielordner; Standard config.BEAMNG_DIR_TEXTURES
        library_dir: Quellordner; Standard config.TEXTURE_LIBRARY_DIR

    Returns:
        {Textur-Name: {"baseColorMap", "normalMap", "roughnessMap"}} mit Pfaden relativ zum BeamNG-Userordner (für
        materials.json). Texturen mit fehlenden PNGs fehlen im Ergebnis (Warnung).
    """
    output_dir = Path(output_dir or config.BEAMNG_DIR_TEXTURES)
    library_dir = Path(library_dir or config.TEXTURE_LIBRARY_DIR)
    hash_path = output_dir / HASH_FILE
    known = json.loads(hash_path.read_text(encoding="utf-8")) if hash_path.exists() else {}

    result: Dict[str, Dict[str, str]] = {}
    for name in load_manifest(library_dir):
        folder = library_dir / name
        missing = missing_files(name, library_dir)
        if missing:
            logger.warning(f"  [!] Texture '{name}' skipped: {', '.join(missing)} missing in {folder}")
            continue

        source = _source_hash(folder)
        for channel, key, suffix, dds_format in _CHANNELS:
            dds_name = f"{name}{suffix}"
            if known.get(dds_name) != source or not (output_dir / f"{dds_name}.dds").exists():
                pixels = np.asarray(Image.open(folder / f"{channel}.png").convert("RGB"), dtype=np.uint8)
                dds_export.write_dds(pixels, output_dir, dds_name, dds_format, 0)  # volle Mip-Kette: kachelnde Texturen
                known[dds_name] = source
            result.setdefault(name, {})[key] = str(config.RELATIVE_DIR_TEXTURES / f"{dds_name}.dds")

    if result:
        output_dir.mkdir(parents=True, exist_ok=True)
        hash_path.write_text(json.dumps(known, indent=2, sort_keys=True), encoding="utf-8")
    return result
