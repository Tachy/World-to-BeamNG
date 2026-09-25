"""
Texture library: textures from `data/textures` (part of the repository) that were generated once or are photo-based
are written into the level as DDS.

Layout: `data/textures/manifest.json` ({"textures": {name: {"tile_m": meters per tile, "source": origin}}}) and, per
texture, a folder `data/textures/<name>/` with `color.png`, `normal.png`, `roughness.png`. A new texture is thus
just a new folder plus a manifest entry. The DDS files are only rewritten when the PNGs change.
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

# (channel = PNG name, material key, DDS suffix, DDS format)
_CHANNELS = (
    ("color", "baseColorMap", "_b.color", dds_export.COLOR),
    ("normal", "normalMap", "_nm.normal", dds_export.NORMAL),
    ("roughness", "roughnessMap", "_r.data", dds_export.DATA),
)


def load_manifest(library_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """{texture name: manifest entry}; empty if the library does not exist (yet)."""
    manifest_path = Path(library_dir or config.TEXTURE_LIBRARY_DIR) / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8")).get("textures", {})


def texture_tile_m(name: str, default: float, library_dir: Optional[Path] = None) -> float:
    """Meters per tile of the texture according to the manifest; `default` if it is not listed there."""
    entry = load_manifest(library_dir).get(name)
    return float(entry["tile_m"]) if entry and "tile_m" in entry else default


def store_texture(name: str, maps: Dict[str, np.ndarray], tile_m: float, source: str, library_dir: Optional[Path] = None) -> Path:
    """
    Stores a texture in the library (PNGs + manifest entry); replaces one of the same name.

    Args:
        maps: {"color", "normal", "roughness"} as uint8 RGB images
        tile_m: real edge length of a tile in meters
        source: origin (e.g. "photo taken 2026-09-21, rubble stone wall, sample town" or "procedural, seed 4242")
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
    """Names of the PNGs (color.png, normal.png, roughness.png) that are missing in the texture's folder."""
    folder = Path(library_dir or config.TEXTURE_LIBRARY_DIR) / name
    return [f"{channel}.png" for channel, *_ in _CHANNELS if not (folder / f"{channel}.png").exists()]


def is_complete(name: str, library_dir: Optional[Path] = None) -> bool:
    """Manifest entry and all three PNGs are present."""
    return name in load_manifest(library_dir) and not missing_files(name, library_dir)


def _source_hash(folder: Path) -> str:
    digest = hashlib.sha1()
    for channel, *_ in _CHANNELS:
        digest.update((folder / f"{channel}.png").read_bytes())
    return digest.hexdigest()


def ensure_library_textures(output_dir: Optional[Path] = None, library_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Writes the library textures as DDS into the level texture folder (only on change).

    Args:
        output_dir: Target folder; default config.BEAMNG_DIR_TEXTURES
        library_dir: Source folder; default config.TEXTURE_LIBRARY_DIR

    Returns:
        {texture name: {"baseColorMap", "normalMap", "roughnessMap"}} with paths relative to the BeamNG user folder (for
        materials.json). Textures with missing PNGs are absent from the result (warning).
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
                dds_export.write_dds(pixels, output_dir, dds_name, dds_format, 0)  # full mip chain: tiling textures
                known[dds_name] = source
            result.setdefault(name, {})[key] = str(config.RELATIVE_DIR_TEXTURES / f"{dds_name}.dds")

    if result:
        output_dir.mkdir(parents=True, exist_ok=True)
        hash_path.write_text(json.dumps(known, indent=2, sort_keys=True), encoding="utf-8")
    return result
