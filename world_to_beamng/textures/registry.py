"""
Registry of all textures from `data/textures` that the export needs, including the pre-flight check.

Each entry says which object uses the texture and where it comes from:
- procedural (`generate` set): if it is missing, it is generated once and stored in `data/textures`
  (deterministic, belongs in the repository)
- photo (`generate` = None): the pipeline cannot generate it; if it is missing, the export aborts (MissingTexturesError)
  with the command that turns a photo into it

`prepare_textures()` runs once at the start of the export (before the heavy computation), converts everything to DDS and
returns the paths for the materials; `prepared_textures()` later hands them to the materials (flat roof, wall).
A new texture = entry in REGISTRY plus a folder in `data/textures`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from .. import config
from . import library
from .concrete import generate_concrete_texture
from .gravel import generate_gravel_texture
from .steel import generate_railing_texture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextureSpec:
    name: str  # folder name in data/textures
    used_by: str  # which object uses the texture (for messages)
    required: Callable[[], bool]  # only checked while the object is being exported
    generate: Optional[Callable[[Optional[Path]], object]] = None  # None = photo texture, cannot be generated
    hint: str = ""  # command for photo textures


class MissingTexturesError(RuntimeError):
    """A photo texture required by the export is missing in data/textures."""


REGISTRY: Sequence[TextureSpec] = (
    TextureSpec(
        config.FLAT_ROOF_GRAVEL_TEXTURE,
        "flat roofs (gravel)",
        required=lambda: True,
        generate=generate_gravel_texture,
    ),
    TextureSpec(
        config.WALL_TEXTURE_NAME,
        "rubble walls (wall body and cap slabs)",
        required=lambda: config.WALLS_ENABLED,
        hint=f"python tools/make_seamless_texture.py <photo> --name {config.WALL_TEXTURE_NAME} --width-m <real width of the photo in meters>",
    ),
    TextureSpec(
        config.CONCRETE_TEXTURE_NAME,
        "bridges (piers/curbs), tunnels (walls/ceiling/portals), galleries (roof/pillars)",
        required=lambda: config.BRIDGES_ENABLED or config.TUNNELS_ENABLED,
        generate=generate_concrete_texture,
    ),
    TextureSpec(
        config.RAILING_TEXTURE_NAME,
        "bridges (railings)",
        required=lambda: config.BRIDGES_ENABLED,
        generate=generate_railing_texture,
    ),
)

_prepared: Optional[Dict[str, Dict[str, str]]] = None


def _missing_message(specs: Sequence[TextureSpec], library_dir: Path) -> str:
    lines = ["Textures required for the export are missing in data/textures (the pipeline cannot generate photo textures itself):"]
    for spec in specs:
        missing = library.missing_files(spec.name, library_dir)
        detail = ", ".join(missing) if library.load_manifest(library_dir).get(spec.name) else "no manifest entry"
        lines.append(f"  - '{spec.name}' for {spec.used_by}: missing in {library_dir / spec.name} ({detail})")
        lines.append(f"    Create with: {spec.hint}")
    return "\n".join(lines)


def prepare_textures(
    output_dir: Optional[Path] = None,
    library_dir: Optional[Path] = None,
    registry: Optional[Sequence[TextureSpec]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Checks all required textures, generates missing procedural ones once and converts all of them to DDS.

    Returns:
        {texture name: {"baseColorMap", "normalMap", "roughnessMap"}} for the required textures

    Raises:
        MissingTexturesError: at least one photo texture is missing (nothing was written into the level)
    """
    global _prepared
    specs = [spec for spec in (REGISTRY if registry is None else registry) if spec.required()]
    library_dir = Path(library_dir or config.TEXTURE_LIBRARY_DIR)

    missing_photos = []
    for spec in specs:
        if library.is_complete(spec.name, library_dir):
            continue
        if spec.generate is None:
            missing_photos.append(spec)
            continue
        logger.info(f"  [i] Texture '{spec.name}' missing, generating it once in {library_dir / spec.name} (please check it in)")
        spec.generate(library_dir)
    if missing_photos:
        raise MissingTexturesError(_missing_message(missing_photos, library_dir))

    paths = library.ensure_library_textures(output_dir, library_dir)
    result = {spec.name: paths[spec.name] for spec in specs}
    for spec in specs:
        kind = "procedural" if spec.generate else "photo"
        logger.debug(f"  [OK] Texture {spec.name:<22} {library.texture_tile_m(spec.name, 0.0, library_dir):5.2f} m  ({kind})  -> {spec.used_by}")
    _prepared = result
    return result


def prepared_textures() -> Dict[str, Dict[str, str]]:
    """Result of this export's pre-flight check; if it runs here for the first time, it is performed now."""
    return _prepared if _prepared is not None else prepare_textures()


def reset_cache() -> None:
    """Forgets the result of the pre-flight check (for tests)."""
    global _prepared
    _prepared = None
