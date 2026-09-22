"""
Register aller Texturen aus `data/textures`, die der Export braucht, samt Vorab-Prüfung.

Jeder Eintrag sagt, welches Objekt die Textur nutzt und woher sie kommt:
- prozedural (`generate` gesetzt): fehlt sie, wird sie einmalig erzeugt und in `data/textures` abgelegt
  (deterministisch, gehört ins Repository)
- Foto (`generate` = None): kann die Pipeline nicht erzeugen; fehlt sie, bricht der Export ab (MissingTexturesError)
  mit dem Befehl, der sie aus einem Foto macht

`prepare_textures()` läuft einmal zu Beginn des Exports (vor allem Rechenaufwand), wandelt alles in DDS um und
liefert die Pfade für die Materialien; `prepared_textures()` gibt sie später den Materialien (Flachdach, Mauer).
Eine neue Textur = Eintrag in REGISTRY plus Ordner in `data/textures`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

from .. import config
from . import library
from .concrete import generate_concrete_texture
from .gravel import generate_gravel_texture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextureSpec:
    name: str  # Ordnername in data/textures
    used_by: str  # welches Objekt die Textur nutzt (für Meldungen)
    required: Callable[[], bool]  # nur geprüft, solange das Objekt exportiert wird
    generate: Optional[Callable[[Optional[Path]], object]] = None  # None = Foto-Textur, nicht erzeugbar
    hint: str = ""  # Befehl für Foto-Texturen


class MissingTexturesError(RuntimeError):
    """Eine vom Export benötigte Foto-Textur fehlt in data/textures."""


REGISTRY: Sequence[TextureSpec] = (
    TextureSpec(
        config.FLAT_ROOF_GRAVEL_TEXTURE,
        "Flachdächer (Kies)",
        required=lambda: True,
        generate=generate_gravel_texture,
    ),
    TextureSpec(
        config.WALL_TEXTURE_NAME,
        "Bruchsteinmauern (Mauerkörper und Abdeckplatten)",
        required=lambda: config.WALLS_ENABLED,
        hint=f"python tools/make_seamless_texture.py <Foto> --name {config.WALL_TEXTURE_NAME} --width-m <reale Breite des Fotos in Metern>",
    ),
    TextureSpec(
        config.CONCRETE_TEXTURE_NAME,
        "Brücken (Pfeiler), Tunnel (Wände/Decke/Portale), Galerien (Dach/Stützen)",
        required=lambda: config.BRIDGES_ENABLED or config.TUNNELS_ENABLED,
        generate=generate_concrete_texture,
    ),
)

_prepared: Optional[Dict[str, Dict[str, str]]] = None


def _missing_message(specs: Sequence[TextureSpec], library_dir: Path) -> str:
    lines = ["Für den Export fehlen Texturen in data/textures (Foto-Texturen kann die Pipeline nicht selbst erzeugen):"]
    for spec in specs:
        missing = library.missing_files(spec.name, library_dir)
        detail = ", ".join(missing) if library.load_manifest(library_dir).get(spec.name) else "kein Manifest-Eintrag"
        lines.append(f"  - '{spec.name}' für {spec.used_by}: fehlt in {library_dir / spec.name} ({detail})")
        lines.append(f"    Erzeugen mit: {spec.hint}")
    return "\n".join(lines)


def prepare_textures(
    output_dir: Optional[Path] = None,
    library_dir: Optional[Path] = None,
    registry: Optional[Sequence[TextureSpec]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Prüft alle benötigten Texturen, erzeugt fehlende prozedurale einmalig und wandelt alle in DDS um.

    Returns:
        {Textur-Name: {"baseColorMap", "normalMap", "roughnessMap"}} für die benötigten Texturen

    Raises:
        MissingTexturesError: mindestens eine Foto-Textur fehlt (nichts wurde ins Level geschrieben)
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
        logger.info(f"  [i] Textur '{spec.name}' fehlt, erzeuge sie einmalig in {library_dir / spec.name} (bitte einchecken)")
        spec.generate(library_dir)
    if missing_photos:
        raise MissingTexturesError(_missing_message(missing_photos, library_dir))

    paths = library.ensure_library_textures(output_dir, library_dir)
    result = {spec.name: paths[spec.name] for spec in specs}
    for spec in specs:
        kind = "prozedural" if spec.generate else "Foto"
        logger.info(f"  [OK] Textur {spec.name:<22} {library.texture_tile_m(spec.name, 0.0, library_dir):5.2f} m  ({kind})  -> {spec.used_by}")
    _prepared = result
    return result


def prepared_textures() -> Dict[str, Dict[str, str]]:
    """Ergebnis der Vorab-Prüfung dieses Exports; läuft sie hier zum ersten Mal, wird sie jetzt durchgeführt."""
    return _prepared if _prepared is not None else prepare_textures()


def reset_cache() -> None:
    """Vergisst das Ergebnis der Vorab-Prüfung (für Tests)."""
    global _prepared
    _prepared = None
