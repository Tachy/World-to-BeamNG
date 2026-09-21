"""
Macht aus einem Foto eine nahtlos kachelnde Textur (Albedo, Normalmap, Roughness) und legt sie in data/textures/<name> ab.

Foto-Hinweise: frontal und senkrecht, gleichmäßiges Licht (bedeckter Himmel), nur die Fläche im Bild (kein Himmel,
Boden, Bewuchs), mindestens ca. 2500 px breit. Die reale Breite des ganzen Fotos in Metern bestimmt den Maßstab.

Beispiel:
    python tools/make_seamless_texture.py foto.jpg --name rubble_stone_wall --width-m 1.6 --source "Foto Bruchsteinmauer"
    python tools/make_seamless_texture.py foto.jpg --name rubble_stone_wall --width-m 1.6 --crop 400,300,1800
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.textures import library, seamless


def _crop(value: str):
    x, y, side = (int(part) for part in value.split(","))
    return x, y, side


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("photo", type=Path, help="Foto (JPG/PNG)")
    parser.add_argument("--name", required=True, help="Name der Textur in data/textures, z. B. rubble_stone_wall")
    parser.add_argument("--width-m", type=float, required=True, help="reale Breite des ganzen Fotos in Metern")
    parser.add_argument("--size", type=int, default=2048, help="Kantenlänge der Kachel in Pixeln (Zweierpotenz, Standard 2048)")
    parser.add_argument("--crop", type=_crop, help="Quadrat x,y,Kante in Foto-Pixeln (Standard: größtes zentriertes Quadrat)")
    parser.add_argument("--blend", type=float, default=seamless.DEFAULT_BLEND, help="Überblendbreite an den Rändern, 0-0.5")
    parser.add_argument("--source", default=None, help="Herkunftsvermerk fürs Manifest (Standard: Dateiname des Fotos)")
    args = parser.parse_args()

    if args.size & (args.size - 1):
        parser.error("--size muss eine Zweierpotenz sein (BeamNG-Texturen)")
    photo = np.asarray(Image.open(args.photo).convert("RGB"), dtype=np.uint8)
    result = seamless.build_from_photo(photo, args.width_m, args.size, args.crop, args.blend)

    folder = library.store_texture(args.name, result["maps"], result["tile_m"], args.source or f"Foto {args.photo.name}")
    print(f"Textur '{args.name}' abgelegt in {folder} (Kachel {result['tile_m']:.2f} m, {args.size} px)")


if __name__ == "__main__":
    main()
