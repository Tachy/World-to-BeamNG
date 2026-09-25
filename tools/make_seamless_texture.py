"""
Turns a photo into a seamlessly tiling texture (albedo, normal map, roughness) and stores it in data/textures/<name>.

Photo notes: frontal and perpendicular, even lighting (overcast sky), only the surface in the image (no sky, ground,
vegetation), at least approx. 2500 px wide. The real-world width of the whole photo in meters determines the scale.

Example:
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
    parser.add_argument("photo", type=Path, help="Photo (JPG/PNG)")
    parser.add_argument("--name", required=True, help="Name of the texture in data/textures, e.g. rubble_stone_wall")
    parser.add_argument("--width-m", type=float, required=True, help="real-world width of the whole photo in meters")
    parser.add_argument("--size", type=int, default=2048, help="Tile edge length in pixels (power of two, default 2048)")
    parser.add_argument("--crop", type=_crop, help="Square x,y,edge in photo pixels (default: largest centered square)")
    parser.add_argument("--blend", type=float, default=seamless.DEFAULT_BLEND, help="Blend width at the edges, 0-0.5")
    parser.add_argument("--source", default=None, help="Source note for the manifest (default: file name of the photo)")
    args = parser.parse_args()

    if args.size & (args.size - 1):
        parser.error("--size must be a power of two (BeamNG textures)")
    photo = np.asarray(Image.open(args.photo).convert("RGB"), dtype=np.uint8)
    result = seamless.build_from_photo(photo, args.width_m, args.size, args.crop, args.blend)

    folder = library.store_texture(args.name, result["maps"], result["tile_m"], args.source or f"Photo {args.photo.name}")
    print(f"Texture '{args.name}' saved to {folder} (tile {result['tile_m']:.2f} m, {args.size} px)")


if __name__ == "__main__":
    main()
