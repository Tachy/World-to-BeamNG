"""
Generates the gravel texture of flat roofs and stores it in data/textures/roof_gravel (checked in).

Usually not needed: the export generates it once by itself if it is missing (textures/registry.py). This script is
meant for a changed pattern (seed, grain size, colors in textures/gravel.py).

Usage: python tools/generate_gravel_texture.py [--seed 4242]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.textures.gravel import generate_gravel_texture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=4242, help="Random seed of the pattern (default 4242)")
    args = parser.parse_args()

    print(f"Gravel texture saved to {generate_gravel_texture(seed=args.seed)}")


if __name__ == "__main__":
    main()
