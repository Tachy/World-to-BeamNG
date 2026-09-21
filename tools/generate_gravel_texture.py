"""
Erzeugt die Kies-Textur der Flachdächer und legt sie in data/textures/roof_gravel ab (wird eingecheckt).

Normalerweise nicht nötig: der Export erzeugt sie selbst einmalig, falls sie fehlt (textures/registry.py). Dieses
Skript ist für ein geändertes Muster gedacht (Seed, Korngröße, Farben in textures/gravel.py).

Aufruf: python tools/generate_gravel_texture.py [--seed 4242]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.textures.gravel import generate_gravel_texture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=4242, help="Zufallsstartwert des Musters (Standard 4242)")
    args = parser.parse_args()

    print(f"Kies-Textur abgelegt in {generate_gravel_texture(seed=args.seed)}")


if __name__ == "__main__":
    main()
