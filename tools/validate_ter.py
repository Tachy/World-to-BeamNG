#!/usr/bin/env python3
"""
Validates the structure of a .ter file (format, value ranges) WITHOUT starting
BeamNG. Finds format errors in seconds instead of waiting for a BeamNG load
after every test (see spec section 9).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.terrain.ter_writer import read_ter, VALID_SIZES


def validate_ter(path: Path) -> bool:
    print(f"[INFO] Validating {path}")
    heightmap, layer_map, material_names = read_ter(path)

    ok = True

    size = heightmap.shape[0]
    if size not in VALID_SIZES:
        print(f"[ERROR] Size {size} is not a valid power of two (128-8192)")
        ok = False
    else:
        print(f"[OK] Size: {size}x{size}")

    if heightmap.shape != layer_map.shape:
        print(f"[ERROR] heightmap shape {heightmap.shape} != layer_map shape {layer_map.shape}")
        ok = False
    else:
        print(f"[OK] heightmap/layer_map shapes match")

    max_material_index = layer_map[layer_map != 255].max() if (layer_map != 255).any() else -1
    if max_material_index >= len(material_names):
        print(
            f"[ERROR] layer_map references material index {max_material_index}, "
            f"but only {len(material_names)} materials exist"
        )
        ok = False
    else:
        print(f"[OK] All layer_map indices (max {max_material_index}) have a material ({len(material_names)} total)")

    hole_fraction = (layer_map == 255).mean()
    print(f"[INFO] Hole fraction (value 255): {hole_fraction:.1%}")

    print(f"[INFO] Height values (u16 raw): min={heightmap.min()}, max={heightmap.max()}")
    print(f"[INFO] Materials ({len(material_names)}): {material_names}")

    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/validate_ter.py <path-to-.ter-file>")
        sys.exit(1)

    success = validate_ter(Path(sys.argv[1]))
    print("\n[✓] VALIDATION PASSED" if success else "\n[!] VALIDATION FAILED")
    sys.exit(0 if success else 1)
