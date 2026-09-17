#!/usr/bin/env python3
"""
Validiert eine .ter-Datei strukturell (Format, Wertebereiche), OHNE BeamNG
zu starten. Findet Formatfehler in Sekunden statt nach jedem Test einen
BeamNG-Ladevorgang abzuwarten (siehe Spec Abschnitt 9).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world_to_beamng.terrain.ter_writer import read_ter, VALID_SIZES


def validate_ter(path: Path) -> bool:
    print(f"[INFO] Validiere {path}")
    heightmap, layer_map, material_names = read_ter(path)

    ok = True

    size = heightmap.shape[0]
    if size not in VALID_SIZES:
        print(f"[FEHLER] Größe {size} ist keine gültige Zweierpotenz (128-8192)")
        ok = False
    else:
        print(f"[OK] Größe: {size}x{size}")

    if heightmap.shape != layer_map.shape:
        print(f"[FEHLER] heightmap shape {heightmap.shape} != layer_map shape {layer_map.shape}")
        ok = False
    else:
        print(f"[OK] heightmap/layer_map Shapes stimmen überein")

    max_material_index = layer_map[layer_map != 255].max() if (layer_map != 255).any() else -1
    if max_material_index >= len(material_names):
        print(
            f"[FEHLER] layer_map referenziert Material-Index {max_material_index}, "
            f"aber nur {len(material_names)} Materialien vorhanden"
        )
        ok = False
    else:
        print(f"[OK] Alle layer_map-Indizes (max {max_material_index}) haben ein Material ({len(material_names)} total)")

    hole_fraction = (layer_map == 255).mean()
    print(f"[INFO] Hole-Anteil (Wert 255): {hole_fraction:.1%}")

    print(f"[INFO] Höhenwerte (u16 roh): min={heightmap.min()}, max={heightmap.max()}")
    print(f"[INFO] Materialien ({len(material_names)}): {material_names}")

    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/validate_ter.py <pfad-zur-.ter-datei>")
        sys.exit(1)

    success = validate_ter(Path(sys.argv[1]))
    print("\n[✓] VALIDIERUNG BESTANDEN" if success else "\n[!] VALIDIERUNG FEHLGESCHLAGEN")
    sys.exit(0 if success else 1)
